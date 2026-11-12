"""On-policy relabeling: the teacher labels the next action at the STUDENT's own states.

Why: teacher-only trajectories have clean histories and few recoveries, but at test time the
student's history holds its own persisted reasoning and its own mistakes. Here the base student
plays authored tasks (``distill_student_eval``); for every student request (system + history +
status card, exactly as the student saw it) the teacher returns the next action. Each label is
checked against the task gold before it becomes a row:

* ``stop`` label ([])  : kept only if the stock BFCL checker already accepts the state reached so
  far in this turn (turn is complete). A stop earlier than that is dropped (``stop_early``).
* ``act`` label        : replayed from the student's executed calls in a fresh backend; dropped
  if any call errors (``act_error``), is ungrounded (``ungrounded``), has a bad copy/move
  destination (``semantic``), or if the turn is already complete (``act_when_done``).

Rows keep the student's real messages. Target = teacher JSON (``[]`` for stop).

    python -m benchmark.distill_onpolicy label --tasks T --student-dir D --out O --max-cost 1.0
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

from . import teacher_api
from .distill_rows import ungrounded_target
from .distill_tasks import _is_error, semantic_issue
from .iffc import (IFFC_RETRY_USER_MESSAGE, assistant_text, iffc_calls_to_func_strings,
                   try_parse_iffc_response)


def turn_index(messages: list[dict]) -> int:
    """0-based BFCL turn of a request = number of task-text user messages so far - 1."""
    n = sum(1 for m in messages if m["role"] == "user"
            and "Function execution results:" not in str(m["content"])
            and str(m["content"]).strip() != IFFC_RETRY_USER_MESSAGE.strip())
    return max(n - 1, 0)


def align(records: list[dict], flat_steps: list[list[str]]) -> list[dict]:
    """Attach, to each logged request, the executed steps that happened BEFORE it."""
    out, j = [], 0
    for r in records:
        parsed = try_parse_iffc_response(assistant_text(
            {"content": r.get("response_text"), "reasoning_content": r.get("reasoning_content")}))
        out.append({"rec": r, "req_index": len(out), "turn": turn_index(r["messages"]), "executed_before": j,
                    "student_calls": parsed})
        if parsed:
            fs = iffc_calls_to_func_strings(parsed)
            if j < len(flat_steps) and flat_steps[j] == fs:
                j += 1
    return out


def _replay(task: dict, calls_by_step: list[list[str]], extra: list[str] | None = None):
    """Fresh backend; run the executed steps; optionally run ``extra`` and return its results."""
    uid = f"{task['id']}_op_{uuid.uuid4().hex[:8]}"
    classes = list(task["involved_classes"])
    for step in calls_by_step:
        execute_multi_turn_func_call(func_call_list=step, initial_config=task["initial_config"],
                                     involved_classes=classes, model_name="distill_op",
                                     test_entry_id=uid, long_context=False, is_evaL_run=False)
    if extra is None:
        return None
    res, _ = execute_multi_turn_func_call(
        func_call_list=extra, initial_config=task["initial_config"], involved_classes=classes,
        model_name="distill_op", test_entry_id=uid, long_context=False, is_evaL_run=False)
    return res


def turn_complete(task: dict, result: list[list[list[str]]], flat_steps: list[list[str]],
                  executed_before: int, turn: int, step_turns: list[int]) -> bool:
    """Would the stock checker accept the state after the steps executed so far, for turns 0..turn?"""
    per_turn: list[list[list[str]]] = [[] for _ in range(turn + 1)]
    for s, t in zip(flat_steps[:executed_before], step_turns[:executed_before]):
        if t <= turn:
            per_turn[t].append(s)
    entry = {k: task[k] for k in ("id", "question", "initial_config", "involved_classes")}
    entry["id"] = f"{task['id']}_chk_{uuid.uuid4().hex[:8]}"
    try:
        res = multi_turn_checker(
            multi_turn_model_result_list_decoded=per_turn,
            multi_turn_ground_truth_list=task["ground_truth"][: turn + 1], test_entry=entry,
            test_category="multi_turn_base", model_name=f"distill_opchk_{uuid.uuid4().hex[:6]}")
        return bool(res.get("valid"))
    except Exception:  # noqa: BLE001
        return False


def check_label(task: dict, a: dict, flat_steps: list[list[str]], step_turns: list[int],
                label: list[dict] | None) -> tuple[str, str]:
    """Return (decision, kind). decision keep/drop; kind explains."""
    if label is None:
        return "drop", "unparsed"
    done = turn_complete(task, None, flat_steps, a["executed_before"], a["turn"], step_turns)
    if not label:  # teacher says stop
        return ("keep", "stop_correct") if done else ("drop", "stop_early")
    if done:
        return "drop", "act_when_done"
    if ungrounded_target(a["rec"]["messages"], label):
        return "drop", "ungrounded"
    calls = iffc_calls_to_func_strings(label)
    try:
        res = _replay(task, flat_steps[: a["executed_before"]], calls)
    except Exception:  # noqa: BLE001
        return "drop", "act_error"
    for c, r in zip(calls, res):
        if _is_error(r):
            return "drop", "act_error"
        if semantic_issue(c, r):
            return "drop", "semantic"
    return "keep", "act_ok"


def same_effect(task: dict, flat_steps: list[list[str]], executed_before: int,
                a_calls: list[str], b_calls: list[str]) -> bool:
    """Do two proposed call lists give identical execution results from the same replayed state?"""
    try:
        ra = _replay(task, flat_steps[:executed_before], a_calls)
        rb = _replay(task, flat_steps[:executed_before], b_calls)
    except Exception:  # noqa: BLE001
        return False
    return [str(x) for x in ra] == [str(x) for x in rb]


def label_dir(tasks_path: Path, student_dir: Path, out: Path, client, workers: int = 6,
              exclude_tasks: set[str] | None = None) -> dict:
    tasks = {}
    for l in tasks_path.read_text().splitlines():
        if l.strip():
            t = json.loads(l)
            tasks[t["id"]] = t
    verdicts = {json.loads(l)["episode_id"]: json.loads(l)
                for l in (student_dir / "verdicts.jsonl").read_text().splitlines() if l.strip()}
    recs = collections.defaultdict(list)
    for l in (student_dir / "calls.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r.get("episode_id") in verdicts:
            recs[r["episode_id"]].append(r)
    jobs, unconsumed = [], []
    for eid, rs in recs.items():
        v = verdicts[eid]
        task = tasks.get(v["task_id"])
        if task is None or v["task_id"] in (exclude_tasks or set()):
            continue
        flat, step_turns = [], []
        for ti, turn in enumerate(v["result"]):
            for st in turn:
                if isinstance(st, list) and st and not str(st[0]).startswith("ERROR"):
                    flat.append(st); step_turns.append(ti)
        aligned = align(rs, flat)
        if aligned and aligned[-1]["executed_before"] + (1 if aligned[-1]["student_calls"] else 0) < len(flat):
            unconsumed.append(v["episode_id"])
        for a in aligned:
            jobs.append((task, v, a, flat, step_turns))

    def work(job):
        task, v, a, flat, step_turns = job
        try:
            with client.episode(f"label_{v['episode_id']}_{a['executed_before']}", kind="label"):
                data = client.post_chat({"messages": a["rec"]["messages"], "temperature": 0.0})
            text = data["choices"][0]["message"].get("content") or ""
        except teacher_api.BudgetExceeded:
            return {"drop": "budget"}
        except Exception as e:  # noqa: BLE001
            return {"drop": f"api:{type(e).__name__}"}
        label = try_parse_iffc_response(text)
        dec, kind = check_label(task, a, flat, step_turns, label)
        same = label is not None and a["student_calls"] is not None and \
            iffc_calls_to_func_strings(label) == iffc_calls_to_func_strings(a["student_calls"])
        equiv = same
        if not same and dec == "keep" and label and a["student_calls"]:
            equiv = same_effect(task, flat, a["executed_before"], iffc_calls_to_func_strings(label),
                                iffc_calls_to_func_strings(a["student_calls"]))
        return {"decision": dec, "kind": kind, "same_as_student": same, "same_effect": equiv, "label": label,
                "student_calls": a["student_calls"], "task_id": v["task_id"],
                "episode_id": v["episode_id"], "turn": a["turn"],
                "step": a["executed_before"], "req_index": a["req_index"], "messages": a["rec"]["messages"],
                "classes": task["involved_classes"]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(work, jobs))
    out.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            if "decision" in r:
                stats[r["kind"]] += 1
                stats["same_as_student" if r["same_as_student"] else "differs_from_student"] += 1
                if r["same_effect"] and not r["same_as_student"]:
                    stats["differs_but_same_effect"] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                stats[r["drop"]] += 1
    rep = {"requests": len(jobs), "unconsumed_alignment_episodes": unconsumed, "stats": dict(stats),
           "kept": stats["stop_correct"] + stats["act_ok"]}
    (out / "label_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def build_rows(labels_path: Path, out: Path, *, n_stop_same: int = 30, n_act_same: int = 6,
               valid_tasks_n: int = 8, seed: int = 61, max_tokens: int = 8192,
               drop_rows: set[tuple[str, int]] | None = None) -> dict:
    """Training rows from gold-checked labels: all corrections + a sample of confirmed-correct rows."""
    import random
    from . import distill_exclusions as X
    from .distill_rows import mix, token_audit
    rng = random.Random(seed)
    L = [json.loads(l) for l in labels_path.read_text().splitlines() if l.strip()]
    K = [x for x in L if x["decision"] == "keep"
         and (x["task_id"], x["step"]) not in (drop_rows or set())]
    seen, rows = set(), []
    for x in K:
        tgt = "[]" if not x["label"] else json.dumps(x["label"], indent=2, ensure_ascii=False)
        msgs = [*x["messages"], {"role": "assistant", "content": tgt}]
        key = json.dumps(msgs, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        last_user = [m for m in x["messages"] if m["role"] == "user"][-1]["content"]
        rows.append({"messages": msgs, "meta": {
            "task_id": x["task_id"], "episode_id": x["episode_id"], "step": x["step"], "turn": x["turn"],
            "action": "end_turn" if not x["label"] else "execute", "kind": x["kind"],
            "same_as_student": bool(x.get("same_effect", x["same_as_student"])), "classes": x["classes"], "exercised": [],
            "n_calls": len(x["label"] or []), "lookup_calls": 0,
            "after_error": ('-> {"error"' in str(last_user) or "-> Error during" in str(last_user)),
            "student_stopped": x["student_calls"] == []}})
    tok = token_audit(rows)
    # drop rows over the training window (a truncated row would train on a cut prompt)
    lens = _token_lengths(rows)
    over = {i for i, n in enumerate(lens) if n is not None and n > max_tokens}
    rows = [r for i, r in enumerate(rows) if i not in over]
    tasks = sorted({r["meta"]["task_id"] for r in rows})
    rng.shuffle(tasks)
    valid_t = set(tasks[:valid_tasks_n])
    valid = [r for r in rows if r["meta"]["task_id"] in valid_t]
    pool = [r for r in rows if r["meta"]["task_id"] not in valid_t]
    corr = [r for r in pool if not r["meta"]["same_as_student"]]
    ss = [r for r in pool if r["meta"]["same_as_student"] and r["meta"]["action"] == "end_turn"]
    sa = [r for r in pool if r["meta"]["same_as_student"] and r["meta"]["action"] == "execute"]
    rng.shuffle(ss); rng.shuffle(sa)
    train = corr + ss[:n_stop_same] + sa[:n_act_same]
    rng.shuffle(train)
    leak = X.scan_rows(train + valid)
    log = {"labels": str(labels_path), "kept_labels": len(K), "unique_rows": len(rows) + len(over),
           "dropped_over_token_limit": len(over), "token_audit_all": tok,
           "train_rows": len(train), "valid_rows": len(valid), "valid_tasks": sorted(valid_t),
           "train_composition": {"corrections": len(corr), "student_stopped_early_fixed": sum(r["meta"]["student_stopped"] and r["meta"]["action"] == "execute" for r in corr),
                                 "recovery_after_error": sum(r["meta"]["after_error"] and r["meta"]["action"] == "execute" for r in train),
                                 "stop_same": len(ss[:n_stop_same]), "act_same": len(sa[:n_act_same])},
           "train_mix": mix(train), "leak_scan": leak,
           "train_tasks": len({r["meta"]["task_id"] for r in train})}
    out.mkdir(parents=True, exist_ok=True)
    if leak["id_rows"] or leak["ngram_rows"] or leak["exact_turn_rows"]:
        (out / "build_log.json").write_text(json.dumps(log, indent=1))
        raise SystemExit("leak scan failed")
    import hashlib
    for name, rs in (("train", train), ("valid", valid)):
        pth = out / f"{name}.jsonl"
        with open(pth, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log[f"{name}_sha256"] = hashlib.sha256(pth.read_bytes()).hexdigest()
    (out / "build_log.json").write_text(json.dumps(log, indent=1))
    return log


def _token_lengths(rows: list[dict]) -> list[int | None]:
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        p = snapshot_download("google/gemma-4-e2b-it", allow_patterns=["tokenizer*", "*.json", "*.jinja"],
                              local_files_only=True)
        tok = AutoTokenizer.from_pretrained(p)
    except Exception:  # noqa: BLE001
        return [None] * len(rows)
    out = []
    for r in rows:
        t = tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
        out.append(len(t["input_ids"]) if hasattr(t, "keys") else len(t))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label")
    l.add_argument("--tasks", type=Path, required=True)
    l.add_argument("--student-dir", type=Path, required=True)
    l.add_argument("--out", type=Path, required=True)
    l.add_argument("--max-cost", type=float, required=True)
    l.add_argument("--workers", type=int, default=6)
    l.add_argument("--exclude-tasks", default="")
    b = sub.add_parser("build")
    b.add_argument("--labels", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n-stop-same", type=int, default=30)
    b.add_argument("--n-act-same", type=int, default=6)
    b.add_argument("--max-tokens", type=int, default=8192)
    b.add_argument("--drop-rows", default="", help="comma list task_id:step to drop")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build_rows(a.labels, a.out, n_stop_same=a.n_stop_same,
                                    n_act_same=a.n_act_same, max_tokens=a.max_tokens,
                                    drop_rows={(x.split(':')[0], int(x.split(':')[1])) for x in a.drop_rows.split(',') if x}), indent=1))
        return 0
    cfg = teacher_api.TeacherConfig.openrouter(log_dir=a.out, max_cost_usd=a.max_cost,
                                               max_concurrency=a.workers)
    client = teacher_api.TeacherClient(cfg)
    print(json.dumps(label_dir(a.tasks, a.student_dir, a.out, client, a.workers,
                                 {x for x in a.exclude_tasks.split(',') if x}), indent=1))
    print("spent_usd", round(client.stats["cost_usd"], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
