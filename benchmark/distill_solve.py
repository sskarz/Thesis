"""Teacher solves validated tasks blind in the student's loop; stock checker verifies.

For every task from ``distill_tasks`` the teacher runs the same JSON-text loop
the student runs (default ``iffc-ooda-card``, the gate setup) with real tool
execution. The BFCL ``multi_turn_checker`` then compares the end state and
responses with the gold. Every model call (messages in, text out, tokens, cost,
exact API model string) is logged by ``teacher_api`` to ``calls.jsonl``; those
records are the future training rows. Nothing here edits a training file.

    python -m benchmark.distill_solve pilot --tasks T.jsonl --log-dir D --max-cost 1.5
"""

from __future__ import annotations

import argparse
import collections
import uuid
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)

from . import config, teacher_api
from .generate_multi_turn import _load_func_docs, generate_for_entry
from .iffc import try_parse_iffc_response

DEFAULT_MODE = config.FC_IFFC_OODA_CARD
# Name-matching heuristic for "look-up" calls (read-only). Not the regex the
# earlier autopsy used, so shares are indicative, not directly comparable.
LOOKUP_PREFIXES = (
    "get_", "list", "ls", "cat", "find", "grep", "tail", "head", "wc", "diff",
    "du", "pwd", "search", "view", "display", "retrieve", "check", "verify",
    "show", "read", "estimate", "sort",
)


def is_lookup(name: str) -> bool:
    return name.lower().startswith(LOOKUP_PREFIXES)


# One nonce per process: backend instances are cached by (model name, entry id),
# so re-running the same task ids in one interpreter must not reuse old state.
RUN_NONCE = uuid.uuid4().hex[:8]


def _entry(task: dict) -> dict:
    return {k: task[k] for k in ("id", "question", "initial_config", "involved_classes")}


def solve_one(task: dict, client: teacher_api.TeacherClient, fc_mode: str,
              sample: int = 0, temperature: float | None = None) -> dict:
    """One attempt. Backend instances are cached by entry id, so every sample
    runs under its own id (``<task>__s<k>``); the gold is shared."""
    entry = _entry(task)
    task_id = entry["id"]
    entry["id"] = f"{task_id}__s{sample}"
    call_nonce = uuid.uuid4().hex[:8]  # per invocation: backend instances never reused
    t0 = time.time()
    with client.episode(entry["id"], kind="solve", fc_mode=fc_mode, task_id=task_id,
                        sample=sample, temperature=temperature):
        turn_results, timing = generate_for_entry(
            entry, "multi_turn_base", _load_func_docs(entry["involved_classes"]),
            execution_model_name=f"distill_solve_{call_nonce}", fc_mode=fc_mode)
    try:
        res = multi_turn_checker(
            multi_turn_model_result_list_decoded=turn_results,
            multi_turn_ground_truth_list=task["ground_truth"], test_entry=entry,
            test_category="multi_turn_base", model_name=f"distill_{call_nonce}")
        valid, err = bool(res.get("valid")), res.get("error_message", res.get("error"))
    except Exception as e:  # noqa: BLE001
        valid, err = False, f"checker_exception:{e}"
    api_error = any(isinstance(f, str) and f.startswith("ERROR:")
                    for turn in turn_results for step in turn for f in step)
    return {
        "episode_id": entry["id"], "task_id": task_id, "sample": sample,
        "temperature": 0.0 if temperature is None else temperature,
        "valid": valid, "api_error": api_error,
        "error": None if valid else (err if isinstance(err, str) else json.dumps(err))[:300],
        "flags": json.loads(timing["flags"]) if timing.get("flags") else [],
        "api_calls": timing["api_calls"], "wall_s": round(time.time() - t0, 2),
        "classes": task["involved_classes"], "n_turns": len(task["question"]),
        "result": turn_results,
    }


def decision_mix(calls_path: Path, solve_ids: set[str]) -> dict:
    """Stop-target and look-up shares over the teacher's assistant outputs."""
    stop = act = unparsed = 0
    names: collections.Counter = collections.Counter()
    for line in calls_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("episode_id") not in solve_ids:
            continue
        parsed = try_parse_iffc_response(r.get("response_text"))
        if parsed is None:
            unparsed += 1
        elif parsed == []:
            stop += 1
        else:
            act += 1
            for c in parsed:
                names[c["function_name"]] += 1
    n_calls = sum(names.values())
    n_look = sum(v for k, v in names.items() if is_lookup(k))
    rows = stop + act
    return {
        "assistant_outputs": rows + unparsed, "stop_rows": stop, "act_rows": act,
        "unparsed_outputs": unparsed,
        "stop_share": round(stop / rows, 4) if rows else None,
        "tool_calls": n_calls, "lookup_calls": n_look,
        "lookup_share": round(n_look / n_calls, 4) if n_calls else None,
        "top_tools": names.most_common(12),
        "note": "lookup = name-prefix heuristic (LOOKUP_PREFIXES), not the autopsy regex",
    }


def pilot(tasks_path: Path, log_dir: Path, *, max_cost: float, max_minutes: float,
          workers: int, fc_mode: str, limit: int | None,
          base_url: str | None = None, k: int = 1, sample_temp: float = 0.7) -> dict:
    tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
    if limit:
        tasks = tasks[:limit]
    extra = {"base_url": base_url} if base_url else {}
    cfg = teacher_api.TeacherConfig.openrouter(
        log_dir=log_dir, max_cost_usd=max_cost, max_concurrency=workers, **extra)
    client = teacher_api.TeacherClient(cfg)
    teacher_api.install(client)  # route generate_multi_turn posts to the teacher
    deadline = time.time() + max_minutes * 60
    t0 = time.time()
    stop_reason: list[str] = []

    def work(job):
        task, sample = job
        eid = f"{task['id']}__s{sample}"
        if stop_reason:
            return {"episode_id": eid, "valid": False, "error": "skipped:" + stop_reason[0]}
        if time.time() > deadline:
            stop_reason.append("deadline")
            return {"episode_id": eid, "valid": False, "error": "skipped:deadline"}
        if client.stats["calls"] >= 5 and client.stats["cost_usd"] == 0:
            stop_reason.append("cost_not_reported")  # guard would not work
            return {"episode_id": eid, "valid": False, "error": "skipped:cost_not_reported"}
        try:
            # sample 0 is greedy like the student's inference; the rest are sampled
            return solve_one(task, client, fc_mode, sample,
                             None if sample == 0 else sample_temp)
        except teacher_api.BudgetExceeded:
            stop_reason.append("budget")
            return {"episode_id": eid, "valid": False, "error": "skipped:budget"}

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            verdicts = list(ex.map(work, [(t, k_) for t in tasks for k_ in range(k)]))
    finally:
        teacher_api.install(None)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "verdicts.jsonl", "w") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    done = [v for v in verdicts if "wall_s" in v]
    ids = {v["episode_id"] for v in done}
    eps = [json.loads(l) for l in (log_dir / "episodes.jsonl").read_text().splitlines()
           if l.strip()]
    eps = [e for e in eps if e["episode_id"] in ids]
    n = len(done)
    per_class = collections.defaultdict(lambda: [0, 0])
    for v in done:
        for c in v["classes"]:
            per_class[c][1] += 1
            per_class[c][0] += int(v["valid"])
    by_task = collections.defaultdict(list)
    for v in done:
        by_task[v["task_id"]].append(v["valid"])
    report = {
        "fc_mode": fc_mode, "tasks_in": len(tasks), "samples_per_task": k,
        "sample_temp": sample_temp, "episodes_run": n,
        "verified": sum(v["valid"] for v in done),
        "tasks_with_any_verified": sum(any(x) for x in by_task.values()),
        "greedy_sample0_verified": sum(v["valid"] for v in done if v["sample"] == 0),
        "stopped_early": stop_reason or None,
        "wall_s_total": round(time.time() - t0, 1),
        "wall_s_per_episode_mean": round(sum(v["wall_s"] for v in done) / n, 1) if n else None,
        "api_models": sorted({m for e in eps for m in e["api_models"]}),
        "api_providers": sorted({m for e in eps for m in e.get("api_providers", [])}),
        "tokens": {k: sum(e[k] for e in eps) for k in ("prompt_tokens", "completion_tokens",
                                                       "prompt_cache_hit_tokens")},
        "api_calls": sum(e["calls"] for e in eps),
        "reasoning_chars_stripped": sum(e["reasoning_chars_stripped"] for e in eps),
        "cost_usd_solve": round(sum(e["cost_usd"] for e in eps), 4),
        "cost_usd_all_calls_this_process": round(client.stats["cost_usd"], 4),
        "episodes_with_api_error": sum(1 for v in done if v.get("api_error")),
        "episodes_with_flags": sum(1 for v in done if v["flags"]),
        "flag_kinds": dict(collections.Counter(
            str(f).split(":")[0] for v in done for f in v["flags"])),
        "fail_errors": [v["error"][:140] for v in done if not v["valid"]][:10],
        "per_class_verified_of_run": {k: tuple(v) for k, v in per_class.items()},
        "decision_mix": decision_mix(log_dir / "calls.jsonl", ids) if n else None,
    }
    (log_dir / "pilot_report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--max-cost", type=float, required=True)
    p.add_argument("--max-minutes", type=float, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fc-mode", default=DEFAULT_MODE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--sample-temp", type=float, default=0.7)
    a = ap.parse_args(argv)
    rep = pilot(a.tasks, a.log_dir, max_cost=a.max_cost, max_minutes=a.max_minutes,
                workers=a.workers, fc_mode=a.fc_mode, limit=a.limit,
                k=a.k, sample_temp=a.sample_temp)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
