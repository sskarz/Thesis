"""Turn verified teacher episodes into SFT rows + the build log (handoff 4, 5).

Input : a solve log dir from ``distill_solve`` (``calls.jsonl``, ``verdicts.jsonl``)
        and the task files it ran on.
Output: ``<out>/train.jsonl``, ``<out>/valid.jsonl`` (mlx-lm chat rows, one row per
        teacher decision: ``{"messages": [... , assistant target], "meta": {...}}``)
        and ``<out>/build_log.json``.

Rules
* Only calls from episodes the BFCL checker verified. Only teacher text that
  parses as an IFFC call list; the target is that list re-serialised
  (``[]`` for a stop), so no reasoning text and no code fence.
* At most ``max_samples_per_task`` verified samples per task (diversity).
* Stop rows are capped to ``stop_cap`` of all rows (seeded random drop).
* Train/valid split is by TASK, never by row.
* Leak scan (``distill_exclusions.scan_rows``) must report 0 id / 8-gram / exact
  rows or the build exits non-zero and writes nothing but the log.

    python -m benchmark.distill_rows --solve-dirs D1 D2 --out sft/data/distill/mlx-t1
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from . import distill_exclusions as X
from . import distill_tasks as DT
from .distill_solve import is_lookup
from .iffc import try_parse_iffc_response


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



_RES_LINE = re.compile(r"^- .* -> (.*)$")


def _exec_failed(next_user_content: str) -> bool:
    """True if any call in the execution-results block of the NEXT row errored."""
    if "Function execution results:" not in next_user_content:
        return False
    block = next_user_content.split("Function execution results:", 1)[1].split("\n\n", 1)[0]
    for line in block.splitlines():
        m = _RES_LINE.match(line.strip())
        if m and (m.group(1).startswith('{"error"') or m.group(1).startswith("Error during execution")
                  or '"booking_status": false' in m.group(1)):
            return True
    return False


_EXT = {"txt", "pdf", "md", "csv", "log", "json", "doc", "docx", "png", "jpg"}


def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _has_seq(corpus: list[str], seq: list[str]) -> bool:
    n = len(seq)
    return n == 0 or any(corpus[i : i + n] == seq for i in range(len(corpus) - n + 1))


def _values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _values(v)
    elif not isinstance(obj, bool) and obj is not None:
        yield obj


def ungrounded_target(messages: list[dict], parsed: list[dict]) -> list[str]:
    """Argument values in a teacher target that nothing visible at that step supports.

    Visible = every user-role message so far (task text, status cards, execution results)
    plus the system message (tool docs, for enum values). Earlier assistant text is NOT
    visible evidence, so a guess cannot justify itself. Free text of 4+ words must be
    dictated verbatim by a user message.
    """
    users = " ".join(str(m["content"]) for m in messages if m["role"] == "user")
    sysm = " ".join(str(m["content"]) for m in messages if m["role"] == "system")
    corpus = _toks(users) + _toks(sysm)
    users_norm = " ".join(users.lower().split())
    bad = []
    for call in parsed:
        for v in _values(call.get("parameters", {})):
            if isinstance(v, (int, float)) and abs(v) < 2:
                continue
            if isinstance(v, str):
                if len(v) < 3:
                    continue
                if len(v.split()) >= 4:
                    if " ".join(v.lower().split()) not in users_norm:
                        bad.append(f"{call['function_name']}: free text not dictated: {v[:40]}")
                    continue
            if isinstance(v, str) and re.search(r"\.[A-Za-z0-9]{2,4}$", v.strip()):
                # a file name with an extension must appear verbatim (listing or user text)
                if v.strip().lower() not in " ".join((users + " " + sysm).lower().split()):
                    bad.append(f"{call['function_name']}: file name not seen: {v[:40]}")
                continue
            seq = [t for t in _toks(v) if t not in _EXT]
            if not _has_seq(corpus, seq):
                bad.append(f"{call['function_name']}: {str(v)[:40]}")
    return bad


def _tool_class_map() -> dict[str, str]:
    from .generate_multi_turn import _name_to_class_map
    out: dict[str, str] = {}
    for c in DT.CLASSES:
        out.update(_name_to_class_map([c]))
    return out


def collect(solve_dirs: list[Path], max_samples_per_task: int,
            drop_tasks: set[str] | None = None) -> tuple[list[dict], dict]:
    drop_tasks = drop_tasks or set()
    cls_of = _tool_class_map()
    rows: list[dict] = []
    info = {"verified_episodes": 0, "episodes_used": 0, "tasks_used": set(),
            "api_models": set(), "api_providers": set(), "unparsed_dropped": 0,
            "quarantined_episodes_skipped": 0, "failed_action_targets_dropped": 0,
            "ungrounded_targets_dropped": 0,
            "duplicate_rows_dropped": 0}
    seen_keys: set[str] = set()
    for d in solve_dirs:
        verdicts = [json.loads(l) for l in (d / "verdicts.jsonl").read_text().splitlines() if l.strip()]
        ok = [v for v in verdicts if v.get("valid")]
        info["verified_episodes"] += len(ok)
        keep: dict[str, list[dict]] = collections.defaultdict(list)
        for v in sorted(ok, key=lambda v: (v["task_id"], v["sample"])):
            if v["task_id"] in drop_tasks:
                info["quarantined_episodes_skipped"] += 1
                continue
            if len(keep[v["task_id"]]) < max_samples_per_task:
                keep[v["task_id"]].append(v)
        used = {v["episode_id"]: v for vs in keep.values() for v in vs}
        per_ep: dict[str, list[dict]] = collections.defaultdict(list)
        for line in (d / "calls.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("episode_id") in used:
                per_ep[r["episode_id"]].append(r)
        for eid, calls in per_ep.items():
            v = used[eid]
            parsed_calls = []
            for r in calls:
                parsed = try_parse_iffc_response(r.get("response_text"))
                if parsed is None:
                    info["unparsed_dropped"] += 1
                    continue
                parsed_calls.append((r, parsed))
            exercised = sorted({cls_of[c["function_name"]] for _, p in parsed_calls for c in p
                                if c["function_name"] in cls_of})
            for i, (r, parsed) in enumerate(parsed_calls):
                if parsed and ungrounded_target(r["messages"], parsed):
                    info["ungrounded_targets_dropped"] += 1
                    continue
                # a target whose own calls errored is not a good next step
                if parsed and i + 1 < len(parsed_calls):
                    nxt_user = [m for m in parsed_calls[i + 1][0]["messages"] if m["role"] == "user"][-1]["content"]
                    if _exec_failed(str(nxt_user)):
                        info["failed_action_targets_dropped"] += 1
                        continue
                target = "[]" if not parsed else json.dumps(parsed, indent=2, ensure_ascii=False)
                msgs = [*r["messages"], {"role": "assistant", "content": target}]
                key = json.dumps(msgs, sort_keys=True)
                if key in seen_keys:
                    info["duplicate_rows_dropped"] += 1
                    continue
                seen_keys.add(key)
                names = [c["function_name"] for c in parsed]
                rows.append({
                    "messages": msgs,
                    "meta": {
                        "task_id": v["task_id"], "episode_id": eid, "sample": v["sample"],
                        "step": i, "action": "end_turn" if not parsed else "execute",
                        "n_calls": len(parsed), "lookup_calls": sum(is_lookup(n) for n in names),
                        "classes": v["classes"], "exercised": exercised,
                        "api_model": r.get("api_model"), "api_provider": r.get("api_provider"),
                        "date": r.get("ts"),
                    },
                })
                info["api_models"].add(r.get("api_model")); info["api_providers"].add(r.get("api_provider"))
        info["episodes_used"] += len(used)
        info["tasks_used"] |= set(keep)
    return rows, info


def cap_stops(rows: list[dict], cap: float, seed: int) -> tuple[list[dict], int]:
    """Drop random stop rows until stops are at most ``cap`` of all rows."""
    rng = random.Random(seed)
    stops = [i for i, r in enumerate(rows) if r["meta"]["action"] == "end_turn"]
    n, s = len(rows), len(stops)
    drop: set[int] = set()
    rng.shuffle(stops)
    while stops and (s - len(drop)) / (n - len(drop)) > cap:
        drop.add(stops.pop())
    return [r for i, r in enumerate(rows) if i not in drop], len(drop)


def mix(rows: list[dict]) -> dict:
    stop = sum(r["meta"]["action"] == "end_turn" for r in rows)
    calls = sum(r["meta"]["n_calls"] for r in rows)
    look = sum(r["meta"]["lookup_calls"] for r in rows)
    per_class = collections.Counter(c for r in rows for c in set(r["meta"].get("exercised") or r["meta"]["classes"]))
    per_task = collections.Counter(r["meta"]["task_id"] for r in rows)
    return {
        "rows": len(rows), "stop_rows": stop,
        "stop_share": round(stop / len(rows), 4) if rows else None,
        "tool_calls": calls, "lookup_calls": look,
        "lookup_share": round(look / calls, 4) if calls else None,
        "lookup_note": "name-prefix heuristic, not the autopsy regex",
        "rows_by_exercised_backend_(a_row_counts_for_each_backend_its_episode_calls)": dict(per_class),
        "tasks": len(per_task), "max_rows_one_task": max(per_task.values()) if per_task else 0,
        "unique_message_sequences": len({json.dumps(r["messages"][:-1], sort_keys=True) for r in rows}),
        "max_chars_one_row": max((sum(len(str(m["content"])) for m in r["messages"]) for r in rows), default=0),
    }


def select_subset(rows: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """Seeded, stratified null-scale subset: whole episodes round-robin over the
    rarest exercised backend first, then the next episode's first rows fill the rest."""
    rng = random.Random(seed)
    eps: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        eps[r["meta"]["episode_id"]].append(r)
    # at most one episode per task first (diversity), then a second pass
    by_task: dict[str, list[str]] = collections.defaultdict(list)
    for e, rs in eps.items():
        by_task[rs[0]["meta"]["task_id"]].append(e)
    freq = collections.Counter(c for rs in eps.values() for c in set(rs[0]["meta"].get("exercised") or []))
    queues: dict[str, list[str]] = collections.defaultdict(list)
    order = []
    for t, es in by_task.items():
        rng.shuffle(es)
        order.append(es)
    for rank in range(max(len(e) for e in order)):
        for es in order:
            if rank < len(es):
                ex = eps[es[rank]][0]["meta"].get("exercised") or ["?"]
                primary = min(ex, key=lambda c: (freq[c], c))
                queues[primary].append(es[rank])
    for q in queues.values():
        pass
    picked: list[dict] = []
    backends = sorted(queues, key=lambda c: (freq[c], c))
    taken_eps: list[str] = []
    while len(picked) < n and any(queues.values()):
        for b in backends:
            if not queues[b]:
                continue
            e = queues[b].pop(0)
            take = eps[e][: n - len(picked)]
            picked.extend(take)
            taken_eps.append(e)
            if len(picked) >= n:
                break
    return picked, {"episodes": taken_eps, "n": len(picked), "seed": seed, "mix": mix(picked)}


def literal_overlap_report(task_files: list[Path], used: set[str]) -> dict:
    ex = X.load_excluded()
    out, shared_total, vals_total = [], 0, 0
    for f in task_files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t["id"] not in used:
                continue
            lits: set[str] = set()
            for turn in t["ground_truth"]:
                for c in turn:
                    lits |= X._call_literals(c)
            sh = sorted(lits & ex["literals"])
            vals_total += len(lits); shared_total += len(sh)
            if sh:
                out.append({"task": t["id"], "shared": sh})
    return {"gold_string_literals": vals_total, "shared_with_bfcl_gold": shared_total,
            "tasks_with_shared": len(out), "detail": out,
            "note": "string literals of len>=4 only; shared values are mostly backend built-ins "
                    "(order ids, zipcodes, enums) - reviewed list, not a rejection rule"}


def token_audit(rows: list[dict]) -> dict | None:
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        p = snapshot_download("google/gemma-4-e2b-it", allow_patterns=["tokenizer*", "*.json", "*.jinja"],
                              local_files_only=True)
        tok = AutoTokenizer.from_pretrained(p)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}
    n = []
    for r in rows:
        t = tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
        n.append(len(t["input_ids"]) if hasattr(t, "keys") else len(t))
    n.sort()
    return {"rows": len(n), "median": n[len(n) // 2], "max": n[-1], "over_8192": sum(x > 8192 for x in n),
            "tokenizer": "google/gemma-4-e2b-it chat template (HF cache)"}


def build(solve_dirs: list[Path], out: Path, *, max_samples_per_task: int = 2,
          stop_cap: float = 0.25, valid_frac: float = 0.1, seed: int = 61,
          task_files: list[Path] | None = None, subset_n: int | None = None,
          manual_quarantine: dict[str, str] | None = None) -> dict:
    # independent semantic audit of every candidate task's gold (e.g. cp into a non-folder)
    quarantine: dict[str, list[str]] = {}
    for f in task_files or []:
        for line in f.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                issues = DT.audit_task(t)
                if issues:
                    quarantine[t["id"]] = issues
    rows, info = collect(solve_dirs, max_samples_per_task, set(quarantine) | set(manual_quarantine or {}))
    before = mix(rows)
    rows, dropped = cap_stops(rows, stop_cap, seed)
    tasks = sorted({r["meta"]["task_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(tasks)
    n_valid = max(1, round(len(tasks) * valid_frac)) if len(tasks) > 1 else 0
    valid_tasks = set(tasks[:n_valid])
    train = [r for r in rows if r["meta"]["task_id"] not in valid_tasks]
    valid = [r for r in rows if r["meta"]["task_id"] in valid_tasks]
    leak = X.scan_rows(rows)
    subset = None
    if subset_n:
        subset_rows, subset = select_subset(train, subset_n, seed)
    log = {
        "quarantined_tasks": quarantine,
        "failed_action_targets_dropped": info["failed_action_targets_dropped"],
        "ungrounded_targets_dropped": info["ungrounded_targets_dropped"],
        "manual_quarantine": manual_quarantine or {},
        "duplicate_rows_dropped": info["duplicate_rows_dropped"],
        "quarantined_episodes_skipped": info["quarantined_episodes_skipped"],
        "literal_overlap": literal_overlap_report(task_files or [], set(info["tasks_used"])) if task_files else None,
        "token_audit": token_audit(rows),
        "subset": ({k: subset[k] for k in ("n", "seed", "episodes", "mix")} if subset else None),
        "built_at_note": "see file mtime", "solve_dirs": [str(d) for d in solve_dirs],
        "params": {"max_samples_per_task": max_samples_per_task, "stop_cap": stop_cap,
                   "valid_frac": valid_frac, "seed": seed},
        "verified_episodes": info["verified_episodes"], "episodes_used": info["episodes_used"],
        "tasks_used": len(info["tasks_used"]), "unparsed_dropped": info["unparsed_dropped"],
        "api_models": sorted(m for m in info["api_models"] if m),
        "api_providers": sorted(m for m in info["api_providers"] if m),
        "mix_before_stop_cap": before, "stop_rows_dropped": dropped, "mix_final": mix(rows),
        "train_rows": len(train), "valid_rows": len(valid),
        "valid_tasks": sorted(valid_tasks),
        "leak_scan": leak,
        "exclusion_sha256": X.build_report()["sha256"],
        "statement": ("No BFCL multi_turn task id, user-turn 8-gram or exact user turn "
                      "appears in these rows." if not (leak["id_rows"] or leak["ngram_rows"]
                                                       or leak["exact_turn_rows"])
                      else "LEAK FOUND - rows not written"),
    }
    out.mkdir(parents=True, exist_ok=True)
    if leak["id_rows"] or leak["ngram_rows"] or leak["exact_turn_rows"]:
        (out / "build_log.json").write_text(json.dumps(log, indent=1))
        raise SystemExit("leak scan failed; see build_log.json")
    for name, rs in (("train", train), ("valid", valid)):
        p = out / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log[f"{name}_sha256"] = _sha(p)
    if subset:
        p = out / f"train_{subset_n}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in subset_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log["subset_sha256"] = _sha(p)
    (out / "build_log.json").write_text(json.dumps(log, indent=1))
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples-per-task", type=int, default=2)
    ap.add_argument("--stop-cap", type=float, default=0.25)
    ap.add_argument("--tasks", type=Path, nargs="*", default=[])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--manual-quarantine", type=Path, default=None,
                    help="json {task_id: reason}, from a human/Codex review")
    a = ap.parse_args(argv)
    log = build(a.solve_dirs, a.out, max_samples_per_task=a.max_samples_per_task,
                stop_cap=a.stop_cap, task_files=a.tasks, subset_n=a.subset,
                manual_quarantine=json.loads(a.manual_quarantine.read_text()) if a.manual_quarantine else None)
    print(json.dumps({k: log[k] for k in ("verified_episodes", "episodes_used", "tasks_used",
                                          "train_rows", "valid_rows", "mix_final", "leak_scan", "quarantined_tasks",
                                          "failed_action_targets_dropped", "ungrounded_targets_dropped", "duplicate_rows_dropped",
                                          "token_audit", "subset",
                                          "statement")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
