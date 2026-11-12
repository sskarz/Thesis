"""Round 3 native-format rows.

Arm A: the student's OWN successful replies. Episodes that pass the stock checker are kept as they were
generated (reasoning_content + answer, never resampled); the prompt is the exact request the server saw.
Arm B: teacher-informed rows from ``distill_hybrid`` (teacher steps in verified episodes), given a native
reasoning by ``distill_rationalize --hinted 0`` (student sampled without any hint; exact action match).

``build`` writes MLX train/valid files for A and for AB, with the SAME number of optimizer steps recommended
for both (``iters`` in the manifest), a task-disjoint validation split frozen BEFORE filtering by token length,
a leak scan on every row and a token-length audit.

    python -m benchmark.distill_native arm-a --student-dirs D1 D2 --out A.jsonl --drop-tasks FILE
    python -m benchmark.distill_native arm-b --hybrid-dirs H1 --out B_json.jsonl
    python -m benchmark.distill_native build --a A.jsonl --b B_native.jsonl --out DIR
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
from . import distill_rows as DR
from .distill_solve import is_lookup
from .iffc import try_parse_iffc_response

MAX_REASONING_CHARS = 3500
_CLS: dict[str, str] = {}
SWITCH_RE = re.compile(r"__h(\d+)__s\d+$")


def _row(r: dict, parsed: list[dict], v: dict, arm: str, step: int, native_msg: dict) -> dict:
    names = [c["function_name"] for c in parsed]
    if not _CLS:
        _CLS.update(DR._tool_class_map())
    cls_of = _CLS
    return {
        "messages": [*r["messages"], native_msg],
        "meta": {
            "task_id": v["task_id"], "episode_id": v["episode_id"], "sample": v.get("sample"),
            "step": step, "action": "end_turn" if not parsed else "execute", "n_calls": len(parsed),
            "lookup_calls": sum(is_lookup(n) for n in names), "classes": v["classes"],
            "exercised": sorted({cls_of[n] for n in names if n in cls_of}), "arm": arm,
            "api_model": r.get("api_model"),
        },
    }


def arm_a(student_dirs: list[Path], drop_tasks: set[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    info = collections.Counter()
    seen: set[str] = set()
    for d in student_dirs:
        verdicts = [json.loads(l) for l in (d / "verdicts.jsonl").read_text().splitlines() if l.strip()]
        ok = {v["episode_id"]: v for v in verdicts if v.get("valid") and not v.get("api_error")
              and v["task_id"] not in drop_tasks}
        info["episodes_total"] += len(verdicts)
        info["episodes_verified"] += len(ok)
        per_ep: dict[str, list[dict]] = collections.defaultdict(list)
        for line in (d / "calls.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("episode_id") in ok:
                per_ep[r["episode_id"]].append(r)
        for eid, calls in per_ep.items():
            parsed_calls = []
            for r in calls:
                p = try_parse_iffc_response(r.get("response_text"))
                if p is None:
                    info["unparsed_dropped"] += 1
                    continue
                parsed_calls.append((r, p))
            for i, (r, parsed) in enumerate(parsed_calls):
                reasoning = (r.get("reasoning_content") or "").strip()
                if not reasoning or len(reasoning) > MAX_REASONING_CHARS:
                    info["reasoning_missing_or_long_dropped"] += 1
                    continue
                if parsed and DR.ungrounded_target(r["messages"], parsed):
                    info["ungrounded_targets_dropped"] += 1
                    continue
                if parsed and i + 1 < len(parsed_calls):
                    nxt = [m for m in parsed_calls[i + 1][0]["messages"] if m["role"] == "user"][-1]["content"]
                    if DR._exec_failed(str(nxt)):
                        info["failed_action_targets_dropped"] += 1
                        continue
                msg = {"role": "assistant", "content": r["response_text"].strip(), "reasoning_content": reasoning}
                row = _row(r, parsed, ok[eid], "A", i, msg)
                key = json.dumps(row["messages"], sort_keys=True)
                if key in seen:
                    info["duplicate_rows_dropped"] += 1
                    continue
                seen.add(key)
                rows.append(row)
    return rows, dict(info)


def arm_b_json(hybrid_dirs: list[Path], drop_tasks: set[str], max_samples_per_task: int = 2) -> tuple[list[dict], dict]:
    """JSON-only teacher rows from verified hybrid rollouts (teacher-produced steps only), to be given a native
    reasoning by ``distill_rationalize --hinted 0``."""
    rows, info = DR.collect(hybrid_dirs, max_samples_per_task, drop_tasks)
    for r in rows:
        r["meta"]["arm"] = "B"
        m = SWITCH_RE.search(r["meta"]["episode_id"])
        r["meta"]["switch"] = int(m.group(1)) if m else None
        r["meta"]["correction_type"] = ("early_stop_fix" if r["meta"]["action"] == "execute" and r["meta"]["step"] == 0
                                        else r["meta"]["action"])
    info = {k: (sorted(v) if isinstance(v, set) else v) for k, v in info.items() if k != "tasks_used"}
    return rows, info


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(a_path: Path, b_path: Path | None, out: Path, *, stop_cap: float = 0.25, valid_frac: float = 0.1,
          seed: int = 71, max_tokens: int = 9000, n_total: int | None = None) -> dict:
    A = [json.loads(l) for l in a_path.read_text().splitlines() if l.strip()]
    B = [json.loads(l) for l in b_path.read_text().splitlines() if l.strip()] if b_path else []
    for r in A:
        r["meta"]["arm"] = "A"
    # task-disjoint validation split decided BEFORE any length filtering, over the union of tasks
    tasks = sorted({r["meta"]["task_id"] for r in A + B})
    rng = random.Random(seed)
    rng.shuffle(tasks)
    valid_tasks = set(tasks[:max(1, round(len(tasks) * valid_frac))])
    leak = X.scan_rows(A + B)
    if leak["id_rows"] or leak["ngram_rows"] or leak["exact_turn_rows"]:
        raise SystemExit("leak scan failed")
    toks = DR.token_audit(A + B) or {}
    out.mkdir(parents=True, exist_ok=True)

    # token length filter (the longest rows would not fit the memory budget)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_download(
        "google/gemma-4-e2b-it", allow_patterns=["tokenizer*", "*.json", "*.jinja"], local_files_only=True))

    def length(r):
        t = tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
        return len(t["input_ids"]) if hasattr(t, "keys") else len(t)

    def strict(r):  # the permissive IFFC parser accepts markdown fences; training targets must be a bare JSON list
        try:
            return isinstance(json.loads(r["messages"][-1]["content"]), list)
        except Exception:  # noqa: BLE001
            return False

    manual_drop = {("distill_r3_0122", 0, "B")}  # Codex r12: relies on hidden fuel level (prompt never shows it)
    nonstrict = sum(not strict(r) for r in A + B)
    A = [r for r in A if strict(r)]
    B = [r for r in B if strict(r) and (r["meta"]["task_id"], r["meta"]["step"], r["meta"]["arm"]) not in manual_drop]
    lenA = [r for r in A if length(r) <= max_tokens]
    keepB = [r for r in B if length(r) <= max_tokens]
    # B wins prompt ties: an A row whose prompt equals a B prompt is dropped (one target per prompt)
    bprompts = {json.dumps(r["messages"][:-1], sort_keys=True) for r in keepB}
    keepA = [r for r in lenA if json.dumps(r["messages"][:-1], sort_keys=True) not in bprompts]
    tie = len(lenA) - len(keepA)
    keepA, a_stop_dropped = DR.cap_stops(keepA, stop_cap, seed)
    isval = lambda r: r["meta"]["task_id"] in valid_tasks
    A_tr, A_va = [r for r in keepA if not isval(r)], [r for r in keepA if isval(r)]
    B_tr, B_va = [r for r in keepB if not isval(r)], [r for r in keepB if isval(r)]
    rng2 = random.Random(seed + 1)
    rng2.shuffle(A_tr)
    n_total = n_total or len(B_tr) + len(A_tr)
    sets = {"A": A_tr[:n_total], "AB": B_tr + A_tr[:max(0, n_total - len(B_tr))]}
    log = {"non_strict_json_rows_dropped": nonstrict, "manual_drops": sorted(map(list, manual_drop)),
           "a_in": len(A), "b_in": len(B), "a_over_len": len(A) - len(lenA),
           "a_prompt_ties_dropped": tie, "a_stop_rows_dropped": a_stop_dropped,
           "b_over_len": len(B) - len(keepB), "n_total_microbatches": n_total,
           "valid_tasks": sorted(valid_tasks), "token_audit_all": toks, "leak_scan": leak, "seed": seed,
           "stop_cap": stop_cap, "max_tokens": max_tokens, "files": {}}
    for name, tr in sets.items():
        va = A_va + B_va
        for fn, rs in ((f"{name}_train.jsonl", tr), (f"{name}_valid.jsonl", va)):
            with open(out / fn, "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log["files"][fn] = {"rows": len(rs), "sha256": _sha(out / fn)}
        log[name] = {"train": len(tr), "valid": len(va), "mix_train": DR.mix(tr),
                     "by_arm_train": dict(collections.Counter(r["meta"]["arm"] for r in tr)),
                     "tasks_train": len({r["meta"]["task_id"] for r in tr})}
    log["iters_for_equal_updates"] = n_total
    (out / "build_log.json").write_text(json.dumps(log, indent=1))
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm-a")
    a.add_argument("--student-dirs", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--drop-tasks", type=Path, default=None, help="json list of task ids to exclude")
    b = sub.add_parser("arm-b")
    b.add_argument("--hybrid-dirs", type=Path, nargs="+", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--drop-tasks", type=Path, default=None)
    c = sub.add_parser("build")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, default=None)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--n-total", type=int, default=None)
    args = ap.parse_args(argv)
    drop = set(json.loads(args.drop_tasks.read_text())) if getattr(args, "drop_tasks", None) else set()
    if args.cmd == "arm-a":
        rows, info = arm_a(args.student_dirs, drop)
    elif args.cmd == "arm-b":
        rows, info = arm_b_json(args.hybrid_dirs, drop)
    else:
        print(json.dumps(build(args.a, args.b, args.out, n_total=args.n_total), indent=1)[:4000])
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), **info}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
