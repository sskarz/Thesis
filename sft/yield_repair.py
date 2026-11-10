#!/usr/bin/env python3
"""Deterministic repair of the id_collision defect (G2 yield; advisor note 2026-11-10 09:36 ET, item 2). CPU only, no API.

The defect: the authored starting state has a backend counter (TwitterAPI.tweet_counter, TradingBot.order_counter) at or below an id that already exists, so the first object the
gold calls create overwrites an existing one. Repair: set that counter to (largest existing id in that backend's own table) + 1. Nothing else changes: initial state apart from
that one number, gold calls and user turns are byte-identical.
A repaired task is KEPT only if (1) yield_checks.check_task reports no hard flag (id_collision, empty_read_with_state_key, quoted_suffix_not_in_state) and (2) the gold calls still
execute without error (the audit's own condition). It is labelled
  R1  every gold call that is not a creation call returns exactly what it returned before the repair (pure counter fix; only the ids returned by creation calls changed), or
  R2  at least one other call returned something different (the collision had already corrupted what a later call saw; the repair changes the gold results, calls and turns not).
usage: yield_repair.py --tasks audited_all.jsonl --flags det_flags.json --out-dir DIR     (writes repaired.jsonl, repair_report.json)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yield_checks as C  # noqa: E402
import yield_render as R  # noqa: E402

BACKENDS = {  # backend -> (counter key, table key)
    "TwitterAPI": ("tweet_counter", "tweets"),
    "TradingBot": ("order_counter", "orders"),
}
HARD = ("id_collision", "empty_read_with_state_key", "quoted_suffix_not_in_state")


def table_max(table) -> int:
    ids = set()
    if isinstance(table, dict):
        for k, v in table.items():
            if str(k).isdigit():
                ids.add(int(k))
            if isinstance(v, dict):
                for kk in ("id", "order_id", "ticket_id"):
                    if isinstance(v.get(kk), int) and not isinstance(v.get(kk), bool):
                        ids.add(v[kk])
    return max(ids) if ids else 0


def repair_config(ic: dict):
    """-> (new_config, [(backend, old_counter, new_counter)]) or (None, []) if there is nothing to set."""
    new = copy.deepcopy(ic); changes = []
    for be, (ck, tk) in BACKENDS.items():
        b = new.get(be)
        if not isinstance(b, dict) or tk not in b:
            continue
        mx = table_max(b.get(tk))
        cur = b.get(ck)
        if isinstance(cur, int) and not isinstance(cur, bool) and cur <= mx:
            b[ck] = mx + 1; changes.append((be, cur, mx + 1))
        elif ck not in b and mx:
            b[ck] = mx + 1; changes.append((be, None, mx + 1))
    return (new, changes) if changes else (None, [])


def creates(call: str) -> bool:
    return bool(C.CREATE_RE.match(C.call_parts(call)[0]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True); ap.add_argument("--flags", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    flags = json.load(open(a.flags))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out, rep = [], {"id_collision_tasks": 0, "only_id_collision": 0, "repaired_kept": 0, "R1": 0, "R2": 0, "dropped": {}, "tasks": {}}
    for t in tasks:
        f = [x for x in flags.get(t["id"], []) if not x.startswith("info_")]
        if not any(x.startswith("id_collision") for x in f):
            continue
        rep["id_collision_tasks"] += 1
        if all(x.startswith("id_collision") for x in f):
            rep["only_id_collision"] += 1
        def drop(why):
            rep["dropped"][why] = rep["dropped"].get(why, 0) + 1; rep["tasks"][t["id"]] = {"kept": False, "why": why}
        new_ic, ch = repair_config(t["initial_config"])
        if new_ic is None:
            drop("no_counter_to_set"); continue
        t2 = copy.deepcopy(t); t2["initial_config"] = new_ic
        try:
            g0 = R.run_gold(t); g1 = R.run_gold(t2)
        except Exception as e:  # noqa: BLE001
            drop(f"gold_exception:{type(e).__name__}"); continue
        calls0 = [[c for c, _ in turn] for turn in g0]; calls1 = [[c for c, _ in turn] for turn in g1]
        if calls0 != calls1 or t2["question"] != t["question"] or t2["ground_truth"] != t["ground_truth"]:
            drop("calls_or_turns_changed"); continue
        if any(isinstance(r, str) and r.startswith(("Error", "Traceback")) for turn in g1 for _, r in turn):
            drop("error_result_after_repair"); continue
        f2 = [x for x in C.check_task(t2, gold=g1) if not x.startswith("info_")]
        if f2:
            drop("flag_after_repair:" + f2[0].split(":")[0]); continue
        diff = sum(1 for turn0, turn1 in zip(g0, g1) for (c0, r0), (_, r1) in zip(turn0, turn1) if not creates(c0) and r0 != r1)
        cls = "R1" if diff == 0 else "R2"
        t2["meta"] = {**(t.get("meta") or {}), "repair": {"kind": "id_counter", "class": cls, "changes": ch, "orig_id": t["id"]}}
        out.append(t2); rep["repaired_kept"] += 1; rep[cls] += 1
        rep["tasks"][t["id"]] = {"kept": True, "class": cls, "changes": ch, "changed_non_creation_results": diff}
    with open(a.out_dir / "repaired.jsonl", "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    json.dump(rep, open(a.out_dir / "repair_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "tasks"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
