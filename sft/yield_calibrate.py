#!/usr/bin/env python3
"""Calibrate the request-coverage judge and the deterministic checks against the reviewed 40-task sample, and count survivors.

usage: yield_calibrate.py --codex codex_a1_sample_verdicts.md --judge DIR/verdicts.jsonl --det det_flags.json --tasks tasks_F.jsonl
                          --out report.json [--subagent-wrong id,id,...]
Registered calibration rule (docs/DATA_YIELD_FACTORY_FIX.md, set before any judge call): the filter (judge not FAITHFUL OR any deterministic
flag) must catch at least 6 of the 7 tasks Codex marked WRONG and wrongly drop at most 5 of the 27 tasks Codex marked FAITHFUL.
Aggregate counts and task ids only; no task text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys


def codex_labels(path):
    lab = {}
    for l in open(path, encoding="utf-8"):
        m = re.match(r"\s*(distill_y_\d+):\s*(FAITHFUL|AMBIGUOUS|WRONG)\b", l)
        if m:
            lab[m.group(1)] = m.group(2)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subagent-wrong", default="")
    ap.add_argument("--guard-flags", required=True, help="guard_flags.json written by yield_judge.py")
    a = ap.parse_args()
    lab = codex_labels(a.codex)
    jv = {}
    for l in open(a.judge, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            jv[r["id"]] = r["verdict"]
    det = json.load(open(a.det))
    ids = [json.loads(l)["id"] for l in open(a.tasks, encoding="utf-8") if l.strip()]
    sub_wrong = {x for x in a.subagent_wrong.split(",") if x}
    hard = lambda t: [f for f in det.get(t, []) if not f.startswith("info_")]
    dropped = lambda t: jv.get(t) != "FAITHFUL" or bool(hard(t))
    flags = json.load(open(a.guard_flags))
    parsed = ("FAITHFUL", "AMBIGUOUS", "WRONG")
    bad_ids = [t for t in ids if jv.get(t) not in parsed]
    rep = {"incomplete": {"missing_or_unparsed_or_skipped_ids": bad_ids, "guard_flags": flags}, "reviewed": len(lab), "codex_counts": {k: sum(v == k for v in lab.values()) for k in ("FAITHFUL", "AMBIGUOUS", "WRONG")}}
    for name, fn in (("judge_only", lambda t: jv.get(t) != "FAITHFUL"), ("det_only", lambda t: bool(hard(t))), ("union", dropped)):
        rep[name] = {k: {"dropped": sum(fn(t) for t, v in lab.items() if v == k), "of": sum(v == k for v in lab.values())}
                     for k in ("WRONG", "AMBIGUOUS", "FAITHFUL")}
    rep["missed_wrong_ids"] = sorted(t for t, v in lab.items() if v == "WRONG" and not dropped(t))
    rep["false_drop_ids"] = sorted(t for t, v in lab.items() if v == "FAITHFUL" and dropped(t))
    rep["judge_verdict_counts_all"] = {k: sum(v == k for v in jv.values()) for k in sorted(set(jv.values()))}
    w, f = rep["union"]["WRONG"], rep["union"]["FAITHFUL"]
    complete = not bad_ids and not flags.get("missing_cost_responses") and not flags.get("budget_hit") and all(t in jv for t in lab)
    rep["run_complete"] = complete
    rep["calibration_pass"] = bool(complete and w["dropped"] >= 6 and f["dropped"] <= 5)
    exclude = {t for t, v in lab.items() if v == "WRONG"} | sub_wrong
    survive = [t for t in ids if not dropped(t) and t not in exclude]
    rep["survivors"] = {"of": len(ids), "filter_survivors": sum(not dropped(t) for t in ids), "after_excluding_reviewed_wrong": len(survive),
                        "reviewed_among_survivors": sum(t in lab for t in survive), "unreviewed_survivors": sum(t not in lab for t in survive)}
    rep["survivor_ids"] = survive
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("codex_counts", "judge_only", "det_only", "union", "missed_wrong_ids", "false_drop_ids", "run_complete", "calibration_pass", "survivors")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
