#!/usr/bin/env python3
"""G2 phase 2 report: floors on TRAIN rows (plan: 'fewer than 250 kept tasks or 1,500 train rows after all filters'), arm A and arm B counts, spend ledger, halfway projection.
usage: g2_floor2.py --attempt DIR --target N --floor-tasks F --floor-train-rows R --cap USD --out FILE"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g2_tools import ledger  # noqa: E402


def count(p):
    return sum(1 for l in open(p, encoding="utf-8") if l.strip()) if Path(p).exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True); ap.add_argument("--target", type=int, required=True); ap.add_argument("--floor-tasks", type=int, required=True)
    ap.add_argument("--floor-train-rows", type=int, required=True); ap.add_argument("--rows-a-dir", default="rows_clean"); ap.add_argument("--rows-b-dir", default="rows_B"); ap.add_argument("--cap", type=float, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); att = Path(a.attempt)
    surv = count(att / "filtered" / "tasks_F.jsonl")
    tr, va = count(att / a.rows_a_dir / "train.jsonl"), count(att / a.rows_a_dir / "valid.jsonl")
    trb, vab = count(att / a.rows_b_dir / "train.jsonl"), count(att / a.rows_b_dir / "valid.jsonl")
    led = ledger(att)
    rj = json.load(open(att / "author" / "report.json")) if (att / "author" / "report.json").exists() else {}
    half = json.load(open(att / "author" / "halfway_projection.json")) if (att / "author" / "halfway_projection.json").exists() else None
    prefilter = json.load(open(att / "prefilter_report.json")) if (att / "prefilter_report.json").exists() else None
    rep = {"surviving_tasks": surv, "arm_a_rows": {"train": tr, "valid": va}, "arm_b_rows": {"train": trb, "valid": vab},
           "target": a.target, "floor_tasks": a.floor_tasks, "floor_train_rows": a.floor_train_rows,
           "floor_met": surv >= a.floor_tasks and tr >= a.floor_train_rows, "target_met": surv >= a.target,
           "arm_b_train_rows_at_floor": trb >= a.floor_train_rows,
           "spend": led, "g2_cap_usd": a.cap, "within_cap": led["usd"] <= a.cap and led["calls_missing_cost"] == 0 and led["guard_flags_bad"] == 0,
           "author_report": {k: rj.get(k) for k in ("slots", "valid", "spent_usd", "proc_stop_reasons", "slots_lost_to_deadline_or_stop", "slots_lost_share")},
           "prefilter": prefilter, "halfway_projection": half,
           "stop_rule": "fewer than floor_tasks surviving tasks or fewer than floor_train_rows arm A train rows = the data cannot test the question: report and stop; never extended automatically"}
    json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep))


if __name__ == "__main__":
    main()
