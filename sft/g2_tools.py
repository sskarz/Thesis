#!/usr/bin/env python3
"""G2 chain helpers (offline, no API). Subcommands:
  spend      --attempt DIR --root ATTEMPTS_ROOT [--max-attempt USD]     cost ledger of this attempt and of all earlier attempts
  progress   --author-dir D --slots-total S --target N --ratio R --floor F [--write-halfway]   per-500-slot table, halfway projection
  aggregate  --author-dir D --out-dir A                                 yield_report-compatible author/report.json, calls.jsonl, guard_flags.json, tasks_raw.jsonl
  floor      --attempt DIR --target N --floor-tasks F --floor-rows R --cap USD --out FILE          G2_REPORT.json (floors, spend, per-500 table, projection)
Spend rule: the aggregated copy <attempt>/author/log/calls.jsonl is skipped by `spend` (it duplicates the chunk logs)."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


def calls_files(root: Path):
    # <attempt>/author/log/calls.jsonl is an aggregated COPY only in the G2 layout (chunk logs live under author/<proc>/chunk_K/log); in a Part 1 style attempt it is the real log
    aggregated = any((root / "author").glob("*/chunk_*")) if (root / "author").is_dir() else False
    for p in sorted(root.rglob("calls.jsonl")):
        if aggregated and p == root / "author" / "log" / "calls.jsonl":
            continue
        yield p


def ledger(root: Path):
    tot = 0.0; n = 0; missing = 0
    for p in calls_files(root):
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            c = json.loads(l).get("cost_usd"); n += 1
            if c is None:
                missing += 1
            else:
                tot += float(c)
    flags_bad = 0; budget_hits = 0
    for p in root.rglob("guard_flags.json"):
        f = json.load(open(p))
        if f.get("missing_cost_responses"):
            flags_bad += 1                      # a response without cost: the accounting cannot be trusted
        if f.get("budget_hit"):
            budget_hits += 1                    # a cap event (the guard refused requests after a process reached its own cap): reported, NOT a cost-accounting failure (batch 2 lesson)
    return {"usd": round(tot, 4), "calls": n, "calls_missing_cost": missing, "guard_flags_bad": flags_bad, "budget_hit_events": budget_hits}


def cmd_spend(a):
    att = Path(a.attempt); root = Path(a.root)
    mine = ledger(att)
    e_usd = e_miss = e_flags = 0
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.resolve() != att.resolve() and re.match(r"^\d{8}_\d{6}_", d.name):      # real attempts only; replay_r3 is a copy of Round 3's historical logs
            e = ledger(d); e_usd += e["usd"]; e_miss += e["calls_missing_cost"]; e_flags += e["guard_flags_bad"]
    out = {"attempt_usd": mine["usd"], "attempt_calls": mine["calls"], "attempt_missing": mine["calls_missing_cost"], "attempt_flags_bad": mine["guard_flags_bad"], "attempt_budget_hit_events": mine["budget_hit_events"],
           "earlier_usd": round(e_usd, 4), "earlier_missing": e_miss, "earlier_flags_bad": e_flags}
    print(json.dumps(out))
    if mine["calls_missing_cost"] or mine["guard_flags_bad"]:
        return 6
    if a.max_attempt is not None and mine["usd"] > a.max_attempt:
        return 5
    return 0


def load_progress(author_dir: Path):
    rows = []
    for p in sorted(author_dir.glob("*/progress.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def classify(rj: dict):
    gold = sum(v for k, v in rj.items() if k.startswith("gold_"))
    dup = rj.get("dup_in_batch", 0)
    other = sum(v for k, v in rj.items() if not k.startswith(("gold_", "deadline", "api")) and k != "dup_in_batch")
    return {"dup_in_batch": dup, "gold_error": gold, "other_rejects": other, "deadline": rj.get("deadline", 0), "api": sum(v for k, v in rj.items() if k.startswith("api"))}


def cmd_progress(a):
    d = Path(a.author_dir); rows = load_progress(d)
    table = []
    for r in rows:
        c = classify(r.get("reject_reasons") or {})
        table.append({"proc": r["proc"], "chunk": r["chunk"], "slots": r["slots"], "accepted": r.get("valid"), **c, "spent_usd": r.get("spent_usd"), "rc": r["rc"]})
    done = sum(r["slots"] - r.get("deadline_skipped", 0) for r in rows)      # slots actually worked
    valid = sum((r.get("valid") or 0) for r in rows)
    dup = sum(t["dup_in_batch"] for t in table)
    summ = {"slots_worked": done, "valid_so_far": valid, "dup_in_batch_so_far": dup, "dup_share": round(dup / done, 4) if done else None,
            "accepted_share": round(valid / done, 4) if done else None}
    half = a.slots_total / 2
    proj = None
    if done >= half:
        surv_so_far = valid * a.ratio
        proj = {"slots_worked_at_projection": done, "valid_tasks": valid, "ratio_valid_to_survivor": a.ratio,
                "survivors_estimate_so_far": round(surv_so_far, 1), "projected_survivors_at_all_slots": round(surv_so_far * a.slots_total / done, 1),
                "target": a.target, "floor": a.floor,
                "flag": "PROJECTION_BELOW_FLOOR (Sans decides on a top-up; no automatic extension)" if surv_so_far * a.slots_total / done < a.floor else
                        ("below target, above floor" if surv_so_far * a.slots_total / done < a.target else "on or above target"),
                "note": "ratio = 71 survivors / 181 valid tasks in Part 1; this is a projection, not a result"}
        pf = d / "halfway_projection.json"
        if a.write_halfway and not pf.exists():
            json.dump(proj, open(pf, "w"), indent=1)
            print("HALFWAY PROJECTION WRITTEN:", json.dumps(proj))
    print(json.dumps({"summary": summ, "projection": proj}))
    if not a.quiet:
        for t in table:
            print(json.dumps(t))
    return 0


def cmd_aggregate(a):
    d = Path(a.author_dir); out = Path(a.out_dir); (out / "log").mkdir(parents=True, exist_ok=True)
    slots = valid = calls = 0; spent = 0.0; rj = collections.Counter(); miss = 0; budget = False
    with open(out / "log" / "calls.jsonl", "w", encoding="utf-8") as cf, open(out / "tasks_raw.jsonl", "w", encoding="utf-8") as tf:
        for pd in sorted(p for p in d.iterdir() if p.is_dir() and (p / "proc_report.json").exists()):
            pr = json.load(open(pd / "proc_report.json"))
            slots += pr["slots_planned"]
            ns = pr["slots_not_started"]
            if ns:
                rj["deadline_not_started" if pr["stop_reason"] == "deadline" else "deadline_cap_or_failure_not_started"] += ns
            tf.write((pd / "tasks_raw.jsonl").read_text(encoding="utf-8"))
            for ch in sorted(pd.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1])):
                if (ch / "report.json").exists():
                    r = json.load(open(ch / "report.json")); valid += r["valid"]; spent += r["spent_usd"]; calls += r["calls"]; rj.update(r["reject_reasons"])
                gf = ch / "log" / "guard_flags.json"
                if gf.exists():
                    f = json.load(open(gf)); miss += int(f.get("missing_cost_responses") or 0); budget = budget or bool(f.get("budget_hit"))
                else:
                    miss += 1
                cp = ch / "log" / "calls.jsonl"
                if cp.exists():
                    cf.write(cp.read_text(encoding="utf-8"))
    stops = {p.parent.name: json.load(open(p))["stop_reason"] for p in sorted(d.glob("*/proc_report.json"))}
    lost = sum(v for k, v in rj.items() if k.startswith("deadline"))
    json.dump({"slots": slots, "valid": valid, "reject_reasons": dict(rj), "spent_usd": round(spent, 4), "calls": calls, "processes": len(list(d.glob("*/proc_report.json"))),
               "proc_stop_reasons": stops, "slots_lost_to_deadline_or_stop": lost, "slots_lost_share": round(lost / slots, 4) if slots else None},
              open(out / "report.json", "w"), indent=1)
    json.dump({"missing_cost_responses": miss, "budget_hit": budget}, open(out / "log" / "guard_flags.json", "w"))
    print(json.dumps({"slots": slots, "valid": valid, "spent_usd": round(spent, 4), "missing_cost_flags": miss, "budget_hit": budget}))
    return 0


def cmd_floor(a):
    att = Path(a.attempt)
    kept = [l for l in open(att / "filtered" / "tasks_F.jsonl") if l.strip()]
    rows = 0
    for fn in ("train.jsonl", "valid.jsonl"):
        p = att / "rows_clean" / fn
        if p.exists():
            rows += sum(1 for l in open(p) if l.strip())
    led = ledger(att)
    half = json.load(open(att / "author" / "halfway_projection.json")) if (att / "author" / "halfway_projection.json").exists() else None
    import subprocess
    prog = subprocess.run([sys.executable, __file__, "progress", "--author-dir", str(att / "author"), "--slots-total", "1", "--target", "0", "--ratio", "0", "--floor", "0"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    rep = {"survivors": len(kept), "rows_clean": rows, "target": a.target, "floor_tasks": a.floor_tasks, "floor_rows": a.floor_rows,
           "floor_met": len(kept) >= a.floor_tasks and rows >= a.floor_rows, "target_met": len(kept) >= a.target,
           "spend": led, "g2_cap_usd": a.cap, "within_cap": led["usd"] <= a.cap and led["calls_missing_cost"] == 0,
           "halfway_projection": half, "author_progress_summary": json.loads(prog[0]) if prog else None,
           "stop_rule": "fewer than floor_tasks surviving tasks or fewer than floor_rows clean rows = stop and report; never extended automatically"}
    json.dump(rep, open(a.out, "w"), indent=1); print(json.dumps(rep))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("spend"); p.add_argument("--attempt", required=True); p.add_argument("--root", required=True); p.add_argument("--max-attempt", type=float)
    p = sub.add_parser("progress"); p.add_argument("--author-dir", required=True); p.add_argument("--slots-total", type=int, required=True); p.add_argument("--target", type=int, required=True)
    p.add_argument("--ratio", type=float, required=True); p.add_argument("--floor", type=int, required=True); p.add_argument("--write-halfway", action="store_true"); p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("aggregate"); p.add_argument("--author-dir", required=True); p.add_argument("--out-dir", required=True)
    p = sub.add_parser("floor"); p.add_argument("--attempt", required=True); p.add_argument("--target", type=int, required=True); p.add_argument("--floor-tasks", type=int, required=True)
    p.add_argument("--floor-rows", type=int, required=True); p.add_argument("--cap", type=float, required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"spend": cmd_spend, "progress": cmd_progress, "aggregate": cmd_aggregate, "floor": cmd_floor}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
