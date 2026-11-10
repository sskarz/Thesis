#!/usr/bin/env python3
"""Yield-check authoring wrapper: the stock distill_tasks.author with a wall-clock deadline and a refusal to run without cost reporting.

Why a wrapper: `python -m benchmark.distill_tasks author` has no deadline option and writes its output file only at the very end,
so killing it from outside loses every paid call. author() itself supports `deadline` (slots that start after it return reason
"deadline"). This script changes nothing in the pipeline: same prompt, validation, repair loop and dedupe as Round 5's K2 run.

usage: yield_author.py --n N --out tasks_raw.jsonl --log-dir DIR --seed S --prefix P --existing F1 F2 ... --max-cost USD
                       --workers W --deadline-minutes M --report-out FILE
Prints the author() report as JSON, then a line `spent_usd X calls N`. Requests stop at the first response without a reported cost (yield_guard.GuardedClient); exit 3 if that happened.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import distill_tasks, teacher_api  # noqa: E402
from yield_guard import GuardedClient  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--existing", type=Path, nargs="*", default=[])
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--deadline-minutes", type=float, required=True)
    ap.add_argument("--variant", choices=["b1", "b2"], default="b1", help="b2 = G2b: THEMES_B2, collision rule in the prompt, backend weights from --weights-file")
    ap.add_argument("--weights-file", type=Path, default=None)
    ap.add_argument("--report-out", type=Path, required=True, help="author() report as clean JSON (stdout/stderr may carry warnings)")
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    cfg = teacher_api.TeacherConfig.openrouter(log_dir=a.log_dir, max_cost_usd=a.max_cost, max_concurrency=a.workers)
    client = GuardedClient(cfg)
    rep = distill_tasks.author(a.n, a.out, client=client, seed=a.seed, workers=a.workers, prefix=a.prefix,
                               existing=a.existing, deadline=time.time() + a.deadline_minutes * 60,
                               focus="b2" if a.variant == "b2" else None, themes=distill_tasks.THEMES_B2 if a.variant == "b2" else None,
                               weights=json.load(open(a.weights_file)) if a.weights_file else None)
    json.dump({**rep, "spent_usd": round(client.stats["cost_usd"], 4), "calls": client.stats["calls"]}, open(a.report_out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("spent_usd", round(client.stats["cost_usd"], 4), "calls", client.stats["calls"])
    GuardedClient.write_flags(a.log_dir / "guard_flags.json")
    if GuardedClient.MISSING_COST or GuardedClient.BUDGET_HIT or (client.stats["calls"] >= 3 and client.stats["cost_usd"] <= 0):
        print("ERROR: a response without cost or the budget cap was hit; further requests were stopped; this stage is invalid", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
