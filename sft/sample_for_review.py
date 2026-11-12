#!/usr/bin/env python3
"""Draw a seeded random sample of KEPT tasks and write them as plain text for an independent semantic review.

usage: sample_for_review.py --tasks filtered/tasks_F.jsonl --n 40 --seed 9090 --out review_sample.md --ids-out review_ids.json

Each entry shows the backends, the initial state (up to 6000 characters), and for every turn the user text, the gold calls and what each call returned when re-run in the real backend. The reviewer's
question: does the gold call sequence do what the user turn asks, using only values the user, the initial state, or earlier results
provide? Verdict per task: FAITHFUL, AMBIGUOUS (a reasonable assistant could do something else), or WRONG (the gold does not match the
request). Only new synthetic tasks are shown here; nothing from any protected bank.
"""
import argparse
import json
import random
import time

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out", required=True)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    if any(not t["id"].startswith(("distill_y_", "distill_g2")) for t in tasks):
        raise SystemExit("refusing: sample must contain only new yield-check tasks (prefix distill_y_ or distill_g2)")
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(tasks, min(a.n, len(tasks))), key=lambda t: t["id"])
    with open(a.out, "w") as f:
        for t in pick:
            f.write(f"## {t['id']}  backends: {', '.join(t['involved_classes'])}\n")
            ic = json.dumps(t["initial_config"], ensure_ascii=False)
            f.write("initial_config" + ("" if len(ic) <= 6000 else f" (first 6000 of {len(ic)} chars)") + ": " + ic[:6000] + "\n")
            vid = f"{t['id']}_review_{time.time_ns()}"
            for i, (q, g) in enumerate(zip(t["question"], t["ground_truth"])):
                f.write(f"turn {i + 1} user: {q[0]['content']}\n")
                res, _ = execute_multi_turn_func_call(func_call_list=g, initial_config=t["initial_config"], involved_classes=t["involved_classes"],
                                                      model_name="distill_review", test_entry_id=vid, long_context=False, is_evaL_run=False)
                for c, r in zip(g, res):
                    f.write(f"    gold: {c}\n        result: {str(r)[:400]}\n")
            f.write("\n")
    json.dump([t["id"] for t in pick], open(a.ids_out, "w"))
    print(json.dumps({"sampled": len(pick), "of": len(tasks), "seed": a.seed}))


if __name__ == "__main__":
    main()
