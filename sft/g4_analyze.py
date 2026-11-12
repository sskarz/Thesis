#!/usr/bin/env python3
"""G4 statistics: exact two-sided McNemar per pair, the two-pair sign test, and the outcome class of the pre-registration table.

usage: g4_analyze.py PAIRS.json OUT.json
PAIRS.json = {"margin": 8, "runs": {"U1": [passed task ids], "T1": [...], "T2": [...], "U2": [...]}, "all_tasks": [ids], "stack_ok": true|false}
Pair 1 = (U1, T1), pair 2 = (T2, U2). net = tasks only the trained model solves minus tasks only the untuned model solves.
Two pairs: each task is counted once by how many of its two runs each side solved (0 to 2); net = tasks where the trained side solved
more minus tasks where the untuned side did; exact two-sided sign test on those tasks, ties dropped (paper/main.tex, glossary "Sign test").
Outcome table: docs/DISTILLATION_G4_PREREGISTRATION_DRAFT.md section 5.
"""
import json, sys
from math import comb


def binom_two_sided(k_small, n):
    """exact two-sided p of a fair-coin sign test with n informative items and the smaller count k_small"""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, k_small + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(trained, untuned, tasks):
    a = sum(1 for t in tasks if t in trained and t not in untuned)       # only trained
    b = sum(1 for t in tasks if t in untuned and t not in trained)       # only untuned
    both = sum(1 for t in tasks if t in trained and t in untuned)
    neither = len(tasks) - a - b - both
    return {"trained_only": a, "untuned_only": b, "both": both, "neither": neither, "net": a - b, "p": binom_two_sided(min(a, b), a + b)}


def sign_test(t_runs, u_runs, tasks):
    better = sum(1 for t in tasks if sum(t in r for r in t_runs) > sum(t in r for r in u_runs))
    worse = sum(1 for t in tasks if sum(t in r for r in t_runs) < sum(t in r for r in u_runs))
    return {"trained_better": better, "untuned_better": worse, "ties": len(tasks) - better - worse, "net": better - worse, "p": binom_two_sided(min(better, worse), better + worse)}


def classify(p, net, pair_nets, margin):
    both_pairs_positive = all(n > 0 for n in pair_nets)
    if p < 0.05 and net >= margin and both_pairs_positive:
        return "win"
    if p < 0.05 and net >= margin:
        return "suggestive (significant and above the margin, but a pair has net <= 0)"
    if (p < 0.05 and 0 < net < margin) or (net >= margin and p >= 0.05):
        return "suggestive"
    if p < 0.05 and net <= -margin:
        return "arm worse"
    if (p < 0.05 and -margin < net < 0) or (p >= 0.05 and net <= -margin):
        return "suggestively worse"
    return "no difference"          # p >= 0.05 and -margin < net < margin


def analyze(d):
    runs = {k: set(v) for k, v in d["runs"].items()}
    tasks = list(d["all_tasks"])
    pair1 = mcnemar(runs["T1"], runs["U1"], tasks)
    pair2 = mcnemar(runs["T2"], runs["U2"], tasks)
    comb_ = sign_test([runs["T1"], runs["T2"]], [runs["U1"], runs["U2"]], tasks)
    out = {"scores": {k: len(v) for k, v in runs.items()}, "pair1": pair1, "pair2": pair2, "combined_sign_test": comb_, "margin": d["margin"],
           "outcome": classify(comb_["p"], comb_["net"], [pair1["net"], pair2["net"]], d["margin"])}
    if not d.get("stack_ok", True):
        out["outcome"] = "STACK NOT REPRODUCED: " + out["outcome"] + " (relative to the fresh comparator only)"
    return out


if __name__ == "__main__":
    res = analyze(json.load(open(sys.argv[1])))
    json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res))
