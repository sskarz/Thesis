#!/usr/bin/env python3
"""Tests for g4_analyze.py against numbers printed in the paper (paper/main.tex, shorter-list tests): pair 1 26 against 13 gives p = 0.053 (0.053252),
pair 2 26 against 10 gives p = 0.011, the two-pair sign test 33 against 17 gives p = 0.033, pooled McNemar 52 against 23 gives p = 0.001."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g4_analyze import binom_two_sided, classify, analyze

assert abs(binom_two_sided(13, 39) - 0.053252) < 5e-6, binom_two_sided(13, 39)
assert round(binom_two_sided(10, 36), 3) == 0.011, binom_two_sided(10, 36)
assert round(binom_two_sided(17, 50), 3) == 0.033, binom_two_sided(17, 50)
assert round(binom_two_sided(23, 75), 3) == 0.001
assert binom_two_sided(0, 0) == 1.0 and binom_two_sided(5, 10) == 1.0
M = 8
# every cell of the outcome table, including the cells the Codex review found missing
assert classify(0.01, 12, [6, 6], M) == "win"
assert classify(0.01, 12, [14, 0], M).startswith("suggestive (significant")
assert classify(0.03, 5, [3, 2], M) == "suggestive" and classify(0.2, 9, [5, 4], M) == "suggestive"
assert classify(0.3, 3, [2, 1], M) == "no difference" and classify(0.5, -7, [-4, -3], M) == "no difference" and classify(1.0, 0, [0, 0], M) == "no difference"
assert classify(0.01, -9, [-5, -4], M) == "arm worse"
assert classify(0.04, -3, [-2, -1], M) == "suggestively worse" and classify(0.2, -10, [-5, -5], M) == "suggestively worse"
# an end-to-end case with made-up task sets: trained solves 1..100 and 101..110 in both runs, untuned solves 1..100 in both; net +10 in both pairs
tasks = [f"t{i}" for i in range(200)]
base = {f"t{i}" for i in range(100)}; plus = {f"t{i}" for i in range(100, 110)}
res = analyze({"margin": 8, "all_tasks": tasks, "stack_ok": True, "runs": {"U1": base, "T1": base | plus, "T2": base | plus, "U2": base}})
assert res["pair1"]["net"] == 10 and res["pair2"]["net"] == 10 and res["combined_sign_test"]["net"] == 10 and res["combined_sign_test"]["p"] < 0.01 and res["outcome"] == "win", res
res = analyze({"margin": 8, "all_tasks": tasks, "stack_ok": False, "runs": {"U1": base, "T1": base, "T2": base, "U2": base}})
assert res["outcome"].startswith("STACK NOT REPRODUCED: no difference"), res
print("test_g4_stats: OK")
