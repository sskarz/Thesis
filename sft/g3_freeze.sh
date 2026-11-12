#!/bin/bash
# Freeze the G3 training rows (after G2 batch 1 finished with floors and provenance passing): copies the arm A and arm B rows, checks they are task-disjoint between train and valid and that arm B is a subset of arm A, writes sft/g3_expected.sha.
set -eu
cd /Users/USER/Documents/GitHub/Thesis-wt-r6
PY=/Users/USER/Documents/GitHub/Thesis/.venv/bin/python
B1=/Users/USER/Documents/GitHub/Thesis-wt-data/sft/data/distill/yield/attempts/20261109_212657_g2full.AUhok0
F=sft/data/distill/r6g3/final
for a in A B; do mkdir -p $F/$a; done
for s in train valid; do cp $B1/rows_final_A/$s.jsonl $F/A/$s.jsonl; cp $B1/rows_final_B/$s.jsonl $F/B/$s.jsonl; done
$PY -B - $F <<'PYEOF'
import json, sys, hashlib
F = sys.argv[1]
def rows(a, s): return [json.loads(l) for l in open(f"{F}/{a}/{s}.jsonl") if l.strip()]
tasks = {(a, s): {r["meta"]["task_id"] for r in rows(a, s)} for a in "AB" for s in ("train", "valid")}
assert not (tasks["A", "train"] & tasks["A", "valid"]), "A train/valid share tasks"
assert not (tasks["B", "train"] & tasks["B", "valid"]), "B train/valid share tasks"
assert tasks["B", "train"] <= tasks["A", "train"] and tasks["B", "valid"] <= tasks["A", "valid"], "B tasks not a subset of A tasks"
key = lambda r: (r["meta"]["episode_id"], json.dumps(r["messages"][:-1], sort_keys=True))
for s in ("train", "valid"):
    ka = {key(r) for r in rows("A", s)}; kb = {key(r) for r in rows("B", s)}
    assert kb <= ka, f"arm B {s} rows are not a subset of arm A rows"
    print(s, "arm A rows", len(ka), "arm B rows", len(kb), "share", round(len(kb) / len(ka), 4), "tasks A", len(tasks["A", s]), "tasks B", len(tasks["B", s]))
with open("sft/g3_expected.sha", "w") as f:
    for a in "AB":
        for s in ("train", "valid"):
            f.write(f"{a} {s} {hashlib.sha256(open(f'{F}/{a}/{s}.jsonl', 'rb').read()).hexdigest()}\n")
print(open("sft/g3_expected.sha").read())
PYEOF
