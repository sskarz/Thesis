#!/bin/bash
# G3c: build and freeze the COMBINED training rows (batch 1 + batch 2) for both arms, CPU only. Writes sft/g3c_expected.sha.
#   arm A rows = batch 1 final A (rows_final_A) + batch 2 rows_clean;  arm B rows = batch 1 final B (rows_final_B) + batch 2 rows_B minus the 4 rows whose
#   teacher sentences share an 8-gram with a held-out task (batch 2 provenance, positions train 12, 16, 278 and valid 100 of rows_B; results log 8b).
#   Validation split: batch 1's validation tasks stay in validation, batch 2's 18 validation tasks are added; train and validation are task-disjoint.
# usage: bash sft/g3c_freeze.sh [OUTDIR]     (OUTDIR default Thesis-wt-r6/sft/data/distill/r6g3c/final; the folder is gitignored data, outside the arm B attempt)
set -eu
ROOT=/Users/USER/Documents/GitHub
PY=$ROOT/Thesis/.venv/bin/python
SRC=$ROOT/Thesis-wt-training
OUT=${1:-$ROOT/Thesis-wt-r6/sft/data/distill/r6g3c/final}
Y=$ROOT/Thesis-wt-data/sft/data/distill/yield/attempts
B1=$Y/20261109_212657_g2full.AUhok0; B2=$Y/20261110_040120_g2full.wjjYRA
mkdir -p $OUT/A $OUT/B
$PY -B - $B1 $B2 $OUT $SRC/sft/g3c_expected.sha <<'PYEOF'
import json, sys, hashlib
b1, b2, out, shafile = sys.argv[1:5]
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
W = lambda p, rows: open(p, "w", encoding="utf-8").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
a1t, a1v = L(f"{b1}/rows_final_A/train.jsonl"), L(f"{b1}/rows_final_A/valid.jsonl")
b1t, b1v = L(f"{b1}/rows_final_B/train.jsonl"), L(f"{b1}/rows_final_B/valid.jsonl")
a2t, a2v = L(f"{b2}/rows_clean/train.jsonl"), L(f"{b2}/rows_clean/valid.jsonl")
b2t, b2v = L(f"{b2}/rows_B/train.jsonl"), L(f"{b2}/rows_B/valid.jsonl")
pos = json.load(open(f"{b2}/provenance_B.json"))["rows_with_heldout_8gram"]          # positions in train.jsonl followed by valid.jsonl of batch 2 rows_B
assert pos == [12, 16, 278, 1412], pos
n = len(b2t); drop_t = {p for p in pos if p < n}; drop_v = {p - n for p in pos if p >= n}
dropped = [b2t[i] for i in sorted(drop_t)] + [b2v[i] for i in sorted(drop_v)]
b2t = [r for i, r in enumerate(b2t) if i not in drop_t]; b2v = [r for i, r in enumerate(b2v) if i not in drop_v]
rows = {"A": {"train": a1t + a2t, "valid": a1v + a2v}, "B": {"train": b1t + b2t, "valid": b1v + b2v}}
tid = lambda rs: {r["meta"]["task_id"] for r in rs}
for a in "AB":
    assert not (tid(rows[a]["train"]) & tid(rows[a]["valid"])), f"arm {a} train and validation share tasks"
assert not (tid(a1t + a1v) & tid(a2t + a2v)), "batch 1 and batch 2 share tasks"
key = lambda r: (r["meta"]["episode_id"], json.dumps(r["messages"][:-1], sort_keys=True))
for s in ("train", "valid"):
    ka = {key(r) for r in rows["A"][s]}; kb = {key(r) for r in rows["B"][s]}
    assert kb <= ka, f"arm B {s} rows are not a subset of arm A rows"
    print(s, "arm A rows", len(rows["A"][s]), "arm B rows", len(rows["B"][s]), "share", round(len(rows["B"][s]) / len(rows["A"][s]), 4), "tasks A", len(tid(rows["A"][s])), "tasks B", len(tid(rows["B"][s])))
print("dropped from arm B (batch 2 provenance):", [(r["meta"]["task_id"], r["meta"]["episode_id"]) for r in dropped])
lines = []
for a in "AB":
    for s in ("train", "valid"):
        p = f"{out}/{a}/{s}.jsonl"; W(p, rows[a][s]); lines.append(f"{a} {s} {hashlib.sha256(open(p, 'rb').read()).hexdigest()}\n")
open(shafile, "w").writelines(lines); print("".join(lines))
print("tasks combined:", len(tid(rows["A"]["train"]) | tid(rows["A"]["valid"])))
PYEOF
