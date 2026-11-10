#!/bin/bash
# OFFLINE replay of steps 5-8 of data_yield_chain.sh on the Round 3 artifacts (training-pool data; no API, no GPU).
set -u
cd /Users/sanskarthapa/Documents/GitHub/Thesis-wt-data || exit 2
source sft/data_yield_lib.sh
D=$DD; R=$D/r3
HELD="$D/r3/dev_all.jsonl $D/dev/confirm_bank.jsonl $D/f/tasks_F.jsonl $D/k/tasks_K2.jsonl"
A=$YD/attempts/replay_r3; rm -rf $A; mkdir -p $A/author/log $A/solve $A/filtered $A/rows $A/rows_clean $A/runlogs $A/smoke/log
cp $R/author.out $A/author/author.out; cp $R/author/calls.jsonl $A/author/log/calls.jsonl; cp $R/tasks_audited.jsonl $A/audited.jsonl
echo '{}' > $A/semantic_audit.json
$PY -B -c "import sys,json;sys.path.insert(0,'sft');import yield_report as y;rep,s,c=y.parse_author_out(open('$A/author/author.out').read());rep['spent_usd']=s;rep['calls']=c;json.dump(rep,open('$A/author/report.json','w'))"
cp $R/teacher_solve/verdicts.jsonl $R/teacher_solve/pilot_report.json $R/teacher_solve/calls.jsonl $R/teacher_solve/episodes.jsonl $A/solve/
$PY -B sft/build_bank_f.py filter $A/audited.jsonl $A/solve $A/filtered $HELD > $A/runlogs/filter.out 2>&1; echo "filter rc $?"
$PY -B - $A <<'PYEOF'
import json, sys, pathlib
a = pathlib.Path(sys.argv[1])
kept = {json.loads(l)["id"] for l in open(a / "filtered" / "tasks_F.jsonl") if l.strip()}
allv = {json.loads(l)["task_id"] for l in open(a / "solve" / "verdicts.jsonl") if l.strip() and "task_id" in json.loads(l)}
json.dump({t: "x" for t in sorted(allv - kept)}, open(a / "quarantine.json", "w"))
print("quarantined", len(allv - kept), "kept", len(kept))
PYEOF
$PY -B -m benchmark.distill_rows --solve-dirs $A/solve --out $A/rows --tasks $A/filtered/tasks_F.jsonl --manual-quarantine $A/quarantine.json > $A/runlogs/rows.out 2>&1; echo "rows rc $?"; tail -3 $A/runlogs/rows.out | cut -c1-200
$PY -B sft/dedupe_conflicts.py --rows-dir $A/rows --out-dir $A/rows_clean | cut -c1-300
$PY -B sft/check_provenance.py --rows $A/rows_clean/train.jsonl $A/rows_clean/valid.jsonl --heldout $HELD --task-sources $A/filtered/tasks_F.jsonl --out $A/provenance.json > $A/runlogs/provenance.out 2>&1; echo "prov rc $?"
$PY -B sft/yield_report.py --attempt $A --total-cap 5.0 > $A/runlogs/report.out 2>&1; echo "report rc $?"; tail -30 $A/runlogs/report.out | cut -c1-200
# review sampler on real gold execution: relabel 3 replay tasks with the yield prefix in a scratch copy
$PY -B - $A <<'PYEOF'
import json, sys, pathlib
a = pathlib.Path(sys.argv[1])
ts = [json.loads(l) for l in open(a / "filtered" / "tasks_F.jsonl") if l.strip()][:3]
for i, t in enumerate(ts):
    t["id"] = f"distill_y_replay{i}"
(a / "sample_in.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ts))
PYEOF
$PY -B sft/sample_for_review.py --tasks $A/sample_in.jsonl --n 3 --seed 1 --out $A/review_sample.md --ids-out $A/review_ids.json; echo "sample rc $?"; head -c 1500 $A/review_sample.md
