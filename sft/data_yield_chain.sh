#!/bin/bash
# Data-yield check, PART 1 (API + CPU only). usage: bash sft/data_yield_chain.sh part1
# smoke (3 slots, must report a cost) -> author 2000 slots -> audit -> teacher solve k=4 -> held-out filter -> JSON-only rows
# -> provenance gate -> report -> review sample. No GPU, no llama-server. Registered caps: smoke $0.05, author $3.00 (hard stop 160 min),
# solve $0.60 (hard stop 70 min), whole attempt $5.00. The API key is read from the environment by teacher_api.py and is never printed.
set -u
case "${1:-}" in part1) ;; *) echo "usage: $0 part1"; exit 2;; esac
cd /Users/sanskarthapa/Documents/GitHub/Thesis-wt-data || exit 2
source sft/data_yield_lib.sh
D=$DD
# strict-dedupe inputs for authoring: every earlier task file (including F and K2 text; used only for n-gram dedupe, never read by a person)
EX="$D/all_prior_tasks.jsonl $D/all34.jsonl $D/all54.jsonl $D/dev/confirm_bank.jsonl $D/dev/dev_bank.jsonl $D/dev/tasks_audited.jsonl $D/dev/tasks_raw.jsonl $D/dev/train_pool.jsonl $D/pilot/tasks.jsonl $D/pilot2/tasks.jsonl $D/s2a/tasks.jsonl $D/r3/tasks_raw.jsonl $D/r3/tasks_audited.jsonl $D/r3/dev_all.jsonl $D/r3/pool_all.jsonl $D/f/tasks_raw.jsonl $D/f/tasks_audited.jsonl $D/f/tasks_F.jsonl $D/g/tasks_raw.jsonl $D/g/tasks_audited.jsonl $D/k/tasks_raw.jsonl $D/k/tasks_audited.jsonl $D/k/tasks_K2.jsonl"
# held-out banks: exactly the Round 5 provenance set
HELD="$D/r3/dev_all.jsonl $D/dev/confirm_bank.jsonl $D/f/tasks_F.jsonl $D/k/tasks_K2.jsonl"
init_attempt part1
echo "$(date +%T) data-yield part1 start, attempt $ATT"
preflight
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { say "preflight only: ok"; exit 0; }
SEED=7001; PREFIX=distill_y

# 1. smoke: proves the key works, a task can be authored, and the API reports cost (otherwise the dollar cap is meaningless)
bt 600 $PY -B sft/yield_author.py --n 3 --out $ATT/smoke/tasks_raw.jsonl --log-dir $ATT/smoke/log --seed 6999 --prefix distill_ysmoke \
   --existing $EX --max-cost 0.05 --workers 1 --deadline-minutes 8 --report-out $ATT/smoke/report.json > $ATT/smoke/author.out 2>&1 || die "smoke failed (see $ATT/smoke/author.out)"
tail -1 $ATT/smoke/author.out
$PY -B sft/yield_smoke_check.py --calls $ATT/smoke/log/calls.jsonl || die "smoke gate failed: key, cost reporting, model pin or provider differ from the registered values (nothing further was spent)"
# 2. author
bt 9600 $PY -B sft/yield_author.py --n 2000 --out $ATT/author/tasks_raw.jsonl --log-dir $ATT/author/log --seed $SEED --prefix $PREFIX \
   --existing $EX --max-cost 3.00 --workers 4 --deadline-minutes 145 --report-out $ATT/author/report.json > $ATT/author/author.out 2>&1 || die "author failed (see $ATT/author/author.out)"
tail -1 $ATT/author/author.out
# 3. semantic audit of gold (cp/mv destinations etc.)
bt 1800 $PY -B sft/build_bank_f.py audit $ATT/author/tasks_raw.jsonl $ATT/audited.jsonl $ATT/semantic_audit.json > $ATT/runlogs/audit.out 2>&1 || die "audit failed"
# 4. teacher solve in the student's loop (same harness mode as the evaluation), k=4, checker-verified
bt 4200 $PY -B sft/yield_solve.py pilot --tasks $ATT/audited.jsonl --log-dir $ATT/solve --max-cost 0.60 --max-minutes 60 --k 4 --workers 4 > $ATT/solve.out 2>&1 || die "teacher solve failed (see $ATT/solve.out)"
# 5. keep teacher-verified tasks, drop any whose gold-function signature or user text matches a HELD-OUT bank
bt 1800 $PY -B sft/build_bank_f.py filter $ATT/audited.jsonl $ATT/solve $ATT/filtered $HELD > $ATT/runlogs/filter.out 2>&1 || die "filter failed"
# 6. rows: ONLY tasks that survived step 5 (everything else is manually quarantined so solve episodes of dropped tasks cannot leak in)
$PY -B - $ATT <<'PYEOF' || die "quarantine list failed"
import json, sys, pathlib
a = pathlib.Path(sys.argv[1])
kept = {json.loads(l)["id"] for l in open(a / "filtered" / "tasks_F.jsonl") if l.strip()}
allv = {json.loads(l)["task_id"] for l in open(a / "solve" / "verdicts.jsonl") if l.strip() and "task_id" in json.loads(l)}
apierr = {x["task_id"] for x in map(json.loads, open(a / "solve" / "verdicts.jsonl")) if x.get("task_id") and x.get("api_error")}
q = {t: "not in the held-out-filtered kept set" for t in sorted(allv - kept)}
q.update({t: "has an episode with an API error" for t in sorted(apierr & kept)})   # conservative: whole task out (distill_rows does not look at api_error)
json.dump(q, open(a / "quarantine.json", "w"))
print("quarantined", len(q), "kept", len(kept), "of which api-error tasks", len(apierr & kept))
PYEOF
bt 1800 $PY -B -m benchmark.distill_rows --solve-dirs $ATT/solve --out $ATT/rows --tasks $ATT/filtered/tasks_F.jsonl --manual-quarantine $ATT/quarantine.json > $ATT/runlogs/rows.out 2>&1 || die "row build failed (see $ATT/runlogs/rows.out)"
# 6b. rows whose prompt appears with two different targets are dropped entirely (the builder only removes exact duplicates)
bt 600 $PY -B sft/dedupe_conflicts.py --rows-dir $ATT/rows --out-dir $ATT/rows_clean > $ATT/runlogs/dedupe.out 2>&1 || die "conflict dedupe failed"
cat $ATT/runlogs/dedupe.out
# 7. provenance gate: same command shape and held-out banks as Round 5 (rows are JSON-only here, so there is no reasoning text yet)
bt 3600 $PY -B sft/check_provenance.py --rows $ATT/rows_clean/train.jsonl $ATT/rows_clean/valid.jsonl --heldout $HELD --task-sources $ATT/filtered/tasks_F.jsonl --out $ATT/provenance.json > $ATT/runlogs/provenance.out 2>&1
echo "provenance exit $?" 
# 8. report and review sample (a failing provenance is reported, not hidden)
bt 1800 $PY -B sft/yield_report.py --attempt $ATT --total-cap $TOTAL_CAP_USD > $ATT/runlogs/report.out 2>&1; RC=$?
cat $ATT/runlogs/report.out
bt 300 $PY -B sft/sample_for_review.py --tasks $ATT/filtered/tasks_F.jsonl --n 40 --seed 9090 --out $ATT/review_sample.md --ids-out $ATT/review_ids.json || die "review sample failed"
say "part1 finished (report exit $RC). REQUIRED NEXT (Thesis bot): independent semantic review of review_sample.md by Codex and by a subagent; Codex review of YIELD_REPORT.json; then tell Sans."
exit 0
