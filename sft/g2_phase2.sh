#!/bin/bash
# G2 PHASE 2: teacher solve with reasoning sentences, then everything after it. usage: bash sft/g2_phase2.sh ATTEMPT_DIR   (ATTEMPT_DIR = a finished phase 1 attempt of g2_chain.sh full)
# (PREFLIGHT_ONLY=1 checks and stops.) API + CPU only; no GPU, no llama-server, no port 8080.
# Why a separate file: phase 1 (g2_chain.sh) is a running bash script and its worktree must not change while it runs.
# Authorization: advisor note 2026-11-10 (SANS APPROVED 2026-11-09, "Yeah start it and there is $50 total spend cap so don't worry about it.") covers the first solve,
# solve mode iffc-state-ooda-slim, k = 4, teacher writes two or three plain sentences of reasoning before the call list; total cap of the G2 run $15.
set -u
ATT_ARG=${1:-}; [ -d "$ATT_ARG" ] || { echo "usage: $0 ATTEMPT_DIR"; exit 2; }
cd /Users/USER/Documents/GitHub/Thesis-wt-g2p2 || exit 2
source sft/data_yield_lib.sh                       # say, cleanup, bt, YD, PY, F_SHA, K2SHA, DD
ABSLIM=/Users/USER/Documents/GitHub/Thesis-wt-slim
D=$DD
eval "$(sed -n '/^EX=/p;/^HELD=/p' sft/data_yield_chain.sh)"
eval "$(sed -n '/^N_TARGET=/,/^SMOKE_SLOTS=/p' sft/g2_chain.sh)"           # the same registered parameters as phase 1 (single source)
SOLVE_CAP=4.00; SOLVE_MIN=180
G2_CAP_USD=${G2_CAP_OVERRIDE:-$G2_CAP_USD}; SOLVE_CAP=${SOLVE_CAP_OVERRIDE:-$SOLVE_CAP}      # batch 2 (G2b): G2_CAP_OVERRIDE=12.0 SOLVE_CAP_OVERRIDE=1.70; defaults are batch 1's registered values                      # phase 2 values: longer replies cost more (pre-registration section 9)
ATT=$(cd "$ATT_ARG" && pwd); RL="$ATT/runlogs"
PYT="$PY -B sft/g2_tools.py"
CHAIN_CAP_S=$(( 8 * 3600 )); CHAIN_START=$(date +%s)
trap cleanup EXIT
trap 'echo "$(date +%T) ABORT: signal INT"; cleanup; exit 130' INT
trap 'echo "$(date +%T) ABORT: signal TERM"; cleanup; exit 143' TERM
die() { echo "$(date +%T) ABORT: $*"; cleanup; exit 1; }
chain_left() { echo $(( CHAIN_CAP_S - ($(date +%s) - CHAIN_START) )); }
bt2() { local s=$1; shift; local left; left=$(chain_left); [ "$left" -gt 60 ] || die "whole-chain cap reached before: $*"; [ "$s" -le "$left" ] || s=$left; bt "$s" "$@"; }
fl() { $PY -B -c "import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(round(min(a,b),4))" "$1" "$2"; }

# ---- the attempt must be a finished phase 1 and untouched by phase 2 ----
[ -s "$ATT/audited.jsonl" ] && [ -s "$ATT/prefilter_report.json" ] && [ -s "$ATT/author/report.json" ] || { echo "not a finished phase 1 attempt: $ATT"; exit 2; }
[ ! -e "$ATT/solve/verdicts.jsonl" ] && [ ! -e "$ATT/phase2.started" ] || { echo "phase 2 already started in $ATT"; exit 2; }
exec >> "$ATT/chain.log" 2>&1
echo "$(date +%T) g2 phase 2 start, attempt $ATT"

[ "$(git branch --show-current)" = "thesis/g2-phase2" ] || die "wrong branch"
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked files are modified; commit first"
[ "$(shasum -a 256 "$DD/k/tasks_K2.jsonl" | cut -d' ' -f1)" = "$K2SHA" ] || die "K2 hash changed"
[ "$(shasum -a 256 "$DD/f/tasks_F.jsonl" | cut -d' ' -f1)" = "$F_SHA" ] || die "F hash changed"
[ -n "${OPENROUTER_API_KEY:-}" ] || die "OPENROUTER_API_KEY is not set in this environment"
for f in $EX $HELD; do [ -s "$f" ] || die "missing or empty input file $f"; done
diff -rq benchmark "$ABSLIM/benchmark" -x __pycache__ > /dev/null || die "benchmark/ differs from the slim-port worktree"
if pgrep -f "distill_tasks|distill_solve|yield_author|g2_author|yield_judge|yield_solve" > /dev/null; then die "another authoring/solve/judge process is running"; fi
led=$($PYT spend --attempt "$ATT" --root "$YD/attempts") || die "ledger failed (a response without cost or a raised guard flag in this attempt): $led"
USED=$(echo "$led" | $PY -B -c "import json,sys; d=json.loads(sys.stdin.read().strip().splitlines()[-1]); print(d['attempt_usd'])")
SPENT_BEFORE=$(echo "$led" | $PY -B -c "import json,sys; d=json.loads(sys.stdin.read().strip().splitlines()[-1]); print('MISSING' if d['earlier_missing'] or d['earlier_flags_bad'] else d['earlier_usd'])")
[ "$SPENT_BEFORE" != "MISSING" ] || die "an earlier attempt has responses without cost accounting"
SC=$(fl "$SOLVE_CAP" "$($PY -B -c "print($G2_CAP_USD - $USED)")")
$PY -B -c "import sys; sys.exit(0 if $SC >= 0.50 and $SPENT_BEFORE + $G2_CAP_USD <= $TOTAL_API_CAP_USD else 1)" || die "spent $USED of $G2_CAP_USD in this run (solve cap would be $SC), earlier $SPENT_BEFORE of $TOTAL_API_CAP_USD: refusing the solve"
echo "commit $(git rev-parse HEAD); this run spent $USED so far (earlier attempts $SPENT_BEFORE); solve cap $SC (SOLVE_CAP $SOLVE_CAP, G2 cap $G2_CAP_USD); solve mode $SOLVE_MODE, k=4, sentences on"
$PY -B -c "import json; r=json.load(open('$ATT/prefilter_report.json')); print('tasks for the solve:', r['kept_for_solve'])"
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { say "preflight only: ok"; exit 0; }
date +%s > "$ATT/phase2.started"

# ---- teacher solve under the compact-list stack, with reasoning sentences (instruction only in the HTTP body; logged prompts are the slim prompts) ----
bt2 $(( SOLVE_MIN * 60 + 120 )) $PY -B sft/yield_solve_sentences.py pilot --tasks $ATT/audited.jsonl --log-dir $ATT/solve --fc-mode $SOLVE_MODE --max-cost $SC --max-minutes $SOLVE_MIN --k 4 --workers 4 > $ATT/solve.out 2>&1 || die "teacher solve failed or incomplete (see $ATT/solve.out)"
$PY -B - $ATT <<'PYEOF' || die "solve sanity failed: no sentences were produced or the call lists did not parse"
import json, sys, pathlib, collections
a = pathlib.Path(sys.argv[1])
recs = [json.loads(l) for l in open(a / "solve" / "sentences.jsonl") if l.strip()]
c = collections.Counter(("split_ok" if r["split_ok"] else "no_clean_split") + ":" + ((r.get("verdict") or {}).get("reason") or ("valid" if (r.get("verdict") or {}).get("valid") else "-")).split(":")[0] for r in recs)
print(json.dumps({"calls_with_sentence_record": len(recs), "by_outcome": dict(c)}))
valid = sum(1 for r in recs if (r.get("verdict") or {}).get("valid"))
sys.exit(0 if recs and valid / len(recs) >= 0.5 else 1)
PYEOF
bt2 1800 $PY -B sft/build_bank_f.py filter $ATT/audited.jsonl $ATT/solve $ATT/filtered $HELD > $RL/filter.out 2>&1 || die "filter failed"
$PY -B - $ATT <<'PYEOF' || die "quarantine list failed"
import json, sys, pathlib
a = pathlib.Path(sys.argv[1])
kept = {json.loads(l)["id"] for l in open(a / "filtered" / "tasks_F.jsonl") if l.strip()}
allv = {json.loads(l)["task_id"] for l in open(a / "solve" / "verdicts.jsonl") if l.strip() and "task_id" in json.loads(l)}
apierr = {x["task_id"] for x in map(json.loads, open(a / "solve" / "verdicts.jsonl")) if x.get("task_id") and x.get("api_error")}
q = {t: "not in the held-out-filtered kept set" for t in sorted(allv - kept)}
q.update({t: "has an episode with an API error" for t in sorted(apierr & kept)})
json.dump(q, open(a / "quarantine.json", "w")); print("quarantined", len(q), "kept", len(kept))
PYEOF
bt2 3600 $PY -B -m benchmark.distill_rows --solve-dirs $ATT/solve --out $ATT/rows --tasks $ATT/filtered/tasks_F.jsonl --manual-quarantine $ATT/quarantine.json > $RL/rows.out 2>&1 || die "row build failed"
bt2 1200 $PY -B sft/dedupe_conflicts.py --rows-dir $ATT/rows --out-dir $ATT/rows_clean > $RL/dedupe.out 2>&1 || die "conflict dedupe failed"; cat $RL/dedupe.out
bt2 900 $PY -B sft/g2_rows_b.py --rows-dir $ATT/rows_clean --solve-dir $ATT/solve --out-dir $ATT/rows_B > $RL/rows_b.out 2>&1 || die "arm B join failed"; cat $RL/rows_b.out
bt2 10800 $PY -B sft/check_provenance.py --rows $ATT/rows_clean/train.jsonl $ATT/rows_clean/valid.jsonl --heldout $HELD --task-sources $ATT/filtered/tasks_F.jsonl --out $ATT/provenance.json > $RL/provenance.out 2>&1
echo "provenance arm A exit $?"
bt2 10800 $PY -B sft/check_provenance.py --rows $ATT/rows_B/train.jsonl $ATT/rows_B/valid.jsonl --heldout $HELD --task-sources $ATT/filtered/tasks_F.jsonl --out $ATT/provenance_B.json > $RL/provenance_B.out 2>&1
echo "provenance arm B (reads reasoning_content too) exit $?"
TOTCAP=$($PY -B -c "print(round($SPENT_BEFORE + $G2_CAP_USD, 4))")
bt2 1800 $PY -B sft/yield_report.py --attempt $ATT --total-cap $TOTCAP > $RL/report.out 2>&1; RC=$?; tail -30 $RL/report.out
$PY -B sft/g2_floor2.py --attempt $ATT --target $N_TARGET --floor-tasks $FLOOR_TASKS --floor-train-rows $FLOOR_ROWS --cap $G2_CAP_USD --out $ATT/G2_REPORT.json
bt2 300 $PY -B sft/sample_for_review.py --tasks $ATT/filtered/tasks_F.jsonl --n 40 --seed 9092 --out $ATT/review_sample.md --ids-out $ATT/review_ids.json || die "review sample failed"
say "G2 FINISHED (yield_report exit $RC). REQUIRED NEXT (Thesis bot): read G2_REPORT.json (floor_met / target_met / arm B rows), a scan of 40 arm B sentence sets for answer-naming phrases (beyond the regex), accuracy review of review_sample.md (Codex labeling + two subagents; at most 2 WRONG each), Codex review of the report, then tell the advisor and Sans. No top-up runs without a SANS APPROVED line."
