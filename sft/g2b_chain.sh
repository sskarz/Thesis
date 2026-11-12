#!/bin/bash
# G2b (second authoring batch) = copy of g2_chain.sh with: new procs/seeds/ids (e f g), variant b2 prompts (THEMES_B2, collision rule, weights), dedupe against batch 1, own caps. Pre-registration docs/DATA_YIELD_G2B_PREREGISTRATION.md.
# G2 data scale-up chain (API + CPU only; no GPU, no llama-server, no port 8080). usage: bash sft/g2_chain.sh smoke | full   (add PREFLIGHT_ONLY=1 to check everything and stop)
# DRAFT: written 2026-11-10 for review. NOT approved, NOT launched. Launch needs a SANS APPROVED line naming the run, its time estimate and its spend cap.
#
# smoke: 3 authoring processes x 4 slots (12 slots), must show 0 API rejects, every response reporting a cost, the pinned model and provider; writes the marker the full run requires.
# full : 3 parallel authoring processes (disjoint ids and seeds, chunks of 500 slots) -> aggregate -> cross-process dedupe (yield_merge) -> gold audit -> deterministic state checks
#        -> request-coverage judge -> teacher solve under the compact-list stack -> held-out filter -> JSON-only rows -> conflict dedupe -> provenance gate -> report -> floors -> review sample.
# Budget: the TOTAL cap is checked before authoring, enforced during authoring by a monitor over the cost logs of ALL processes (not only per-process caps), and re-checked before judge and solve:
#   authoring spend A + judge cap + solve cap must stay within G2_CAP_USD, else the judge or solve is shrunk or refused. No automatic extension or top-up of any kind.
set -u
MODE=${1:-}; case "$MODE" in smoke|full) ;; solve) [ -d "${2:-}" ] || { echo "usage: $0 solve ATTEMPT_DIR"; exit 2; } ;; *) echo "usage: $0 smoke|full|solve ATTEMPT_DIR"; exit 2;; esac
cd /Users/USER/Documents/GitHub/Thesis-wt-g2b || exit 2
source sft/data_yield_lib.sh                       # say, die, cleanup, bt, init_attempt; YD stays the shared attempts root of the data worktree
ABSLIM=/Users/USER/Documents/GitHub/Thesis-wt-slim
D=$DD
eval "$(sed -n '/^EX=/p;/^HELD=/p' sft/data_yield_chain.sh)"           # same strict-dedupe inputs and held-out banks as Part 1
P1=$YD/attempts/20261109_144037_part1.E11FEx
B1=$YD/attempts/20261109_212657_g2full.AUhok0
EX="$EX $P1/author/tasks_raw.jsonl $B1/author/a/tasks_raw.jsonl $B1/author/c/tasks_raw.jsonl $B1/author/d/tasks_raw.jsonl"
WF=sft/g2b_weights.json                                    # new tasks must not duplicate Part 1's raw tasks

# ---------------- parameters (registered in docs/DATA_YIELD_G2_PREREGISTRATION.md) ----------------
N_TARGET=400            # surviving tasks wanted
SLOTS_TOTAL=10000       # authoring slots over all processes
PROCS="e f g"; SEED_BASES="7201000 7202000 7203000"
WORKERS=4               # per process (Part 1 tested 4 in one process)
CHUNK=500               # slots per chunk (progress is logged per chunk)
DEADLINE_MIN=330        # per process: 3434 slots / 12.5 per min = 275 min, plus 25% slack
G2_CAP_USD=12.0         # total cap of THIS run (authoring + judge + solve)
AUTH_CAP_PER_PROC=3.30  # cumulative per process (3 x 3.60 = 10.80)
JUDGE_CAP=0.40; SOLVE_CAP=1.70; SOLVE_MIN=150
SOLVE_MODE=iffc-state-ooda-slim
TOTAL_API_CAP_USD=50.0  # Sans, 2026-11-09: all API spend of the whole project
PROJ_RATIO=0.392        # valid -> surviving tasks, Part 1: 71 / 181
FLOOR_TASKS=250; FLOOR_ROWS=1500
SWAP_FREE_MIN_MB=300
CHAIN_CAP_S=$(( 6 * 3600 ))        # the whole chain, registered: about 6 h
SMOKE_SLOTS=4; SMOKE_CAP=0.05
# --------------------------------------------------------------------------------------------------
PIDS=()
cleanup_g2() {
  local p
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM -- "-$p" 2>/dev/null; done
  sleep 3
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -KILL -- "-$p" 2>/dev/null; done
  PIDS=()
  cleanup
}
trap cleanup_g2 EXIT
trap 'echo "$(date +%T) ABORT: signal INT"; cleanup_g2; exit 130' INT
trap 'echo "$(date +%T) ABORT: signal TERM"; cleanup_g2; exit 143' TERM
die() { echo "$(date +%T) ABORT: $*"; cleanup_g2; exit 1; }
PYT="$PY -B sft/g2_tools.py"
CHAIN_START=$(date +%s)
chain_left() { echo $(( CHAIN_CAP_S - ($(date +%s) - CHAIN_START) )); }
bt2() {           # bt2 SECONDS CMD... : bt, but never beyond the whole-chain cap
  local s=$1; shift; local left; left=$(chain_left); [ "$left" -gt 60 ] || die "whole-chain cap reached before: $*"
  [ "$s" -le "$left" ] || s=$left
  bt "$s" "$@"
}
fl() { $PY -B -c "import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(round(min(a,b),4))" "$1" "$2"; }

if [ "$MODE" = solve ]; then
  ATT=$(cd "$2" && pwd); RL="$ATT/runlogs"; CHAIN_CAP_S=28800          # phase 2 continues an existing attempt; own 8 h cap
  [ -s "$ATT/audited.jsonl" ] && [ -s "$ATT/prefilter_report.json" ] || { echo "not a finished phase 1 attempt: $ATT"; exit 2; }
  [ ! -e "$ATT/solve/verdicts.jsonl" ] || { echo "phase 2 already started in $ATT"; exit 2; }
  exec >> "$ATT/chain.log" 2>&1
else
  init_attempt "g2$MODE"
  RL="$ATT/runlogs"
fi
echo "$(date +%T) g2 $MODE start, attempt $ATT"

preflight_g2() {
  [ "$(git branch --show-current)" = "thesis/g2b-scale" ] || die "wrong branch"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked files are modified; commit first"
  [ "$(shasum -a 256 "$DD/k/tasks_K2.jsonl" | cut -d' ' -f1)" = "$K2SHA" ] || die "K2 hash changed"
  [ "$(shasum -a 256 "$DD/f/tasks_F.jsonl" | cut -d' ' -f1)" = "$F_SHA" ] || die "F hash changed"
  [ -n "${OPENROUTER_API_KEY:-}" ] || die "OPENROUTER_API_KEY is not set in this environment"        # presence only; never printed
  for f in $EX $HELD; do [ -s "$f" ] || die "missing or empty input file $f"; done
  [ "$(df -k "$YD" | awk 'NR==2{print $4}')" -gt 5000000 ] || die "less than 5 GB free"
  if pgrep -f "distill_tasks|distill_solve|yield_author|g2_author" > /dev/null; then die "another authoring/solve process is running"; fi        # batch 1 judge (API only) may run in parallel
  [ "$(diff -rq benchmark "$ABSLIM/benchmark" -x __pycache__ | sed "s/ and .*//" | tr -d "\n")" = "Files benchmark/distill_tasks.py" ] || die "benchmark/ must differ from the slim-port worktree in distill_tasks.py only (batch 2 prompts)"
  command -v timeout > /dev/null || die "timeout(1) not found"
  local free; free=$(sysctl -n vm.swapusage | $PY -B -c "import re,sys; m=re.search(r'free = ([0-9.]+)M', sys.stdin.read()); print(int(float(m.group(1))))")
  [ "$free" -ge "$SWAP_FREE_MIN_MB" ] || die "swap free is ${free} MB (< $SWAP_FREE_MIN_MB): too little headroom while another job may hold the GPU memory"
  # ledger across ALL earlier attempts, then the cap arithmetic
  local led; led=$($PYT spend --attempt "$ATT" --root "$YD/attempts") || die "ledger failed: $led"
  SPENT_BEFORE=$(echo "$led" | $PY -B -c "import json,sys; d=json.loads(sys.stdin.read().strip().splitlines()[-1]); print('MISSING' if d['earlier_missing'] or d['earlier_flags_bad'] else d['earlier_usd'])")
  [ "$SPENT_BEFORE" != "MISSING" ] || die "an earlier attempt has responses without cost accounting; reconcile with the provider first"
  echo "spent in earlier attempts: $SPENT_BEFORE"
  # the key's own usage counter is the authoritative project total (earlier rounds are not in the attempt ledger); presence of the key is not printed
  KEYUSE=$(curl -s -m 15 -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/auth/key | $PY -B -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])") || die "key usage lookup failed"
  echo "key usage so far: $KEYUSE"
  $PY -B -c "import sys; sys.exit(0 if $KEYUSE + 4.0 + $G2_CAP_USD <= $TOTAL_API_CAP_USD else 1)" || die "key usage $KEYUSE + 4.0 (rest of batch 1) + $G2_CAP_USD exceeds the project total $TOTAL_API_CAP_USD"
  $PY -B - "$SPENT_BEFORE" "$G2_CAP_USD" "$TOTAL_API_CAP_USD" "$AUTH_CAP_PER_PROC" "$JUDGE_CAP" "$SOLVE_CAP" <<'PYEOF' || die "cap arithmetic: per-process author caps + judge cap + solve cap exceed G2_CAP_USD, or earlier spend + G2_CAP_USD exceed the project total"
import sys
sb, g2, tot, ap, j, s = map(float, sys.argv[1:7])
sys.exit(0 if 3 * ap + j + s <= g2 + 1e-9 and sb + g2 <= tot + 1e-9 else 1)
PYEOF
  echo "commit $(git rev-parse HEAD); F $F_SHA; K2 $K2SHA; caps: G2 $G2_CAP_USD (author 3x$AUTH_CAP_PER_PROC, judge $JUDGE_CAP, solve $SOLVE_CAP), project total $TOTAL_API_CAP_USD"
}
preflight_g2
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { say "preflight only: ok"; exit 0; }

launch_authors() {   # launch_authors SLOTS_PER_PROC CAP SEED_OFFSET CHUNKSIZE DEADLINE_EPOCH SUBDIR
  local spp=$1 cap=$2 seedoff=$3 chunk=$4 dl=$5 sub=$6 i=0 p sb
  mkdir -p "$ATT/$sub"
  set -m
  for p in $PROCS; do
    sb=$(( $(echo $SEED_BASES | cut -d' ' -f$((i+1))) + seedoff ))
    timeout -k 30 $(( dl - $(date +%s) + 60 )) nice -n 10 $PY -B sft/g2_author.py --proc $p --seed-base $sb --slots $spp --chunk $chunk \
      --deadline-epoch $dl --cap $cap --workers $WORKERS --out-dir "$ATT/$sub/$p" --author-cmd "$PY -B sft/yield_author.py" --variant b2 --weights-file $WF --existing $EX > "$ATT/$sub/$p.out" 2>&1 &
    PIDS+=($!); i=$((i+1))
  done
  set +m
}
wait_authors() {     # wait_authors MAX_ATTEMPT_USD [WRITE_HALFWAY]: monitor spend over ALL logs of this attempt every 60 s; kills all processes on a breach
  local maxusd=$1 halfway=${2:-0} alive p rc
  while :; do
    alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    [ "$alive" -gt 0 ] || break
    sleep 60
    $PYT spend --attempt "$ATT" --root "$YD/attempts" --max-attempt "$maxusd" > "$RL/spend_now.json" 2>&1; rc=$?
    [ $rc -eq 0 ] || { cleanup_g2; die "spend monitor rc $rc (5 = attempt spend above $maxusd, 6 = a response without cost): $(tail -c 300 "$RL/spend_now.json")"; }
    [ "$halfway" = 1 ] && $PYT progress --author-dir "$ATT/author" --slots-total "$SLOTS_TOTAL" --target "$N_TARGET" --ratio "$PROJ_RATIO" --floor "$FLOOR_TASKS" --write-halfway --quiet | grep "HALFWAY" 
  done
  local bad=0
  for p in "${PIDS[@]}"; do wait "$p"; rc=$?; [ $rc -eq 0 ] || { say "author process $p exited with rc $rc"; bad=1; }; done
  PIDS=()
  [ $bad -eq 0 ] || die "an authoring process failed or hit the cost guard (partial chunks stay on disk; nothing downstream was spent)"
}

if [ "$MODE" = smoke ]; then
  SPP=$SMOKE_SLOTS; DL=$(( $(date +%s) + 480 ))
  launch_authors $SPP $SMOKE_CAP 9000 $SPP $DL author
  wait_authors 0.20
  $PYT aggregate --author-dir "$ATT/author" --out-dir "$ATT/author" || die "aggregate failed"
  $PY -B sft/yield_smoke_check.py --calls "$ATT/author/log/calls.jsonl" || die "smoke gate failed: cost reporting, model pin or provider"
  $PY -B - "$ATT/author/report.json" <<'PYEOF' || die "smoke: an API reject appeared"
import json, sys
r = json.load(open(sys.argv[1])); api = sum(v for k, v in r["reject_reasons"].items() if k.startswith("api"))
print(json.dumps({"slots": r["slots"], "valid": r["valid"], "api_rejects": api, "spent_usd": r["spent_usd"], "calls": r["calls"]}))
sys.exit(0 if api == 0 and r["calls"] > 0 else 1)
PYEOF
  echo "{\"ok\": true, \"commit\": \"$(git rev-parse HEAD)\", \"attempt\": \"$ATT\"}" > "$YD/g2b_smoke_ok.json"
  say "G2 SMOKE COMPLETE: $ATT"; exit 0
fi

# ------------------------------------ full run = PHASE 1 (authoring ... judge and pre-solve filter); the solve is PHASE 2 (mode solve) ------------------------------------
if [ "$MODE" = full ]; then
[ -f "$YD/g2b_smoke_ok.json" ] || die "no smoke marker; run 'g2b_chain.sh smoke' first"
$PY -B -c "import json,sys; sys.exit(0 if json.load(open('$YD/g2b_smoke_ok.json'))['commit']=='$(git rev-parse HEAD)' else 1)" || die "the smoke ran on a different commit than HEAD; re-run the smoke"
SPP=$(( (SLOTS_TOTAL + 2) / 3 ))
AUTH_TOTAL=$($PY -B -c "print(round(3*$AUTH_CAP_PER_PROC,4))")
DL=$(( $(date +%s) + DEADLINE_MIN * 60 ))
say "authoring: 3 processes x $SPP slots, chunks of $CHUNK, deadline $(date -r $DL +%T), caps 3 x $AUTH_CAP_PER_PROC (monitor limit $AUTH_TOTAL)"
launch_authors $SPP $AUTH_CAP_PER_PROC 0 $CHUNK $DL author
wait_authors "$AUTH_TOTAL" 1
$PYT aggregate --author-dir "$ATT/author" --out-dir "$ATT/author" || die "aggregate failed"
$PY -B - "$ATT/author/report.json" "$SLOTS_TOTAL" <<'PYEOF' || die "API errors above 2% of slots (see report.json reject_reasons); judge and solve are NOT started"
import json, sys
r = json.load(open(sys.argv[1])); n = int(sys.argv[2])
bad = {k: v for k, v in r["proc_stop_reasons"].items() if v not in ("all_chunks_done", "deadline")}
print(json.dumps({"proc_stop_reasons": r["proc_stop_reasons"], "slots_lost_to_deadline_or_stop": r["slots_lost_to_deadline_or_stop"], "share": r["slots_lost_share"]}))
api = sum(v for k, v in r["reject_reasons"].items() if k.startswith("api"))
print(json.dumps({"api_rejects": api, "api_share": round(api / n, 4)}))
sys.exit(0 if api <= 0.02 * n else 1)       # G2b rule: only API errors above 2% stop the chain here; deadline loss is reported, a crash or process cap already stopped the chain in wait_authors
PYEOF
say "per-chunk table (accepted, dup_in_batch, gold errors, other rejects, deadline, api per 500 slots):"
$PYT progress --author-dir "$ATT/author" --slots-total "$SLOTS_TOTAL" --target "$N_TARGET" --ratio "$PROJ_RATIO" --floor "$FLOOR_TASKS"
A=$($PYT spend --attempt "$ATT" --root "$YD/attempts" | $PY -B -c "import json,sys; print(json.loads(sys.stdin.read().strip().splitlines()[-1])['attempt_usd'])") || die "ledger after authoring failed"
say "authoring spent $A"
JC=$(fl "$JUDGE_CAP" "$($PY -B -c "print($G2_CAP_USD - $A - $SOLVE_CAP)")")
SC=$(fl "$SOLVE_CAP" "$($PY -B -c "print($G2_CAP_USD - $A - $JC)")")
$PY -B -c "import sys; sys.exit(0 if $JC >= 0.10 and $SC >= 0.50 else 1)" || die "authoring spent $A of $G2_CAP_USD; not enough left for the judge ($JC) and the solve ($SC): refusing to start them"
say "caps for the rest: judge $JC, solve $SC (authoring $A + $JC + $SC <= $G2_CAP_USD)"

P_IN=""; for p in $PROCS; do P_IN="$P_IN $ATT/author/$p/tasks_raw.jsonl"; done
bt2 600 $PY -B sft/yield_merge.py --inputs $P_IN --existing $P1/author/tasks_raw.jsonl --out $ATT/merged.jsonl --report $ATT/merge_report.json > $RL/merge.out 2>&1 || die "merge failed"; cat $RL/merge.out
bt2 5400 $PY -B sft/build_bank_f.py audit $ATT/merged.jsonl $ATT/audited_all.jsonl $ATT/semantic_audit.json > $RL/audit.out 2>&1 || die "audit failed"
bt2 5400 $PY -B sft/yield_checks.py --tasks $ATT/audited_all.jsonl --out $ATT/det_flags.json > $RL/det.out 2>&1 || die "deterministic checks failed"; cat $RL/det.out
bt2 2520 $PY -B sft/yield_judge.py --tasks $ATT/audited_all.jsonl --out $ATT/judge --max-cost $JC --workers 4 --deadline-minutes 40 > $RL/judge.out 2>&1 || die "judge failed or incomplete (see $RL/judge.out)"; tail -1 $RL/judge.out
$PY -B - $ATT <<'PYEOF' || die "pre-solve filter failed"
import json, sys, pathlib
a = pathlib.Path(sys.argv[1])
tasks = [json.loads(l) for l in open(a / "audited_all.jsonl") if l.strip()]
det = json.load(open(a / "det_flags.json")); jv = {json.loads(l)["id"]: json.loads(l)["verdict"] for l in open(a / "judge" / "verdicts.jsonl") if l.strip()}
hard = lambda t: [f for f in det.get(t, []) if not f.startswith("info_")]
keep = [t for t in tasks if jv.get(t["id"]) == "FAITHFUL" and not hard(t["id"])]
with open(a / "audited.jsonl", "w", encoding="utf-8") as f:       # name kept for yield_report.py: here 'audited' = passed audit + state checks + judge
    for t in keep:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
rep = {"audited_all": len(tasks), "det_dropped": sum(bool(hard(t["id"])) for t in tasks), "judge_not_faithful": sum(jv.get(t["id"]) != "FAITHFUL" for t in tasks),
       "judge_counts": {k: sum(v == k for v in jv.values()) for k in sorted(set(jv.values()))}, "kept_for_solve": len(keep)}
json.dump(rep, open(a / "prefilter_report.json", "w"), indent=1); print(json.dumps(rep))
sys.exit(0 if keep else 1)
PYEOF
say "PHASE 1 COMPLETE: $ATT. The solve is phase 2 ('g2_chain.sh solve $ATT') and waits for the teacher-sentence solve (amendment + Codex review). Spend so far: $($PYT spend --attempt "$ATT" --root "$YD/attempts" | tail -1)"
exit 0
fi

# ------------------------------------ PHASE 2 (mode solve): teacher solve with two or three plain sentences of reasoning, then rows ... report ------------------------------------
SOLVE_SENTENCES_READY=0       # set to 1 only by the reviewed amendment that implements the teacher sentences (docs/DATA_YIELD_G2_PREREGISTRATION.md section 9)
[ "$SOLVE_SENTENCES_READY" = 1 ] || die "the teacher-sentence solve is not implemented/reviewed yet; phase 2 refuses to run the JSON-only solve"
USED=$($PYT spend --attempt "$ATT" --root "$YD/attempts" | $PY -B -c "import json,sys; print(json.loads(sys.stdin.read().strip().splitlines()[-1])['attempt_usd'])") || die "ledger failed"
SC=$(fl "$SOLVE_CAP" "$($PY -B -c "print($G2_CAP_USD - $USED)")")
$PY -B -c "import sys; sys.exit(0 if $SC >= 0.50 else 1)" || die "spent $USED of $G2_CAP_USD; not enough left for the solve ($SC)"
say "phase 2: solve cap $SC (spent so far $USED, cap $G2_CAP_USD)"
bt2 $(( SOLVE_MIN * 60 + 120 )) $PY -B sft/yield_solve.py pilot --tasks $ATT/audited.jsonl --log-dir $ATT/solve --fc-mode $SOLVE_MODE --max-cost $SC --max-minutes $SOLVE_MIN --k 4 --workers 4 > $ATT/solve.out 2>&1 || die "teacher solve failed (see $ATT/solve.out)"
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
bt2 10800 $PY -B sft/check_provenance.py --rows $ATT/rows_clean/train.jsonl $ATT/rows_clean/valid.jsonl --heldout $HELD --task-sources $ATT/filtered/tasks_F.jsonl --out $ATT/provenance.json > $RL/provenance.out 2>&1
echo "provenance exit $?"
TOTCAP=$($PY -B -c "print(round($SPENT_BEFORE + $G2_CAP_USD, 4))")
bt2 1800 $PY -B sft/yield_report.py --attempt $ATT --total-cap $TOTCAP > $RL/report.out 2>&1; RC=$?; tail -30 $RL/report.out
$PYT floor --attempt $ATT --target $N_TARGET --floor-tasks $FLOOR_TASKS --floor-rows $FLOOR_ROWS --cap $G2_CAP_USD --out $ATT/G2_REPORT.json
bt2 300 $PY -B sft/sample_for_review.py --tasks $ATT/filtered/tasks_F.jsonl --n 40 --seed 9092 --out $ATT/review_sample.md --ids-out $ATT/review_ids.json || die "review sample failed"
say "G2 FINISHED (yield_report exit $RC). REQUIRED NEXT (Thesis bot): read G2_REPORT.json (floor_met / target_met / projection), accuracy review of review_sample.md (Codex labeling + two subagents; at most 2 WRONG each), Codex review of the report, then tell the advisor and Sans. No top-up runs without a SANS APPROVED line."
