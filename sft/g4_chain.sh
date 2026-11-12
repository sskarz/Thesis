#!/bin/bash
# G4: BFCL v4 multi_turn_base (200 tasks) evaluation of ONE trained model against the untuned model, one session of four runs:
#   U1 (untuned, thinking on) -> stack verdict -> T1 (trained) -> T2 (trained) -> U2 (untuned, thinking on).
# Pre-registration: docs/DISTILLATION_G4_PREREGISTRATION_DRAFT.md. Approval: ledger A-003 (Sans, 2026-11-11), relayed by the advisor; the chain refuses to run without G4_APPROVAL=A-003.
# usage: G4_APPROVAL=A-003 bash sft/g4_chain.sh run      |      bash sft/g4_chain.sh dry   (all CPU checks, no server, no model call, no run on the BFCL 200)
# Harness: worktree Thesis-wt-g4 at dfdbc2b (the slim port); every run has its OWN results folder and database, and is scored in its own process (the checker keeps state across calls).
set -u
MODE=${1:-}; [ "$MODE" = dry ] || [ "$MODE" = run ] || { echo "usage: $0 dry|run"; exit 2; }
ROOT=/Users/USER/Documents/GitHub
SRC=$ROOT/Thesis-wt-training                       # this script, the helpers and g3lib.sh are committed here
source $SRC/sft/g3lib.sh                          # server lifecycle helpers (start_server, check_identity, stop_server, bt, die, abort, chain clock)
D6=$ROOT/Thesis-wt-r6/sft/data/distill/r6g4        # G4 attempts live here (outside git; the folder is new)
R6G3=$ROOT/Thesis-wt-r6/sft/data/distill/r6g3/attempts
A_ATT=$R6G3/20261110_062035_run_A.lBKtpA
B_ATT=""; NB=0                                    # arm B folder: the one that holds G3_B_RUN1.json; two such folders -> the chain refuses (no guessing after a restart)
for d in $R6G3/*_run_B.*; do [ -d "$d" ] || continue; if [ -f "$d/G3_B_RUN1.json" ]; then NB=$((NB+1)); B_ATT=$d; elif [ -z "$B_ATT" ]; then B_ATT=$d; fi; done
HARNESS=$ROOT/Thesis-wt-g4; HARNESS_COMMIT=dfdbc2b
RUN_CAP=16200                                      # 4.5 h per run: 1.5 x the longest recorded run (2 h 59 min); a run above it is void
CHAIN_CAP=68400                                    # 19 h = 4 runs x 4.5 h + 1 h
BAND_LO=82; BAND_HI=94; MARGIN=8; MAX_ERR=10
DB84=$ROOT/Thesis/benchmark/results/benchmark.db
FMODE=iffc-state-ooda-slim
RENDER_REF=$HOME/thesis-backups/render/slim_iffc-state-ooda-slim.jsonl
MONPID=""
unset G4_SERVER BFCL_SERVER_URL                     # the harness and the render capture must talk to the server this chain starts on 8080, never to an inherited override
CHECKER_VERSION=2026.3.23; CHECKER_SHA=fc86f04e343793cdc5d99fc59eab85d1f9125cb78d90ef5f1d6e708b244b856e   # registered scorer pin (pre-registration section 2)
cleanup() {                                         # every exit: stop only what this chain started, then verify the port is free
  [ -n "$MONPID" ] && kill "$MONPID" 2>/dev/null
  stop_server
  for _ in $(seq 1 20); do lsof -nP -iTCP:8080 -sTCP:LISTEN > /dev/null 2>&1 || break; sleep 1; done
  if lsof -nP -iTCP:8080 -sTCP:LISTEN > /dev/null 2>&1; then echo "$(date +%T) CLEANUP FAILED: something still listens on 8080"; exit 4; else echo "$(date +%T) cleanup ok: no listener on 8080"; fi
}
trap cleanup EXIT
swap_gb() { sysctl -n vm.swapusage | $PY -c "import re,sys; m=re.search(r'used = ([0-9.]+)([MG])', sys.stdin.read()); v=float(m.group(1)); print(v/1024 if m.group(2)=='M' else v)"; }

# ---------------------------------------------------------------- preflight (dry and run)
mkdir -p $D6/attempts || abort "cannot create $D6"
echo $(date +%s) > $D6/chain_start_epoch          # fresh clock at the very start of every invocation (dry runs remove it at the end)
init_attempt g4$MODE
chain_clock
say "G4 chain ($MODE) attempt $ATT; arm A attempt $A_ATT; arm B attempt ${B_ATT:-none}"
FAILS=0
SRC_COMMIT=$(git -C $SRC rev-parse --short HEAD 2>/dev/null)
say "code commit (Thesis-wt-training): $SRC_COMMIT; harness commit: $HARNESS_COMMIT"
chk() { if [ "$1" = ok ]; then say "check ok: $2"; else say "CHECK FAILED: $2"; FAILS=$((FAILS+1)); fi; }
if [ "$MODE" = run ]; then [ "${G4_APPROVAL:-}" = "A-003" ] || abort "G4_APPROVAL=A-003 is not set (ledger A-003, Sans 2026-11-11)"; fi
$PY -c "import datetime,sys; sys.exit(0 if datetime.datetime.now() < datetime.datetime(2026,11,25,23,59) else 1)" && chk ok "date before the Nov 25 stop" || chk fail "date is past Nov 25"
[ -z "$(git -C $SRC status --porcelain --untracked-files=no 2>/dev/null)" ] && chk ok "code worktree Thesis-wt-training clean at $SRC_COMMIT" || chk fail "Thesis-wt-training has uncommitted tracked changes"
[ $NB -le 1 ] && chk ok "at most one arm B folder holds a run-1 result ($NB)" || chk fail "two arm B folders hold a run-1 result"
[ "$(git -C $HARNESS rev-parse --short HEAD 2>/dev/null)" = "$HARNESS_COMMIT" ] && chk ok "harness worktree at $HARNESS_COMMIT" || chk fail "harness worktree $HARNESS missing or not at $HARNESS_COMMIT"
[ -z "$(git -C $HARNESS status --porcelain --untracked-files=no 2>/dev/null)" ] && chk ok "harness worktree clean" || chk fail "harness worktree has tracked changes"
[ "$(shasum -a 256 "$K" | cut -d' ' -f1)" = "$K2SHA" ] && [ "$(shasum -a 256 "$F" | cut -d' ' -f1)" = "$F_SHA" ] && chk ok "F and K2 hashes unchanged" || chk fail "F or K2 hash changed"
[ "$(shasum -a 256 "$BAR" | cut -c1-16)" = "$BAR_SHA16" ] && chk ok "bar GGUF prefix $BAR_SHA16" || chk fail "bar GGUF prefix"
$PY -c "import sys; sys.exit(0 if float('$(swap_gb)') < 8.0 else 1)" && chk ok "swap used $(swap_gb) GB < 8" || chk fail "swap used above 8 GB"
if lsof -nP -iTCP:8080 -sTCP:LISTEN > /dev/null 2>&1; then chk fail "port 8080 is in use"; else chk ok "port 8080 free"; fi
if pgrep -f "g3_train.py|mlx_lm" > /dev/null; then chk fail "a training process is running (expected in dry mode while arm B trains)"; else chk ok "no training process"; fi
# selection of the single trained model (needs arm B's screen)
bt $PY -B $SRC/sft/g4_tools.py select --a-attempt $A_ATT --b-attempt "${B_ATT:-none}" --out $ATT/G4_SELECTION.json > $RL/select.out 2>&1 && chk ok "model selection: $(tail -1 $RL/select.out | cut -c1-200)" || chk fail "model selection: $(tail -c 300 $RL/select.out | tr '\n' ' ')"
# scorer pin: stored Runs 84 and 85 must re-score to 88 and 72, each run in its own process, database read-only
for pair in 84:88 85:72; do
  r=${pair%%:*}; want=${pair##*:}
  (cd $SRC && bt $PY -B $SRC/sft/g4_rescore.py $DB84 $ATT/rescore_run$r.json $r > $RL/rescore$r.out 2>&1)
  $PY - $ATT/rescore_run$r.json $r $want $CHECKER_VERSION $CHECKER_SHA <<'PYEOF' && chk ok "scorer pin: stored Run $r re-scores to $want of 200, 0 differing verdicts, checker $CHECKER_VERSION and file hash as registered" || chk fail "scorer pin for Run $r"
import json, sys
d = json.load(open(sys.argv[1])); v = d["runs"][sys.argv[2]]
ok = v["rescored_passes"] == int(sys.argv[3]) and v["tasks_with_different_verdict"] == 0 and v["checker_exceptions"] == 0
ok = ok and d["bfcl_eval_version"] == sys.argv[4] and d["checker_sha256_of_file_hashes"] == sys.argv[5]
sys.exit(0 if ok else 1)
PYEOF
done
$PY -B $SRC/sft/test_g4_stats.py > $RL/stats_test.out 2>&1 && chk ok "statistics tests (paper numbers)" || chk fail "statistics tests"
(cd $HARNESS && BENCHMARK_RESULTS_DIR=$ATT/helpcheck PYTHONDONTWRITEBYTECODE=1 $PY -m benchmark.run_all --help > /dev/null 2>&1) && chk ok "harness imports and run_all --help" || chk fail "harness does not import"
if [ "$MODE" = dry ]; then
  rm -f $D6/chain_start_epoch
  say "DRY RUN DONE: $FAILS check(s) failed (in dry mode, while arm B trains, the training-process check, the model selection and port checks are expected to fail until arm B's screen is finished)"
  say "plan: U1 untuned thinking on -> verdict (band $BAND_LO-$BAND_HI) -> T1 -> T2 -> U2; caps $RUN_CAP s per run, $CHAIN_CAP s for the chain; margin $MARGIN; void above $MAX_ERR errored tasks"
  exit 0
fi
[ $FAILS -eq 0 ] || die "preflight failed ($FAILS checks); nothing started"

# ---------------------------------------------------------------- run
echo $(date +%s) > $D6/chain_start_epoch; chain_clock
ARM=$($PY -c "import json; print(json.load(open('$ATT/G4_SELECTION.json'))['chosen'])")
GGUF=$($PY -c "import json; s=json.load(open('$ATT/G4_SELECTION.json')); print(s['arms'][s['chosen']]['gguf'])")
SHA=$($PY -c "import json; s=json.load(open('$ATT/G4_SELECTION.json')); print(s['arms'][s['chosen']]['gguf_sha256'])")
THINK=$($PY -c "import json; s=json.load(open('$ATT/G4_SELECTION.json')); print(s['arms'][s['chosen']]['think'])")
say "trained model: arm $ARM, thinking $THINK, GGUF $GGUF sha256 $SHA"
[ "$(shasum -a 256 $GGUF | cut -d' ' -f1)" = "$SHA" ] || die "GGUF hash differs from the screened one"

source $SRC/sft/g4_lib.sh
parity_phase
run_valid U1 U
stack_verdict

if [ $VRC -ne 0 ]; then say "STACK NOT REPRODUCED: U1 is outside $BAND_LO-$BAND_HI. The trained model is NOT run. Report to Sans; he decides (pre-registration section 3)."; exit 3; fi
run_valid T1 T
run_valid T2 T
run_valid U2 U
final_analysis
say "G4 COMPLETE: $ATT (G4_RESULT.json). REQUIRED NEXT (Thesis bot, not the chain): report counts with the limits of the pre-registration section 8, Codex review of the result, backup."
