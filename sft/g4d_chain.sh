#!/bin/bash
# G4 DIAGNOSTICS (EXPLORATORY, outside the G4 registration): three single runs on the BFCL 200 in this order:
#   D1 untuned GGUF, thinking OFF -> D3 arm A pass-1 checkpoint (converted + identity-checked), thinking OFF -> D2 arm B screened GGUF, thinking ON.
# Pre-registration: docs/DISTILLATION_G4_DIAGNOSTICS_PREREGISTRATION.md. Approval: ledger A-011 (Sans, 2026-11-12, "2. Yes"); the chain refuses to run without G4D_APPROVAL=A-011.
# usage: G4D_APPROVAL=A-011 bash sft/g4d_chain.sh run   |   bash sft/g4d_chain.sh dry (CPU checks, converts and audits the pass-1 GGUF, no server, no model call)
# Pieces of sft/g4_chain.sh and sft/g4_lib.sh; serve() and parity_phase() are overridden below; no automatic rerun of a void run (one run each).
set -u
MODE=${1:-}; [ "$MODE" = dry ] || [ "$MODE" = run ] || { echo "usage: $0 dry|run"; exit 2; }
ROOT=/Users/USER/Documents/GitHub
SRC=$ROOT/Thesis-wt-training                       # this script, the helpers and g3lib.sh are committed here
source $SRC/sft/g3lib.sh                          # server lifecycle helpers (start_server, check_identity, stop_server, bt, die, abort, chain clock)
D6=$ROOT/Thesis-wt-r6/sft/data/distill/r6g4d       # diagnostics attempts live here (outside git; new folder; G4 attempts stay untouched in r6g4)
R6G3=$ROOT/Thesis-wt-r6/sft/data/distill/r6g3/attempts
A_ATT=$R6G3/20261110_062035_run_A.lBKtpA
B_ATT=$R6G3/20261111_020051_run_B.U98dKQ; G4ATT=$ROOT/Thesis-wt-r6/sft/data/distill/r6g4/attempts/20261111_203247_g4run.JoHtbc
B_GGUF=$B_ATT/G3B.gguf; B_SHA=2c0bb899884ce2ce5cbead46abea6a0f1f73572230218770a6472c06cd4f1e80   # the screened file; same hash recorded at both of its screens
A_PASS1=$A_ATT/train/adapters_pass1; A_PASS1_SHA=65f14318c5faf940dba88b82a6b31c835fd3a6d4fb9418f28ee95c2db0d6bcfb; PROBE_GGUF_SHA=d847d57497092b3fac43f18580aecc90169d124621b5e024fbfae34c3e299c89   # pass-1 adapter file and the GGUF the Nov 10 probe made from it
HARNESS=$ROOT/Thesis-wt-g4; HARNESS_COMMIT=dfdbc2b
RUN_CAP=16200                                      # 4.5 h per run (D2 is expected at about 2.5 h)
CHAIN_CAP=28800                                    # 8 h for three runs, conversion and restarts (estimate about 4.7 h)
MAX_ERR=10
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
say "G4 diagnostics chain ($MODE) attempt $ATT; arm A attempt $A_ATT; arm B attempt $B_ATT"
FAILS=0
SRC_COMMIT=$(git -C $SRC rev-parse --short HEAD 2>/dev/null)
say "code commit (Thesis-wt-training): $SRC_COMMIT; harness commit: $HARNESS_COMMIT"
chk() { if [ "$1" = ok ]; then say "check ok: $2"; else say "CHECK FAILED: $2"; FAILS=$((FAILS+1)); fi; }
if [ "$MODE" = run ]; then [ "${G4D_APPROVAL:-}" = "A-011" ] || abort "G4D_APPROVAL=A-011 is not set (ledger A-011, Sans 2026-11-12)"; fi
$PY -c "import datetime,sys; sys.exit(0 if datetime.datetime.now() < datetime.datetime(2026,11,25,23,59) else 1)" && chk ok "date before the Nov 25 stop" || chk fail "date is past Nov 25"
[ -z "$(git -C $SRC status --porcelain --untracked-files=no 2>/dev/null)" ] && chk ok "code worktree Thesis-wt-training clean at $SRC_COMMIT" || chk fail "Thesis-wt-training has uncommitted tracked changes"
[ "$(git -C $HARNESS rev-parse --short HEAD 2>/dev/null)" = "$HARNESS_COMMIT" ] && chk ok "harness worktree at $HARNESS_COMMIT" || chk fail "harness worktree $HARNESS missing or not at $HARNESS_COMMIT"
[ -z "$(git -C $HARNESS status --porcelain --untracked-files=no 2>/dev/null)" ] && chk ok "harness worktree clean" || chk fail "harness worktree has tracked changes"
[ "$(shasum -a 256 "$K" | cut -d' ' -f1)" = "$K2SHA" ] && [ "$(shasum -a 256 "$F" | cut -d' ' -f1)" = "$F_SHA" ] && chk ok "F and K2 hashes unchanged" || chk fail "F or K2 hash changed"
[ "$(shasum -a 256 "$BAR" | cut -c1-16)" = "$BAR_SHA16" ] && chk ok "bar GGUF prefix $BAR_SHA16" || chk fail "bar GGUF prefix"
$PY -c "import sys; sys.exit(0 if float('$(swap_gb)') < 8.0 else 1)" && chk ok "swap used $(swap_gb) GB < 8" || chk fail "swap used above 8 GB"
if lsof -nP -iTCP:8080 -sTCP:LISTEN > /dev/null 2>&1; then chk fail "port 8080 is in use"; else chk ok "port 8080 free"; fi
if pgrep -f "g3_train.py|mlx_lm|g3c_arm|g4_chain|llama-server" > /dev/null; then chk fail "a training, chain or server process is running"; else chk ok "no training, chain or server process"; fi
# model files: arm B screened GGUF by hash; arm A pass-1 adapter by hash, converted with the G3 converter and audited (CPU only)
[ "$(shasum -a 256 $B_GGUF | cut -d' ' -f1)" = "$B_SHA" ] && chk ok "arm B GGUF equals the screened file ($B_SHA)" || chk fail "arm B GGUF hash differs from the screened file"
[ "$(shasum -a 256 $A_PASS1/adapters.safetensors | cut -d' ' -f1)" = "$A_PASS1_SHA" ] && chk ok "arm A pass-1 adapter equals the file the Nov 10 probe used" || chk fail "arm A pass-1 adapter hash differs"
D3_GGUF=$ATT/D3_pass1.gguf; mkdir -p $ATT/pass1_src && cp $A_PASS1/adapters.safetensors $A_PASS1/adapter_config.json $ATT/pass1_src/
(cd $SRC && bt nice -n 10 $PY -B sft/mlx_adapter_to_gguf_lora.py $ATT/pass1_src $D3_GGUF > $RL/convert_pass1.log 2>&1) && (cd $SRC && bt nice -n 10 $PY -B sft/r6_tools.py gguf-check --adapter-dir $ATT/pass1_src --src-dir $ATT/pass1_src --gguf $D3_GGUF > $RL/gguf_check_pass1.log 2>&1) && chk ok "pass-1 GGUF converted and audited: $(tail -1 $RL/gguf_check_pass1.log | cut -c1-160)" || chk fail "pass-1 conversion or audit"
D3_SHA=$(shasum -a 256 $D3_GGUF 2>/dev/null | cut -d' ' -f1); say "pass-1 GGUF sha256 $D3_SHA (the Nov 10 probe file had $PROBE_GGUF_SHA: $([ "$D3_SHA" = "$PROBE_GGUF_SHA" ] && echo identical || echo different))"
[ -f $G4ATT/RUN_U1.json ] && [ -f $G4ATT/RUN_U2.json ] && [ -f $G4ATT/RUN_T1.json ] && chk ok "G4 reference runs present" || chk fail "G4 reference runs missing"
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
$PY -B $SRC/sft/test_g4d_readings.py > $RL/readings_test.out 2>&1 && chk ok "reading-rule tests" || chk fail "reading-rule tests"
(cd $HARNESS && BENCHMARK_RESULTS_DIR=$ATT/helpcheck PYTHONDONTWRITEBYTECODE=1 $PY -m benchmark.run_all --help > /dev/null 2>&1) && chk ok "harness imports and run_all --help" || chk fail "harness does not import"
if [ "$MODE" = dry ]; then
  rm -f $D6/chain_start_epoch
  say "DRY RUN DONE: $FAILS check(s) failed"
  say "plan: D1 untuned thinking off -> D3 arm A pass 1 thinking off -> D2 arm B thinking on; caps $RUN_CAP s per run, $CHAIN_CAP s for the chain; void above $MAX_ERR errored tasks; one run each"
  exit 0
fi
[ $FAILS -eq 0 ] || die "preflight failed ($FAILS checks); nothing started"

# ---------------------------------------------------------------- run
echo $(date +%s) > $D6/chain_start_epoch; chain_clock
THINK=off; ARM=diag; GGUF=$D3_GGUF; SHA=$D3_SHA
source $SRC/sft/g4_lib.sh
serve() {            # serve NAME KIND : D1 untuned thinking off; D3 arm A pass 1 thinking off; D2 arm B thinking on (each with its identity check and file hash)
  case "$2" in
    D1) start_server G4D$1 "" off; check_identity G4D$1 1.0 "";;
    D3) start_server G4D$1 $D3_GGUF off; check_identity G4D$1 1.0 $D3_GGUF; [ "$(shasum -a 256 $D3_GGUF | cut -d' ' -f1)" = "$D3_SHA" ] || die "pass-1 GGUF hash changed before $1";;
    D2) start_server G4D$1 $B_GGUF on; check_identity G4D$1 1.0 $B_GGUF; [ "$(shasum -a 256 $B_GGUF | cut -d' ' -f1)" = "$B_SHA" ] || die "arm B GGUF hash changed before $1";;
    *) die "unknown kind $2";;
  esac
}
parity_phase() {     # render capture for each server mode before any run
  for kind in D1 D3 D2; do
    serve PAR$kind $kind
    th=off; ref=""; [ $kind = D2 ] && { th=on; ref="--ref $RENDER_REF"; }
    (cd $HARNESS && bt $PY -B $SRC/sft/g4_tools.py parity --think $th --out $ATT/parity_$kind.json $ref > $RL/parity_$kind.out 2>&1) || die "prompt-render capture failed on the $kind server: $(tail -c 300 $RL/parity_$kind.out)"
    say "render capture ok ($kind, thinking $th): $(tr -d '\n ' < $ATT/parity_$kind.json | cut -c1-260)"
    stop_server
  done
}
one_run() {          # one_run NAME : exactly one run; a void run stops the chain (no rerun without a new approval)
  run_one $1 $1 || die "run $1 void or stopped; chain stops; report to Sans (one run each, no repeats without a new approval)"
}
parity_phase
one_run D1
one_run D3
one_run D2
$PY -B $SRC/sft/g4d_readings.py $ATT > $RL/readings.out 2>&1 || die "readings failed: $(tail -c 300 $RL/readings.out)"
say "readings: $(tr -d '\n' < $RL/readings.out | cut -c1-600)"
say "G4 DIAGNOSTICS COMPLETE: $ATT (DIAG_RESULT.json). REQUIRED NEXT (Thesis bot, not the chain): report scores with the preregistered readings, label exploratory, back up the attempt folder, Codex review."
