#!/bin/bash
# Offline test of the G4 chain functions (sft/g4_lib.sh) with a FAKE harness and stubbed servers: no llama-server, no GPU, no model call, nothing written to the real databases
# (the fake harness copies rows of stored Runs 82 to 85 from benchmark.db, opened read-only, into a throwaway database).
# usage: bash sft/test_g4_chain.sh        (cwd anywhere; uses the Mac paths of the chain)
set -u
ROOT=/Users/USER/Documents/GitHub
SRC=${SRC:-$ROOT/Thesis-wt-training}
source $SRC/sft/g3lib.sh
PY=/Users/USER/Documents/GitHub/Thesis/.venv/bin/python
DB84=$ROOT/Thesis/benchmark/results/benchmark.db
T=$(mktemp -d); trap 'rm -rf $T; pkill -f G4RUN_TEST 2>/dev/null' EXIT
mkdir -p $T/h/benchmark; touch $T/h/benchmark/__init__.py
cat > $T/h/benchmark/run_all.py <<'PYEOF'
import os, sqlite3, sys, time, json
res = os.environ["BENCHMARK_RESULTS_DIR"]; os.makedirs(res, exist_ok=True)
assert os.environ.get("BFCL_SERVER_URL") == "http://127.0.0.1:8080", os.environ.get("BFCL_SERVER_URL")
mode = open(os.environ["FAKE_MODE_FILE"]).read().split() if os.path.exists(os.environ["FAKE_MODE_FILE"]) else ["84", "0", "0"]
src_run, n_err, sleep = int(mode[0]), int(mode[1]), int(mode[2])
import re
if re.search(r"_\w+b ", " ".join(sys.argv)): n_err = 0          # the rerun of a void run is clean
src = sqlite3.connect("file:" + os.environ["FAKE_DB"] + "?mode=ro", uri=True)
dst = sqlite3.connect(res + "/benchmark.db")
for t in ("runs", "responses", "scores", "eval_errors"):
    dst.execute(src.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0])
dst.execute("insert into runs select * from runs where 0")
cols = lambda t: [r[1] for r in src.execute(f"pragma table_info({t})")]
def copy(t, where):
    rows = src.execute(f"select * from {t} where {where}", (src_run,)).fetchall()
    dst.executemany(f"insert into {t} values ({','.join('?' * len(cols(t)))})", rows)
    return rows
copy("runs", "id=?"); copy("scores", "run_id=?"); copy("eval_errors", "run_id=?")
rows = copy("responses", "run_id=?")
if n_err:
    ids = [r[2] for r in rows][:n_err]
    for i in ids: dst.execute("update responses set result = result || ' ERROR: x' where entry_id=? and run_id=?", (i, src_run))
dst.commit(); time.sleep(sleep)
dst.execute("update runs set finished_at = '2026-11-11T00:00:00+00:00'"); dst.commit()
PYEOF
export FAKE_DB=$DB84 FAKE_MODE_FILE=$T/mode
ATT=$T/TEST; RL=$ATT/runlogs; mkdir -p $RL
say() { echo "$(date +%T) $*" >> $ATT/chain.log; }
die() { echo "DIE: $*" >> $ATT/chain.log; exit 9; }
start_server() { :; }; check_identity() { :; }; stop_server() { :; }; chain_left() { echo 99999; }
conn_ok() { return 0; }; swap_gb() { echo 1; }
HARNESS=$T/h; SRC=$SRC; RUN_CAP=1200; MAX_ERR=10; FMODE=iffc-state-ooda-slim; BAND_LO=82; BAND_HI=94; MARGIN=8; ARM=A; SRC_COMMIT=testcommit; HARNESS_COMMIT=dfdbc2b; echo fake > $T/g.gguf; GGUF=$T/g.gguf; SHA=$(shasum -a 256 $GGUF | cut -d' ' -f1); THINK=off; RENDER_REF=/nonexistent
echo '{"chosen": "A", "arms": {"A": {"gguf": "x", "gguf_sha256": "x", "think": "off"}}}' > $ATT/G4_SELECTION.json
source $SRC/sft/g4_lib.sh
export G4_MON_TICK=1 G4_EARLY=2 G4_NOPROG=4
fail=0; ok() { if [ "$1" = ok ]; then echo "ok   $2"; else echo "FAIL $2"; fail=$((fail+1)); fi; }
q() { $PY -c "import json; d=json.load(open('$ATT/$1')); print($2)"; }

echo "84 0 0" > $T/mode; run_valid U1 U > /dev/null 2>&1
[ "$(q RUN_U1.json "d['valid'], d['counted_passes'], d['errored_tasks']")" = "True 88 0" ] && ok ok "U1 recorded: 88 counted, 0 errored, valid" || ok no "U1 recorded ($(q RUN_U1.json "d['valid'], d['counted_passes']" 2>&1))"
stack_verdict > /dev/null 2>&1
[ "$VRC" = 0 ] && [ "$(q STACK_VERDICT.json "d['stack_reproduced']")" = "True" ] && ok ok "stack verdict: 88 is inside 82 to 94" || ok no "stack verdict inside the band"
echo "82 0 0" > $T/mode; rm -rf $ATT/results_U1 $ATT/RUN_U1.json; run_valid U1 U > /dev/null 2>&1; stack_verdict > /dev/null 2>&1
[ "$VRC" = 3 ] && ok ok "stack verdict: 71 is outside the band, VRC 3 (the chain would stop before the trained model)" || ok no "stack verdict outside the band (VRC=$VRC)"
echo "84 0 0" > $T/mode; rm -rf $ATT/results_U1 $ATT/RUN_U1.json; run_valid U1 U > /dev/null 2>&1
echo "83 0 0" > $T/mode; run_valid T1 T > /dev/null 2>&1
echo "85 0 0" > $T/mode; run_valid T2 T > /dev/null 2>&1
echo "82 0 0" > $T/mode; run_valid U2 U > /dev/null 2>&1
final_analysis > /dev/null 2>&1
[ "$(q G4_RESULT.json "d['scores']['U1'], d['scores']['T1'], d['scores']['T2'], d['scores']['U2'], d['trained_arm']")" = "88 84 72 71 A" ] && ok ok "four runs analysed (scores 88, 84, 72, 71; arm A)" || ok no "analysis scores"
# void path: 12 errored tasks in the first attempt, then a clean rerun
rm -rf $ATT/results_V1* $ATT/RUN_V1*; echo "84 12 0" > $T/mode
run_valid V1 U > /dev/null 2>&1
[ -f $ATT/RUN_V1.void.json ] && [ "$(q RUN_V1.json "d['valid'], d['errored_tasks']")" = "True 0" ] && ok ok "void run (12 errored) rerun once, then valid" || ok no "void rerun path"
# monitor kill path: a run that never connects is stopped by the monitor, twice, then the chain stops
conn_ok() { return 1; }
echo "84 0 300" > $T/mode; rm -rf $ATT/results_W1*; t0=$(date +%s)
( run_valid W1 U ) > /dev/null 2>&1; rc=$?
el=$(( $(date +%s) - t0 ))
[ $rc -ne 0 ] && [ $el -lt 60 ] && [ -f $ATT/STOP_W1 ] && ok ok "monitor stopped the idle run within ${el}s; run void twice -> chain stops (rc $rc)" || ok no "monitor kill path (rc=$rc, ${el}s)"
pgrep -f G4RUN_TEST > /dev/null && ok no "leftover fake harness process" || ok ok "no leftover harness process"
echo "failures: $fail"; echo "--- chain.log tail"; tail -12 $ATT/chain.log; exit $fail
