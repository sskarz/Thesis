# G4 chain functions (sourced by g4_chain.sh; tests source it with stubs). Needs g3lib.sh helpers and the variables set in g4_chain.sh.
conn_ok() { lsof -nP -iTCP:8080 -sTCP:ESTABLISHED 2>/dev/null | grep -q .; }          # a client is connected to the server (tests override this)
serve() {            # serve NAME KIND : server + identity (+ adapter hash for the trained kind)
  if [ "$2" = U ]; then start_server G4$1 "" on; check_identity G4$1 1.0 ""
  else start_server G4$1 $GGUF $THINK; check_identity G4$1 1.0 $GGUF; [ "$(shasum -a 256 $GGUF | cut -d' ' -f1)" = "$SHA" ] || die "GGUF hash changed before $1"; fi
}
parity_phase() {     # render capture on both servers before any run
  for kind in U T; do
    serve PAR$kind $kind
    th=on; [ $kind = T ] && th=$THINK
    ref=""; [ $kind = U ] && ref="--ref $RENDER_REF"
    (cd $HARNESS && bt $PY -B $SRC/sft/g4_tools.py parity --think $th --out $ATT/parity_$kind.json $ref > $RL/parity_$kind.out 2>&1) || die "prompt-render capture failed on the $kind server: $(tail -c 300 $RL/parity_$kind.out)"
    say "render capture ok ($kind, thinking $th): $(tr -d '\n ' < $ATT/parity_$kind.json | cut -c1-260)"
    stop_server
  done
}
monitor() {          # monitor NAME RES TAG : watches the run every 60 s; kills only this run's harness process on a stop condition
  local name=$1 res=$2 tag=$3 t0 n="" e="" el; t0=$(date +%s)
  while sleep ${G4_MON_TICK:-60}; do
    el=$(( $(date +%s) - t0 ))
    $PY -c "import sys; sys.exit(0 if float('$(swap_gb)') > 8.0 else 1)" && { echo "swap above 8 GB" > $ATT/STOP_$name; pkill -TERM -f "$tag"; break; }
    if [ $el -ge ${G4_EARLY:-150} ] && [ ! -e $ATT/early_$name ]; then
      touch $ATT/early_$name
      conn_ok || { echo "no connection to 8080 after 150 s" > $ATT/STOP_$name; pkill -TERM -f "$tag"; break; }
    fi
    if [ -f $res/benchmark.db ]; then
      $PY -B $SRC/sft/g4_tools.py collect --results-dir $res --out $ATT/partial_$name.json --name $name --partial > /dev/null 2>&1
      n=$($PY -c "import json; d=json.load(open('$ATT/partial_$name.json')); print(d['responses'], d['errored_tasks'])" 2>/dev/null)
      echo "$(date +%T) $name elapsed $el s responses/errored $n swap $(swap_gb)" >> $ATT/monitor_$name.log
      e=${n##* }
      [ -n "$e" ] && [ "$e" -gt $MAX_ERR ] 2>/dev/null && { echo "more than $MAX_ERR errored tasks ($e)" > $ATT/STOP_$name; pkill -TERM -f "$tag"; break; }
    fi
    if [ $el -ge ${G4_NOPROG:-1500} ] && { [ -z "$n" ] || [ "${n%% *}" = "0" ]; } ; then echo "no task finished after 1500 s" > $ATT/STOP_$name; pkill -TERM -f "$tag"; break; fi
  done
}
run_one() {          # run_one NAME KIND -> 0 valid run recorded in RUN_NAME.json, 1 void
  local name=$1 kind=$2 res tag to
  res=$ATT/results_$name; tag="G4RUN_$(basename $ATT)_$name"
  serve $name $kind
  [ ! -e $res ] || die "results folder $res exists"; mkdir -p $res
  to=$(min2 $RUN_CAP "$(chain_left)"); [ "$to" -gt 600 ] || die "no time left for $name"
  rm -f $ATT/STOP_$name; monitor $name $res $tag & MONPID=$!
  say "START $name ($kind) cap ${to}s"
  bt timeout -k 30 "$to" bash -c "cd $HARNESS && BENCHMARK_RESULTS_DIR=$res BFCL_SERVER_URL=http://127.0.0.1:8080 PYTHONDONTWRITEBYTECODE=1 exec $PY -m benchmark.run_all --model gemma4-e2b --only-multi-turn --categories multi_turn_base --fc-mode $FMODE --notes '$tag $kind $FMODE bar Q4_K_M'" > $RL/run_$name.out 2>&1
  local rc=$?; kill $MONPID 2>/dev/null; MONPID=""
  stop_server
  say "END $name rc=$rc"
  if [ -f $ATT/STOP_$name ]; then say "run $name stopped by the monitor: $(cat $ATT/STOP_$name)"; return 1; fi
  [ $rc -eq 0 ] || { say "run $name exited rc=$rc"; return 1; }
  $PY -B $SRC/sft/g4_tools.py collect --results-dir $res --out $ATT/RUN_$name.json --name $name > $RL/collect_$name.out 2>&1 || return 1
  $PY -c "import json,sys; d=json.load(open('$ATT/RUN_$name.json')); print('$name', 'stored', d['score'], 'counted', d['counted_passes'], 'errored', d['errored_tasks'], 'valid', d['valid'], d.get('void_reason')); sys.exit(0 if d['valid'] else 1)"
}
run_valid() {        # run_valid NAME KIND : a void run is rerun once under the same plan (registered)
  run_one $1 $2 && return 0
  say "run $1 void; rerunning once as ${1}b"; mv $ATT/RUN_$1.json $ATT/RUN_$1.void.json 2>/dev/null
  run_one ${1}b $2 || die "run $1 void twice; chain stops"
  cp $ATT/RUN_${1}b.json $ATT/RUN_$1.json
}

stack_verdict() {     # writes STACK_VERDICT.json before any trained run; sets VRC (0 reproduced, 3 not)
$PY - $ATT <<PYEOF
import json, sys
a = sys.argv[1]; d = json.load(open(f"{a}/RUN_U1.json")); s = len(d["passed_ids"])
v = {"run": "U1", "score": s, "band": [$BAND_LO, $BAND_HI], "stack_reproduced": $BAND_LO <= s <= $BAND_HI, "run_84_score": 88, "run_83_score": 84}
import datetime; v["written_at"] = datetime.datetime.now().isoformat(timespec="seconds"); v["note"] = "written before any trained-model run starts"
json.dump(v, open(f"{a}/STACK_VERDICT.json", "w"), indent=1); print(json.dumps(v)); sys.exit(0 if v["stack_reproduced"] else 3)
PYEOF
VRC=$?
}
final_analysis() {
$PY - $ATT <<PYEOF
import json, sys
sys.path.insert(0, "$SRC/sft")
from g4_analyze import analyze
a = sys.argv[1]; runs = {k: json.load(open(f"{a}/RUN_{k}.json"))["passed_ids"] for k in ("U1", "T1", "T2", "U2")}
d = {"margin": $MARGIN, "all_tasks": [f"multi_turn_base_{i}" for i in range(200)], "stack_ok": True, "runs": runs}
json.dump(d, open(f"{a}/PAIRS.json", "w")); r = analyze(d); r["trained_arm"] = "$ARM"; r["code_commit"] = "$SRC_COMMIT"; r["harness_commit"] = "$HARNESS_COMMIT"; r["selection"] = json.load(open(f"{a}/G4_SELECTION.json"))["arms"]
json.dump(r, open(f"{a}/G4_RESULT.json", "w"), indent=1); print(json.dumps({k: r[k] for k in ("scores", "pair1", "pair2", "combined_sign_test", "outcome")}))
PYEOF
}
