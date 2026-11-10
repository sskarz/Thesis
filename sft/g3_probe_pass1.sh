#!/bin/bash
# v2 (attempt 2 of the same approved run; attempt 1 in probe_pass1/ is void: 0 requests reached the model because distill_student_eval ignores the server port). Exploratory probe of arm A's PASS-1 checkpoint on bank F only. Not the registered screen. Pre-registration: docs/DISTILLATION_G3_PREREGISTRATION.md, amendment "Probe of the pass-1 checkpoint".
# Own server on port 8081 (bar GGUF + pass-1 LoRA, slim mode, thinking off, -np 2, -c 32768). Never touches port 8080 or the training process.
# known gap (recorded): the slowdown guard reads train_log.jsonl, which does not move during the end-of-pass validation sweep.
# usage: bash sft/g3_probe_pass1.sh <arm A attempt dir>      (waits for adapters_pass1/, then runs)
export PYTHONDONTWRITEBYTECODE=1
A=$1; [ -d "$A/train" ] || { echo "usage: $0 <attempt dir>"; exit 2; }
R6=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6; cd $R6 || exit 2
SLIM=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-slim
PY=/Users/sanskarthapa/Documents/GitHub/Thesis/.venv/bin/python
LLAMA=/Users/sanskarthapa/Documents/GitHub/Thesis/llama.cpp/build/bin/llama-server
BAR=/Users/sanskarthapa/.cache/huggingface/hub/models--bartowski--google_gemma-4-E2B-it-GGUF/snapshots/81012ba3538e061d5ee003f11f25335b17f82e2d/google_gemma-4-E2B-it-Q4_K_M.gguf
F=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/data/distill/f/tasks_F.jsonl
F_SHA=7a45cd08b7b2226ebad135f1e7e9aa8b4fdd098801742ddd673df08262f13709
BUILD=b10894-d344123fe
P=$A/probe_pass1_r2; mkdir -p $P; LOG=$P/probe.log; exec >> $LOG 2>&1
say() { echo "$(date +%T) $*"; }
SRV=""; EVP=""
stop_all() { [ -n "$EVP" ] && kill -TERM "$EVP" 2>/dev/null && pkill -TERM -P "$EVP" 2>/dev/null; sleep 3; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; SRV=""; EVP=""; }
trap stop_all EXIT
die() { say "ABORT: $*"; echo "$*" > $P/ABORTED; stop_all; exit 1; }
swap_used_gb() { sysctl -n vm.swapusage | $PY -c "import re,sys; m=re.search(r'used = ([0-9.]+)([MG])', sys.stdin.read()); v=float(m.group(1)); print(v/1024 if m.group(2)=='M' else v)"; }
mean20() { $PY - $A/train/train_log.jsonl <<'PYEOF'
import json, sys
r = []
for l in open(sys.argv[1]):
    if '"micro"' in l and '"s"' in l:
        try: r.append(json.loads(l)["s"])
        except Exception: pass
r = r[-20:]; print(round(sum(r) / len(r), 3) if r else 0)
PYEOF
}
say "waiting for the pass-1 checkpoint in $A/train/adapters_pass1"
while [ ! -f $A/train/adapters_pass1/adapter_config.json ]; do sleep 30; pgrep -f g3_train.py > /dev/null || die "training process is not running and no checkpoint exists"; done
sleep 20; CK_TIME=$(date +%T); say "CHECKPOINT on disk at $CK_TIME"; echo $CK_TIME > $P/checkpoint_time.txt
SW=$(swap_used_gb); say "free-memory record: $(vm_stat | head -3 | tr '\n' ' ') | swap used GB $SW"; sysctl vm.swapusage > $P/swap_before.txt; vm_stat > $P/vmstat_before.txt
$PY -c "import sys; sys.exit(0 if float('$SW') <= 8.0 else 1)" || die "swap used $SW GB is above 8 GB; refusing to start"
[ "$(shasum -a 256 $F | cut -d' ' -f1)" = "$F_SHA" ] || die "F hash changed"
lsof -nP -iTCP:8081 -sTCP:LISTEN > /dev/null 2>&1 && die "port 8081 busy"
BASE_S=$(mean20); say "pre-probe mean seconds per micro-batch (last 20): $BASE_S"; echo $BASE_S > $P/baseline_s_per_micro.txt
cp -r $A/train/adapters_pass1 $P/adapter_src            # a copy, so the probe never reads a file the trainer may rewrite
nice -n 10 $PY -B sft/mlx_adapter_to_gguf_lora.py $P/adapter_src $P/pass1.gguf > $P/convert.log 2>&1 || die "convert failed"
nice -n 10 $PY -B sft/r6_tools.py gguf-check --adapter-dir $P/adapter_src --src-dir $P/adapter_src --gguf $P/pass1.gguf > $P/gguf_check.log 2>&1 || die "GGUF audit failed"
say "converted and audited: $(tail -1 $P/gguf_check.log | cut -c1-200)"
SW=$(swap_used_gb); $PY -c "import sys; sys.exit(0 if float('$SW') <= 8.0 else 1)" || die "swap used $SW GB above 8 GB after conversion; refusing to start the server"
nohup nice -n 5 $LLAMA --jinja -fa on -m $BAR --lora $P/pass1.gguf --chat-template-kwargs '{"enable_thinking":false}' --host 127.0.0.1 --port 8081 -ngl 99 -c 32768 --temp 0.0 -np 2 -cb -ub 1024 > $P/server.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do kill -0 $SRV 2>/dev/null || die "probe server exited during start"; curl -s --max-time 5 http://127.0.0.1:8081/health | grep -q '"ok"' && break; sleep 2; done
curl -s --max-time 5 http://127.0.0.1:8081/health | grep -q '"ok"' || die "probe server not healthy"
[ "$(lsof -nP -tiTCP:8081 -sTCP:LISTEN | head -1)" = "$SRV" ] || die "port 8081 listener is not our server"
curl -s --max-time 10 http://127.0.0.1:8081/props > $P/props.json; curl -s --max-time 10 http://127.0.0.1:8081/lora-adapters > $P/lora.json
ps -ww -o command= -p $SRV > $P/argv.txt; grep -q enable_thinking $P/argv.txt || die "server not started with thinking off"
$PY - $P/props.json $P/lora.json $BAR $BUILD $P/pass1.gguf <<'PYEOF' || die "identity check failed"
import json, os, sys
props = json.load(open(sys.argv[1])); la = json.load(open(sys.argv[2])); bar, build, lora = sys.argv[3:6]
ok = os.path.realpath(props.get("model_path", "")) == os.path.realpath(bar) and props.get("build_info") == build and props.get("total_slots") == 2
ok = ok and len(la) == 1 and os.path.realpath(la[0]["path"]) == os.path.realpath(lora) and abs(float(la[0]["scale"]) - 1.0) < 1e-9
sys.exit(0 if ok else 1)
PYEOF
shasum -a 256 $P/pass1.gguf > $P/adapter_sha.txt; say "identity ok (port 8081, 2 slots, scale 1.0)"
LAUNCH='import sys
from benchmark import teacher_api as T, distill_student_eval as E
_o = T.TeacherConfig.local
def _l(**kw):
    kw.setdefault("base_url", "http://127.0.0.1:8081/v1")
    return _o(**kw)
T.TeacherConfig.local = staticmethod(_l)
sys.exit(E.main(sys.argv[1:]))'
# prove the wrapper before the bank: the config must point at 8081, kwargs must pass through, and one non-bank request must reach the model
(cd $SLIM && $PY -c "$LAUNCH" --help > /dev/null 2>&1) || die "launcher does not import the harness"
(cd $SLIM && $PY -c 'import sys
from benchmark import teacher_api as T
_o = T.TeacherConfig.local
def _l(**kw):
    kw.setdefault("base_url", "http://127.0.0.1:8081/v1")
    return _o(**kw)
T.TeacherConfig.local = staticmethod(_l)
c = T.TeacherConfig.local(max_concurrency=2)
print(c.base_url, c.max_concurrency, c.backend)
sys.exit(0 if c.base_url == "http://127.0.0.1:8081/v1" and c.max_concurrency == 2 and c.backend == "local" else 1)') > $P/wrapper_check.out 2>&1 || die "wrapper check failed: $(cat $P/wrapper_check.out)"
say "wrapper check: $(cat $P/wrapper_check.out)"
curl -s --max-time 120 http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}' > $P/hi.json
$PY -c "import json,sys; d=json.load(open('$P/hi.json')); sys.exit(0 if d.get('choices') else 1)" || die "the hi request got no completion from 8081: $(head -c 200 $P/hi.json)"
say "hi request completed on 8081: $(head -c 160 $P/hi.json)"
(cd $SLIM && exec nice -n 5 timeout -k 30 7200 $PY -c "$LAUNCH" --tasks $F --log-dir $P/evals/F_pass1 --label probe_pass1_r2_F --samples 0 --workers 2 --fc-mode iffc-state-ooda-slim) > $P/eval.out 2>&1 &
EVP=$!; T0=$(date +%s); say "bank F probe started"
while kill -0 $EVP 2>/dev/null; do
  sleep 60
  M=$(mean20); SW=$(swap_used_gb); echo "$(date +%T) mean20_s_per_micro $M swap_GB $SW" >> $P/monitor.log
  $PY -c "import sys; sys.exit(0 if float('$SW') > 8.0 else 1)" && { say "STOP: swap used $SW GB above 8 GB during the run"; echo "swap $SW" > $P/STOPPED; stop_all; exit 3; }
  if [ -z "$EARLY_DONE" ] && [ $(( $(date +%s) - T0 )) -ge 120 ]; then
    EARLY_DONE=1
    # this llama.cpp build logs no request lines, so completed requests are read from the harness's own calls.jsonl (one line per completed model call, with usage and the server build)
    $PY - $P/evals/F_pass1/calls.jsonl $BUILD <<'PYEOF' || die "early check: no completed model call recorded from 8081 after 120 s"
import json, sys
n = 0
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("finish_reason") and (r.get("usage") or {}).get("completion_tokens") is not None and r.get("system_fingerprint") == sys.argv[2]: n += 1
print("completed model calls:", n); sys.exit(0 if n >= 1 else 1)
PYEOF
    lsof -nP -iTCP:8080 -sTCP:LISTEN > /dev/null 2>&1 && die "early check: something listens on 8080"
    say "early check ok: completed model calls recorded, 8080 has no listener"
  fi
  pgrep -f g3_train.py > /dev/null || { say "TRAINING PROCESS GONE"; echo "training process gone" > $P/STOPPED; stop_all; exit 3; }
  if $PY -c "import sys; sys.exit(0 if float('$M') > 3 * float('$BASE_S') else 1)"; then say "STOP: training mean $M > 3 x baseline $BASE_S"; echo "slowdown $M vs $BASE_S" > $P/STOPPED; stop_all; exit 3; fi
done
wait $EVP; say "probe eval ended rc $?"; stop_all
$PY - $P/evals/F_pass1 <<'PYEOF' || die "probe result invalid"
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); v = [json.loads(l) for l in open(d / "verdicts.jsonl")]
u = len({x.get("task_id") or x.get("id") for x in v}); api = sum(bool(x.get("api_error")) for x in v); p = sum(bool(x.get("valid")) for x in v)
out = {"bank": "F", "rows": len(v), "unique_tasks": u, "api_error_episodes": api, "passes": p, "base_F_on": [30, 33], "base_F_off": 27, "note": "exploratory, not the registered screen"}
json.dump(out, open(d.parent.parent / "PROBE_RESULT.json", "w"), indent=1); print(json.dumps(out)); sys.exit(0 if (len(v) == 81 and u == 81 and api == 0) else 1)
PYEOF
say "PROBE DONE $(( $(date +%s) - T0 )) s"
