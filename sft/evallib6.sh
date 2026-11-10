# Round 6 helpers (source from the Thesis-wt-r6 chain). Read-only data root: the Round 5 worktree (R5).
# v3: answers Codex r36. Owned-server lifecycle (never pkill), exact identity checks, enforced caps, unique attempt dirs.
export PYTHONDONTWRITEBYTECODE=1                      # imports from the Round 5 worktree must not create __pycache__ there
R5=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill
ABS6=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6
BAR=/Users/sanskarthapa/.cache/huggingface/hub/models--bartowski--google_gemma-4-E2B-it-GGUF/snapshots/81012ba3538e061d5ee003f11f25335b17f82e2d/google_gemma-4-E2B-it-Q4_K_M.gguf
BAR_SHA16=923c4c86177d2ee1
BUILD=b10894-d344123fe
LLAMA=/Users/sanskarthapa/Documents/GitHub/Thesis/llama.cpp/build/bin/llama-server
PY=/Users/sanskarthapa/Documents/GitHub/Thesis/.venv/bin/python
D6=$ABS6/sft/data/distill/r6
F=$R5/sft/data/distill/f/tasks_F.jsonl
K=$R5/sft/data/distill/k/tasks_K2.jsonl
F_SHA=7a45cd08b7b2226ebad135f1e7e9aa8b4fdd098801742ddd673df08262f13709
K2SHA=e2ac40e4ece912d07a3646a3b6a787c4dd880bd8a2b7dc1f2a4e966bfc98c8ef
FK_CAP=9000            # seconds per FK composite (F and K2 together), registered: 2.5 h
CHAIN_CAP=57600        # seconds for the whole chain across stages, registered: 16 h
OWNED_PID=""
BTPID=""
PROBE_INVALID=""

kill_tree() {          # kill_tree PID SIGNAL : the process and all its descendants (our own children only)
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c" "$2"; done
  kill -"$2" "$1" 2>/dev/null || true
}
say() { echo "$(date +%T) $*"; }
stop_server() {        # kills ONLY the processes this chain started (server and the current bounded child)
  if [ -n "$BTPID" ]; then
    kill -TERM -- "-$BTPID" 2>/dev/null || true       # the bounded job runs in its own process group (set -m in bt)
    kill_tree "$BTPID" TERM
    for _ in $(seq 1 10); do kill -0 -- "-$BTPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$BTPID" 2>/dev/null || true       # TERM-resistant members and orphaned descendants do not survive cleanup
    kill_tree "$BTPID" KILL
  fi
  BTPID=""
  if [ -n "$OWNED_PID" ] && kill -0 "$OWNED_PID" 2>/dev/null; then
    kill "$OWNED_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$OWNED_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$OWNED_PID" 2>/dev/null || true
  fi
  OWNED_PID=""
}
trap stop_server EXIT
trap 'echo "$(date +%T) ABORT: signal INT"; stop_server; exit 130' INT
trap 'echo "$(date +%T) ABORT: signal TERM"; stop_server; exit 143' TERM
die() { echo "$(date +%T) ABORT: $*"; stop_server; exit 1; }
abort() { echo "$(date +%T) ABORT (nothing started, nothing killed): $*"; exit 1; }   # before the chain owns anything

init_attempt() {       # init_attempt STAGE : atomically allocated unique attempt dir, all output goes there
  mkdir -p "$D6/attempts" || abort "cannot create $D6/attempts"
  ATT=$(mktemp -d "$D6/attempts/$(date +%Y%m%d_%H%M%S)_$1.XXXXXX") || abort "cannot allocate an attempt directory"
  mkdir -p "$ATT/runlogs" "$ATT/evals" || abort "cannot create attempt subdirectories"
  RL=$ATT/runlogs
  exec >> "$ATT/chain.log" 2>&1 || abort "cannot redirect to the attempt log"
}

# bt CMD...: run any blocking phase under the REMAINING whole-chain budget (TERM, then KILL after 30 s); fails if the budget is spent
bt() {
  local left; left=$(chain_left)
  [ "$left" -gt 0 ] || die "whole-chain cap (16 h) reached before: $*"
  set -m                              # own process group for the bounded job, so cleanup can signal the whole group
  timeout -k 30 "$left" "$@" &       # background + wait, so INT/TERM traps run immediately instead of after the child finishes
  BTPID=$!
  set +m
  wait "$BTPID"
  local rc=$?
  BTPID=""
  return $rc
}

chain_clock() {        # persists the chain start across stages; must be a plain integer
  if [ ! -f "$D6/chain_start_epoch" ]; then date +%s > "$D6/chain_start_epoch"; fi
  CHAIN_START=$(cat "$D6/chain_start_epoch")
  case "$CHAIN_START" in ''|*[!0-9]*) abort "chain_start_epoch is not an integer: '$CHAIN_START'";; esac
}
chain_left() { echo $(( CHAIN_CAP - ($(date +%s) - CHAIN_START) )); }
cu() {                 # curl bounded by min(10 s, remaining chain budget)
  local left; left=$(chain_left); [ "$left" -gt 0 ] || die "whole-chain cap reached (http call)"
  curl -s --max-time "$(min2 10 "$left")" "$@"
}
min2() { [ "$1" -lt "$2" ] && echo "$1" || echo "$2"; }

preflight() {
  [ "$(git branch --show-current)" = "thesis/distill-r6-diagnostics" ] || abort "wrong branch"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || abort "tracked files are modified; commit first so the run is reproducible"
  [ "$(shasum -a 256 "$K" | cut -d' ' -f1)" = "$K2SHA" ] || abort "K2 hash changed"
  [ "$(shasum -a 256 "$F" | cut -d' ' -f1)" = "$F_SHA" ] || abort "F hash changed"
  [ "$(shasum -a 256 "$BAR" | cut -c1-16)" = "$BAR_SHA16" ] || abort "bar GGUF sha256 prefix is not $BAR_SHA16"
  if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then abort "port 8080 already in use (another server running); not touching it"; fi
  if pgrep -f "mlx_lm|train_mlx" >/dev/null; then abort "an MLX training process is running"; fi
  # the Run 69 flags used by sft/serve_gguf.sh must be the ones this chain uses
  $PY - <<'PYEOF' || abort "serve_gguf.sh flags differ from this chain's flags"
import shlex
text = open("sft/serve_gguf.sh").read().replace("\\\n", " ")          # join backslash-continued lines
line = [l for l in text.splitlines() if "llama-server" in l and "nohup" in l][0]
toks = shlex.split(line.replace('${3:+--lora "$3"}', "").replace('"$1"', "MODEL"))
toks = toks[toks.index("--jinja"):]
toks = toks[:toks.index(">")] if ">" in toks else toks
want = "--jinja -fa on -m MODEL --host 127.0.0.1 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024".split()
raise SystemExit(0 if toks == want else 1)
PYEOF
  echo "F sha256 $F_SHA, K2 sha256 $K2SHA, bar prefix $BAR_SHA16, commit $(git rev-parse HEAD)"
}

# start_server NAME [LORA_GGUF]  (owned process; cleanup kills only this PID)
start_server() {
  local name=$1 lora=${2:-}
  [ -z "$OWNED_PID" ] || die "a server is already owned"
  if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then die "port 8080 became busy; not touching it"; fi
  nohup "$LLAMA" --jinja -fa on -m "$BAR" ${lora:+--lora "$lora"} --host 127.0.0.1 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024 > "$RL/server_$name.log" 2>&1 &
  OWNED_PID=$!
  for _ in $(seq 1 120); do
    [ "$(chain_left)" -gt 0 ] || die "whole-chain cap reached during server start"
    kill -0 "$OWNED_PID" 2>/dev/null || die "server $name exited during start"
    cu http://127.0.0.1:8080/health | grep -q '"ok"' && break
    sleep 2
  done
  cu http://127.0.0.1:8080/health | grep -q '"ok"' || die "server $name not healthy"
  [ "$(lsof -nP -tiTCP:8080 -sTCP:LISTEN | head -1)" = "$OWNED_PID" ] || die "port 8080 listener is not the server this chain started"
  ps -ww -o command= -p "$OWNED_PID" > "$RL/argv_$name.txt"
}

# check_identity NAME EXPECTED_SCALE [LORA_GGUF]: structured checks on /props and /lora-adapters; JSON persisted
check_identity() {
  local name=$1 scale=$2 lora=${3:-}
  [ "$(chain_left)" -gt 0 ] || die "whole-chain cap reached before identity check"
  if [ -n "$lora" ] && [ "$scale" != "1.0" ]; then
    cu -X POST http://127.0.0.1:8080/lora-adapters -H 'Content-Type: application/json' -d "[{\"id\":0,\"scale\":$scale}]" > "$RL/setscale_$name.out" || die "set scale failed $name"
  fi
  cu http://127.0.0.1:8080/props > "$RL/props_$name.json"
  cu http://127.0.0.1:8080/lora-adapters > "$RL/lora_$name.json"
  $PY - "$RL/props_$name.json" "$RL/lora_$name.json" "$BAR" "$BUILD" "$scale" "$lora" <<'PYEOF' || die "identity check failed for $name (see $RL/props_$name.json and lora_$name.json)"
import json, os, sys
props = json.load(open(sys.argv[1])); la = json.load(open(sys.argv[2]))
bar, build, scale, lora = sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6]
ok = os.path.realpath(props.get("model_path", "")) == os.path.realpath(bar) and props.get("build_info") == build and props.get("total_slots") == 4
if lora:
    ok = ok and len(la) == 1 and la[0].get("id") == 0 and os.path.realpath(la[0]["path"]) == os.path.realpath(lora) and abs(float(la[0]["scale"]) - scale) < 1e-9
else:
    ok = ok and la == []
sys.exit(0 if ok else 1)
PYEOF
  [ -z "$lora" ] || shasum -a 256 "$lora" > "$RL/adapter_sha_$name.txt"
  say "identity ok: $name (scale $scale)"
}

# run6 NAME BANKFILE EXPECTED_N OUTDIR TIMEOUT_SECONDS
run6() {
  local to=$5
  [ "$to" -gt 0 ] || die "no time left for $1 (cap)"
  bt timeout -k 30 "$to" $PY -m benchmark.distill_student_eval --tasks "$2" --log-dir "$4" --label "r6_$1" --samples 0 --workers 4 --fc-mode iffc-ooda-card > "$RL/eval_$1.out" 2>&1 || die "eval $1 failed or exceeded its cap"
  $PY - "$4" "$3" <<'PYEOF' || die "eval $1 invalid (counts/api errors)"
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); n = int(sys.argv[2]); v = [json.loads(l) for l in open(d / "verdicts.jsonl")]
u = len({x.get("task_id") or x.get("id") for x in v}); api = sum(bool(x.get("api_error")) for x in v)
p = sum(bool(x.get("valid")) for x in v)
print(json.dumps({"rows": len(v), "unique_tasks": u, "api_error_episodes": api, "passes": p}))
sys.exit(0 if (len(v) == n and u == n and api == 0) else 1)
PYEOF
  say "$1 finished"
}

# fk6 NAME : F then K2 under one 2.5 h composite deadline and the chain deadline
fk6() {
  local t0 left to
  t0=$(date +%s)
  to=$(min2 "$FK_CAP" "$(chain_left)"); run6 "$1_F" "$F" 81 "$ATT/evals/F_$1" "$to"
  left=$(( FK_CAP - ($(date +%s) - t0) ))
  to=$(min2 "$left" "$(chain_left)"); run6 "$1_K" "$K" 74 "$ATT/evals/K2_$1" "$to"
}

# probe6 NAME : 70 frozen prompts (exactly one think marker asserted by the script); run with the Round 5 worktree as cwd, absolute output
probe6() {
  pushd "$R5" > /dev/null || die "cannot enter R5"
  bt $PY -B sft/r5_precision_probe.py run --frozen "$R5/sft/data/distill/r5/probe_frozen.json" --tag "$1" --out "$ATT/probe_$1.json" > "$RL/probe_$1.out" 2>&1
  local rc=$?
  popd > /dev/null
  if [ $rc -ne 0 ]; then say "WARNING: probe $1 FAILED to run or was invalid (rc $rc); exploratory, FK continues"; PROBE_INVALID="$PROBE_INVALID $1"; return 0; fi
  say "probe $1 done: $(tail -1 "$RL/probe_$1.out" | cut -c1-200)"
  $PY -B sft/r6_tools.py probe-check --probe "$ATT/probe_$1.json" > "$RL/probe_check_$1.out" 2>&1 || { say "WARNING: probe $1 INVALID (exploratory; FK continues): $(head -c 300 "$RL/probe_check_$1.out")"; PROBE_INVALID="$PROBE_INVALID $1"; }
}

# arm NAME SCALE [LORA] : own server -> identity -> probe -> FK -> stop
arm() {
  local name=$1 scale=$2 lora=${3:-}
  [ "$(chain_left)" -gt 0 ] || die "whole-chain cap reached before $name"
  start_server "$name" "$lora"
  check_identity "$name" "$scale" "$lora"
  probe6 "$name"
  fk6 "$name"
  stop_server
}
