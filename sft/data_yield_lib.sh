# Data-yield check helpers (source from sft/data_yield_chain.sh). API + CPU only: no GPU, no llama-server, no port 8080.
export PYTHONDONTWRITEBYTECODE=1
R5=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill
ABSW=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-data
PY=/Users/sanskarthapa/Documents/GitHub/Thesis/.venv/bin/python
DD=$R5/sft/data/distill
YD=$ABSW/sft/data/distill/yield
F_SHA=7a45cd08b7b2226ebad135f1e7e9aa8b4fdd098801742ddd673df08262f13709
K2SHA=e2ac40e4ece912d07a3646a3b6a787c4dd880bd8a2b7dc1f2a4e966bfc98c8ef
TOTAL_CAP_USD=5.0
CHILD=""

say() { echo "$(date +%T) $*"; }
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c" "$2"; done; kill -"$2" "$1" 2>/dev/null || true; }
cleanup() {
  if [ -n "$CHILD" ]; then
    kill -TERM -- "-$CHILD" 2>/dev/null || true; kill_tree "$CHILD" TERM
    for _ in $(seq 1 10); do kill -0 -- "-$CHILD" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$CHILD" 2>/dev/null || true; kill_tree "$CHILD" KILL
  fi
  CHILD=""
}
trap cleanup EXIT
trap 'echo "$(date +%T) ABORT: signal INT"; cleanup; exit 130' INT
trap 'echo "$(date +%T) ABORT: signal TERM"; cleanup; exit 143' TERM
die() { echo "$(date +%T) ABORT: $*"; cleanup; exit 1; }

# bt SECONDS CMD... : run a blocking step in its own process group with a hard time limit (TERM, then KILL after 30 s)
bt() {
  local secs=$1; shift
  set -m
  timeout -k 30 "$secs" nice -n 10 "$@" &
  CHILD=$!
  set +m
  wait "$CHILD"; local rc=$?
  CHILD=""
  return $rc
}

init_attempt() {
  mkdir -p "$YD/attempts" || { echo "cannot create $YD/attempts"; exit 1; }
  ATT=$(mktemp -d "$YD/attempts/$(date +%Y%m%d_%H%M%S)_$1.XXXXXX") || { echo "cannot allocate attempt dir"; exit 1; }
  mkdir -p "$ATT/author" "$ATT/solve" "$ATT/filtered" "$ATT/rows" "$ATT/rows_clean" "$ATT/smoke" "$ATT/runlogs" || { echo "mkdir failed"; exit 1; }
  exec >> "$ATT/chain.log" 2>&1 || { echo "cannot redirect"; exit 1; }
}

preflight() {
  [ "$(git branch --show-current)" = "thesis/distill-data-scale" ] || die "wrong branch"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked files are modified; commit first"
  [ "$(shasum -a 256 "$DD/k/tasks_K2.jsonl" | cut -d' ' -f1)" = "$K2SHA" ] || die "K2 hash changed"
  [ "$(shasum -a 256 "$DD/f/tasks_F.jsonl" | cut -d' ' -f1)" = "$F_SHA" ] || die "F hash changed"
  [ -n "${OPENROUTER_API_KEY:-}" ] || die "OPENROUTER_API_KEY is not set in this environment"   # presence only; never printed
  for f in $EX $HELD; do [ -s "$f" ] || die "missing or empty input file $f"; done
  [ "$(df -k "$YD" | awk 'NR==2{print $4}')" -gt 5000000 ] || die "less than 5 GB free"
  if pgrep -f "distill_tasks|distill_solve|yield_author" > /dev/null; then die "another authoring/solve process is running"; fi
  diff -rq "$ABSW/benchmark" "/Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6/benchmark" -x __pycache__ > /dev/null || die "benchmark/ differs from the Round 6 diagnostics worktree (harness code must be identical to what is evaluated)"
  command -v timeout > /dev/null || die "timeout(1) not found"
  # cross-attempt spend ledger: every earlier real attempt counts against the same $5 total
  SPENT_BEFORE=$($PY -B - "$YD/attempts" <<'PYEOF'
import glob, json, sys
tot, missing = 0.0, 0
for p in glob.glob(sys.argv[1] + "/*_part1.*/*/calls.jsonl") + glob.glob(sys.argv[1] + "/*_part1.*/*/log/calls.jsonl"):
    for l in open(p):
        if l.strip():
            c = json.loads(l).get("cost_usd")
            if c is None:
                missing += 1
            else:
                tot += float(c)
for p in glob.glob(sys.argv[1] + "/*_part1.*/*/log/guard_flags.json") + glob.glob(sys.argv[1] + "/*_part1.*/*/guard_flags.json"):
    if json.load(open(p)).get("missing_cost_responses"):
        missing += 1
print("MISSING" if missing else round(tot, 4))
PYEOF
)
  [ "$SPENT_BEFORE" != "MISSING" ] || die "an earlier attempt has responses without cost accounting; reconcile its charges with the provider before any new paid run"
  echo "spent in earlier attempts: $SPENT_BEFORE"
  $PY -B -c "import sys; sys.exit(0 if float('$SPENT_BEFORE') + 3.65 <= $TOTAL_CAP_USD else 1)" || die "earlier attempts spent \$$SPENT_BEFORE; another run (caps 3.65) would exceed the \$$TOTAL_CAP_USD total"
  echo "commit $(git rev-parse HEAD); F $F_SHA; K2 $K2SHA"
  for f in $EX $HELD; do echo "input $(shasum -a 256 "$f" | cut -c1-16) $f"; done
}
