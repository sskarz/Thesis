#!/bin/bash
# Round 6 registered follow-up: ONE additional base run (B3c) on FK with the same procedure as the stage 1 base arms.
# Why: stage 1 raised two drift flags (B3 = 64 is outside the registered base band 65 to 70; the contemporaneous S0 = 64 is outside 56 to 62);
# the registered follow-up for both is "one additional base run, all attempts listed, drift flag" (plan section 4 and 5).
# usage: bash sft/r6_extrabase.sh extrabase     Reuses sft/evallib6.sh unchanged (owned server, identity checks, caps, probe, F then K2).
set -u
case "${1:-}" in extrabase) ;; *) echo "usage: $0 extrabase"; exit 2;; esac
cd /Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6 || exit 2
source sft/evallib6.sh
init_attempt extrabase
echo "$(date +%T) extrabase start, attempt $ATT"
preflight
chain_clock
arm B3c 1.0
$PY -B - $ATT <<'PYEOF'
import json, sys, pathlib
a = pathlib.Path(sys.argv[1])
def passes(p):
    v = [json.loads(l) for l in open(p) if l.strip()]
    return sum(bool(x.get("valid")) for x in v), len(v), len({x.get("task_id") for x in v}), sum(bool(x.get("api_error")) for x in v)
f = passes(a / "evals" / "F_B3c" / "verdicts.jsonl"); k = passes(a / "evals" / "K2_B3c" / "verdicts.jsonl")
ok = f[1:] == (81, 81, 0) and k[1:] == (74, 74, 0)
print(json.dumps({"B3c": {"F": f[0], "K2": k[0], "composite": f[0] + k[0], "valid": ok}}))
sys.exit(0 if ok else 1)
PYEOF
RC=$?
[ $RC -eq 0 ] && say "extrabase COMPLETE: $ATT" || die "extrabase invalid (counts/ids/api errors)"
