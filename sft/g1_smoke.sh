#!/bin/bash
# G1 end-to-end smoke (training plan P0 / advisor item 2): 12 real micro-batches -> adapter -> GGUF -> audit -> thinking-off server WITH the adapter -> identity -> render parity.
# Not a result. usage: bash sft/g1_smoke.sh   Uses the same owned-server helpers as the chain. $0, about 15 min of GPU.
set -u
cd /Users/USER/Documents/GitHub/Thesis-wt-r6 || exit 2
source sft/evallib6.sh
D6=$ABS6/sft/data/distill/r6g1; CHAIN_CAP=3600
ROWS=$ABS6/sft/data/distill/r6/g1/dry/g1_rows.jsonl
init_attempt smoke; chain_clock
echo "$(date +%T) smoke start, attempt $ATT"
lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && abort "port 8080 busy"
bt $PY -B sft/g1_train.py --rows $ROWS --out-dir $ATT/train --smoke 12 > $RL/train.out 2>&1 || die "smoke training failed: $(tail -c 400 $RL/train.out)"
say "train: $(tail -2 $RL/train.out | tr '\n' ' ' | cut -c1-300)"
$PY -B - $ATT/train/train_log.jsonl <<'PYEOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
print("micro-batches", len(L), "mean s", round(sum(x["s"] for x in L) / len(L), 2), "mean tokens", round(sum(x["tokens"] for x in L) / len(L)),
      "s per 1k tokens", round(1000 * sum(x["s"] for x in L) / sum(x["tokens"] for x in L), 2), "losses", [x["loss"] for x in L])
PYEOF
bt $PY -B sft/mlx_adapter_to_gguf_lora.py $ATT/train/adapters $ATT/smoke.gguf > $RL/convert.log 2>&1 || die "convert failed"
bt $PY -B sft/r6_tools.py gguf-check --adapter-dir $ATT/train/adapters --src-dir $ATT/train/adapters --gguf $ATT/smoke.gguf || die "gguf audit failed"
$PY -B - $ATT/smoke.gguf <<'PYEOF'
import sys; sys.path.insert(0, "/Users/USER/Documents/GitHub/Thesis/llama.cpp/gguf-py")
from gguf import GGUFReader
n = len(GGUFReader(sys.argv[1]).tensors); print("GGUF tensors", n, "expected", 149 * 2); sys.exit(0 if n == 298 else 1)
PYEOF
[ $? -eq 0 ] || die "GGUF tensor count is not 2 x 149 modules"
[ -z "$OWNED_PID" ] || die "server owned"
nohup "$LLAMA" --jinja -fa on -m "$BAR" --lora $ATT/smoke.gguf --chat-template-kwargs '{"enable_thinking":false}' --host 127.0.0.1 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024 > "$RL/server_smoke.log" 2>&1 &
OWNED_PID=$!
for _ in $(seq 1 120); do kill -0 "$OWNED_PID" 2>/dev/null || die "server exited"; cu http://127.0.0.1:8080/health | grep -q '"ok"' && break; sleep 2; done
cu http://127.0.0.1:8080/health | grep -q '"ok"' || die "server not healthy"
ps -ww -o command= -p "$OWNED_PID" > "$RL/argv_smoke.txt"
check_identity smoke 1.0 $ATT/smoke.gguf
grep -i -c "lora" "$RL/server_smoke.log" | sed 's/^/server log lines mentioning lora: /'
grep -i "lora" "$RL/server_smoke.log" | head -5 | cut -c1-200
bt $PY -B sft/g1_parity.py --rows $ROWS --out $ATT/parity.json > $RL/parity.out 2>&1 || die "parity failed: $(tail -c 300 $RL/parity.out)"
say "parity: $(tr -d '\n ' < $ATT/parity.json)"
stop_server
say "SMOKE COMPLETE (not a result): $ATT"
