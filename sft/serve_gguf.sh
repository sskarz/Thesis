#!/bin/bash
# usage: serve_gguf.sh <gguf> <logfile> [lora.gguf]   (optional runtime LoRA applied on top of the base GGUF)   -> llama-server on :8080 with the Run 69 argv (only -m differs)
set -euo pipefail
pkill -f "llama-server.*--port 8080" 2>/dev/null || true
for i in $(seq 1 30); do lsof -tnP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
nohup /Users/sanskarthapa/Documents/GitHub/Thesis/llama.cpp/build/bin/llama-server --jinja -fa on -m "$1" \
  ${3:+--lora "$3"} --host 127.0.0.1 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024 > "$2" 2>&1 &
for i in $(seq 1 120); do curl -s http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
echo "health: $(curl -s http://127.0.0.1:8080/health)"; echo "model: $(curl -s http://127.0.0.1:8080/props | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model_path"), d.get("total_slots"), d.get("build_info"))')"
