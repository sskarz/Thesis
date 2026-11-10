#!/bin/bash
# usage: fuse_q4.sh <adapter_dir_with_adapters.safetensors> <name>   -> sft/fused/<name>-Q4_K_M.gguf
# Recipe as R74: mlx_lm fuse -> convert_hf_to_gguf F16 -> llama-quantize Q4_K_M; deletes the fused dir and F16.
set -euo pipefail
T=/Users/sanskarthapa/Documents/GitHub/Thesis
AD="$1"; NAME="$2"
FD=$T/sft/fused/$NAME
F16=$T/sft/fused/$NAME-F16.gguf
Q4=$T/sft/fused/$NAME-Q4_K_M.gguf
cd "$T"
echo "===== FUSE $NAME START $(date +%Y-%m-%dT%H:%M:%S%z) adapter=$AD ====="
shasum -a 256 "$AD/adapters.safetensors"
[ ! -e "$Q4" ] || { echo "exists: $Q4 (reuse)"; exit 0; }
rm -rf "$FD" "$F16"
FREE_KB=$(df -k "$T" | tail -1 | awk '{print $4}')
[ "$FREE_KB" -gt $((22*1024*1024)) ] || { echo "BLOCKER: free ${FREE_KB}KB < 22GiB"; exit 5; }
.venv/bin/python -m mlx_lm fuse --model "$T/sft/models/gemma-4-e2b-it-mlx" --adapter-path "$AD" --save-path "$FD"
.venv/bin/python llama.cpp/convert_hf_to_gguf.py "$FD" --outfile "$F16"
rm -rf "$FD"
./llama.cpp/build/bin/llama-quantize "$F16" "$Q4" Q4_K_M
rm -f "$F16"
shasum -a 256 "$Q4"; ls -l "$Q4"; df -h "$T" | tail -1
echo "===== FUSE $NAME DONE $(date +%Y-%m-%dT%H:%M:%S%z) ====="
