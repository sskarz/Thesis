#!/bin/bash
# G1 overfit gate (training plan 2026-11-09, section G1). usage: bash sft/g1_chain.sh g1
# Question: can an adapter learn its own training tasks? 40 tasks (sha256-selected, see docs/DISTILLATION_G1_PREREGISTRATION.md), thinking OFF everywhere.
# Steps: verify frozen inputs -> base server (thinking off) -> render-parity check on every row -> score base on the 40 tasks -> train (wide LoRA, 4 passes, <=1000 micro-batches)
#        -> convert + audit GGUF -> adapter server (thinking off) -> parity -> score adapter on the same 40 tasks -> apply the registered pass rule.
# Fail-fast only on INVALID runs; the outcome (PASS/FAIL) never stops anything. Owned-server lifecycle and caps come from sft/evallib6.sh (reviewed). Own attempt dir and own 10 h clock.
set -u
case "${1:-}" in g1) ;; *) echo "usage: $0 g1"; exit 2;; esac
cd /Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6 || exit 2
source sft/evallib6.sh
D6=$ABS6/sft/data/distill/r6g1            # separate attempts directory and separate clock from the stage 1 chain
CHAIN_CAP=36000                           # registered: G1 must finish within 10 h of its start (training itself is capped at 8 h)
G1DIR=$ABS6/sft/data/distill/r6/g1/final
T=sft/r6_tools.py
init_attempt g1
echo "$(date +%T) g1 start, attempt $ATT"
preflight
chain_clock

# 0. frozen inputs
[ "$(shasum -a 256 $G1DIR/g1_rows.jsonl | cut -d' ' -f1)" = "$(sed -n 1p sft/g1_expected.sha)" ] || die "g1_rows.jsonl hash differs from sft/g1_expected.sha"
[ "$(shasum -a 256 $G1DIR/g1_tasks.jsonl | cut -d' ' -f1)" = "$(sed -n 2p sft/g1_expected.sha)" ] || die "g1_tasks.jsonl hash differs from sft/g1_expected.sha"
[ "$(wc -l < $G1DIR/g1_tasks.jsonl | tr -d ' ')" = "40" ] || die "g1_tasks.jsonl is not 40 tasks"
bt $PY -B sft/g1_train.py --rows $G1DIR/g1_rows.jsonl --out-dir $ATT/dry --dry-tokenize > $RL/dry_tokenize.log 2>&1 || die "row rendering check failed (see $RL/dry_tokenize.log)"
say "inputs frozen and render-checked: $(tail -1 $RL/dry_tokenize.log | cut -c1-200)"

# thinking-off server (same flags as the reviewed stage 1 chain plus the chat-template kwarg); never a second server
start_g1_server() {    # NAME [LORA_GGUF]
  local name=$1 lora=${2:-}
  [ -z "$OWNED_PID" ] || die "a server is already owned"
  if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then die "port 8080 became busy; not touching it"; fi
  nohup "$LLAMA" --jinja -fa on -m "$BAR" ${lora:+--lora "$lora"} --chat-template-kwargs '{"enable_thinking":false}' --host 127.0.0.1 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024 > "$RL/server_$name.log" 2>&1 &
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
  grep -q 'enable_thinking' "$RL/argv_$name.txt" || die "server $name was not started with thinking off"
}

g1_parity() {          # NAME
  bt $PY -B sft/g1_parity.py --rows $G1DIR/g1_rows.jsonl --out $ATT/parity_$1.json > $RL/parity_$1.out 2>&1 || die "render parity failed for $1: $(tail -c 400 $RL/parity_$1.out)"
  say "parity ok ($1): $(cat $ATT/parity_$1.json | tr -d '\n ' | cut -c1-200)"
}

rowcheck() {           # NAME : diagnostic only (reported, never gating): row-level in-sample agreement with the training targets
  bt timeout -k 30 $(min2 2400 "$(chain_left)") $PY -B sft/g1_rowcheck.py --rows $G1DIR/g1_rows.jsonl --out $ATT/rowcheck_$1.json > $RL/rowcheck_$1.out 2>&1 \
    && say "rowcheck $1 (diagnostic): $(tail -1 $RL/rowcheck_$1.out | cut -c1-300)" || say "WARNING: rowcheck $1 failed or timed out (diagnostic only; chain continues)"
}

# 1. base, thinking off: parity, then score the 40 tasks
start_g1_server base
check_identity G1base 1.0
g1_parity base
rowcheck base
run6 G1_base $G1DIR/g1_tasks.jsonl 40 $ATT/evals/G1_base $(min2 3600 "$(chain_left)")
stop_server

# 2. train (wide LoRA, layers 8-34, rank 16, lr 1e-4); one registered restart at 5e-5 if it diverges
TRAIN_T0=$(date +%s)                      # ONE 8 h training budget shared by the first attempt and the registered restart
bt $PY -B sft/g1_train.py --rows $G1DIR/g1_rows.jsonl --out-dir $ATT/train --max-seconds $(min2 28800 "$(chain_left)") > $RL/train.out 2>&1
RC=$?
TR=$ATT/train
if [ $RC -eq 5 ]; then
  say "TRAINING DIVERGED (registered): restarting once at lr 5e-5. See $RL/train.out"
  TRAIN_LEFT=$(( 28800 - ($(date +%s) - TRAIN_T0) ))
  [ "$TRAIN_LEFT" -gt 3600 ] || die "less than 1 h of the shared 8 h training budget left; restart not possible (INVALID)"
  bt $PY -B sft/g1_train.py --rows $G1DIR/g1_rows.jsonl --out-dir $ATT/train_lr5e-5 --lr 5e-5 --max-seconds $(min2 "$TRAIN_LEFT" "$(chain_left)") > $RL/train_lr5e-5.out 2>&1
  RC=$?; TR=$ATT/train_lr5e-5
fi
[ $RC -eq 0 ] || die "training failed (rc $RC): $(tail -c 300 $RL/train.out)"
[ -f $TR/adapters/adapters.safetensors ] || die "no adapter was written"
say "training done: $(tail -1 $RL/train.out | cut -c1-300)"

# 3. convert + audit
bt $PY -B sft/mlx_adapter_to_gguf_lora.py $TR/adapters $ATT/G1.gguf > $RL/convert_G1.log 2>&1 || die "convert G1 failed"
bt $PY -B $T gguf-check --adapter-dir $TR/adapters --src-dir $TR/adapters --gguf $ATT/G1.gguf || die "GGUF audit of the G1 adapter failed"

# 4. adapter, thinking off: parity, then score the same 40 tasks
start_g1_server G1 $ATT/G1.gguf
check_identity G1adapter 1.0 $ATT/G1.gguf
g1_parity adapter
rowcheck adapter
run6 G1_adapter $G1DIR/g1_tasks.jsonl 40 $ATT/evals/G1_adapter $(min2 3600 "$(chain_left)")
stop_server

# 5. registered rule
$PY -B sft/g1_summarize.py --base $ATT/evals/G1_base --adapter $ATT/evals/G1_adapter --out $ATT/G1_REPORT.json || die "summarize failed"
say "G1 COMPLETE: $(cat $ATT/G1_REPORT.json | tr -d '\n ' | cut -c1-600)"
say "REQUIRED NEXT (Thesis bot, not the chain): report the counts, Codex review of the result"
