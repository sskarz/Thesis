#!/bin/bash
# Arms G4: waits for arm B's chain to finish (its FK screen), then starts the G4 chain once, under ledger entry A-003 (Sans, 2026-11-11, relayed by the advisor).
# It starts nothing else and never kills anything. The chain itself re-checks every registered condition and refuses to run if one fails (nothing is started in that case).
# usage: nohup bash sft/g4_wait_launch.sh > ~/g2dev/g4/wait.log 2>&1 &
ROOT=/Users/USER/Documents/GitHub
export PATH=/opt/homebrew/bin:$PATH
say() { echo "$(date '+%F %T') $*"; }
B=""
until [ -n "$B" ]; do B=$(ls -d $ROOT/Thesis-wt-r6/sft/data/distill/r6g3/attempts/*_run_B.* 2>/dev/null | head -1); sleep 30; done
say "waiting for arm B to complete: $B"
while true; do
  if grep -q "G3 ARM B COMPLETE" $B/chain.log 2>/dev/null && [ -f $B/G3_B_REPORT.json ]; then break; fi
  if grep -q "ABORT" $B/chain.log 2>/dev/null; then say "arm B chain ABORTED; G4 not started. See $B/chain.log"; exit 1; fi
  if [ $(date +%s) -gt $(date -j -f "%Y-%m-%d %H:%M" "2026-11-25 23:00" +%s) ]; then say "past Nov 25; not starting"; exit 1; fi
  sleep 120
done
say "arm B complete; waiting for the GPU to be free"
for i in $(seq 1 60); do pgrep -f "g3_arm.sh|g3_train.py|mlx_lm|llama-server" > /dev/null || break; sleep 20; done
if pgrep -f "g3_arm.sh|g3_train.py|mlx_lm|llama-server" > /dev/null; then say "GPU not free after 20 min; G4 not started"; exit 1; fi
say "starting the G4 chain"
cd $ROOT/Thesis-wt-training && G4_APPROVAL=A-003 exec caffeinate -i bash sft/g4_chain.sh run
