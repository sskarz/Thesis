#!/bin/bash
# Arms G3c (ledger A-005): waits for the G4 chain to print "G4 COMPLETE", copies the G4 attempt folder to a dated backup and verifies it, then starts the combined retrain once.
# If G4 aborts, stops on a failed stack check, or the backup does not verify, it starts nothing and says so. It never kills anything.
# usage: nohup bash sft/g3c_wait_launch.sh > ~/g2dev/g3c_wait.log 2>&1 &
ROOT=/Users/USER/Documents/GitHub
export PATH=/opt/homebrew/bin:$PATH
say() { echo "$(date '+%F %T') $*"; }
deadline=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-25 23:00" +%s)
G=""
until [ -n "$G" ]; do G=$(ls -dt $ROOT/Thesis-wt-r6/sft/data/distill/r6g4/attempts/*_g4run.* 2>/dev/null | head -1); [ -n "$G" ] && break; [ $(date +%s) -gt $deadline ] && { say "past Nov 25 and no G4 run; not starting"; exit 1; }; sleep 60; done
say "G4 attempt: $G"
while true; do
  grep -q "G4 COMPLETE" $G/chain.log 2>/dev/null && break
  if grep -q -E "ABORT|STACK NOT REPRODUCED" $G/chain.log 2>/dev/null; then say "G4 did not complete (see $G/chain.log); combined retrain NOT started"; exit 1; fi
  if ! pgrep -f "g4_chain.sh" > /dev/null && ! pgrep -f "g4_wait_launch.sh" > /dev/null && ! grep -q "G4 COMPLETE" $G/chain.log 2>/dev/null; then say "G4 chain is not running and did not complete; combined retrain NOT started"; exit 1; fi
  [ $(date +%s) -gt $deadline ] && { say "past Nov 25; not starting"; exit 1; }
  sleep 120
done
say "G4 complete; backing up its outputs"
BK=$HOME/thesis-backups/$(date +%F)-g4; mkdir -p $BK
rsync -a $G/ $BK/g4_attempt/ || { say "rsync failed; combined retrain NOT started"; exit 1; }
diff -rq $G $BK/g4_attempt > $BK/verify.txt 2>&1; DRC=$?
{ [ $DRC -eq 0 ] && [ ! -s $BK/verify.txt ] && grep -q "G4 COMPLETE" $BK/g4_attempt/chain.log; } || { say "backup did not verify (diff rc $DRC, $BK/verify.txt); combined retrain NOT started"; exit 1; }
say "backup verified: $BK"
for i in $(seq 1 30); do pgrep -f "g4_chain.sh|llama-server|g3_train.py" > /dev/null || break; sleep 20; done
pgrep -f "g4_chain.sh|llama-server|g3_train.py" > /dev/null && { say "GPU not free after 10 min; not starting"; exit 1; }
[ $(date +%s) -gt $deadline ] && { say "past Nov 25; not starting"; exit 1; }
say "starting the combined retrain"
cd $ROOT/Thesis-wt-training && G3C_APPROVAL=A-005 G3C_G4_ATT=$G G3C_BACKUP=$BK exec caffeinate -i bash sft/g3c_arm.sh run
