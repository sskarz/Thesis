#!/bin/bash
# Round 6 diagnostics chain, STAGE 1 ONLY (stage 2 lives in its own, separately reviewed script).
# usage: bash sft/r6_chain.sh stage1
#   builds the spectrum-preserving random adapter (D1), then for each arm: own server -> identity checks -> 70-prompt probe -> FK (F then K2):
#   B3 (base), S0-1.0c (S0, scale 1.0), R-S0 (random direction), S0-0.5, S0-0.25, B3b (closing base)
# Fail-fast only on INVALID runs. Outcomes (any band) never stop it. All output goes to one new, unique attempt directory.
# The server is started and stopped by this script only (never pkill); caps: 2.5 h per FK composite, 16 h whole chain.
set -u
case "${1:-}" in stage1) ;; *) echo "usage: $0 stage1"; exit 2;; esac
cd /Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6 || exit 2
source sft/evallib6.sh
init_attempt stage1
echo "$(date +%T) chain stage1 start, attempt $ATT"
preflight
chain_clock
S0A=$R5/sft/adapters-r5-S0; S0G=$R5/sft/lora-gguf/r5_S0.gguf
T=sft/r6_tools.py

bt $PY -B sft/test_r6_random_adapter.py && bt $PY -B sft/test_r6_tools.py || die "selftests failed"
bt $PY -B sft/r6_random_adapter.py $S0A $ATT/R-S0 --seed 6061 || die "random adapter build failed"
bt $PY -B sft/mlx_adapter_to_gguf_lora.py $ATT/R-S0 $ATT/R-S0.gguf > $RL/convert_R-S0.log 2>&1 || die "convert R-S0 failed"
bt $PY -B $T gguf-check --adapter-dir $ATT/R-S0 --src-dir $ATT/R-S0 --gguf $ATT/R-S0.gguf || die "GGUF audit of R-S0 failed"
bt $PY -B $T gguf-check --adapter-dir $S0A --src-dir $S0A --gguf $S0G || die "GGUF audit of S0 failed"
bt $PY -B $T norms --adapters $S0A $ATT/R-S0 --out $ATT/norms.json || die "norms failed"

arm B3 1.0
arm S0-1.0c 1.0 "$S0G"
arm R-S0 1.0 "$ATT/R-S0.gguf"
arm S0-0.5 0.5 "$S0G"
arm S0-0.25 0.25 "$S0G"
arm B3b 1.0

bt $PY -B $T probes --ref $ATT/probe_B3.json --runs $ATT/probe_B3.json $ATT/probe_B3b.json $ATT/probe_S0-1.0c.json $ATT/probe_R-S0.json $ATT/probe_S0-0.5.json $ATT/probe_S0-0.25.json --out $ATT/probe_report.json > $RL/probe_report.out 2>&1; PROBE_RC=$?
[ $PROBE_RC -eq 0 ] || say "WARNING: probe report has invalid probes or comparisons (rc $PROBE_RC); FK results are unaffected"
bt $PY -B $T summarize --attempt $ATT --out $ATT/STAGE_REPORT.json > $RL/summarize.out 2>&1; rc=$?
cat $RL/summarize.out
[ $rc -eq 0 ] || die "stage report incomplete or invalid (see $RL/summarize.out)"
[ -s $ATT/probe_report.json ] && [ -s $ATT/STAGE_REPORT.json ] || die "reports missing"
if [ $PROBE_RC -eq 0 ] && [ -z "$PROBE_INVALID" ]; then
  say "stage1 COMPLETE. Attempt: $ATT. REQUIRED NEXT (Thesis bot, not the chain): Codex review of STAGE_REPORT.json and probe_report.json against plan section 4; message Sans with counts and any FOLLOW-UP flag; start nothing else until that review has no open BLOCKER."
else
  say "stage1 FINISHED WITH INVALID PROBES (FK summary valid): $ATT; invalid probe runs:$PROBE_INVALID. Probe comparisons must not be interpreted. REQUIRED NEXT: as above, and tell Sans the probes are invalid."
  exit 3
fi
