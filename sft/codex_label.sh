#!/bin/bash
# Codex LABELING run (accuracy review of sampled tasks): same model/sandbox as sft/codex_review.sh but WITHOUT the major-problems-only policy,
# because labeling needs one verdict per task. usage: sft/codex_label.sh <prompt_file> <out_file>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <prompt_file> <out_file>"; exit 2; }
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:$PATH
cat "$1" | codex exec -m gpt-6.1-sol -c model_reasoning_effort=medium -s read-only -c shell_environment_policy.inherit=none -o "$2" -
