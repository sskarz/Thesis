#!/bin/bash
# Run a Codex review. usage: sft/codex_review.sh <prompt_file> <out_file>
# Model gpt-6.1-sol, reasoning effort medium, read-only sandbox, no inherited environment (so no API keys are visible).
# The severity policy (sft/codex_review_policy.txt: major problems only, at most 5 findings) is prepended to every prompt.
# Never edit ~/.codex/config.toml. Run from the worktree root so Codex can read the files the prompt names.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <prompt_file> <out_file>"; exit 2; }
export PATH=/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:$PATH
HERE=$(cd "$(dirname "$0")" && pwd)
{ cat "$HERE/codex_review_policy.txt"; echo; echo "TASK:"; cat "$1"; } | codex exec -m gpt-6.1-sol -c model_reasoning_effort=medium -s read-only \
  -c shell_environment_policy.inherit=none -o "$2" -
