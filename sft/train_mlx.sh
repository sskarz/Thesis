#!/usr/bin/env bash
# Train Gemma-4-E2B LoRA on the frozen IFFC mix (Apple Silicon / mlx-lm).
#
#   ./sft/train_mlx.sh --smoke    # 100 iters, tiny unrolled slice
#   ./sft/train_mlx.sh --full     # default full iters
#   ./sft/train_mlx.sh --full --run
#
# Does not download data. Run download → format → split first (see README.md).
# Unroll applies the SFT v3 empty-after-exec drop + recovery curriculum unless
# --keep-terminal-empty-after-exec (v2) or --keep-nonterminal-empty-after-exec
# (v1) is passed through EXTRA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="--smoke"
RUN=()
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --smoke) MODE="--smoke" ;;
    --full) MODE="--full" ;;
    --run) RUN=(--run) ;;
    --fuse) EXTRA+=(--fuse) ;;
    --prepare-only) EXTRA+=(--prepare-only) ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [[ ! -f sft/data/splits/train.jsonl ]]; then
  echo "missing sft/data/splits/train.jsonl" >&2
  echo "Run: uv run python -m sft.download_datasets && uv run python -m sft.format_iffc && uv run python -m sft.split_data" >&2
  exit 1
fi

exec uv run python -m sft.train_mlx "$MODE" "${RUN[@]}" "${EXTRA[@]}"
