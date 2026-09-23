#!/usr/bin/env bash
# Train Gemma-4-E2B LoRA on the frozen IFFC mix (Apple Silicon / mlx-lm).
#
#   ./sft/train_mlx.sh --smoke    # 100 iters, tiny unrolled slice
#   ./sft/train_mlx.sh --full     # default full iters
#   ./sft/train_mlx.sh --full --run
#   ./sft/train_mlx.sh --smoke --run --recipe v7
#
# Does not download data. Run download → format → split first (see README.md).
# Unroll applies the SFT v5 curriculum (v4 empty-after-exec + recovery +
# first-assistant / clarifying empty drops, plus turn-0 completeness and
# env-state consistency) unless --recipe v7/v6/v4/v3/v2/v1 or keep-flags are
# passed through EXTRA. v6 = v4 empty/recovery + drop_state_inconsistent only.
# v7 = APIGen-MT mix (defaults splits-v7 / mlx-v7 / adapters-v7).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="--smoke"
RUN=()
EXTRA=()
RECIPE=""
args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--recipe" && $((i + 1)) -lt ${#args[@]} ]]; then
    RECIPE="${args[$((i + 1))]}"
  fi
done

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

SPLITS_FILE="sft/data/splits/train.jsonl"
if [[ "$RECIPE" == "v7" ]]; then
  SPLITS_FILE="sft/data/splits-v7/train.jsonl"
fi

if [[ ! -f "$SPLITS_FILE" ]]; then
  echo "missing $SPLITS_FILE" >&2
  if [[ "$RECIPE" == "v7" ]]; then
    echo "Run: huggingface-cli login && accept Salesforce/APIGen-MT-5k (CC-BY-NC)" >&2
    echo "     uv run python -m sft.download_datasets --apigen-mt-only" >&2
    echo "     uv run python -m sft.format_iffc --apigen-mt" >&2
    echo "     uv run python -m sft.build_mix_v7 --out sft/data/splits-v7" >&2
  else
    echo "Run: uv run python -m sft.download_datasets && uv run python -m sft.format_iffc && uv run python -m sft.split_data" >&2
  fi
  exit 1
fi

exec uv run python -m sft.train_mlx "$MODE" "${RUN[@]}" "${EXTRA[@]}"
