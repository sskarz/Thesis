"""MLX LoRA SFT pipeline for IFFC-style multi-turn tool calling.

This package prepares a frozen ToolACE + xLAM mix and trains Gemma-4-E2B
on Apple Silicon with mlx-lm. It does **not** include BFCL prompts or scores.
"""

from sft.config import (
    CORPUS_TEST_FRAC,
    HF_INSTRUCT_ID,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    XLAM_SAMPLE_SIZE,
)

__all__ = [
    "CORPUS_TEST_FRAC",
    "HF_INSTRUCT_ID",
    "SEED",
    "TRAIN_FRAC",
    "VAL_FRAC",
    "XLAM_SAMPLE_SIZE",
]
