"""Paths and locked SFT v1 constants.

Data mix and splits are frozen for the thesis — do not reopen without a
new experiment id. See ``sft/README.md``.
"""

from __future__ import annotations

from pathlib import Path

from benchmark.config import MODELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SFT_ROOT = Path(__file__).resolve().parent
DATA_DIR = SFT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FORMATTED_DIR = DATA_DIR / "formatted"
SPLITS_DIR = DATA_DIR / "splits"
MLX_DATA_DIR = DATA_DIR / "mlx"
ADAPTER_DIR = SFT_ROOT / "adapters"
FUSED_DIR = SFT_ROOT / "fused"
MLX_MODEL_DIR = SFT_ROOT / "models" / "gemma-4-e2b-it-mlx"

TRAIN_JSONL = SPLITS_DIR / "train.jsonl"
VALID_JSONL = SPLITS_DIR / "valid.jsonl"
CORPUS_TEST_JSONL = SPLITS_DIR / "corpus_test.jsonl"
MANIFEST_PATH = SPLITS_DIR / "MANIFEST.json"

RAW_TOOLACE = RAW_DIR / "toolace.jsonl"
RAW_XLAM = RAW_DIR / "xlam.jsonl"
FORMATTED_TOOLACE = FORMATTED_DIR / "toolace.jsonl"
FORMATTED_XLAM = FORMATTED_DIR / "xlam.jsonl"

LORA_CONFIG_PATH = SFT_ROOT / "lora_config.yaml"

# ---------------------------------------------------------------------------
# HuggingFace datasets
# ---------------------------------------------------------------------------
TOOLACE_HF_ID = "Team-ACE/ToolACE"
XLAM_HF_ID = "Salesforce/xlam-function-calling-60k"
SOURCE_TOOLACE = "toolace"
SOURCE_XLAM = "xlam"

# ---------------------------------------------------------------------------
# Locked split / sample policy (SFT v1)
# ---------------------------------------------------------------------------
SEED = 42
XLAM_SAMPLE_SIZE = 25_000
TRAIN_FRAC = 0.90
VAL_FRAC = 0.05
CORPUS_TEST_FRAC = 0.05
N_TOOLS_BIN_SINGLE = "1"  # n_tools_in_gold <= 1
N_TOOLS_BIN_PARALLEL = "ge2"  # n_tools_in_gold >= 2

# ---------------------------------------------------------------------------
# Base model (matches benchmark/config.py gemma4-e2b GGUF lineage)
# ---------------------------------------------------------------------------
# Instruct checkpoint used for GGUF serving:
#   bartowski/google_gemma-4-E2B-it-GGUF : Q4_K_M
# HuggingFace instruct weights (gated; needs HF_TOKEN / huggingface-cli):
HF_INSTRUCT_ID = "google/gemma-4-e2b-it"
GGUF_REPO = MODELS["gemma4-e2b"]["hf_repo"]
GGUF_QUANT = MODELS["gemma4-e2b"]["quant"]

# There is **no** mlx-lm id in benchmark/config.py. The popular
# ``mlx-community/gemma-4-e2b-it-4bit`` repo is an mlx-vlm conversion and is
# missing text-only PLE weights — do not LoRA-train that file with mlx-lm.
# Convert the instruct checkpoint on the Mac instead (see README).
MLX_VLM_4BIT_ID = "mlx-community/gemma-4-e2b-it-4bit"  # not for mlx-lm LoRA
MLX_CONVERT_HINT = (
    "python -m mlx_lm.convert "
    f"--hf-path {HF_INSTRUCT_ID} "
    f"--mlx-path {MLX_MODEL_DIR}"
)
MLX_CONVERT_QLORA_HINT = MLX_CONVERT_HINT + " -q --q-bits 4"

# ---------------------------------------------------------------------------
# mlx-lm LoRA defaults (Apple Silicon; not run in CI)
# ---------------------------------------------------------------------------
LORA_RANK = 16  # 16–32 is the intended band; raise via --rank
LORA_SCALE = 32.0  # ToolACE-8B used rank 16 / alpha 32
LORA_DROPOUT = 0.0
LORA_NUM_LAYERS = 16
LORA_BATCH_SIZE = 1
LORA_LEARNING_RATE = 1e-5
LORA_MAX_SEQ_LENGTH = 4096
LORA_ITERS_SMOKE = 100
LORA_ITERS_FULL = 2000
SMOKE_TRAIN_ROWS = 200
SMOKE_VALID_ROWS = 20

# ---------------------------------------------------------------------------
# Chat format
# ---------------------------------------------------------------------------
CHAT_FORMAT = "iffc"
THESIS_TEST = "BFCL v4 multi_turn_base (never in train/val/corpus_test)"

# IDs / sources that must never appear in the SFT mix.
# Checked at split time and again when unrolling mlx rows (SFT v4).
BFCL_FORBIDDEN_ID_PATTERNS = (
    "bfcl_v",
    "bfcl-v",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel_multiple",
)
# Stronger content markers only (avoid false positives on generic words).
BFCL_FORBIDDEN_CONTENT_PATTERNS = (
    "bfcl_v",
    "bfcl-v",
    "multi_turn_base_",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "gorillafilesystem",
)

# Skipped for v1 (docs only).
SKIPPED_DATASETS_V1 = (
    "Salesforce/APIGen-MT-5k (NC license)",
    "glaiveai/glaive-function-calling-v2",
    "NousResearch/hermes-function-calling-v1 (optional backup only)",
)
