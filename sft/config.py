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
RAW_APIGEN_MT = RAW_DIR / "apigen_mt.jsonl"
FORMATTED_TOOLACE = FORMATTED_DIR / "toolace.jsonl"
FORMATTED_XLAM = FORMATTED_DIR / "xlam.jsonl"
FORMATTED_APIGEN_MT = FORMATTED_DIR / "apigen_mt.jsonl"

# SFT v7 experiment dirs — do **not** reuse frozen v1 ``splits/`` or v4–v6 adapters.
SPLITS_DIR_V7 = DATA_DIR / "splits-v7"
MLX_DATA_DIR_V7 = DATA_DIR / "mlx-v7"
ADAPTER_DIR_V7 = SFT_ROOT / "adapters-v7"
TRAIN_JSONL_V7 = SPLITS_DIR_V7 / "train.jsonl"
VALID_JSONL_V7 = SPLITS_DIR_V7 / "valid.jsonl"
CORPUS_TEST_JSONL_V7 = SPLITS_DIR_V7 / "corpus_test.jsonl"
MANIFEST_PATH_V7 = SPLITS_DIR_V7 / "MANIFEST.json"

# SFT v8 experiment dirs — lever 1 (identical-call drop + higher non-tool band).
# Do **not** reuse splits-v7 / mlx-v7 / adapters-v7 (Run 21 collapsed).
SPLITS_DIR_V8 = DATA_DIR / "splits-v8"
MLX_DATA_DIR_V8 = DATA_DIR / "mlx-v8"
ADAPTER_DIR_V8 = SFT_ROOT / "adapters-v8"
TRAIN_JSONL_V8 = SPLITS_DIR_V8 / "train.jsonl"
VALID_JSONL_V8 = SPLITS_DIR_V8 / "valid.jsonl"
CORPUS_TEST_JSONL_V8 = SPLITS_DIR_V8 / "corpus_test.jsonl"
MANIFEST_PATH_V8 = SPLITS_DIR_V8 / "MANIFEST.json"

LORA_CONFIG_PATH = SFT_ROOT / "lora_config.yaml"

# ---------------------------------------------------------------------------
# HuggingFace datasets
# ---------------------------------------------------------------------------
TOOLACE_HF_ID = "Team-ACE/ToolACE"
XLAM_HF_ID = "Salesforce/xlam-function-calling-60k"
SOURCE_TOOLACE = "toolace"
SOURCE_XLAM = "xlam"
FILLER_SOURCES = frozenset({SOURCE_TOOLACE, SOURCE_XLAM})

# Salesforce/APIGen-MT-5k — v7 mix core (not v1).
# License: CC-BY-NC-4.0 + Hub gated acknowledgment. Card notes the GPT-4-
# generated portion must not be used to compete with OpenAI. Thesis academic
# use is the intended path — Sanskar must click Agree on the Hub page, set
# HF_TOKEN, and cite Prabhakar et al. Do **not** commit raw/formatted JSONL.
# Cite: Prabhakar et al., "APIGen-MT: Agentic Pipeline for Multi-Turn Data
# Generation via Simulated Agent-Human Interplay," arXiv:2504.03601, 2025
# (NeurIPS). Official Hub only — do not use Beryex/APIGen-MT-5k-sharegpt.
APIGEN_MT_HF_ID = "Salesforce/APIGen-MT-5k"
APIGEN_MT_LICENSE = "CC-BY-NC-4.0"
APIGEN_MT_CITATION = (
    "Prabhakar et al., APIGen-MT: Agentic Pipeline for Multi-Turn Data "
    "Generation via Simulated Agent-Human Interplay, arXiv:2504.03601, 2025."
)
APIGEN_MT_CONFIG_NAME = "dataset"  # Hub config; download falls back to default
SOURCE_APIGEN_MT = "apigen_mt"
APIGEN_MT_POLICY_MAX_CHARS = 2000

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
# SFT v7 mix (new experiment id; seed 47; do not reopen v1 seed-42 splits)
# ---------------------------------------------------------------------------
SEED_V7 = 47
V7_TARGET_APIGEN_FRAC = 0.55
V7_MIN_APIGEN_FRAC = 0.50
V7_TARGET_MID_FC_FRAC = 0.15
V7_TARGET_CLARIFY_FRAC = 0.10
V7_TARGET_IRRELEVANCE_FRAC = 0.09
V7_MAX_IRRELEVANCE_FRAC = 0.10
V7_ABORT_IRRELEVANCE_FRAC = 0.12
V7_TARGET_FINAL_FRAC = 0.05
V7_NONTOOL_MIN = 0.10
V7_NONTOOL_MAX = 0.20
V7_NONTOOL_TARGET = 0.15  # unroll → mlx-v7 cap; conversations in splits-v7 stay full
V7_EXPERIMENT = "sft-v7"

# ---------------------------------------------------------------------------
# SFT v8 mix (lever 1; seed 48; do not reopen v7 seed-47 or v1 seed-42)
# ---------------------------------------------------------------------------
# Run 21 (v7 IFFC) collapsed unfinished: hit_step_limit / identical-call loops
# (ls / VehicleControl). Hypothesis: ~85% tool_call / ~15% non-tool taught
# observe→re-emit the same call. v8 keeps the APIGen-MT conversation mix,
# drops identical consecutive assistant *calls* at unroll, and raises the
# mlx non-tool / stop target into [0.25, 0.35] (default 0.30).
# Unscored — do not invent a v8 BFCL number. No finish-200 / BoN on v7.
SEED_V8 = 48
V8_TARGET_APIGEN_FRAC = V7_TARGET_APIGEN_FRAC
V8_MIN_APIGEN_FRAC = V7_MIN_APIGEN_FRAC  # keep ≥50% APIGen-MT after rebalance
V8_TARGET_MID_FC_FRAC = V7_TARGET_MID_FC_FRAC
V8_TARGET_CLARIFY_FRAC = V7_TARGET_CLARIFY_FRAC
V8_TARGET_IRRELEVANCE_FRAC = V7_TARGET_IRRELEVANCE_FRAC
V8_MAX_IRRELEVANCE_FRAC = V7_MAX_IRRELEVANCE_FRAC
V8_ABORT_IRRELEVANCE_FRAC = V7_ABORT_IRRELEVANCE_FRAC
V8_TARGET_FINAL_FRAC = V7_TARGET_FINAL_FRAC
V8_NONTOOL_MIN = 0.25
V8_NONTOOL_MAX = 0.35
V8_NONTOOL_TARGET = 0.30  # unroll → mlx-v8 cap; conversations in splits-v8 stay full
V8_EXPERIMENT = "sft-v8"
# Conversation-level loop filter: drop trajs with this many consecutive
# identical assistant tool-calls (observations in between still count).
V8_LOOP_MIN_RUN = 3

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
# Checked at split time and again when unrolling mlx rows (SFT v5 default; v6 selectable).
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

# Skipped for v1 (docs only). v7 **explicitly uses** APIGen-MT-5k under the
# CC-BY-NC Hub gate (see APIGEN_MT_* above). Glaive / Hermes stay skipped.
SKIPPED_DATASETS_V1 = (
    "Salesforce/APIGen-MT-5k (NC license)",
    "glaiveai/glaive-function-calling-v2",
    "NousResearch/hermes-function-calling-v1 (optional backup only)",
)
