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

# SFT v8 dirs (path constants only on main — full v8 mix lives on lever1 branches).
# Listed so v9 can refuse to overwrite them.
SPLITS_DIR_V8 = DATA_DIR / "splits-v8"
MLX_DATA_DIR_V8 = DATA_DIR / "mlx-v8"
ADAPTER_DIR_V8 = SFT_ROOT / "adapters-v8"

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

# ---------------------------------------------------------------------------
# SFT v9 mix (lever 2; seed 49; do not reopen v7/v8 dirs or seeds)
# ---------------------------------------------------------------------------
# lever1 (v8 / v8.1) failed: teacher observe→same / consecutive-identical
# drops did not transfer stop/advance into Gemma-4-E2B LoRA. lever2 pivots
# to ToolACE-style **action-class quotas** for state_mismatch + under-call.
# Conversation pool: ToolACE 45–55% + xLAM 15–20% + APIGen 20–30% (cap 35%).
# Unroll quotas replace v7/v8 single pct_nontool band. No BoN / no v7 resume /
# no full LoRA on failed v8 / no teacher loop-filter as primary / no v5 synth.
SEED_V9 = 49
SPLITS_DIR_V9 = DATA_DIR / "splits-v9"
MLX_DATA_DIR_V9 = DATA_DIR / "mlx-v9"
ADAPTER_DIR_V9 = SFT_ROOT / "adapters-v9"
TRAIN_JSONL_V9 = SPLITS_DIR_V9 / "train.jsonl"
VALID_JSONL_V9 = SPLITS_DIR_V9 / "valid.jsonl"
CORPUS_TEST_JSONL_V9 = SPLITS_DIR_V9 / "corpus_test.jsonl"
MANIFEST_PATH_V9 = SPLITS_DIR_V9 / "MANIFEST.json"
V9_EXPERIMENT = "sft-v9"
V9_LEVER = "lever2"
V9_MAX_SEQ_LENGTH = 8192  # + existing budget_seq; do not silently reuse 4096

# Conversation-level pool targets (shares of conversations).
V9_TARGET_TOOLACE_FRAC = 0.50
V9_MIN_TOOLACE_FRAC = 0.45
V9_MAX_TOOLACE_FRAC = 0.55
V9_TARGET_XLAM_FRAC = 0.175
V9_MIN_XLAM_FRAC = 0.15
V9_MAX_XLAM_FRAC = 0.20
V9_TARGET_APIGEN_FRAC = 0.25
V9_MIN_APIGEN_FRAC = 0.15
V9_MAX_APIGEN_FRAC = 0.35  # hard ceiling (CONFIRM)
V9_MIN_TOOLACE_XLAM_FRAC = 0.55  # ToolACE + xLAM floor
V9_ABORT_IRRELEVANCE_FRAC = 0.12
V9_MAX_IRRELEVANCE_CONV_FRAC = 0.10

# Unrolled assistant-turn action-class quotas (PRIMARY lever).
# Each value is (target_lo, target_hi, hard_lo, hard_hi).
# refuse hard_lo is a soft floor (2% OK); abort only if refuse ≈ 0.
V9_ACTION_MT_TOOL_CHAIN_ADVANCE = (0.30, 0.38, 0.28, 0.42)
V9_ACTION_TOOL_CALL_START = (0.12, 0.18, 0.10, 0.20)
V9_ACTION_FINAL_AFTER_OBS = (0.20, 0.24, 0.18, 0.26)
V9_ACTION_ASK = (0.08, 0.12, 0.06, 0.14)
V9_ACTION_REFUSE = (0.03, 0.05, 0.02, 0.06)  # floor 2% OK; abort if ≈0
V9_ACTION_IRRELEVANCE = (0.05, 0.08, 0.0, 0.10)  # abort >0.12 separately
V9_ABORT_IRRELEVANCE_UNROLL_FRAC = 0.12
V9_ABORT_REFUSE_MIN = 0.005  # ≈0 → abort; 2% floor is soft
V9_MAX_OBSERVE_SAME_OF_TOOL = 0.01  # sampler exclusion working

# Formal 10-ID plain-IFFC gate (denominators /10). Do NOT fold VC into this.
V9_PROBE_IDS = (
    "_4",
    "_38",
    "_46",
    "_64",
    "_84",
    "_95",
    "_30",
    "_82",
    "_3",
    "_163",
)
# VehicleControl regressors: required diagnostic sidecar; never inflate the 10-ID denom.
V9_PROBE_IDS_VC_SIDECAR = (
    "_52",
    "_60",
    "_74",
    "_76",
    "_83",
    "_96",
)
# Optional later (not required for scaffold): "_186"
V9_PROBE_IDS_OPTIONAL = ("_186",)

# Stage B DiaTool-DPO stub only — no training in this PR.
V9_PREF_PAIRS_SCHEMA_VERSION = 1
