"""Shared constants for the BFCL benchmark pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_URL = "http://localhost:8080"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS: dict[str, dict[str, str]] = {
    "qwen2.5-3b": {
        "hf_repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "quant": "Q4_K_M",
        "api_name": "qwen2.5-3b",
        "display": "Qwen2.5-3B-Instruct (Q4_K_M)",
        "slug": "Qwen2.5-3B-Instruct-Q4_K_M",
    },
    "llama-3.2-3b": {
        "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "quant": "Q4_K_M",
        "api_name": "llama-3.2-3b",
        "display": "Llama-3.2-3B-Instruct (Q4_K_M)",
        "slug": "Llama-3.2-3B-Instruct-Q4_K_M",
    },
    "qwen3-4b": {
        "hf_repo": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
        "quant": "Q4_K_M",
        "api_name": "qwen3-4b",
        "display": "Qwen3-4B-Instruct-2507 (Q4_K_M)",
        "slug": "Qwen3-4B-Instruct-2507-Q4_K_M",
    },
    "qwen3.5-4b": {
        "hf_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "quant": "Q4_K_M",
        "api_name": "qwen3.5-4b",
        "display": "Qwen3.5-4B-Instruct (Q4_K_M)",
        "slug": "Qwen3.5-4B-Instruct-Q4_K_M",
    },
    "qwen3.5-27b": {
        "hf_repo": "bartowski/Qwen_Qwen3.5-27B-GGUF",
        "quant": "Q4_K_M",
        "api_name": "qwen3.5-27b",
        "display": "Qwen3.5-27B-Instruct (Q4_K_M)",
        "slug": "Qwen3.5-27B-Instruct-Q4_K_M",
    },
    "gemma4-e2b": {
        "hf_repo": "bartowski/google_gemma-4-E2B-it-GGUF",
        "quant": "Q4_K_M",
        "api_name": "gemma4-e2b",
        "display": "Gemma-4-E2B-it (Q4_K_M)",
        "slug": "Gemma-4-E2B-it-Q4_K_M",
    },
    # Mac-local MLX LoRA SFT (IFFC mix, best val @ iter 1000). Not on HuggingFace.
    # llama-server is started with `-m <local.gguf>` (see CLAUDE.md / sft/README.md);
    # `--model gemma4-e2b-sft-iffc` is scoring identity only — do not reuse gemma4-e2b.
    "gemma4-e2b-sft-iffc": {
        "hf_repo": "",  # placeholder: serve with -m, not -hf
        "gguf_path": "sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "api_name": "gemma4-e2b-sft-iffc",
        "display": "Gemma-4-E2B-it SFT-IFFC (Q4_K_M)",
        "slug": "Gemma-4-E2B-it-SFT-IFFC-Q4_K_M",
    },
    # Filtered mix (drop non-terminal [] after tool-exec), LoRA best@2000 val 0.211, PR #24.
    # Distinct api_name/slug so Run 13 does not overwrite Run 12 (gemma4-e2b-sft-iffc).
    "gemma4-e2b-sft-iffc-v2": {
        "hf_repo": "",  # placeholder: serve with -m, not -hf
        "gguf_path": "sft/fused/gemma-4-e2b-it-iffc-v2-best2000-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "api_name": "gemma4-e2b-sft-iffc-v2",
        "display": "Gemma-4-E2B-it SFT-IFFC-v2 filtered-[] (Q4_K_M)",
        "slug": "Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M",
    },
    # SFT v4 (PR #31 unroll: v3 empty-after-exec + drop first-assistant-empty
    # and clarifying []). LoRA best@1600 val 0.196. Distinct api_name/slug so
    # Run 15 does not overwrite Runs 12–14.
    "gemma4-e2b-sft-iffc-v4": {
        "hf_repo": "",  # placeholder: serve with -m, not -hf
        "gguf_path": "sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "api_name": "gemma4-e2b-sft-iffc-v4",
        "display": "Gemma-4-E2B-it SFT-IFFC-v4 (Q4_K_M)",
        "slug": "Gemma-4-E2B-it-SFT-IFFC-v4-Q4_K_M",
    },
    # SFT v5 (PR #35 unroll: v4 empty-recovery + turn-0 completeness +
    # env-state consistency). LoRA best@1800 val 0.147. Distinct api_name/slug
    # so Run 17 does not overwrite Runs 12–16. Run 17 scored 23/200 = 11.5%
    # (scored regression vs Run 15 15.0%). Do not overwrite; do not present
    # as a method win; no BoN / no more epochs on this mix.
    "gemma4-e2b-sft-iffc-v5": {
        "hf_repo": "",  # serve with -m, not -hf
        "gguf_path": "sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "api_name": "gemma4-e2b-sft-iffc-v5",
        "display": "Gemma-4-E2B-it SFT-IFFC-v5 (Q4_K_M)",
        "slug": "Gemma-4-E2B-it-SFT-IFFC-v5-Q4_K_M",
    },
    # SFT v6 (PR #38: v4 empty/recovery + drop_state_inconsistent only;
    # turn-0 rewrites off). LoRA best@1800 val 0.142. Distinct api_name/slug
    # so Run 18 does not overwrite Runs 12–17.
    "gemma4-e2b-sft-iffc-v6": {
        "hf_repo": "",  # serve with -m, not -hf
        "gguf_path": "sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "api_name": "gemma4-e2b-sft-iffc-v6",
        "display": "Gemma-4-E2B-it SFT-IFFC-v6 (Q4_K_M)",
        "slug": "Gemma-4-E2B-it-SFT-IFFC-v6-Q4_K_M",
    },
    "functiongemma-270m": {
        "hf_repo": "ggml-org/functiongemma-270m-it-GGUF",
        "quant": "",
        "api_name": "functiongemma-270m",
        "display": "FunctionGemma-270M-IT",
        "slug": "FunctionGemma-270M-IT",
    },
}
DEFAULT_MODEL = "qwen2.5-3b"

# ---------------------------------------------------------------------------
# Model identity (populated by select_model)
# ---------------------------------------------------------------------------
MODEL_HF_REPO: str = ""
MODEL_QUANT: str = ""
MODEL_NAME: str = ""
MODEL_DISPLAY: str = ""
MODEL_SLUG: str = ""
RUN_ID: int | None = None

# ---------------------------------------------------------------------------
# Paths  (results are organized per model)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
MODEL_RESULTS_DIR: Path = RESULTS_DIR
RESPONSES_DIR: Path = RESULTS_DIR
METRICS_DIR: Path = RESULTS_DIR
SCORES_DIR: Path = RESULTS_DIR
FIGURES_DIR: Path = RESULTS_DIR


def select_model(key: str) -> None:
    """Activate a model from the MODELS registry, setting module-level constants."""
    global MODEL_HF_REPO, MODEL_QUANT, MODEL_NAME, MODEL_DISPLAY, MODEL_SLUG
    global MODEL_RESULTS_DIR, RESPONSES_DIR, METRICS_DIR, SCORES_DIR, FIGURES_DIR

    if key not in MODELS:
        raise ValueError(
            f"Unknown model {key!r}. Choose from: {', '.join(MODELS)}"
        )
    m = MODELS[key]
    MODEL_HF_REPO = m["hf_repo"]
    MODEL_QUANT = m["quant"]
    MODEL_NAME = m["api_name"]
    MODEL_DISPLAY = m["display"]
    MODEL_SLUG = m["slug"]

    MODEL_RESULTS_DIR = RESULTS_DIR / MODEL_SLUG
    RESPONSES_DIR = MODEL_RESULTS_DIR / "responses"
    METRICS_DIR = MODEL_RESULTS_DIR / "metrics"
    SCORES_DIR = MODEL_RESULTS_DIR / "scores"
    FIGURES_DIR = MODEL_RESULTS_DIR / "figures"


# Apply defaults at import time so existing code works unchanged.
select_model(DEFAULT_MODEL)

import bfcl_eval  # noqa: E402

BFCL_DATA_DIR = Path(bfcl_eval.__file__).parent / "data"
BFCL_POSSIBLE_ANSWER_DIR = BFCL_DATA_DIR / "possible_answer"

# ---------------------------------------------------------------------------
# BFCL categories (single-turn, AST-evaluated)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "simple": {"language": "Python", "file": "BFCL_v4_simple_python.json"},
    "multiple": {"language": "Python", "file": "BFCL_v4_multiple.json"},
    "parallel": {"language": "Python", "file": "BFCL_v4_parallel.json"},
    "parallel_multiple": {
        "language": "Python",
        "file": "BFCL_v4_parallel_multiple.json",
    },
    "irrelevance": {"language": "Python", "file": "BFCL_v4_irrelevance.json"},
    "java": {"language": "Java", "file": "BFCL_v4_simple_java.json"},
    "javascript": {"language": "JavaScript", "file": "BFCL_v4_simple_javascript.json"},
}

# ---------------------------------------------------------------------------
# BFCL multi-turn categories (execution-evaluated)
# ---------------------------------------------------------------------------
MULTI_TURN_CATEGORIES = {
    "multi_turn_base": {"file": "BFCL_v4_multi_turn_base.json"},
    "multi_turn_miss_func": {"file": "BFCL_v4_multi_turn_miss_func.json"},
    "multi_turn_miss_param": {"file": "BFCL_v4_multi_turn_miss_param.json"},
    "multi_turn_long_context": {"file": "BFCL_v4_multi_turn_long_context.json"},
}

MULTI_TURN_FUNC_DOC_DIR = BFCL_DATA_DIR / "multi_turn_func_doc"
MULTI_TURN_MAX_STEPS = 20

# ---------------------------------------------------------------------------
# Function-calling mode
# nfc  = native OpenAI tools / tool_calls (llama-server --jinja; may apply
#        lazy tools GBNF — this is NOT the CD experiment)
# iffc = Instruction-Followed Function Calling (arXiv:2608.22472 Appendix B)
#        JSON-in-text; do not send tools/tool_choice
# cd   = TagDispatch-style two-pass constrained decoding (arXiv:2601.04426
#        approximation on stock llama-server). Never sends tools/tool_choice.
#        Pass 1: GBNF grammar over live tool names. Pass 2: json_schema for
#        the selected tool's arguments. Not XGrammar-2; LLGuidance not required.
# gst  = Guided-Structured Templates (arXiv:2509.18076). Prompt-only overlay
#        on the NFC channel: still sends tools/tool_choice; model writes a
#        short [GST]...[/GST] scaffold in content before native tool_calls.
#        Not GBNF / XGrammar. GST text is ignored by the checker.
# iffc-state = IFFC JSON-in-text plus a harness-written [STATE] snapshot
#        from actual tool execution results (METHODS §5 working memory).
#        Not GST, not LLM summarization, not CD/GBNF, not shell→GFS.
# iffc-state-bon = Best-of-N majority voting overlay on IFFC+STATE
#        (METHODS §4 / PLAN #2; Snell et al. arXiv:2408.03314). Sample N
#        completions at temperature=0.7, vote on (function_name, arguments)
#        per step, execute the winner once. Not NFC/GST/CD.
# ---------------------------------------------------------------------------
FC_NFC = "nfc"
FC_IFFC = "iffc"
FC_CD = "cd"
FC_GST = "gst"
FC_IFFC_STATE = "iffc-state"
FC_IFFC_STATE_BON = "iffc-state-bon"
FC_MODES = (FC_NFC, FC_IFFC, FC_CD, FC_GST, FC_IFFC_STATE, FC_IFFC_STATE_BON)
FC_JSON_TEXT_MODES = (FC_IFFC, FC_CD, FC_IFFC_STATE, FC_IFFC_STATE_BON)
FC_IFFC_STATE_MODES = (FC_IFFC_STATE, FC_IFFC_STATE_BON)
FC_MODE: str = FC_NFC

# Best-of-N (only used by --fc-mode iffc-state-bon)
BON_N_DEFAULT = 5
BON_N: int = BON_N_DEFAULT
BON_TEMPERATURE = 0.7

# Diagnostic-only: drop trailing IFFC assistant ``[]`` stop turns from
# chat_history before the next BFCL user turn. Default **off** so Run 5–13
# IFFC bookkeeping is unchanged. Not a training method; not a new fc-mode.
IFFC_STRIP_EMPTY_ASSISTANT: bool = False
IFFC_STRIP_EMPTY_ASSISTANT_DIRNAME = "iffc-strip-empty-assistant"


def set_fc_mode(mode: str) -> None:
    """Set the process-wide function-calling mode."""
    global FC_MODE
    if mode not in FC_MODES:
        raise ValueError(
            f"Unknown fc-mode {mode!r}. Choose from: {', '.join(FC_MODES)}"
        )
    FC_MODE = mode


def set_bon_n(n: int) -> None:
    """Set the Best-of-N sample count (``--bon-n``; default 5)."""
    global BON_N
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"bon-n must be an integer >= 1, got {n!r}")
    BON_N = n


def set_iffc_strip_empty_assistant(enabled: bool) -> None:
    """Enable or disable diagnostic stripping of trailing IFFC assistant ``[]``."""
    global IFFC_STRIP_EMPTY_ASSISTANT
    IFFC_STRIP_EMPTY_ASSISTANT = bool(enabled)


def apply_iffc_strip_empty_assistant_result_paths() -> None:
    """Nest on-disk metrics/scores/figures under a diagnostic subdirectory.

    Scoring identity (``--model`` / ``model_key``) is unchanged so Run 12/13
    keys stay intact. Charts for this ablation must not overwrite standard
    IFFC figures for the same model. No-op when the flag is off.
    """
    global FIGURES_DIR, METRICS_DIR, SCORES_DIR
    if not IFFC_STRIP_EMPTY_ASSISTANT:
        return
    root = MODEL_RESULTS_DIR / IFFC_STRIP_EMPTY_ASSISTANT_DIRNAME
    FIGURES_DIR = root / "figures"
    METRICS_DIR = root / "metrics"
    SCORES_DIR = root / "scores"
    for path in (FIGURES_DIR, METRICS_DIR, SCORES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_multi_turn_categories(
    names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Return a subset of ``MULTI_TURN_CATEGORIES``, or all if *names* is empty.

    Individual tokens may be comma-separated (e.g. ``multi_turn_base``).
    """
    if not names:
        return dict(MULTI_TURN_CATEGORIES)

    expanded: list[str] = []
    for name in names:
        expanded.extend(part.strip() for part in name.split(",") if part.strip())

    unknown = [n for n in expanded if n not in MULTI_TURN_CATEGORIES]
    if unknown:
        raise ValueError(
            f"Unknown multi-turn categor(y/ies): {', '.join(unknown)}. "
            f"Choose from: {', '.join(MULTI_TURN_CATEGORIES)}"
        )

    ordered: list[str] = []
    seen: set[str] = set()
    for n in expanded:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return {n: MULTI_TURN_CATEGORIES[n] for n in ordered}

# ---------------------------------------------------------------------------
# Concurrency (must match llama-server -np flag)
# ---------------------------------------------------------------------------
PARALLEL_SLOTS = 4  # IFFC multi-turn: system prompt alone ~6k tok; need ~16k/slot (65536/-np 4)

# ---------------------------------------------------------------------------
# BFCL → JSON-Schema type mapping
# ---------------------------------------------------------------------------
GORILLA_TO_OPENAPI: dict[str, str] = {
    # Gorilla / Python types
    "dict": "object",
    "float": "number",
    "list": "array",
    "tuple": "array",
    # Java types
    "String": "string",
    "int": "integer",
    "double": "number",
    "long": "integer",
    "char": "string",
    "byte": "integer",
    "short": "integer",
    "Boolean": "boolean",
    "HashMap": "object",
    "ArrayList": "array",
    # JavaScript types
    "Number": "number",
    "Object": "object",
    "Array": "array",
    # Already-valid JSON-Schema types pass through
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
    "any": None,
}
