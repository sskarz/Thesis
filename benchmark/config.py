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
# ---------------------------------------------------------------------------
FC_NFC = "nfc"
FC_IFFC = "iffc"
FC_CD = "cd"
FC_GST = "gst"
FC_IFFC_STATE = "iffc-state"
FC_MODES = (FC_NFC, FC_IFFC, FC_CD, FC_GST, FC_IFFC_STATE)
FC_JSON_TEXT_MODES = (FC_IFFC, FC_CD, FC_IFFC_STATE)
FC_MODE: str = FC_NFC


def set_fc_mode(mode: str) -> None:
    """Set the process-wide function-calling mode (``nfc``, ``iffc``, ``cd``, ``gst``, or ``iffc-state``)."""
    global FC_MODE
    if mode not in FC_MODES:
        raise ValueError(
            f"Unknown fc-mode {mode!r}. Choose from: {', '.join(FC_MODES)}"
        )
    FC_MODE = mode


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
