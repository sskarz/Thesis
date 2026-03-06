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
    "qwen3.5-4b": {
        "hf_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "quant": "Q4_K_M",
        "api_name": "qwen3.5-4b",
        "display": "Qwen3.5-4B-Instruct (Q4_K_M)",
        "slug": "Qwen3.5-4B-Instruct-Q4_K_M",
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
    "simple": {"language": "Python", "file": "BFCL_v3_simple.json"},
    "multiple": {"language": "Python", "file": "BFCL_v3_multiple.json"},
    "parallel": {"language": "Python", "file": "BFCL_v3_parallel.json"},
    "parallel_multiple": {
        "language": "Python",
        "file": "BFCL_v3_parallel_multiple.json",
    },
    "irrelevance": {"language": "Python", "file": "BFCL_v3_irrelevance.json"},
    "java": {"language": "Java", "file": "BFCL_v3_java.json"},
    "javascript": {"language": "JavaScript", "file": "BFCL_v3_javascript.json"},
}

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
    "any": "string",
}
