"""Shared constants for the BFCL benchmark pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_URL = "http://localhost:8080"

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_HF_REPO = "bartowski/Qwen2.5-3B-Instruct-GGUF"
MODEL_QUANT = "Q4_K_M"
MODEL_NAME = "qwen2.5-3b"  # API name reported by llama-server
MODEL_DISPLAY = f"Qwen2.5-3B-Instruct ({MODEL_QUANT})"
MODEL_SLUG = f"Qwen2.5-3B-Instruct-{MODEL_QUANT}"

# ---------------------------------------------------------------------------
# Paths  (results are organized per model)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
MODEL_RESULTS_DIR = RESULTS_DIR / MODEL_SLUG
RESPONSES_DIR = MODEL_RESULTS_DIR / "responses"
METRICS_DIR = MODEL_RESULTS_DIR / "metrics"
SCORES_DIR = MODEL_RESULTS_DIR / "scores"
FIGURES_DIR = MODEL_RESULTS_DIR / "figures"

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
