"""Score BFCL responses using the AST checker."""

import csv
import json
from pathlib import Path

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

LANGUAGE_MAP: dict[str, Language] = {
    "Python": Language.PYTHON,
    "Java": Language.JAVA,
    "JavaScript": Language.JAVASCRIPT,
}

from . import config


def _ensure_model_registered() -> None:
    """Register the active model in BFCL's config if not already present."""
    key = config.MODEL_NAME
    if key not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[key] = ModelConfig(
            model_name=key,
            display_name=f"{config.MODEL_DISPLAY} (local)",
            org="local",
            underscore_to_dot=False,
            is_fc_model=True,
            url="",
            license="Apache-2.0",
            input_price=0,
            output_price=0,
            model_handler=None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_function_calling_format(output) -> bool:
    """True when *output* looks like ``[{"func": {"arg": val}}, ...]``."""
    if not isinstance(output, list) or len(output) == 0:
        return False
    for item in output:
        if not isinstance(item, dict) or len(item) != 1:
            return False
        if not isinstance(list(item.values())[0], dict):
            return False
    return True


# ---------------------------------------------------------------------------
# Per-category evaluation
# ---------------------------------------------------------------------------

def _evaluate_irrelevance(responses: dict[str, dict]) -> dict:
    """For irrelevance, correct = model did NOT produce tool calls."""
    correct = 0
    total = 0
    for _id, resp in responses.items():
        total += 1
        result_str = resp["result"]
        try:
            parsed = json.loads(result_str)
            if is_function_calling_format(parsed):
                continue  # model incorrectly called functions
        except (json.JSONDecodeError, TypeError):
            pass  # text response → correct
        correct += 1

    return {
        "category": "irrelevance",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
    }


def evaluate_category(category: str, category_info: dict) -> dict:
    """Evaluate one BFCL category. Returns accuracy dict."""
    # Load response file
    response_file = config.RESPONSES_DIR / f"{category}_responses.jsonl"
    if not response_file.exists():
        return {"category": category, "correct": 0, "total": 0, "accuracy": 0}
    responses = {e["id"]: e for e in load_jsonl(response_file)}

    if category == "irrelevance":
        return _evaluate_irrelevance(responses)

    language = LANGUAGE_MAP[category_info["language"]]

    # Load test data (for func_description)
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    test_entries = {e["id"]: e for e in load_jsonl(data_file)}

    # Load possible answers
    answer_file = config.BFCL_POSSIBLE_ANSWER_DIR / category_info["file"]
    if not answer_file.exists():
        print(f"  [warn] No possible-answer file for {category}, skipping.")
        return {"category": category, "correct": 0, "total": 0, "accuracy": 0}
    answers = {e["id"]: e for e in load_jsonl(answer_file)}

    correct = 0
    total = 0
    errors: list[dict] = []

    for entry_id, resp in responses.items():
        total += 1
        result_str = resp["result"]

        # Parse model output
        try:
            model_output = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            errors.append({"id": entry_id, "error": "Could not parse result as JSON"})
            continue

        if not isinstance(model_output, list):
            errors.append({"id": entry_id, "error": "Result is not a list"})
            continue

        test_entry = test_entries.get(entry_id)
        answer_entry = answers.get(entry_id)
        if not test_entry or not answer_entry:
            errors.append({"id": entry_id, "error": "Missing test or answer data"})
            continue

        func_description = test_entry["function"]
        possible_answer = answer_entry["ground_truth"]

        try:
            result = ast_checker(
                func_description,
                model_output,
                possible_answer,
                language,
                category,
                config.MODEL_NAME,
            )
            if result["valid"]:
                correct += 1
            else:
                errors.append({"id": entry_id, "error": result.get("error", [])})
        except Exception as e:
            errors.append({"id": entry_id, "error": str(e)})

    accuracy = correct / total if total else 0
    return {
        "category": category,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> Path:
    """Evaluate all categories. Returns path to accuracy CSV."""
    _ensure_model_registered()
    config.SCORES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for category, info in config.CATEGORIES.items():
        print(f"[evaluate] Category: {category}")
        result = evaluate_category(category, info)
        n_errors = len(result.get("errors", []))
        print(
            f"  → {result['correct']}/{result['total']} correct "
            f"({result['accuracy']:.1%})"
            + (f", {n_errors} errors" if n_errors else "")
        )
        rows.append({
            "category": result["category"],
            "correct": result["correct"],
            "total": result["total"],
            "accuracy": round(result["accuracy"], 4),
        })

    csv_path = config.SCORES_DIR / "accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate] Scores saved to {csv_path.name}")

    return csv_path


if __name__ == "__main__":
    run()
