"""Score BFCL responses using the AST checker."""

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

from . import config, db


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
    run_id = config.RUN_ID
    db_rows = db.get_responses(run_id, category=category, mode="single") if run_id else []
    if db_rows:
        responses = {r["entry_id"]: {"id": r["entry_id"], "result": r["result"]} for r in db_rows}
    else:
        # Fallback to flat files for backward compatibility / tests
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

        # Parse model output.  Raw Gemma markup (saved when generation-time
        # parsing missed it) is recovered here so already-stored rows still
        # score instead of counting as 0%.
        try:
            model_output = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            from .parse_gemma_tool_calls import parse_gemma_tool_call_content
            model_output = parse_gemma_tool_call_content(result_str)
            if model_output is None:
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

def run() -> list[dict]:
    """Evaluate all categories. Returns list of score rows."""
    _ensure_model_registered()
    run_id = config.RUN_ID

    rows: list[dict] = []

    for category, info in config.CATEGORIES.items():
        print(f"[evaluate] Category: {category}")
        result = evaluate_category(category, info)
        errors = result.get("errors", [])
        n_errors = len(errors)
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
        if run_id and errors:
            db.insert_eval_errors(run_id, category, errors)

    if run_id:
        db.insert_scores(run_id, rows)
    print(f"[evaluate] Scores saved to DB")

    return rows


if __name__ == "__main__":
    run()
