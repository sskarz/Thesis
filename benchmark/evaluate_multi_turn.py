"""Score BFCL multi-turn responses using the execution-based checker."""

import json

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)

from . import config, db
from .evaluate_responses import _ensure_model_registered, load_jsonl


# ---------------------------------------------------------------------------
# Per-category evaluation
# ---------------------------------------------------------------------------

def evaluate_category(category: str, category_info: dict) -> dict:
    """Evaluate one multi-turn category. Returns accuracy dict."""
    run_id = config.RUN_ID
    db_rows = db.get_responses(run_id, category=category, mode="multi") if run_id else []
    if db_rows:
        responses = {}
        for r in db_rows:
            result = r["result"]
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass
            responses[r["entry_id"]] = {"id": r["entry_id"], "result": result}
    else:
        # Fallback to flat files for backward compatibility / tests
        response_file = config.RESPONSES_DIR / f"{category}_responses.jsonl"
        if not response_file.exists():
            print(f"  [warn] No response file for {category}, skipping.")
            return {"category": category, "correct": 0, "total": 0, "accuracy": 0}
        responses = {e["id"]: e for e in load_jsonl(response_file)}

    # Load test entries
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    test_entries = {e["id"]: e for e in load_jsonl(data_file)}

    # Load ground truth
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
        test_entry = test_entries.get(entry_id)
        answer_entry = answers.get(entry_id)
        if not test_entry or not answer_entry:
            errors.append({"id": entry_id, "error": "Missing test or answer data"})
            continue

        model_result = resp["result"]  # list[list[list[str]]] (turns × steps × funcs)
        ground_truth = answer_entry["ground_truth"]  # list[list[str]] (turns × funcs)

        try:
            result = multi_turn_checker(
                multi_turn_model_result_list_decoded=model_result,
                multi_turn_ground_truth_list=ground_truth,
                test_entry=test_entry,
                test_category=category,
                model_name=config.MODEL_NAME,
            )
            if result["valid"]:
                correct += 1
            else:
                errors.append({
                    "id": entry_id,
                    "error": result.get("error_message", result.get("error", [])),
                })
        except Exception as e:
            errors.append({"id": entry_id, "error": str(e)})

    accuracy = correct / total if total else 0
    if config.RUN_ID and errors:
        db.insert_eval_errors(config.RUN_ID, category, errors)
        print(f"  → {len(errors)} errors saved to DB")
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

def run(categories: dict[str, dict] | None = None) -> list[dict]:
    """Evaluate selected multi-turn categories. Returns list of accuracy rows."""
    _ensure_model_registered()
    run_id = config.RUN_ID
    cats = categories or config.resolve_multi_turn_categories(None)

    rows: list[dict] = []

    for category, info in cats.items():
        print(f"[multi-turn evaluate] Category: {category}")
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

    if run_id:
        db.insert_scores(run_id, rows)
    print(f"[multi-turn evaluate] Scores saved to DB")

    return rows


if __name__ == "__main__":
    run()
