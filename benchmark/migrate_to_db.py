"""One-time migration: import existing flat-file results into the SQLite database.

For each model directory under benchmark/results/:
1. Create a synthetic run (notes: "Migrated from flat files")
2. Import responses, timings, scores, eval errors, inference metrics

Run: uv run python -m benchmark.migrate_to_db
"""

import csv
import json
from pathlib import Path

from . import config, db


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def migrate_model(model_key: str, model_info: dict) -> int | None:
    """Migrate one model's flat-file results into the DB. Returns run_id or None."""
    slug = model_info["slug"]
    model_dir = config.RESULTS_DIR / slug
    if not model_dir.exists():
        print(f"  [skip] {slug}: no results directory")
        return None

    scores_path = model_dir / "scores" / "accuracy.csv"
    if not scores_path.exists():
        print(f"  [skip] {slug}: no accuracy.csv")
        return None

    print(f"  Migrating {slug}...")

    # Create synthetic run
    run_id = db.create_run(
        model_key=model_key,
        model_slug=slug,
        model_display=model_info["display"],
        notes="Migrated from flat files",
        hardware={},
    )

    # --- Responses (single-turn) ---
    responses_dir = model_dir / "responses"
    for cat in config.CATEGORIES:
        resp_file = responses_dir / f"{cat}_responses.jsonl"
        rows = _read_jsonl(resp_file)
        if rows:
            db.insert_responses(run_id, cat, "single", rows)
            print(f"    {cat}: {len(rows)} single-turn responses")

    # --- Responses (multi-turn) ---
    for cat in config.MULTI_TURN_CATEGORIES:
        resp_file = responses_dir / f"{cat}_responses.jsonl"
        rows = _read_jsonl(resp_file)
        if rows:
            db.insert_responses(run_id, cat, "multi", rows)
            print(f"    {cat}: {len(rows)} multi-turn responses")

    # --- Single-turn timings ---
    st_timings_path = responses_dir / "all_timings.csv"
    st_timings = _read_csv(st_timings_path)
    if st_timings:
        # CSV reader returns strings; convert numeric fields
        for row in st_timings:
            row["prompt_n"] = int(row.get("prompt_n", 0))
            row["prompt_ms"] = float(row.get("prompt_ms", 0))
            row["prompt_per_second"] = float(row.get("prompt_per_second", 0))
            row["predicted_n"] = int(row.get("predicted_n", 0))
            row["predicted_ms"] = float(row.get("predicted_ms", 0))
            row["predicted_per_second"] = float(row.get("predicted_per_second", 0))
        db.insert_single_turn_timings(run_id, st_timings)
        print(f"    Single-turn timings: {len(st_timings)} rows")

    # --- Multi-turn timings ---
    mt_timings_path = responses_dir / "multi_turn_timings.csv"
    mt_timings = _read_csv(mt_timings_path)
    if mt_timings:
        for row in mt_timings:
            row["turns"] = int(row.get("turns", 0))
            row["api_calls"] = int(row.get("api_calls", 0))
            row["total_prompt_n"] = int(row.get("total_prompt_n", 0))
            row["total_prompt_ms"] = float(row.get("total_prompt_ms", 0))
            row["total_predicted_n"] = int(row.get("total_predicted_n", 0))
            row["total_predicted_ms"] = float(row.get("total_predicted_ms", 0))
            # Handle missing columns in older CSVs (e.g., Qwen2.5 lacks first_prompt_ms)
            row["first_prompt_ms"] = float(row.get("first_prompt_ms", 0) or 0)
            row["flags"] = row.get("flags", "")
        db.insert_multi_turn_timings(run_id, mt_timings)
        print(f"    Multi-turn timings: {len(mt_timings)} rows")

    # --- Scores ---
    score_rows = _read_csv(scores_path)
    if score_rows:
        for row in score_rows:
            row["correct"] = int(row["correct"])
            row["total"] = int(row["total"])
            row["accuracy"] = float(row["accuracy"])
        db.insert_scores(run_id, score_rows)
        print(f"    Scores: {len(score_rows)} categories")

    # --- Eval errors ---
    scores_dir = model_dir / "scores"
    for cat in config.MULTI_TURN_CATEGORIES:
        err_file = scores_dir / f"{cat}_eval_errors.jsonl"
        errors = _read_jsonl(err_file)
        if errors:
            db.insert_eval_errors(run_id, cat, errors)
            print(f"    {cat}: {len(errors)} eval errors")

    # --- Inference metrics (summary) ---
    metrics_dir = model_dir / "metrics"
    metrics_json = metrics_dir / "inference_metrics.json"
    if metrics_json.exists():
        summary = json.loads(metrics_json.read_text())
        # Older results may lack GPU VRAM fields — add defaults
        mem = summary.get("memory", {})
        mem.setdefault("baseline_gpu_vram_mb", 0)
        mem.setdefault("peak_gpu_vram_mb", 0)
        db.insert_inference_metrics(run_id, summary)
        print(f"    Inference metrics: imported")

    # --- Metrics per request ---
    metrics_csv = metrics_dir / "inference_metrics.csv"
    per_req = _read_csv(metrics_csv)
    if per_req:
        for row in per_req:
            row["prompt_n"] = int(row.get("prompt_n", 0))
            row["prompt_ms"] = float(row.get("prompt_ms", 0))
            row["prompt_per_second"] = float(row.get("prompt_per_second", 0))
            row["predicted_n"] = int(row.get("predicted_n", 0))
            row["predicted_ms"] = float(row.get("predicted_ms", 0))
            row["predicted_per_second"] = float(row.get("predicted_per_second", 0))
        db.insert_metrics_per_request(run_id, per_req)
        print(f"    Per-request metrics: {len(per_req)} rows")

    # Finish the run
    db.finish_run(run_id)
    print(f"    Done (run_id={run_id})")
    return run_id


def main():
    print("Migrating flat-file results to SQLite database...")
    print(f"Database: {db.DB_PATH}")
    print()

    migrated = 0
    for key, info in config.MODELS.items():
        run_id = migrate_model(key, info)
        if run_id is not None:
            migrated += 1

    print(f"\nMigrated {migrated} model(s).")
    runs = db.list_runs()
    print(f"Total runs in database: {len(runs)}")


if __name__ == "__main__":
    main()
