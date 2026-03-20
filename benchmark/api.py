"""FastAPI backend for the benchmark dashboard.

Thin read-only wrapper around db.py.
Run with: uv run uvicorn benchmark.api:app --reload --port 8000
"""

import json

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from . import config, db

app = FastAPI(title="Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _parse_json_fields(row: dict) -> dict:
    """Parse hardware and server_params JSON strings into dicts."""
    for key in ("hardware", "server_params"):
        val = row.get(key)
        if isinstance(val, str):
            try:
                row[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    return row


# --- Runs ---


@app.get("/api/runs")
def list_runs(model_key: str | None = Query(None)):
    return [_parse_json_fields(r) for r in db.list_runs(model_key)]


@app.get("/api/runs/{run_id}")
def get_run(run_id: int):
    row = db.get_run(run_id)
    if row is None:
        return {"error": "not found"}
    return _parse_json_fields(row)


@app.get("/api/runs/latest/{model_key}")
def get_latest_run(model_key: str):
    row = db.get_latest_run(model_key)
    if row is None:
        return {"error": "not found"}
    return _parse_json_fields(row)


# --- Models & Categories ---


@app.get("/api/models")
def get_models():
    return config.MODELS


@app.get("/api/categories")
def get_categories():
    return {
        "single_turn": config.CATEGORIES,
        "multi_turn": config.MULTI_TURN_CATEGORIES,
    }


# --- Scores ---


@app.get("/api/scores/{run_id}")
def get_scores(run_id: int):
    return db.get_scores(run_id)


# --- Inference Metrics ---


@app.get("/api/metrics/{run_id}")
def get_metrics(run_id: int):
    return db.get_inference_metrics(run_id)


# --- Timings ---


@app.get("/api/timings/single/{run_id}")
def get_single_turn_timings(run_id: int):
    return db.get_single_turn_timings(run_id)


@app.get("/api/timings/multi/{run_id}")
def get_multi_turn_timings(run_id: int):
    return db.get_multi_turn_timings(run_id)


# --- Eval Errors ---


@app.get("/api/errors/{run_id}")
def get_errors(run_id: int, category: str | None = Query(None)):
    return db.get_eval_errors(run_id, category)


# --- Per-request Metrics ---


@app.get("/api/metrics/per-request/{run_id}")
def get_metrics_per_request(run_id: int):
    return db.get_metrics_per_request(run_id)


# --- Comparison ---


@app.get("/api/comparison")
def get_comparison():
    results = db.get_all_model_results()
    for r in results:
        run = db.get_run(r["run_id"])
        if run:
            r["run"] = _parse_json_fields(run)
    return results
