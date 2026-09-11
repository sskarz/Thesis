"""SQLite data-access layer for benchmark results.

Provides a thin wrapper around stdlib sqlite3 with:
- Auto-initializing schema (CREATE TABLE IF NOT EXISTS)
- Module-level connection singleton
- Insert/get functions for every table
- Metadata auto-detection (GPU, CPU, versions)
"""

import json
import os
import platform
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import config

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

DB_PATH: Path = config.RESULTS_DIR / "benchmark.db"
_conn: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    """Return the module-level connection, creating it if needed."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH))
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(_conn)
    return _conn


def close() -> None:
    """Close the module-level connection."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    conn.executescript(_SCHEMA)


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key       TEXT NOT NULL,
    model_slug      TEXT NOT NULL,
    model_display   TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    hardware        TEXT,
    server_params   TEXT,
    llama_cpp_version TEXT,
    bfcl_version    TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    entry_id    TEXT NOT NULL,
    category    TEXT NOT NULL,
    mode        TEXT NOT NULL CHECK(mode IN ('single', 'multi')),
    result      TEXT NOT NULL,
    flags       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS single_turn_timings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES runs(id),
    entry_id             TEXT NOT NULL,
    category             TEXT NOT NULL,
    prompt_n             INTEGER NOT NULL DEFAULT 0,
    prompt_ms            REAL NOT NULL DEFAULT 0,
    prompt_per_second    REAL NOT NULL DEFAULT 0,
    predicted_n          INTEGER NOT NULL DEFAULT 0,
    predicted_ms         REAL NOT NULL DEFAULT 0,
    predicted_per_second REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS multi_turn_timings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES runs(id),
    entry_id             TEXT NOT NULL,
    category             TEXT NOT NULL,
    turns                INTEGER NOT NULL DEFAULT 0,
    api_calls            INTEGER NOT NULL DEFAULT 0,
    total_prompt_n       INTEGER NOT NULL DEFAULT 0,
    total_prompt_ms      REAL NOT NULL DEFAULT 0,
    total_predicted_n    INTEGER NOT NULL DEFAULT 0,
    total_predicted_ms   REAL NOT NULL DEFAULT 0,
    first_prompt_ms      REAL NOT NULL DEFAULT 0,
    flags                TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    category    TEXT NOT NULL,
    correct     INTEGER NOT NULL,
    total       INTEGER NOT NULL,
    accuracy    REAL NOT NULL,
    UNIQUE(run_id, category)
);

CREATE TABLE IF NOT EXISTS eval_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    entry_id    TEXT NOT NULL,
    category    TEXT NOT NULL,
    error       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_metrics (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL UNIQUE REFERENCES runs(id),
    num_requests         INTEGER NOT NULL,
    prompt_tps_mean      REAL, prompt_tps_median   REAL, prompt_tps_p95 REAL, prompt_tps_p99 REAL,
    gen_tps_mean         REAL, gen_tps_median       REAL, gen_tps_p95   REAL, gen_tps_p99   REAL,
    ttft_mean_ms         REAL, ttft_median_ms       REAL, ttft_p95_ms   REAL, ttft_p99_ms   REAL,
    baseline_rss_mb      REAL, peak_rss_mb          REAL,
    baseline_gpu_vram_mb REAL, peak_gpu_vram_mb     REAL,
    power_watts          REAL
);

CREATE TABLE IF NOT EXISTS metrics_per_request (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES runs(id),
    entry_id             TEXT NOT NULL,
    category             TEXT NOT NULL,
    prompt_n             INTEGER NOT NULL DEFAULT 0,
    prompt_ms            REAL NOT NULL DEFAULT 0,
    prompt_per_second    REAL NOT NULL DEFAULT 0,
    predicted_n          INTEGER NOT NULL DEFAULT 0,
    predicted_ms         REAL NOT NULL DEFAULT 0,
    predicted_per_second REAL NOT NULL DEFAULT 0
);

-- Idempotent multi-turn checkpoint/resume: one row per run/entry/category.
CREATE UNIQUE INDEX IF NOT EXISTS idx_responses_run_entry_cat_mode
    ON responses(run_id, entry_id, category, mode);
CREATE UNIQUE INDEX IF NOT EXISTS idx_multi_turn_timings_run_entry_cat
    ON multi_turn_timings(run_id, entry_id, category);
"""

# ---------------------------------------------------------------------------
# Metadata auto-detection
# ---------------------------------------------------------------------------


def _detect_hardware() -> dict:
    """Detect GPU, CPU, RAM, OS info."""
    hw: dict = {
        "cpu": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
    }
    # RAM via psutil (already a dependency)
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass
    # NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            hw["gpu"] = parts[0]
            hw["gpu_vram_gb"] = round(float(parts[1]) / 1024, 1) if len(parts) > 1 else None
    except Exception:
        pass
    return hw


def _detect_llama_cpp_version() -> str | None:
    """Get git short hash from llama.cpp directory."""
    llama_dir = config.PROJECT_ROOT / "llama.cpp"
    if not llama_dir.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(llama_dir),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _detect_bfcl_version() -> str | None:
    """Get bfcl_eval package version."""
    try:
        import bfcl_eval
        return getattr(bfcl_eval, "__version__", None)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def create_run(
    model_key: str,
    model_slug: str | None = None,
    model_display: str | None = None,
    notes: str | None = None,
    server_params: dict | None = None,
    hardware: dict | None = None,
) -> int:
    """Create a new run and return its ID."""
    conn = get_connection()
    m = config.MODELS.get(model_key, {})
    slug = model_slug or m.get("slug", model_key)
    display = model_display or m.get("display", model_key)

    hw = hardware or _detect_hardware()
    llama_ver = _detect_llama_cpp_version()
    bfcl_ver = _detect_bfcl_version()

    cursor = conn.execute(
        """INSERT INTO runs
           (model_key, model_slug, model_display, started_at,
            hardware, server_params, llama_cpp_version, bfcl_version, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_key,
            slug,
            display,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(hw),
            json.dumps(server_params) if server_params else None,
            llama_ver,
            bfcl_ver,
            notes,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def finish_run(run_id: int) -> None:
    """Set the finished_at timestamp for a run."""
    conn = get_connection()
    conn.execute(
        "UPDATE runs SET finished_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), run_id),
    )
    conn.commit()


def get_run(run_id: int) -> dict | None:
    """Return a single run as a dict, or None."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_runs(model_key: str | None = None) -> list[dict]:
    """List all runs, optionally filtered by model_key."""
    conn = get_connection()
    if model_key:
        rows = conn.execute(
            "SELECT * FROM runs WHERE model_key = ? ORDER BY id DESC",
            (model_key,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_latest_run(model_key: str) -> dict | None:
    """Return the most recent finished run for a model, or None."""
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM runs
           WHERE model_key = ? AND finished_at IS NOT NULL
           ORDER BY id DESC LIMIT 1""",
        (model_key,),
    ).fetchone()
    return dict(row) if row else None


def get_unfinished_runs(model_key: str | None = None) -> list[dict]:
    """List unfinished runs (finished_at IS NULL), newest first."""
    conn = get_connection()
    if model_key:
        rows = conn.execute(
            """SELECT * FROM runs
               WHERE model_key = ? AND finished_at IS NULL
               ORDER BY id DESC""",
            (model_key,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM runs
               WHERE finished_at IS NULL
               ORDER BY id DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def insert_responses(
    run_id: int,
    category: str,
    mode: str,
    rows: list[dict],
) -> None:
    """Bulk-insert response rows. Each dict must have 'id' and 'result'."""
    conn = get_connection()
    conn.executemany(
        """INSERT INTO responses (run_id, entry_id, category, mode, result, flags)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id,
                r["id"],
                category,
                mode,
                r["result"] if isinstance(r["result"], str) else json.dumps(r["result"]),
                r.get("flags", ""),
            )
            for r in rows
        ],
    )
    conn.commit()


def get_responses(
    run_id: int,
    category: str | None = None,
    mode: str | None = None,
) -> list[dict]:
    """Retrieve responses, optionally filtered by category and/or mode."""
    conn = get_connection()
    query = "SELECT * FROM responses WHERE run_id = ?"
    params: list = [run_id]
    if category:
        query += " AND category = ?"
        params.append(category)
    if mode:
        query += " AND mode = ?"
        params.append(mode)
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_response_entry_ids(
    run_id: int,
    category: str,
    mode: str | None = None,
) -> set[str]:
    """Return entry_ids already stored for a run/category (optionally mode)."""
    conn = get_connection()
    query = "SELECT entry_id FROM responses WHERE run_id = ? AND category = ?"
    params: list = [run_id, category]
    if mode:
        query += " AND mode = ?"
        params.append(mode)
    return {row["entry_id"] for row in conn.execute(query, params).fetchall()}


def insert_multi_turn_entry(
    run_id: int,
    category: str,
    result_row: dict,
    timing_row: dict,
) -> None:
    """Atomically upsert one completed multi-turn response + timing row.

    Used as a per-entry checkpoint so a killed process does not lose
    finished entries. ``ON CONFLICT`` keeps resume idempotent if the same
    entry is flushed twice.
    """
    conn = get_connection()
    result = result_row["result"]
    result_json = result if isinstance(result, str) else json.dumps(result)
    flags = result_row.get("flags", "")
    try:
        conn.execute(
            """INSERT INTO responses (run_id, entry_id, category, mode, result, flags)
               VALUES (?, ?, ?, 'multi', ?, ?)
               ON CONFLICT(run_id, entry_id, category, mode) DO UPDATE SET
                 result = excluded.result,
                 flags = excluded.flags""",
            (run_id, result_row["id"], category, result_json, flags),
        )
        conn.execute(
            """INSERT INTO multi_turn_timings
               (run_id, entry_id, category, turns, api_calls,
                total_prompt_n, total_prompt_ms, total_predicted_n, total_predicted_ms,
                first_prompt_ms, flags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, entry_id, category) DO UPDATE SET
                 turns = excluded.turns,
                 api_calls = excluded.api_calls,
                 total_prompt_n = excluded.total_prompt_n,
                 total_prompt_ms = excluded.total_prompt_ms,
                 total_predicted_n = excluded.total_predicted_n,
                 total_predicted_ms = excluded.total_predicted_ms,
                 first_prompt_ms = excluded.first_prompt_ms,
                 flags = excluded.flags""",
            (
                run_id,
                timing_row["id"],
                timing_row.get("category", category),
                timing_row.get("turns", 0),
                timing_row.get("api_calls", 0),
                timing_row.get("total_prompt_n", 0),
                timing_row.get("total_prompt_ms", 0),
                timing_row.get("total_predicted_n", 0),
                timing_row.get("total_predicted_ms", 0),
                timing_row.get("first_prompt_ms", 0),
                timing_row.get("flags", flags),
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Single-turn timings
# ---------------------------------------------------------------------------


def insert_single_turn_timings(run_id: int, rows: list[dict]) -> None:
    """Bulk-insert single-turn timing rows."""
    conn = get_connection()
    conn.executemany(
        """INSERT INTO single_turn_timings
           (run_id, entry_id, category, prompt_n, prompt_ms, prompt_per_second,
            predicted_n, predicted_ms, predicted_per_second)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id,
                r["id"],
                r["category"],
                r.get("prompt_n", 0),
                r.get("prompt_ms", 0),
                r.get("prompt_per_second", 0),
                r.get("predicted_n", 0),
                r.get("predicted_ms", 0),
                r.get("predicted_per_second", 0),
            )
            for r in rows
        ],
    )
    conn.commit()


def get_single_turn_timings(run_id: int) -> list[dict]:
    """Retrieve all single-turn timings for a run."""
    conn = get_connection()
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM single_turn_timings WHERE run_id = ?", (run_id,)
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# Multi-turn timings
# ---------------------------------------------------------------------------


def insert_multi_turn_timings(run_id: int, rows: list[dict]) -> None:
    """Bulk-insert multi-turn timing rows."""
    conn = get_connection()
    conn.executemany(
        """INSERT INTO multi_turn_timings
           (run_id, entry_id, category, turns, api_calls,
            total_prompt_n, total_prompt_ms, total_predicted_n, total_predicted_ms,
            first_prompt_ms, flags)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id,
                r["id"],
                r["category"],
                r.get("turns", 0),
                r.get("api_calls", 0),
                r.get("total_prompt_n", 0),
                r.get("total_prompt_ms", 0),
                r.get("total_predicted_n", 0),
                r.get("total_predicted_ms", 0),
                r.get("first_prompt_ms", 0),
                r.get("flags", ""),
            )
            for r in rows
        ],
    )
    conn.commit()


def get_multi_turn_timings(run_id: int) -> list[dict]:
    """Retrieve all multi-turn timings for a run."""
    conn = get_connection()
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM multi_turn_timings WHERE run_id = ?", (run_id,)
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def insert_scores(run_id: int, rows: list[dict]) -> None:
    """Bulk-insert score rows. Each dict has category, correct, total, accuracy."""
    conn = get_connection()
    conn.executemany(
        """INSERT OR REPLACE INTO scores (run_id, category, correct, total, accuracy)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (run_id, r["category"], r["correct"], r["total"], r["accuracy"])
            for r in rows
        ],
    )
    conn.commit()


def get_scores(run_id: int) -> list[dict]:
    """Retrieve all scores for a run."""
    conn = get_connection()
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM scores WHERE run_id = ? ORDER BY category", (run_id,)
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# Eval errors
# ---------------------------------------------------------------------------


def insert_eval_errors(
    run_id: int, category: str, errors: list[dict]
) -> None:
    """Bulk-insert evaluation errors."""
    if not errors:
        return
    conn = get_connection()
    conn.executemany(
        """INSERT INTO eval_errors (run_id, entry_id, category, error)
           VALUES (?, ?, ?, ?)""",
        [
            (
                run_id,
                e["id"],
                category,
                e["error"] if isinstance(e["error"], str) else json.dumps(e["error"]),
            )
            for e in errors
        ],
    )
    conn.commit()


def get_eval_errors(run_id: int, category: str | None = None) -> list[dict]:
    """Retrieve eval errors, optionally filtered by category."""
    conn = get_connection()
    if category:
        rows = conn.execute(
            "SELECT * FROM eval_errors WHERE run_id = ? AND category = ?",
            (run_id, category),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_errors WHERE run_id = ?", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Inference metrics (summary)
# ---------------------------------------------------------------------------


def insert_inference_metrics(run_id: int, summary: dict) -> None:
    """Insert the flattened inference metrics summary for a run."""
    conn = get_connection()
    p = summary.get("prompt_tokens_per_sec", {})
    g = summary.get("generation_tokens_per_sec", {})
    t = summary.get("ttft_ms", {})
    m = summary.get("memory", {})
    conn.execute(
        """INSERT OR REPLACE INTO inference_metrics
           (run_id, num_requests,
            prompt_tps_mean, prompt_tps_median, prompt_tps_p95, prompt_tps_p99,
            gen_tps_mean, gen_tps_median, gen_tps_p95, gen_tps_p99,
            ttft_mean_ms, ttft_median_ms, ttft_p95_ms, ttft_p99_ms,
            baseline_rss_mb, peak_rss_mb,
            baseline_gpu_vram_mb, peak_gpu_vram_mb,
            power_watts)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            summary.get("num_requests", 0),
            p.get("mean"), p.get("median"), p.get("p95"), p.get("p99"),
            g.get("mean"), g.get("median"), g.get("p95"), g.get("p99"),
            t.get("mean"), t.get("median"), t.get("p95"), t.get("p99"),
            m.get("baseline_rss_mb"), m.get("peak_rss_mb"),
            m.get("baseline_gpu_vram_mb"), m.get("peak_gpu_vram_mb"),
            summary.get("power_watts"),
        ),
    )
    conn.commit()


def get_inference_metrics(run_id: int) -> dict:
    """Retrieve inference metrics, re-nested into the original JSON structure."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM inference_metrics WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return {}
    r = dict(row)
    return {
        "num_requests": r["num_requests"],
        "prompt_tokens_per_sec": {
            "mean": r["prompt_tps_mean"] or 0,
            "median": r["prompt_tps_median"] or 0,
            "p95": r["prompt_tps_p95"] or 0,
            "p99": r["prompt_tps_p99"] or 0,
        },
        "generation_tokens_per_sec": {
            "mean": r["gen_tps_mean"] or 0,
            "median": r["gen_tps_median"] or 0,
            "p95": r["gen_tps_p95"] or 0,
            "p99": r["gen_tps_p99"] or 0,
        },
        "ttft_ms": {
            "mean": r["ttft_mean_ms"] or 0,
            "median": r["ttft_median_ms"] or 0,
            "p95": r["ttft_p95_ms"] or 0,
            "p99": r["ttft_p99_ms"] or 0,
        },
        "memory": {
            "baseline_rss_mb": r["baseline_rss_mb"] or 0,
            "peak_rss_mb": r["peak_rss_mb"] or 0,
            "baseline_gpu_vram_mb": r["baseline_gpu_vram_mb"] or 0,
            "peak_gpu_vram_mb": r["peak_gpu_vram_mb"] or 0,
        },
        "power_watts": r["power_watts"],
    }


# ---------------------------------------------------------------------------
# Metrics per request
# ---------------------------------------------------------------------------


def insert_metrics_per_request(run_id: int, rows: list[dict]) -> None:
    """Bulk-insert per-request metrics rows."""
    conn = get_connection()
    conn.executemany(
        """INSERT INTO metrics_per_request
           (run_id, entry_id, category, prompt_n, prompt_ms, prompt_per_second,
            predicted_n, predicted_ms, predicted_per_second)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id,
                r["id"],
                r["category"],
                r.get("prompt_n", 0),
                r.get("prompt_ms", 0),
                r.get("prompt_per_second", 0),
                r.get("predicted_n", 0),
                r.get("predicted_ms", 0),
                r.get("predicted_per_second", 0),
            )
            for r in rows
        ],
    )
    conn.commit()


def get_metrics_per_request(run_id: int) -> list[dict]:
    """Retrieve per-request metrics for a run."""
    conn = get_connection()
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM metrics_per_request WHERE run_id = ?", (run_id,)
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# Convenience: get all model results (for compare.py)
# ---------------------------------------------------------------------------


def get_all_model_results() -> list[dict]:
    """For each distinct model_key, find its latest finished run and return
    scores + metrics. Drop-in replacement for compare._load_models()."""
    conn = get_connection()
    # Get distinct model keys that have finished runs
    model_keys = conn.execute(
        """SELECT DISTINCT model_key FROM runs
           WHERE finished_at IS NOT NULL
           ORDER BY model_key"""
    ).fetchall()

    results = []
    for row in model_keys:
        key = row["model_key"]
        run = get_latest_run(key)
        if run is None:
            continue
        scores = get_scores(run["id"])
        metrics = get_inference_metrics(run["id"])
        if not scores:
            continue
        results.append({
            "key": key,
            "display": run["model_display"],
            "slug": run["model_slug"],
            "run_id": run["id"],
            "scores": scores,
            "metrics": metrics,
        })
    return results
