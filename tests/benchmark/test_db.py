"""Round-trip tests for benchmark.db — every table insert/get pair."""

import json

from benchmark import db


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_create_and_get_run(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {"cpu": "test"})
    monkeypatch.setattr(db, "_detect_llama_cpp_version", lambda: "abc123")
    monkeypatch.setattr(db, "_detect_bfcl_version", lambda: "2026.3.11")

    run_id = db.create_run("qwen2.5-3b", notes="test run")
    assert isinstance(run_id, int)

    run = db.get_run(run_id)
    assert run is not None
    assert run["model_key"] == "qwen2.5-3b"
    assert run["model_slug"] == "Qwen2.5-3B-Instruct-Q4_K_M"
    assert run["model_display"] == "Qwen2.5-3B-Instruct (Q4_K_M)"
    assert run["notes"] == "test run"
    assert run["finished_at"] is None
    assert run["llama_cpp_version"] == "abc123"
    assert run["bfcl_version"] == "2026.3.11"
    assert json.loads(run["hardware"])["cpu"] == "test"


def test_create_run_sft_iffc_uses_distinct_slug(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    base_id = db.create_run("gemma4-e2b")
    sft_id = db.create_run("gemma4-e2b-sft-iffc")
    base = db.get_run(base_id)
    sft = db.get_run(sft_id)
    assert sft["model_key"] == "gemma4-e2b-sft-iffc"
    assert sft["model_slug"] == "Gemma-4-E2B-it-SFT-IFFC-Q4_K_M"
    assert sft["model_display"] == "Gemma-4-E2B-it SFT-IFFC (Q4_K_M)"
    assert sft["model_slug"] != base["model_slug"]
    assert sft["model_key"] != base["model_key"]


def test_finish_run(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")
    db.finish_run(run_id)
    run = db.get_run(run_id)
    assert run["finished_at"] is not None


def test_list_runs_filters_by_model(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    db.create_run("qwen2.5-3b")
    db.create_run("llama-3.2-3b")
    db.create_run("qwen2.5-3b")

    all_runs = db.list_runs()
    assert len(all_runs) == 3
    qwen_runs = db.list_runs(model_key="qwen2.5-3b")
    assert len(qwen_runs) == 2
    assert all(r["model_key"] == "qwen2.5-3b" for r in qwen_runs)


def test_get_latest_run_returns_most_recent_finished(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    r1 = db.create_run("qwen2.5-3b", notes="first")
    db.finish_run(r1)
    r2 = db.create_run("qwen2.5-3b", notes="second")
    db.finish_run(r2)
    # r3 not finished
    db.create_run("qwen2.5-3b", notes="third")

    latest = db.get_latest_run("qwen2.5-3b")
    assert latest is not None
    assert latest["id"] == r2
    assert latest["notes"] == "second"


def test_get_latest_run_returns_none_when_no_finished():
    assert db.get_latest_run("nonexistent-model") is None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def test_insert_and_get_responses(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    rows = [
        {"id": "simple_0", "result": '[{"weather": {"city": "Paris"}}]'},
        {"id": "simple_1", "result": "text response"},
    ]
    db.insert_responses(run_id, "simple", "single", rows)

    got = db.get_responses(run_id, category="simple")
    assert len(got) == 2
    assert got[0]["entry_id"] == "simple_0"
    assert got[0]["mode"] == "single"
    assert got[0]["result"] == '[{"weather": {"city": "Paris"}}]'


def test_insert_responses_with_list_result(monkeypatch):
    """Result can be a list (multi-turn) — should be JSON-serialized."""
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    rows = [{"id": "mt_0", "result": [[["func(a=1)"]]], "flags": "overflow"}]
    db.insert_responses(run_id, "multi_turn_base", "multi", rows)

    got = db.get_responses(run_id, mode="multi")
    assert len(got) == 1
    assert json.loads(got[0]["result"]) == [[["func(a=1)"]]]
    assert got[0]["flags"] == "overflow"


def test_get_responses_filters(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    db.insert_responses(run_id, "simple", "single", [{"id": "s1", "result": "ok"}])
    db.insert_responses(run_id, "multi_turn_base", "multi", [{"id": "m1", "result": "ok"}])

    single = db.get_responses(run_id, mode="single")
    multi = db.get_responses(run_id, mode="multi")
    by_cat = db.get_responses(run_id, category="simple")
    all_resp = db.get_responses(run_id)

    assert len(single) == 1
    assert len(multi) == 1
    assert len(by_cat) == 1
    assert len(all_resp) == 2


# ---------------------------------------------------------------------------
# Single-turn timings
# ---------------------------------------------------------------------------


def test_insert_and_get_single_turn_timings(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    rows = [
        {
            "id": "simple_0", "category": "simple",
            "prompt_n": 100, "prompt_ms": 10.5, "prompt_per_second": 9523.8,
            "predicted_n": 20, "predicted_ms": 50.0, "predicted_per_second": 400.0,
        },
        {
            "id": "simple_1", "category": "simple",
            "prompt_n": 150, "prompt_ms": 15.0, "prompt_per_second": 10000.0,
            "predicted_n": 30, "predicted_ms": 75.0, "predicted_per_second": 400.0,
        },
    ]
    db.insert_single_turn_timings(run_id, rows)

    got = db.get_single_turn_timings(run_id)
    assert len(got) == 2
    assert got[0]["entry_id"] == "simple_0"
    assert got[0]["prompt_n"] == 100
    assert got[0]["prompt_ms"] == 10.5


# ---------------------------------------------------------------------------
# Multi-turn timings
# ---------------------------------------------------------------------------


def test_insert_and_get_multi_turn_timings(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    rows = [
        {
            "id": "mt_0", "category": "multi_turn_base",
            "turns": 4, "api_calls": 9,
            "total_prompt_n": 4733, "total_prompt_ms": 304.33,
            "total_predicted_n": 382, "total_predicted_ms": 1404.94,
            "first_prompt_ms": 50.0, "flags": "",
        },
    ]
    db.insert_multi_turn_timings(run_id, rows)

    got = db.get_multi_turn_timings(run_id)
    assert len(got) == 1
    assert got[0]["turns"] == 4
    assert got[0]["api_calls"] == 9
    assert got[0]["first_prompt_ms"] == 50.0


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def test_insert_and_get_scores(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    rows = [
        {"category": "simple", "correct": 374, "total": 400, "accuracy": 0.935},
        {"category": "multiple", "correct": 186, "total": 200, "accuracy": 0.93},
    ]
    db.insert_scores(run_id, rows)

    got = db.get_scores(run_id)
    assert len(got) == 2
    assert got[1]["category"] == "simple"  # ordered by category
    assert got[1]["correct"] == 374


def test_insert_scores_upserts_on_duplicate(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    db.insert_scores(run_id, [{"category": "simple", "correct": 100, "total": 400, "accuracy": 0.25}])
    db.insert_scores(run_id, [{"category": "simple", "correct": 374, "total": 400, "accuracy": 0.935}])

    got = db.get_scores(run_id)
    assert len(got) == 1
    assert got[0]["correct"] == 374


# ---------------------------------------------------------------------------
# Eval errors
# ---------------------------------------------------------------------------


def test_insert_and_get_eval_errors(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    errors = [
        {"id": "mt_0", "error": "state mismatch"},
        {"id": "mt_1", "error": ["multi", "errors"]},
    ]
    db.insert_eval_errors(run_id, "multi_turn_base", errors)

    got = db.get_eval_errors(run_id, "multi_turn_base")
    assert len(got) == 2
    assert got[0]["error"] == "state mismatch"
    assert json.loads(got[1]["error"]) == ["multi", "errors"]

    # Filter by category
    assert db.get_eval_errors(run_id, "nonexistent") == []


def test_insert_eval_errors_empty_is_noop(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")
    db.insert_eval_errors(run_id, "simple", [])
    assert db.get_eval_errors(run_id) == []


# ---------------------------------------------------------------------------
# Inference metrics
# ---------------------------------------------------------------------------


def test_insert_and_get_inference_metrics(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    summary = {
        "num_requests": 50,
        "prompt_tokens_per_sec": {"mean": 13000, "median": 12000, "p95": 16000, "p99": 19000},
        "generation_tokens_per_sec": {"mean": 270, "median": 273, "p95": 294, "p99": 303},
        "ttft_ms": {"mean": 21.4, "median": 17.7, "p95": 43.8, "p99": 46.5},
        "memory": {
            "baseline_rss_mb": 11671.4,
            "peak_rss_mb": 11958.8,
            "baseline_gpu_vram_mb": 2048.0,
            "peak_gpu_vram_mb": 2100.0,
        },
        "power_watts": 250.0,
    }
    db.insert_inference_metrics(run_id, summary)

    got = db.get_inference_metrics(run_id)
    assert got["num_requests"] == 50
    assert got["prompt_tokens_per_sec"]["mean"] == 13000
    assert got["generation_tokens_per_sec"]["median"] == 273
    assert got["ttft_ms"]["p95"] == 43.8
    assert got["memory"]["peak_rss_mb"] == 11958.8
    assert got["memory"]["baseline_gpu_vram_mb"] == 2048.0
    assert got["power_watts"] == 250.0


def test_get_inference_metrics_empty(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")
    assert db.get_inference_metrics(run_id) == {}


# ---------------------------------------------------------------------------
# Metrics per request
# ---------------------------------------------------------------------------


def test_insert_and_get_metrics_per_request(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")

    rows = [
        {
            "id": "bench_0", "category": "simple",
            "prompt_n": 100, "prompt_ms": 10, "prompt_per_second": 10000,
            "predicted_n": 20, "predicted_ms": 50, "predicted_per_second": 400,
        },
    ]
    db.insert_metrics_per_request(run_id, rows)

    got = db.get_metrics_per_request(run_id)
    assert len(got) == 1
    assert got[0]["entry_id"] == "bench_0"
    assert got[0]["prompt_per_second"] == 10000


# ---------------------------------------------------------------------------
# Unfinished runs / checkpoint helpers
# ---------------------------------------------------------------------------


def test_get_unfinished_runs_newest_first(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    first = db.create_run("qwen2.5-3b", notes="fc_mode=cd")
    second = db.create_run("qwen2.5-3b", notes="fc_mode=iffc")
    db.create_run("llama-3.2-3b")
    db.finish_run(first)

    qwen = db.get_unfinished_runs("qwen2.5-3b")
    assert [r["id"] for r in qwen] == [second]
    assert all(r["finished_at"] is None for r in db.get_unfinished_runs())


def test_get_response_entry_ids_filters_by_category_and_mode(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")
    db.insert_responses(
        run_id, "multi_turn_base", "multi", [{"id": "mt-1", "result": [[[]]]}]
    )
    db.insert_responses(
        run_id, "multi_turn_base", "multi", [{"id": "mt-2", "result": [[[]]]}]
    )
    db.insert_responses(
        run_id, "simple", "single", [{"id": "s-1", "result": "ok"}]
    )

    assert db.get_response_entry_ids(run_id, "multi_turn_base", mode="multi") == {
        "mt-1",
        "mt-2",
    }
    assert db.get_response_entry_ids(run_id, "simple", mode="single") == {"s-1"}
    assert db.get_response_entry_ids(run_id, "multi_turn_miss_func") == set()


def test_insert_multi_turn_entry_upserts_response_and_timing(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")
    result_row = {"id": "mt-0", "result": [[["cd(folder='docs')"]]], "flags": ""}
    timing = {
        "id": "mt-0",
        "category": "multi_turn_base",
        "turns": 2,
        "api_calls": 3,
        "total_prompt_n": 10,
        "total_prompt_ms": 1.5,
        "total_predicted_n": 4,
        "total_predicted_ms": 2.5,
        "first_prompt_ms": 0.5,
        "flags": "",
    }
    db.insert_multi_turn_entry(run_id, "multi_turn_base", result_row, timing)

    result_row["result"] = [[["ls()"]]]
    result_row["flags"] = "retried"
    timing["api_calls"] = 4
    timing["flags"] = "retried"
    db.insert_multi_turn_entry(run_id, "multi_turn_base", result_row, timing)

    responses = db.get_responses(run_id, category="multi_turn_base", mode="multi")
    assert len(responses) == 1
    assert json.loads(responses[0]["result"]) == [[["ls()"]]]
    assert responses[0]["flags"] == "retried"

    timings = db.get_multi_turn_timings(run_id)
    assert len(timings) == 1
    assert timings[0]["api_calls"] == 4
    assert timings[0]["flags"] == "retried"


def test_get_all_model_results(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})

    # Create two models, each with a finished run
    r1 = db.create_run("qwen2.5-3b")
    db.insert_scores(r1, [{"category": "simple", "correct": 374, "total": 400, "accuracy": 0.935}])
    db.insert_inference_metrics(r1, {
        "num_requests": 50,
        "prompt_tokens_per_sec": {"mean": 13000, "median": 12000, "p95": 16000, "p99": 19000},
        "generation_tokens_per_sec": {"mean": 270, "median": 273, "p95": 294, "p99": 303},
        "ttft_ms": {"mean": 21, "median": 17, "p95": 43, "p99": 46},
        "memory": {"baseline_rss_mb": 100, "peak_rss_mb": 150,
                    "baseline_gpu_vram_mb": 0, "peak_gpu_vram_mb": 0},
        "power_watts": None,
    })
    db.finish_run(r1)

    r2 = db.create_run("llama-3.2-3b")
    db.insert_scores(r2, [{"category": "simple", "correct": 300, "total": 400, "accuracy": 0.75}])
    db.finish_run(r2)

    # Unfinished run should not appear
    db.create_run("qwen3.5-4b")

    results = db.get_all_model_results()
    assert len(results) == 2
    keys = {r["key"] for r in results}
    assert keys == {"qwen2.5-3b", "llama-3.2-3b"}

    qwen = next(r for r in results if r["key"] == "qwen2.5-3b")
    assert len(qwen["scores"]) == 1
    assert qwen["metrics"]["num_requests"] == 50
