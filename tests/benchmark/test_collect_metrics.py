import json
from pathlib import Path

from benchmark import collect_metrics
from benchmark import config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_sample_requests_respects_single_and_multi_selection(
    monkeypatch,
    bfcl_fixture_dirs,
):
    data_dir, _ = bfcl_fixture_dirs
    monkeypatch.setattr(
        config,
        "CATEGORIES",
        {
            "simple": {"file": "BFCL_v3_simple.json"},
            "multiple": {"file": "BFCL_v3_multiple.json"},
        },
    )
    monkeypatch.setattr(
        config,
        "MULTI_TURN_CATEGORIES",
        {
            "multi_turn_base": {"file": "BFCL_v3_multi_turn_base.json"},
        },
    )
    _write_jsonl(Path(data_dir) / "BFCL_v3_simple.json", [{"id": "s1", "question": [[]]}])
    _write_jsonl(Path(data_dir) / "BFCL_v3_multiple.json", [{"id": "s2", "question": [[]]}])
    _write_jsonl(
        Path(data_dir) / "BFCL_v3_multi_turn_base.json",
        [{"id": "m1", "question": [[]]}],
    )

    single_only = collect_metrics.sample_requests(2, include_single=True, include_multi=False)
    multi_only = collect_metrics.sample_requests(1, include_single=False, include_multi=True)
    mixed = collect_metrics.sample_requests(3, include_single=True, include_multi=True)

    assert {entry["_mode"] for entry in single_only} == {"single"}
    assert {entry["_mode"] for entry in multi_only} == {"multi"}
    assert {entry["_mode"] for entry in mixed} == {"single", "multi"}


def test_measure_request_normalizes_multi_turn_timing_row(monkeypatch):
    monkeypatch.setattr(
        collect_metrics.generate_multi_turn,
        "_load_func_docs",
        lambda classes: [{"type": "function", "function": {"name": "weather"}}],
    )
    monkeypatch.setattr(
        collect_metrics.generate_multi_turn,
        "generate_for_entry",
        lambda *args, **kwargs: (
            [[["weather(city='Paris')"]]],
            {
                "total_prompt_n": 200,
                "total_prompt_ms": 100,
                "total_predicted_n": 20,
                "total_predicted_ms": 50,
            },
        ),
    )

    row = collect_metrics.measure_request(
        {
            "id": "mt-1",
            "_mode": "multi",
            "_category": "multi_turn_base",
            "involved_classes": ["weather"],
        }
    )

    assert row == {
        "id": "mt-1",
        "category": "multi_turn_base",
        "prompt_n": 200,
        "prompt_ms": 0,
        "prompt_per_second": 2000.0,
        "predicted_n": 20,
        "predicted_ms": 50,
        "predicted_per_second": 400.0,
    }


def test_run_writes_metrics_summary_and_csv(monkeypatch):
    monkeypatch.setattr(collect_metrics, "WARMUP_REQUESTS", 1)
    monkeypatch.setattr(collect_metrics, "BENCHMARK_REQUESTS", 2)
    monkeypatch.setattr(collect_metrics, "find_llama_server_pid", lambda: None)
    monkeypatch.setattr(
        collect_metrics,
        "sample_requests",
        lambda *args, **kwargs: [
            {"id": "warmup", "_category": "simple"},
            {"id": "bench-1", "_category": "simple"},
            {"id": "bench-2", "_category": "multi_turn_base"},
        ],
    )

    calls = []

    def fake_measure(entry):
        calls.append(entry["id"])
        return {
            "id": entry["id"],
            "category": entry["_category"],
            "prompt_n": 10,
            "prompt_ms": 5 if entry["id"] != "bench-2" else 0,
            "prompt_per_second": 2000 if entry["id"] != "bench-2" else 0,
            "predicted_n": 4,
            "predicted_ms": 10,
            "predicted_per_second": 400,
        }

    monkeypatch.setattr(collect_metrics, "measure_request", fake_measure)
    monkeypatch.setattr(collect_metrics, "try_power_draw", lambda: 12.5)

    out = collect_metrics.run(include_single=True, include_multi=True)

    summary = json.loads(out.read_text())
    csv_path = collect_metrics.config.METRICS_DIR / "inference_metrics.csv"

    assert calls == ["warmup", "bench-1", "bench-2"]
    assert out.name == "inference_metrics.json"
    assert csv_path.exists()
    assert summary["num_requests"] == 2
    assert summary["power_watts"] == 12.5
    assert summary["ttft_ms"]["mean"] == 5
