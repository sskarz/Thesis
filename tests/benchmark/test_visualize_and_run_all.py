import json
import sys

import pandas as pd

from benchmark import db
from benchmark import run_all
from benchmark import visualize


def test_load_timings_filters_sources_by_mode():
    pd.DataFrame(
        [
            {
                "id": "s1",
                "category": "simple",
                "prompt_n": 10,
                "prompt_ms": 5,
                "prompt_per_second": 2000,
                "predicted_n": 4,
                "predicted_ms": 10,
                "predicted_per_second": 400,
            }
        ]
    ).to_csv(visualize.config.RESPONSES_DIR / "all_timings.csv", index=False)
    pd.DataFrame(
        [
            {
                "id": "m1",
                "category": "multi_turn_base",
                "turns": 1,
                "api_calls": 1,
                "total_prompt_n": 20,
                "total_prompt_ms": 10,
                "total_predicted_n": 5,
                "total_predicted_ms": 25,
            }
        ]
    ).to_csv(visualize.config.RESPONSES_DIR / "multi_turn_timings.csv", index=False)

    single = visualize._load_timings(include_single=True, include_multi=False)
    multi = visualize._load_timings(include_single=False, include_multi=True)
    combined = visualize._load_timings(include_single=True, include_multi=True)

    assert list(single["id"]) == ["s1"]
    assert list(multi["id"]) == ["m1"]
    assert multi.iloc[0]["prompt_ms"] == 0
    assert multi.iloc[0]["prompt_per_second"] == 2000.0
    assert set(combined["id"]) == {"s1", "m1"}


def test_visualize_run_creates_charts_for_multi_turn_only():
    pd.DataFrame(
        [{"category": "multi_turn_base", "correct": 1, "total": 1, "accuracy": 1.0}]
    ).to_csv(visualize.config.SCORES_DIR / "accuracy.csv", index=False)
    pd.DataFrame(
        [
            {
                "id": "m1",
                "category": "multi_turn_base",
                "turns": 1,
                "api_calls": 1,
                "total_prompt_n": 20,
                "total_prompt_ms": 10,
                "total_predicted_n": 5,
                "total_predicted_ms": 25,
            }
        ]
    ).to_csv(visualize.config.RESPONSES_DIR / "multi_turn_timings.csv", index=False)
    (visualize.config.METRICS_DIR / "inference_metrics.json").write_text(
        json.dumps(
            {
                "memory": {"baseline_rss_mb": 100, "peak_rss_mb": 150},
                "prompt_tokens_per_sec": {"mean": 1000},
                "generation_tokens_per_sec": {"mean": 200},
                "ttft_ms": {"median": 0},
                "power_watts": None,
            }
        )
    )

    visualize.run(include_single=False, include_multi=True)

    assert (visualize.config.FIGURES_DIR / "accuracy_by_category.png").exists()
    assert (visualize.config.FIGURES_DIR / "throughput_distribution.png").exists()
    assert (visualize.config.FIGURES_DIR / "ttft_distribution.png").exists()
    assert (visualize.config.FIGURES_DIR / "summary_dashboard.png").exists()


def test_run_all_only_multi_turn_wires_correct_stages(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})

    calls = []

    def record(name):
        def inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            if name == "evaluate_multi_turn.run":
                return [{"category": "multi_turn_base", "correct": 1, "total": 1, "accuracy": 1.0}]
            if name == "evaluate_responses.run":
                return [{"category": "simple", "correct": 1, "total": 1, "accuracy": 1.0}]
            return None

        return inner

    monkeypatch.setattr(run_all, "check_server", lambda: True)
    monkeypatch.setattr(run_all.generate_responses, "run", record("generate_responses.run"))
    monkeypatch.setattr(run_all.generate_multi_turn, "run", record("generate_multi_turn.run"))
    monkeypatch.setattr(run_all.evaluate_responses, "run", record("evaluate_responses.run"))
    monkeypatch.setattr(run_all.evaluate_multi_turn, "run", record("evaluate_multi_turn.run"))
    monkeypatch.setattr(run_all.collect_metrics, "run", record("collect_metrics.run"))
    monkeypatch.setattr(run_all.visualize, "run", record("visualize.run"))
    monkeypatch.setattr(sys, "argv", ["benchmark.run_all", "--only-multi-turn"])

    run_all.main()

    called_names = [name for name, _, _ in calls]
    assert "generate_responses.run" not in called_names
    assert "evaluate_responses.run" not in called_names
    assert called_names[:2] == ["generate_multi_turn.run", "evaluate_multi_turn.run"]

    mt_gen = next(c for c in calls if c[0] == "generate_multi_turn.run")
    assert mt_gen[2]["fc_mode"] == "nfc"
    assert list(mt_gen[2]["categories"]) == list(run_all.config.MULTI_TURN_CATEGORIES)

    metrics = next(c for c in calls if c[0] == "collect_metrics.run")
    assert metrics[2]["include_single"] is False
    assert metrics[2]["include_multi"] is True
    assert ("visualize.run", (), {"include_single": False, "include_multi": True}) in calls
    # Verify run was created and finished in DB
    run_id = run_all.config.RUN_ID
    assert run_id is not None
    run = db.get_run(run_id)
    assert run["finished_at"] is not None


def test_run_all_iffc_base_only_wires_flags(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})

    calls = []

    def record(name):
        def inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            return None

        return inner

    monkeypatch.setattr(run_all, "check_server", lambda: True)
    monkeypatch.setattr(run_all.generate_responses, "run", record("generate_responses.run"))
    monkeypatch.setattr(run_all.generate_multi_turn, "run", record("generate_multi_turn.run"))
    monkeypatch.setattr(run_all.evaluate_responses, "run", record("evaluate_responses.run"))
    monkeypatch.setattr(run_all.evaluate_multi_turn, "run", record("evaluate_multi_turn.run"))
    monkeypatch.setattr(run_all.collect_metrics, "run", record("collect_metrics.run"))
    monkeypatch.setattr(run_all.visualize, "run", record("visualize.run"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.run_all",
            "--only-multi-turn",
            "--categories",
            "multi_turn_base",
            "--fc-mode",
            "iffc",
        ],
    )

    run_all.main()

    mt_gen = next(c for c in calls if c[0] == "generate_multi_turn.run")
    assert mt_gen[2]["fc_mode"] == "iffc"
    assert list(mt_gen[2]["categories"]) == ["multi_turn_base"]

    mt_eval = next(c for c in calls if c[0] == "evaluate_multi_turn.run")
    assert list(mt_eval[2]["categories"]) == ["multi_turn_base"]

    metrics = next(c for c in calls if c[0] == "collect_metrics.run")
    assert list(metrics[2]["multi_categories"]) == ["multi_turn_base"]
    assert "generate_responses.run" not in [c[0] for c in calls]

    run = db.get_run(run_all.config.RUN_ID)
    assert "fc_mode=iffc" in (run["notes"] or "")
    assert "multi_turn_base" in (run["notes"] or "")

