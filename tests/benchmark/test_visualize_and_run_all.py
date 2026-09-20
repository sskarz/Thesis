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


def test_run_all_sft_iffc_model_is_cli_selectable_with_distinct_slug(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(run_all, "check_server", lambda: True)
    monkeypatch.setattr(run_all.generate_responses, "run", noop)
    monkeypatch.setattr(run_all.generate_multi_turn, "run", noop)
    monkeypatch.setattr(run_all.evaluate_responses, "run", noop)
    monkeypatch.setattr(run_all.evaluate_multi_turn, "run", noop)
    monkeypatch.setattr(run_all.collect_metrics, "run", noop)
    monkeypatch.setattr(run_all.visualize, "run", noop)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.run_all",
            "--model",
            "gemma4-e2b-sft-iffc",
            "--only-multi-turn",
            "--categories",
            "multi_turn_base",
            "--fc-mode",
            "iffc",
        ],
    )

    run_all.main()

    run = db.get_run(run_all.config.RUN_ID)
    assert run["model_key"] == "gemma4-e2b-sft-iffc"
    assert run["model_slug"] == "Gemma-4-E2B-it-SFT-IFFC-Q4_K_M"
    assert run["model_display"] == "Gemma-4-E2B-it SFT-IFFC (Q4_K_M)"
    assert run["model_slug"] != run_all.config.MODELS["gemma4-e2b"]["slug"]
    assert "fc_mode=iffc" in (run["notes"] or "")


def test_run_all_cd_base_only_wires_flags(monkeypatch):
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
            "cd",
        ],
    )

    run_all.main()

    mt_gen = next(c for c in calls if c[0] == "generate_multi_turn.run")
    assert mt_gen[2]["fc_mode"] == "cd"
    assert list(mt_gen[2]["categories"]) == ["multi_turn_base"]
    assert "generate_responses.run" not in [c[0] for c in calls]

    run = db.get_run(run_all.config.RUN_ID)
    assert "fc_mode=cd" in (run["notes"] or "")
    assert "multi_turn_base" in (run["notes"] or "")


def test_run_all_gst_base_only_wires_flags(monkeypatch):
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
            "gst",
            "--notes",
            "GST+NFC vs NFC 27% / IFFC 30.5% / CD 10.5%",
        ],
    )

    run_all.main()

    mt_gen = next(c for c in calls if c[0] == "generate_multi_turn.run")
    assert mt_gen[2]["fc_mode"] == "gst"
    assert list(mt_gen[2]["categories"]) == ["multi_turn_base"]
    assert "generate_responses.run" not in [c[0] for c in calls]

    run = db.get_run(run_all.config.RUN_ID)
    assert "fc_mode=gst" in (run["notes"] or "")
    assert "multi_turn_base" in (run["notes"] or "")


def test_run_all_iffc_state_base_only_wires_flags(monkeypatch):
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
            "iffc-state",
            "--notes",
            "IFFC+STATE vs NFC 27% / IFFC 30.5% / GST 30.5% / CD 10.5%",
        ],
    )

    run_all.main()

    mt_gen = next(c for c in calls if c[0] == "generate_multi_turn.run")
    assert mt_gen[2]["fc_mode"] == "iffc-state"
    assert list(mt_gen[2]["categories"]) == ["multi_turn_base"]
    assert "generate_responses.run" not in [c[0] for c in calls]

    run = db.get_run(run_all.config.RUN_ID)
    assert "fc_mode=iffc-state" in (run["notes"] or "")
    assert "multi_turn_base" in (run["notes"] or "")
    assert "bon_n=" not in (run["notes"] or "")


def test_run_all_iffc_state_bon_wires_flags_and_bon_n(monkeypatch):
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
            "iffc-state-bon",
            "--bon-n",
            "5",
            "--notes",
            "BoN N=5 on IFFC+STATE vs 34% / IFFC 30.5% / GST 30.5% / NFC 27% / CD 10.5%",
        ],
    )

    run_all.main()

    mt_gen = next(c for c in calls if c[0] == "generate_multi_turn.run")
    assert mt_gen[2]["fc_mode"] == "iffc-state-bon"
    assert list(mt_gen[2]["categories"]) == ["multi_turn_base"]
    assert "generate_responses.run" not in [c[0] for c in calls]
    assert run_all.config.BON_N == 5
    assert run_all.config.FC_MODE == "iffc-state-bon"

    run = db.get_run(run_all.config.RUN_ID)
    notes = run["notes"] or ""
    assert "fc_mode=iffc-state-bon" in notes
    assert "bon_n=5" in notes
    assert "multi_turn_base" in notes


def test_fc_mode_from_notes():
    assert run_all.fc_mode_from_notes(None) == "nfc"
    assert run_all.fc_mode_from_notes("RTX 4090 baseline") == "nfc"
    assert run_all.fc_mode_from_notes("fc_mode=cd categories=multi_turn_base") == "cd"
    assert run_all.fc_mode_from_notes("TagDispatch; fc_mode=iffc") == "iffc"
    assert run_all.fc_mode_from_notes("GST+NFC; fc_mode=gst") == "gst"
    assert run_all.fc_mode_from_notes("IFFC+STATE; fc_mode=iffc-state") == "iffc-state"
    assert (
        run_all.fc_mode_from_notes(
            "BoN; fc_mode=iffc-state-bon bon_n=5 categories=multi_turn_base"
        )
        == "iffc-state-bon"
    )
    assert run_all.bon_n_from_notes(None) is None
    assert run_all.bon_n_from_notes("fc_mode=iffc-state") is None
    assert run_all.bon_n_from_notes("fc_mode=iffc-state-bon bon_n=5") == 5
    assert run_all.bon_n_from_notes("bon_n=3 fc_mode=iffc-state-bon") == 3


def test_resolve_pipeline_run_resume_matches_fc_mode(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    nfc = db.create_run("qwen2.5-3b", notes="baseline")
    cd = db.create_run("qwen2.5-3b", notes="fc_mode=cd categories=multi_turn_base")
    db.create_run("llama-3.2-3b", notes="fc_mode=cd")

    run_id, resumed = run_all.resolve_pipeline_run(
        model_key="qwen2.5-3b",
        fc_mode="cd",
        notes="ignored",
        resume=True,
    )
    assert resumed is True
    assert run_id == cd

    nfc_id, nfc_resumed = run_all.resolve_pipeline_run(
        model_key="qwen2.5-3b",
        fc_mode="nfc",
        notes=None,
        resume=True,
    )
    assert nfc_resumed is True
    assert nfc_id == nfc


def test_resolve_pipeline_run_resume_run_id_and_errors(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b", notes="fc_mode=cd")

    got, resumed = run_all.resolve_pipeline_run(
        model_key="qwen2.5-3b",
        fc_mode="cd",
        notes=None,
        resume_run_id=run_id,
    )
    assert resumed is True
    assert got == run_id

    try:
        run_all.resolve_pipeline_run(
            model_key="llama-3.2-3b",
            fc_mode="cd",
            notes=None,
            resume_run_id=run_id,
        )
        raise AssertionError("expected model mismatch")
    except ValueError as exc:
        assert "model_key" in str(exc)

    try:
        run_all.resolve_pipeline_run(
            model_key="qwen2.5-3b",
            fc_mode="iffc",
            notes=None,
            resume_run_id=run_id,
        )
        raise AssertionError("expected fc_mode mismatch")
    except ValueError as exc:
        assert "fc_mode=cd" in str(exc)

    db.finish_run(run_id)
    try:
        run_all.resolve_pipeline_run(
            model_key="qwen2.5-3b",
            fc_mode="cd",
            notes=None,
            resume=True,
        )
        raise AssertionError("expected no unfinished run")
    except ValueError as exc:
        assert "No unfinished run" in str(exc)


def test_run_all_resume_reuses_unfinished_run(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    existing = db.create_run("qwen2.5-3b", notes="fc_mode=cd categories=multi_turn_base")

    monkeypatch.setattr(run_all, "check_server", lambda: True)
    monkeypatch.setattr(run_all.generate_responses, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.generate_multi_turn, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.evaluate_responses, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.evaluate_multi_turn, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.collect_metrics, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.visualize, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.run_all",
            "--only-multi-turn",
            "--categories",
            "multi_turn_base",
            "--fc-mode",
            "cd",
            "--resume",
        ],
    )

    run_all.main()

    assert run_all.config.RUN_ID == existing
    run = db.get_run(existing)
    assert run["finished_at"] is not None
    assert len(db.list_runs("qwen2.5-3b")) == 1


def test_run_all_resume_restores_bon_n_from_notes(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    existing = db.create_run(
        "qwen2.5-3b",
        notes="fc_mode=iffc-state-bon bon_n=3 categories=multi_turn_base",
    )

    monkeypatch.setattr(run_all, "check_server", lambda: True)
    monkeypatch.setattr(run_all.generate_responses, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.generate_multi_turn, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.evaluate_responses, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.evaluate_multi_turn, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.collect_metrics, "run", lambda *a, **k: None)
    monkeypatch.setattr(run_all.visualize, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.run_all",
            "--only-multi-turn",
            "--categories",
            "multi_turn_base",
            "--fc-mode",
            "iffc-state-bon",
            "--resume",
        ],
    )

    run_all.main()

    assert run_all.config.RUN_ID == existing
    assert run_all.config.BON_N == 3
    assert run_all.config.FC_MODE == "iffc-state-bon"

