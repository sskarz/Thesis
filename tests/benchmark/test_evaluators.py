from pathlib import Path

from benchmark import db
from benchmark import evaluate_multi_turn
from benchmark import evaluate_responses

from tests.helpers import write_jsonl


def test_evaluate_responses_category_counts_valid_and_invalid_entries(
    monkeypatch,
    bfcl_fixture_dirs,
):
    data_dir, answers_dir = bfcl_fixture_dirs
    write_jsonl(
        Path(data_dir) / "BFCL_v4_simple_python.json",
        [
            {"id": "ok", "function": [{"name": "weather"}]},
            {"id": "bad", "function": [{"name": "weather"}]},
        ],
    )
    write_jsonl(
        Path(answers_dir) / "BFCL_v4_simple_python.json",
        [
            {"id": "ok", "ground_truth": [{"weather": {"city": "Paris"}}]},
            {"id": "bad", "ground_truth": [{"weather": {"city": "Paris"}}]},
        ],
    )
    write_jsonl(
        evaluate_responses.config.RESPONSES_DIR / "simple_responses.jsonl",
        [
            {"id": "ok", "result": '[{"weather": {"city": "Paris"}}]'},
            {"id": "bad", "result": '[{"weather": {"city": "Berlin"}}]'},
        ],
    )

    def fake_ast_checker(*args, **kwargs):
        model_output = args[1]
        return {"valid": model_output[0]["weather"]["city"] == "Paris"}

    monkeypatch.setattr(evaluate_responses, "ast_checker", fake_ast_checker)

    result = evaluate_responses.evaluate_category(
        "simple",
        {"language": "Python", "file": "BFCL_v4_simple_python.json"},
    )

    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["accuracy"] == 0.5
    assert result["errors"] == [{"id": "bad", "error": []}]


def test_evaluate_responses_reparses_raw_gemma_markup(
    monkeypatch,
    bfcl_fixture_dirs,
):
    data_dir, answers_dir = bfcl_fixture_dirs
    write_jsonl(
        Path(data_dir) / "BFCL_v4_simple_python.json",
        [{"id": "ok", "function": [{"name": "weather"}]}],
    )
    write_jsonl(
        Path(answers_dir) / "BFCL_v4_simple_python.json",
        [{"id": "ok", "ground_truth": [{"weather": {"city": "Paris"}}]}],
    )
    write_jsonl(
        evaluate_responses.config.RESPONSES_DIR / "simple_responses.jsonl",
        [
            {
                "id": "ok",
                "result": (
                    '<|tool_call>call:weather{city:<|"|>Paris<|"|>}<tool_call|>'
                ),
            }
        ],
    )

    def fake_ast_checker(*args, **kwargs):
        model_output = args[1]
        return {"valid": model_output == [{"weather": {"city": "Paris"}}]}

    monkeypatch.setattr(evaluate_responses, "ast_checker", fake_ast_checker)

    result = evaluate_responses.evaluate_category(
        "simple",
        {"language": "Python", "file": "BFCL_v4_simple_python.json"},
    )

    assert result["correct"] == 1
    assert result["total"] == 1
    assert result["accuracy"] == 1.0
    assert result.get("errors", []) == []


def test_eval_errors_saved_to_db(monkeypatch):
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")
    db.insert_eval_errors(run_id, "multi_turn_base", [{"id": "mt-1", "error": "boom"}])

    errors = db.get_eval_errors(run_id, "multi_turn_base")
    assert len(errors) == 1
    assert errors[0]["entry_id"] == "mt-1"
    assert errors[0]["error"] == "boom"


def test_evaluate_multi_turn_category_records_checker_failures(
    monkeypatch,
    bfcl_fixture_dirs,
):
    data_dir, answers_dir = bfcl_fixture_dirs
    write_jsonl(
        Path(data_dir) / "BFCL_v4_multi_turn_base.json",
        [{"id": "mt-1", "question": [[{"role": "user", "content": "hello"}]]}],
    )
    write_jsonl(
        Path(answers_dir) / "BFCL_v4_multi_turn_base.json",
        [{"id": "mt-1", "ground_truth": [["weather(city='Paris')"]]}],
    )
    write_jsonl(
        evaluate_multi_turn.config.RESPONSES_DIR / "multi_turn_base_responses.jsonl",
        [{"id": "mt-1", "result": [[["weather(city='Berlin')"]]]}],
    )

    def fake_checker(**kwargs):
        assert kwargs["test_category"] == "multi_turn_base"
        return {"valid": False, "error_message": "wrong city"}

    monkeypatch.setattr(evaluate_multi_turn, "multi_turn_checker", fake_checker)

    result = evaluate_multi_turn.evaluate_category(
        "multi_turn_base",
        {"file": "BFCL_v4_multi_turn_base.json"},
    )

    assert result["correct"] == 0
    assert result["total"] == 1
    assert result["errors"] == [{"id": "mt-1", "error": "wrong city"}]
