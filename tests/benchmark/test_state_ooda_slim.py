"""R81 iffc-state-ooda-slim: compact tool list keeps every schema and the tool order."""

import json

import pytest

from benchmark import config, generate_multi_turn as gm, ooda_card, run_metrics
from benchmark.compact_tools import (
    class_preambles,
    expand_compact,
    format_available_functions_compact,
)
from benchmark.generate_responses import load_jsonl
from benchmark.iffc import IFFC_SYSTEM_PROMPT, _unwrap_function_schemas

_BASE = config.BFCL_DATA_DIR / "BFCL_v4_multi_turn_base.json"
SLIM = config.FC_IFFC_STATE_OODA_SLIM


def _entries():
    if not _BASE.exists():
        pytest.skip("BFCL multi_turn_base data not present")
    return load_jsonl(_BASE)


def test_slim_mode_registered_like_state_ooda():
    assert SLIM == "iffc-state-ooda-slim"
    for group in (
        config.FC_MODES,
        config.FC_JSON_TEXT_MODES,
        config.FC_IFFC_STATE_MODES,
        config.FC_IFFC_STOP_REPEAT_MODES,
        config.FC_IFFC_OODA_MODES,
    ):
        assert (SLIM in group) == (config.FC_IFFC_STATE_OODA in group)
    assert SLIM not in config.FC_IFFC_DECISION_MODES


def test_schema_preserved_and_order_unchanged_all_200_tasks():
    entries = _entries()
    assert len(entries) == 200
    for e in entries:
        tools = gm._load_func_docs(e["involved_classes"])
        funcs = _unwrap_function_schemas(tools)
        af = format_available_functions_compact(tools)
        assert "\n  " not in af.split("Functions (JSON):\n", 1)[-1]  # no indentation
        back = expand_compact(af)
        assert [f["name"] for f in back] == [f["name"] for f in funcs]
        assert back == funcs  # every schema byte-for-byte after re-expansion
        # every class with tools in this task is hoisted exactly once
        assert af.startswith("Tool classes:\n")
        head = af.split("\nFunctions (JSON):\n", 1)[0].splitlines()[1:]
        assert len(head) == len(set(head)) == len(e["involved_classes"])


def test_multi_turn_base_classes_all_strictly_hoistable():
    name_to_class, hoist = class_preambles()
    for c in (
        "GorillaFileSystem", "MathAPI", "MessageAPI", "TicketAPI",
        "TradingBot", "TravelAPI", "TwitterAPI", "VehicleControlAPI",
    ):
        assert hoist[c] is not None


def test_slim_system_prompt_only_changes_function_block():
    e = _entries()[0]
    tools = gm._load_func_docs(e["involved_classes"])
    off = ooda_card.build_state_ooda_system_prompt(tools)
    on = ooda_card.build_state_ooda_system_prompt(tools, compact=True)
    pre, post = IFFC_SYSTEM_PROMPT.split("{available_functions}")
    assert off.startswith(pre) and on.startswith(pre)
    assert off.split(post, 1)[1] == on.split(post, 1)[1]
    assert len(on) < len(off)


def test_prepare_step_messages_slim_vs_off():
    e = _entries()[0]
    tools = gm._load_func_docs(e["involved_classes"])
    h_off = [{"role": "user", "content": "hi"}]
    h_on = [{"role": "user", "content": "hi"}]
    gm._prepare_step_messages(h_off, tools, config.FC_IFFC_STATE_OODA)
    gm._prepare_step_messages(h_on, tools, SLIM)
    assert h_off[0]["content"] == ooda_card.build_state_ooda_system_prompt(tools)
    assert h_on[0]["content"] == ooda_card.build_state_ooda_system_prompt(
        tools, compact=True
    )
    assert h_off[1:] == h_on[1:]
    p = gm._build_chat_payload(h_on, tools, SLIM)
    assert "tools" not in p and p["temperature"] == 0.0


def test_run_metrics_counts(tmp_path, monkeypatch):
    out = tmp_path / "m.jsonl"
    monkeypatch.setenv("RUN_METRICS_JSONL", str(out))
    tools = [{"type": "function", "function": {"name": "cd", "parameters": {
        "type": "dict", "properties": {"folder": {"type": "string"}},
        "required": ["folder"]}}}]
    m = run_metrics.TaskMetrics("x", SLIM)
    m.start_request()
    m.end_request({"timings": {"prompt_n": 10, "prompt_ms": 5.0}}, 0, 0)
    m.on_step("retry", None)
    m.on_step("end_turn", "iffc_unparseable")
    m.check_calls([
        {"function_name": "cd", "parameters": {"folder": "a"}},
        {"function_name": "ls", "parameters": {}},
        {"function_name": "cd", "parameters": {"dir": "a"}},
        {"function_name": "cd", "parameters": {}},
    ], tools)
    m.write()
    r = json.loads(out.read_text())
    assert (r["calls"], r["invalid_name"], r["invalid_args"]) == (4, 1, 2)
    assert (r["format_retries"], r["format_unparseable"], r["format_errors"]) == (1, 1, 2)
    assert r["first_prompt_n"] == 10 and r["n_requests"] == 1


def test_entry_id_filter(monkeypatch):
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    monkeypatch.delenv("BFCL_ENTRY_IDS", raising=False)
    assert gm._filter_entry_ids(rows) == rows
    monkeypatch.setenv("BFCL_ENTRY_IDS", "c, a")
    assert [r["id"] for r in gm._filter_entry_ids(rows)] == ["a", "c"]
