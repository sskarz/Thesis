"""Hermetic tests for --fc-mode iffc-ptc-a (name DFA, arguments unconstrained)."""

import copy
import json

import pytest

from benchmark import config, generate_multi_turn
from benchmark.cd import (
    build_name_dispatch_grammar,
    grammar_allowed_names,
    tool_names,
)
from benchmark.ptc_a import (
    apply_name_dfa_constraints,
    assert_ptc_a_payload,
    build_name_dfa_grammar,
    grammar_constrains_argument_schemas,
    instance_matches_name_dfa,
    text_matches_name_dfa,
)
from benchmark.smoke_ptc_a import run_offline_smoke


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
        },
    },
}


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _timings(prompt_n=10, prompt_ms=50, predicted_n=5, predicted_ms=25):
    return {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
    }


def test_default_fc_mode_is_still_nfc():
    assert config.FC_MODE == config.FC_NFC
    assert config.FC_IFFC_PTC_A != config.FC_NFC
    assert config.FC_IFFC_PTC_A != config.FC_CD
    assert config.FC_IFFC_PTC_A != config.FC_IFFC_STOP_REPEAT
    assert config.FC_IFFC_PTC_A != "iffc-ooda-card"


def test_set_fc_mode_accepts_iffc_ptc_a_and_rejects_collisions():
    config.set_fc_mode("iffc-ptc-a")
    assert config.FC_MODE == config.FC_IFFC_PTC_A
    assert config.FC_IFFC_PTC_A in config.FC_MODES
    assert config.FC_IFFC_PTC_A in config.FC_JSON_TEXT_MODES
    config.set_fc_mode("nfc")
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("ptc-a")
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("iffc-ooda-card")


def test_grammar_accepts_live_names_and_empty_rejects_out_of_set():
    grammar = build_name_dfa_grammar([WEATHER, TIME])
    assert grammar_allowed_names(grammar) == ["get_weather", "get_time"]
    assert "not_a_tool" not in grammar_allowed_names(grammar)
    assert instance_matches_name_dfa([WEATHER, TIME], [])
    assert text_matches_name_dfa([WEATHER, TIME], "[]")
    assert instance_matches_name_dfa(
        [WEATHER, TIME],
        [{"function_name": "get_weather", "parameters": {"city": "SF"}}],
    )
    assert instance_matches_name_dfa(
        [WEATHER, TIME],
        [{"function_name": "get_time", "parameters": {"tz": "UTC"}}],
    )
    assert not instance_matches_name_dfa(
        [WEATHER, TIME],
        [{"function_name": "not_a_tool", "parameters": {"city": "SF"}}],
    )
    assert not text_matches_name_dfa(
        [WEATHER, TIME],
        '[{"function_name": "hack", "parameters": {}}]',
    )
    empty = build_name_dfa_grammar([])
    assert grammar_allowed_names(empty) == []
    assert instance_matches_name_dfa([], [])
    assert not instance_matches_name_dfa(
        [], [{"function_name": "get_weather", "parameters": {}}]
    )


def test_grammar_does_not_constrain_arguments_or_become_cd():
    grammar = build_name_dfa_grammar([WEATHER, TIME])
    cd_grammar = build_name_dispatch_grammar([WEATHER, TIME])
    assert grammar != cd_grammar
    assert "parameters" in grammar
    assert "json-object" in grammar
    assert not grammar_constrains_argument_schemas(grammar)
    # Tool schema keys must not appear as GBNF tokens (args are free).
    assert "city" not in grammar
    assert "tz" not in grammar
    # Empty / extra / wrong-typed args still match the name DFA.
    assert instance_matches_name_dfa(
        [WEATHER, TIME],
        [{"function_name": "get_weather", "parameters": {}}],
    )
    assert instance_matches_name_dfa(
        [WEATHER, TIME],
        [
            {
                "function_name": "get_weather",
                "parameters": {"city": "SF", "bonus": [1, 2], "ok": False},
            }
        ],
    )
    # Name-only CD objects (no parameters) are not PTC-A IFFC shape.
    assert not instance_matches_name_dfa(
        [WEATHER, TIME],
        [{"function_name": "get_weather"}],
    )


def test_payload_never_includes_json_schema_or_native_tools():
    payload = {
        "model": "m",
        "messages": [],
        "temperature": 0.0,
        "tools": [WEATHER],
        "tool_choice": "auto",
        "json_schema": {"type": "array"},
    }
    apply_name_dfa_constraints(payload, [WEATHER, TIME])
    assert_ptc_a_payload(payload, [WEATHER, TIME])
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "json_schema" not in payload
    assert payload["grammar"]
    assert grammar_allowed_names(payload["grammar"]) == tool_names([WEATHER, TIME])

    with pytest.raises(AssertionError, match="tools/tool_choice"):
        assert_ptc_a_payload({"tools": [WEATHER], "grammar": payload["grammar"]})
    with pytest.raises(AssertionError, match="json_schema"):
        assert_ptc_a_payload(
            {"grammar": payload["grammar"], "json_schema": {"type": "object"}}
        )


def test_build_chat_payload_ptc_a_attaches_name_grammar_not_cd_schema():
    history = [{"role": "user", "content": "hi"}]
    tools = [WEATHER, TIME]
    payload = generate_multi_turn._build_chat_payload(
        history, tools, config.FC_IFFC_PTC_A
    )
    assert_ptc_a_payload(payload, tools)
    assert payload["temperature"] == 0.0
    assert "tools" not in payload
    assert "json_schema" not in payload
    # CD payload builder still does not attach grammar (two-pass complete_step).
    cd = generate_multi_turn._build_chat_payload(history, tools, config.FC_CD)
    assert "grammar" not in cd
    assert "json_schema" not in cd
    assert "tools" not in cd
    # Default NFC still sends native tools.
    nfc = generate_multi_turn._build_chat_payload(history, tools, config.FC_NFC)
    assert nfc["tools"] == tools
    assert nfc["tool_choice"] == "auto"
    assert "grammar" not in nfc


def test_generate_for_entry_ptc_a_single_pass_live_names_no_json_schema(
    monkeypatch,
):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '[{"function_name": "cd",'
                                ' "parameters": {"folder": "docs"}}]'
                            )
                        }
                    }
                ],
                "timings": _timings(),
            },
            {
                "choices": [{"message": {"content": "[]"}}],
                "timings": _timings(15, 30, 2, 10),
            },
        ]
    )

    def fake_post(url, json, timeout):  # noqa: A002
        payloads.append(copy.deepcopy(json))
        return _FakeResponse(next(responses))

    executed = []

    def fake_execute(**kwargs):
        executed.append(kwargs)
        return ["tool-result"], None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn,
        "execute_multi_turn_func_call",
        fake_execute,
    )

    entry = {
        "id": "ptc-a-1",
        "initial_config": {"cwd": "/tmp"},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "Open docs"}]],
    }
    all_tools = [
        {
            "type": "function",
            "function": {
                "name": "cd",
                "description": "change directory",
                "parameters": {
                    "type": "object",
                    "properties": {"folder": {"type": "string"}},
                    "required": ["folder"],
                },
            },
        },
        {"type": "function", "function": {"name": "ls"}},
    ]

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        all_tools,
        fc_mode=config.FC_IFFC_PTC_A,
    )

    assert turn_results == [[["cd(folder='docs')"]]]
    assert len(payloads) == 2  # not CD's three (dispatch + args + [])
    for payload in payloads:
        assert_ptc_a_payload(payload, all_tools)
        assert "json_schema" not in payload
        assert "tools" not in payload
        assert grammar_allowed_names(payload["grammar"]) == ["cd", "ls"]
        assert "folder" not in payload["grammar"]
    followup = payloads[1]["messages"]
    assert followup[-1]["role"] == "user"
    assert "tool-result" in followup[-1]["content"]
    assert not any(m.get("role") == "tool" for m in followup)
    assert executed[0]["func_call_list"] == ["cd(folder='docs')"]
    assert timing["id"] == "ptc-a-1"
    assert timing["api_calls"] == 2
    assert timing["flags"] == ""


def test_generate_for_entry_ptc_a_grammar_follows_live_tool_restore(
    monkeypatch,
):
    payloads: list[dict] = []
    responses = iter(
        [
            {"choices": [{"message": {"content": "[]"}}], "timings": _timings()},
            {
                "choices": [{"message": {"content": "[]"}}],
                "timings": _timings(8, 20, 3, 12),
            },
        ]
    )

    def fake_post(url, json, timeout):  # noqa: A002
        payloads.append(copy.deepcopy(json))
        return _FakeResponse(next(responses))

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn,
        "execute_multi_turn_func_call",
        lambda **kwargs: ([], None),
    )

    entry = {
        "id": "ptc-a-restore",
        "initial_config": {},
        "involved_classes": ["fs"],
        "missed_function": {"1": ["restored_tool"]},
        "question": [
            [{"role": "user", "content": "Open docs"}],
            [],
        ],
    }
    all_tools = [
        {"type": "function", "function": {"name": "initial_tool"}},
        {"type": "function", "function": {"name": "restored_tool"}},
    ]

    generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_miss_func",
        all_tools,
        fc_mode=config.FC_IFFC_PTC_A,
    )

    assert grammar_allowed_names(payloads[0]["grammar"]) == ["initial_tool"]
    assert "restored_tool" not in grammar_allowed_names(payloads[0]["grammar"])
    assert grammar_allowed_names(payloads[1]["grammar"]) == [
        "initial_tool",
        "restored_tool",
    ]
    first_sys = payloads[0]["messages"][0]["content"]
    assert "initial_tool" in first_sys
    assert "restored_tool" not in first_sys


def test_offline_smoke_script_passes():
    run_offline_smoke()
