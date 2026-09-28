"""Hermetic tests for --fc-mode iffc-stop-repeat."""

from __future__ import annotations

import copy
import json

from benchmark import config, generate_multi_turn
from benchmark.stop_on_repeat import (
    REASON_CONSECUTIVE,
    REASON_OBSERVE_SAME,
    force_stop_reason,
    is_exec_observation,
)


def _timings(pn=10, pms=50, yn=5, yms=25):
    return {
        "prompt_n": pn,
        "prompt_ms": pms,
        "predicted_n": yn,
        "predicted_ms": yms,
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_force_stop_observe_same_and_consecutive():
    ls = '[{"function_name": "ls", "parameters": {}}]'
    ls_a = '[{"function_name": "ls", "parameters": {"a": true}}]'
    obs = (
        "Function execution results:\n- ls() -> ok\n"
        'Respond with a JSON list of function calls (each with "function_name" '
        'and "parameters"), or [] if no further function calling is needed.'
    )
    history = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": ls},
        {"role": "user", "content": obs},
    ]
    assert is_exec_observation(obs)
    assert force_stop_reason(history, ls) == REASON_CONSECUTIVE

    history2 = [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": ls},
        {"role": "user", "content": obs},
        {"role": "assistant", "content": ls_a},
        {
            "role": "user",
            "content": (
                "Function execution results:\n- ls(a=True) -> ok\n"
                'Respond with a JSON list of function calls (each with "function_name" '
                'and "parameters"), or [] if no further function calling is needed.'
            ),
        },
    ]
    # A-B-A: re-propose ls after ls_a → observe_same (not consecutive)
    assert force_stop_reason(history2, ls) == REASON_OBSERVE_SAME

    # New BFCL user question must NOT force-stop
    history3 = history + [{"role": "user", "content": "now sort them"}]
    assert force_stop_reason(history3, ls) is None


def test_baseline_iffc_still_reexecutes_identical_calls(monkeypatch):
    """Baseline iffc must not force-stop — regression guard."""
    payloads: list[dict] = []
    replies = [
        '[{"function_name": "ls", "parameters": {}}]',
        '[{"function_name": "ls", "parameters": {}}]',  # re-emit
        "[]",
    ]
    responses = iter(
        {
            "choices": [{"message": {"content": c}}],
            "timings": _timings(),
        }
        for c in replies
    )

    def fake_post(url, json, timeout):  # noqa: A002
        payloads.append(copy.deepcopy(json))
        return _FakeResponse(next(responses))

    executed = []

    def fake_execute(**kwargs):
        executed.append(list(kwargs["func_call_list"]))
        return ["ok"], None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn, "execute_multi_turn_func_call", fake_execute
    )

    entry = {
        "id": "iffc-baseline-repeat",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "list"}]],
    }
    tools = [{"type": "function", "function": {"name": "ls"}}]
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry, "multi_turn_base", tools, fc_mode="iffc"
    )
    assert turn_results == [[["ls()"], ["ls()"]]]
    assert executed == [["ls()"], ["ls()"]]
    assert timing["flags"] == ""


def test_iffc_stop_repeat_forces_stop_on_identical_reemit(monkeypatch):
    payloads: list[dict] = []
    replies = [
        '[{"function_name": "ls", "parameters": {}}]',
        '[{"function_name": "ls", "parameters": {}}]',  # would loop — force stop
        # should never be consumed
        '[{"function_name": "sort", "parameters": {}}]',
    ]
    responses = iter(
        {
            "choices": [{"message": {"content": c}}],
            "timings": _timings(),
        }
        for c in replies
    )

    def fake_post(url, json, timeout):  # noqa: A002
        payloads.append(copy.deepcopy(json))
        return _FakeResponse(next(responses))

    executed = []

    def fake_execute(**kwargs):
        executed.append(list(kwargs["func_call_list"]))
        return ["ok"], None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn, "execute_multi_turn_func_call", fake_execute
    )

    entry = {
        "id": "iffc-stop-1",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "list"}]],
    }
    tools = [{"type": "function", "function": {"name": "ls"}}]
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry, "multi_turn_base", tools, fc_mode=config.FC_IFFC_STOP_REPEAT
    )
    assert turn_results == [[["ls()"]]]
    assert executed == [["ls()"]]
    assert timing["api_calls"] == 2
    assert "stop_on_repeat_consecutive_identical:turn_0" in timing["flags"]
    # Forced [] stop in history before end
    last_msgs = payloads[1]["messages"]
    assert last_msgs[-1]["role"] == "user"  # observation fed into 2nd call
    # After force-stop, no third API call
    assert len(payloads) == 2


def test_iffc_stop_repeat_catches_abab_observe_same(monkeypatch):
    replies = [
        '[{"function_name": "ls", "parameters": {}}]',
        '[{"function_name": "ls", "parameters": {"a": true}}]',
        '[{"function_name": "ls", "parameters": {}}]',  # A after A-B → stop
        "[]",
    ]
    responses = iter(
        {
            "choices": [{"message": {"content": c}}],
            "timings": _timings(),
        }
        for c in replies
    )

    def fake_post(url, json, timeout):  # noqa: A002
        return _FakeResponse(next(responses))

    executed = []

    def fake_execute(**kwargs):
        executed.append(list(kwargs["func_call_list"]))
        return ["ok"], None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn, "execute_multi_turn_func_call", fake_execute
    )

    entry = {
        "id": "iffc-stop-abab",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "list"}]],
    }
    tools = [{"type": "function", "function": {"name": "ls"}}]
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry, "multi_turn_base", tools, fc_mode="iffc-stop-repeat"
    )
    assert turn_results == [[["ls()"], ["ls(a=True)"]]]
    assert executed == [["ls()"], ["ls(a=True)"]]
    assert "stop_on_repeat_observe_same_call:turn_0" in timing["flags"]


def test_fc_modes_include_stop_repeat_and_json_text():
    assert config.FC_IFFC_STOP_REPEAT in config.FC_MODES
    assert config.FC_IFFC_STOP_REPEAT in config.FC_JSON_TEXT_MODES
    assert config.FC_IFFC != config.FC_IFFC_STOP_REPEAT
