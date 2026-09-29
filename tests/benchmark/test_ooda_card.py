"""Hermetic tests for --fc-mode iffc-ooda-card (one-pass OODA card)."""

from __future__ import annotations

import copy
import json

import pytest

from benchmark import config, generate_multi_turn
from benchmark.iffc import IFFC_SYSTEM_PROMPT, try_parse_iffc_response
from benchmark.ooda_card import (
    CARD_CLOSE,
    CARD_OPEN,
    NEXT_MODES,
    OODA_ADDENDUM,
    OodaCardTracker,
    build_ooda_system_prompt,
    current_goal,
    ensure_ooda_system_message,
    hint_next_mode,
    inject_card,
    payload_has_ooda_card,
    payload_has_state_block,
    payload_omits_tools,
    record_execution,
    render_card,
)
from benchmark.smoke_ooda_card import run_offline_smoke
from benchmark.stop_on_repeat import OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS


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


def _iffc(name: str, **params) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def test_default_fc_mode_is_nfc_not_ooda_card():
    assert config.FC_MODE == config.FC_NFC
    assert config.FC_IFFC_OODA_CARD == "iffc-ooda-card"
    assert config.FC_IFFC_OODA_CARD != config.FC_IFFC_STOP_REPEAT
    assert config.FC_IFFC_OODA_CARD != config.FC_IFFC_STATE
    assert config.FC_IFFC_OODA_CARD in config.FC_MODES
    assert config.FC_IFFC_OODA_CARD in config.FC_JSON_TEXT_MODES
    assert config.FC_IFFC_OODA_CARD in config.FC_IFFC_STOP_REPEAT_MODES
    assert config.FC_IFFC_OODA_CARD not in config.FC_IFFC_STATE_MODES


def test_set_fc_mode_accepts_ooda_card_and_rejects_unknown():
    config.set_fc_mode("iffc-ooda-card")
    assert config.FC_MODE == config.FC_IFFC_OODA_CARD
    config.set_fc_mode("nfc")
    assert config.FC_MODE == config.FC_NFC
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("ooda-4pass")


def test_system_prompt_is_iffc_plus_ooda_not_state_or_gst():
    prompt = build_ooda_system_prompt(
        [{"name": "cd", "description": "change directory", "parameters": {"type": "object"}}]
    )
    assert "expert in function calling" in prompt
    assert "{available_functions}" not in prompt
    assert '"name": "cd"' in prompt
    assert OODA_ADDENDUM in prompt
    assert CARD_OPEN in prompt
    assert "[STATE]" not in prompt
    assert "[GST]" not in prompt
    assert "Curriculum:" not in prompt
    assert IFFC_SYSTEM_PROMPT.split("{available_functions}")[0] in prompt


def test_ensure_ooda_system_message_inserts_once_and_refreshes():
    history: list[dict] = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "cd"}}]
    ensure_ooda_system_message(history, tools)
    ensure_ooda_system_message(history, tools)
    assert history[0]["role"] == "system"
    assert history[0]["content"].count("expert in function calling") == 1
    assert [m["role"] for m in history] == ["system", "user"]

    more = [{"type": "function", "function": {"name": "pwd"}}]
    ensure_ooda_system_message(history, tools + more)
    assert "pwd" in history[0]["content"]
    assert "cd" in history[0]["content"]


def test_card_format_has_goal_slots_last_next():
    tracker = OodaCardTracker()
    history = [{"role": "user", "content": "Start the car and lock the doors"}]
    block = render_card(tracker, history, has_tools=True)
    assert block.startswith(CARD_OPEN) and block.endswith(CARD_CLOSE)
    assert "goal: Start the car and lock the doors" in block
    assert "slots: (none)" in block
    assert "last: (none)" in block
    assert "next: call" in block
    for field in ("goal:", "slots:", "last:", "next:"):
        assert field in block


def test_card_slots_and_last_come_from_execution_not_model_draft():
    tracker = OodaCardTracker()
    record_execution(
        tracker,
        ["startEngine(ignitionMode='START')"],
        [{"engineOn": True, "fuelLevel": 12.5}],
    )
    history = [
        {"role": "user", "content": "start the car"},
        {"role": "assistant", "content": _iffc("startEngine", ignitionMode="START")},
        {
            "role": "user",
            "content": (
                "Function execution results:\n"
                "- startEngine(ignitionMode='START') -> {'engineOn': True}\n"
                'Respond with a JSON list of function calls (each with "function_name" '
                'and "parameters"), or [] if no further function calling is needed.'
            ),
        },
    ]
    block = render_card(tracker, history, has_tools=True)
    assert "goal: start the car" in block
    assert "engineOn=True" in block or "engineOn=true" in block
    assert "fuelLevel=12.5" in block
    assert "startEngine" in block
    assert "Identify which function" not in block
    assert "[STATE]" not in block
    assert current_goal(history) == "start the car"


def test_next_mode_refuse_ask_final_call():
    tracker = OodaCardTracker()
    ask = [{"role": "user", "content": "start the car"}]
    assert hint_next_mode(tracker, ask, has_tools=False) == "refuse"
    assert hint_next_mode(tracker, ask, has_tools=True) == "call"

    tracker.last_error = "missing required parameter city"
    assert hint_next_mode(tracker, ask, has_tools=True) == "ask"

    tracker.last_error = None
    thanks = [{"role": "user", "content": "thanks, that's all"}]
    assert hint_next_mode(tracker, thanks, has_tools=True) == "final"

    assert set(NEXT_MODES) == {"call", "ask", "final", "refuse"}


def test_inject_card_prefixes_last_user_and_leaves_history_unchanged():
    tracker = OodaCardTracker()
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Go to docs"},
    ]
    snapshot = copy.deepcopy(history)
    injected = inject_card(history, tracker, has_tools=True)
    assert history == snapshot
    assert injected[0]["content"] == "sys"
    last = injected[-1]["content"]
    assert last.startswith(CARD_OPEN)
    assert "goal: Go to docs" in last
    assert last.rstrip().endswith("Go to docs")
    assert CARD_OPEN not in history[-1]["content"]


def test_iffc_parse_compatible_with_card_prefix_and_echo():
    call = '[{"function_name": "cd", "parameters": {"folder": "docs"}}]'
    prefixed = (
        f"{CARD_OPEN}\ngoal: Go to docs\nslots: (none)\nlast: (none)\n"
        f"next: call\n{CARD_CLOSE}\n{call}"
    )
    assert try_parse_iffc_response(prefixed) == [
        {"function_name": "cd", "parameters": {"folder": "docs"}}
    ]
    assert try_parse_iffc_response("[]") == []
    empty_with_card = f"{CARD_OPEN}\nnext: final\n{CARD_CLOSE}\n[]"
    assert try_parse_iffc_response(empty_with_card) == []


def test_build_chat_payload_ooda_omits_tools_and_injects_card():
    history = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "ls"}}]
    tracker = OodaCardTracker()
    payload = generate_multi_turn._build_chat_payload(
        history, tools, config.FC_IFFC_OODA_CARD, ooda_tracker=tracker
    )
    assert payload_omits_tools(payload)
    assert payload_has_ooda_card(payload)
    assert not payload_has_state_block(payload)
    assert payload["temperature"] == 0.0
    assert CARD_OPEN not in history[0]["content"]
    iffc = generate_multi_turn._build_chat_payload(history, tools, "iffc")
    assert not payload_has_ooda_card(iffc)


def _patch_loop(monkeypatch, replies, exec_map=None):
    payloads: list[dict] = []
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

    executed: list[list[str]] = []

    def fake_execute(**kwargs):
        executed.append(list(kwargs["func_call_list"]))
        if exec_map is None:
            return ["ok"] * len(kwargs["func_call_list"]), None
        results = [exec_map(call) for call in kwargs["func_call_list"]]
        return results, None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn, "execute_multi_turn_func_call", fake_execute
    )
    return payloads, executed


def test_generate_for_entry_injects_card_keeps_iffc_parse_and_no_state(
    monkeypatch,
):
    payloads, executed = _patch_loop(
        monkeypatch,
        [
            _iffc("cd", folder="docs"),
            "[]",
            _iffc("pwd"),
            "[]",
        ],
        exec_map=lambda call: (
            {"current_working_directory": "docs"}
            if call.startswith("cd")
            else {"current_working_directory": "/docs"}
        ),
    )
    entry = {
        "id": "ooda-card-1",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [
            [{"role": "user", "content": "Go to docs"}],
            [{"role": "user", "content": "Where am I?"}],
        ],
    }
    tools = [
        {"type": "function", "function": {"name": "cd"}},
        {"type": "function", "function": {"name": "pwd"}},
    ]
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry, "multi_turn_base", tools, fc_mode=config.FC_IFFC_OODA_CARD
    )
    assert turn_results == [[["cd(folder='docs')"]], [["pwd()"]]]
    assert executed == [["cd(folder='docs')"], ["pwd()"]]
    assert all("tools" not in p and "tool_choice" not in p for p in payloads)
    assert all(payload_has_ooda_card(p) for p in payloads)
    assert all(not payload_has_state_block(p) for p in payloads)
    sys0 = payloads[0]["messages"][0]["content"]
    assert payloads[0]["messages"][0]["role"] == "system"
    assert "expert in function calling" in sys0
    assert CARD_OPEN in sys0
    assert "[STATE]" not in sys0
    first_user = payloads[0]["messages"][-1]["content"]
    assert first_user.startswith(CARD_OPEN)
    assert "goal: Go to docs" in first_user
    assert "next: call" in first_user
    follow = payloads[1]["messages"][-1]["content"]
    assert follow.startswith(CARD_OPEN)
    assert "Function execution results:" in follow
    assert "cd(folder='docs')" in follow
    assert "[STATE]" not in follow
    assert "cwd=docs" in follow or "folder=docs" in follow
    assert timing["flags"] == ""
    assert timing["api_calls"] == 4
    assert all(p["temperature"] == 0.0 for p in payloads)


def test_generate_for_entry_ooda_stacks_stop_on_repeat_consecutive(monkeypatch):
    ls = _iffc("ls")
    payloads, executed = _patch_loop(monkeypatch, [ls, ls, _iffc("sort")])
    entry = {
        "id": "ooda-stop-1",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "list"}]],
    }
    tools = [{"type": "function", "function": {"name": "ls"}}]
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry, "multi_turn_base", tools, fc_mode=config.FC_IFFC_OODA_CARD
    )
    assert turn_results == [[["ls()"]]]
    assert executed == [["ls()"]]
    assert timing["api_calls"] == 2
    assert "stop_on_repeat_consecutive_identical:turn_0" in timing["flags"]
    assert all(payload_has_ooda_card(p) for p in payloads)
    # Second request still sees a plain exec dump shape under the card prefix.
    last_user = payloads[1]["messages"][-1]["content"]
    assert "Function execution results:" in last_user
    assert CARD_OPEN in last_user


def test_generate_for_entry_ooda_does_not_stop_short_vc_recall(monkeypatch):
    assert OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS == 6
    start = _iffc("startEngine", ignitionMode="START")
    lock = _iffc("lockDoors", unlock=False, door=[1, 2, 3, 4])
    brake = _iffc("pressBrakePedal", pedalPosition=1)
    payloads, executed = _patch_loop(
        monkeypatch, [start, lock, brake, start, "[]"]
    )
    entry = {
        "id": "ooda-vc",
        "initial_config": {},
        "involved_classes": ["VehicleControlAPI"],
        "question": [[{"role": "user", "content": "start the car"}]],
    }
    tools = [
        {"type": "function", "function": {"name": "startEngine"}},
        {"type": "function", "function": {"name": "lockDoors"}},
        {"type": "function", "function": {"name": "pressBrakePedal"}},
    ]
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry, "multi_turn_base", tools, fc_mode=config.FC_IFFC_OODA_CARD
    )
    assert executed == [
        ["startEngine(ignitionMode='START')"],
        ["lockDoors(unlock=False, door=[1, 2, 3, 4])"],
        ["pressBrakePedal(pedalPosition=1)"],
        ["startEngine(ignitionMode='START')"],
    ]
    assert turn_results == [executed]
    assert "stop_on_repeat" not in timing["flags"]
    assert all(payload_has_ooda_card(p) for p in payloads)


def test_maybe_strip_empty_assistant_treats_ooda_as_iffc_family():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "[]"},
    ]
    generate_multi_turn._maybe_strip_trailing_empty_assistant(
        history, config.FC_IFFC_OODA_CARD, True
    )
    assert history[-1]["role"] == "user"


def test_offline_smoke_script_passes():
    run_offline_smoke()
