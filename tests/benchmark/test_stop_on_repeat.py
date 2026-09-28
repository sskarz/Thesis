"""Hermetic tests for --fc-mode iffc-stop-repeat."""

from __future__ import annotations

import copy
import json

from benchmark import config, generate_multi_turn
from benchmark.stop_on_repeat import (
    OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS,
    REASON_CONSECUTIVE,
    REASON_OBSERVE_SAME,
    assistant_tool_turns_so_far,
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


def _iffc(name: str, **params) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _obs(*calls: str) -> str:
    body = "".join(f"- {c} -> ok\n" for c in calls)
    return (
        "Function execution results:\n"
        f"{body}"
        'Respond with a JSON list of function calls (each with "function_name" '
        'and "parameters"), or [] if no further function calling is needed.'
    )


LS = _iffc("ls")
LS_A = _iffc("ls", a=True)
START_ENGINE = _iffc("startEngine", ignitionMode="START")
LOCK_DOORS = _iffc("lockDoors", unlock=False, door=[1, 2, 3, 4])
PRESS_BRAKE = _iffc("pressBrakePedal", pedalPosition=1)


def _exec_loop(steps: list[tuple[str, str]], *, ask: str = "do it") -> list[dict]:
    """Build history: user ask, then (assistant call, exec obs) for each step."""
    history = [{"role": "user", "content": ask}]
    for content, obs_call in steps:
        history.append({"role": "assistant", "content": content})
        history.append({"role": "user", "content": _obs(obs_call)})
    return history


def _patch_loop(monkeypatch, replies):
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
        return ["ok"], None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn, "execute_multi_turn_func_call", fake_execute
    )
    return payloads, executed


def test_observe_same_min_is_six():
    """Document the gate: 6 keeps 3–5 step VC; first fire at the 7th proposal."""
    assert OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS == 6


def test_force_stop_consecutive_is_immediate():
    history = _exec_loop([(LS, "ls()")])
    assert is_exec_observation(history[-1]["content"])
    assert force_stop_reason(history, LS) == REASON_CONSECUTIVE
    assert assistant_tool_turns_so_far(history, len(history)) == 1


def test_force_stop_short_abab_does_not_observe_same():
    """A-B-A after 2 tool-turns is below MIN — not a long decode loop."""
    history = _exec_loop([(LS, "ls()"), (LS_A, "ls(a=True)")])
    assert assistant_tool_turns_so_far(history, len(history)) == 2
    assert force_stop_reason(history, LS) is None


def test_force_stop_vc_startengine_recall_after_intervening_does_not_stop():
    """Legitimate VehicleControl: startEngine after lockDoors / pressBrakePedal."""
    history = _exec_loop(
        [
            (START_ENGINE, "startEngine(ignitionMode='START')"),
            (LOCK_DOORS, "lockDoors(unlock=False, door=[1, 2, 3, 4])"),
            (PRESS_BRAKE, "pressBrakePedal(pedalPosition=1)"),
        ],
        ask="start the car",
    )
    assert assistant_tool_turns_so_far(history, len(history)) == 3
    assert force_stop_reason(history, START_ENGINE) is None


def test_force_stop_observe_same_boundary_min_minus_one_then_min():
    """Subset-of-executed is silent at MIN-1; fires at MIN prior tool-turns.

    Last assistant must differ from the proposed re-call or consecutive wins.
    """
    intervening = [(_iffc(f"step{i}"), f"step{i}()") for i in range(4)]
    short = _exec_loop([(LS, "ls()")] + intervening)
    assert assistant_tool_turns_so_far(short, len(short)) == 5
    assert 5 < OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS
    assert force_stop_reason(short, LS) is None

    long_h = _exec_loop(
        [(LS, "ls()")] + [(_iffc(f"step{i}"), f"step{i}()") for i in range(5)]
    )
    assert assistant_tool_turns_so_far(long_h, len(long_h)) == 6
    assert force_stop_reason(long_h, LS) == REASON_OBSERVE_SAME


def test_force_stop_long_abab_observe_same():
    """Hit-step style A-B-A-B…: after MIN tool-turns, re-calling A stops."""
    ab = [(LS, "ls()"), (LS_A, "ls(a=True)")]
    history = _exec_loop(ab * (OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS // 2))
    assert assistant_tool_turns_so_far(history, len(history)) == 6
    assert force_stop_reason(history, LS) == REASON_OBSERVE_SAME
    # Consecutive must still win when the last assistant set equals proposed.
    assert force_stop_reason(history, LS_A) == REASON_CONSECUTIVE


def test_force_stop_new_user_question_resets_window_and_does_not_stop():
    """Later BFCL ask: short VC-like startEngine re-call must not inherit MIN."""
    # startEngine in ask 1 so subset WOULD match if the window were entry-wide.
    first_with_engine = _exec_loop(
        [(START_ENGINE, "startEngine(ignitionMode='START')")]
        + [(_iffc(f"step{i}"), f"step{i}()") for i in range(5)],
        ask="first question",
    )
    first_with_engine.append({"role": "assistant", "content": "[]"})
    first_with_engine.append(
        {"role": "user", "content": "now lock doors then start again"}
    )
    first_with_engine.append({"role": "assistant", "content": LOCK_DOORS})
    first_with_engine.append(
        {
            "role": "user",
            "content": _obs("lockDoors(unlock=False, door=[1, 2, 3, 4])"),
        }
    )
    assert assistant_tool_turns_so_far(
        first_with_engine, len(first_with_engine)
    ) == 1
    # Subset holds (startEngine already executed in ask 1) but window is 1.
    assert force_stop_reason(first_with_engine, START_ENGINE) is None
    # A fresh BFCL user question (not an exec obs) never force-stops.
    first_with_engine.append({"role": "user", "content": "now sort them"})
    assert force_stop_reason(first_with_engine, START_ENGINE) is None


def test_baseline_iffc_still_reexecutes_identical_calls(monkeypatch):
    """Baseline iffc must not force-stop — regression guard."""
    replies = [
        LS,
        LS,  # re-emit
        "[]",
    ]
    _payloads, executed = _patch_loop(monkeypatch, replies)

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
    payloads, executed = _patch_loop(
        monkeypatch,
        [
            LS,
            LS,  # would loop — force stop
            _iffc("sort"),  # should never be consumed
        ],
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
    last_msgs = payloads[1]["messages"]
    assert last_msgs[-1]["role"] == "user"  # observation fed into 2nd call
    assert len(payloads) == 2


def test_iffc_stop_repeat_vc_startengine_recall_is_executed(monkeypatch):
    """Short VC path must execute the startEngine re-call, not observe_same."""
    _payloads, executed = _patch_loop(
        monkeypatch,
        [
            START_ENGINE,
            LOCK_DOORS,
            PRESS_BRAKE,
            START_ENGINE,  # legitimate re-call
            "[]",
        ],
    )

    entry = {
        "id": "iffc-stop-vc",
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
        entry, "multi_turn_base", tools, fc_mode=config.FC_IFFC_STOP_REPEAT
    )
    assert executed == [
        ["startEngine(ignitionMode='START')"],
        ["lockDoors(unlock=False, door=[1, 2, 3, 4])"],
        ["pressBrakePedal(pedalPosition=1)"],
        ["startEngine(ignitionMode='START')"],
    ]
    assert turn_results == [executed]
    assert "stop_on_repeat" not in timing["flags"]
    assert timing["api_calls"] == 5


def test_iffc_stop_repeat_short_abab_is_not_force_stopped(monkeypatch):
    """Existing ungated A-B-A (2 prior) must now execute the third call."""
    _payloads, executed = _patch_loop(
        monkeypatch,
        [LS, LS_A, LS, "[]"],
    )

    entry = {
        "id": "iffc-stop-abab-short",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "list"}]],
    }
    tools = [{"type": "function", "function": {"name": "ls"}}]
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry, "multi_turn_base", tools, fc_mode="iffc-stop-repeat"
    )
    assert turn_results == [[["ls()"], ["ls(a=True)"], ["ls()"]]]
    assert executed == [["ls()"], ["ls(a=True)"], ["ls()"]]
    assert "stop_on_repeat" not in timing["flags"]


def test_iffc_stop_repeat_catches_long_abab_observe_same(monkeypatch):
    """Long A-B-A-B (MIN prior tool-turns) still force-stops on observe_same."""
    replies = [LS, LS_A] * 3 + [LS, "[]"]
    _payloads, executed = _patch_loop(monkeypatch, replies)

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
    assert executed == [["ls()"], ["ls(a=True)"]] * 3
    assert turn_results == [executed]
    assert "stop_on_repeat_observe_same_call:turn_0" in timing["flags"]
    assert "stop_on_repeat_consecutive_identical" not in timing["flags"]


def test_fc_modes_include_stop_repeat_and_json_text():
    assert config.FC_IFFC_STOP_REPEAT in config.FC_MODES
    assert config.FC_IFFC_STOP_REPEAT in config.FC_JSON_TEXT_MODES
    assert config.FC_IFFC != config.FC_IFFC_STOP_REPEAT
