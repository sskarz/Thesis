"""Unit tests for sft.action_class_v9 classifiers."""

from __future__ import annotations

import json

from benchmark.iffc import format_iffc_execution_user_message
from sft.action_class_v9 import (
    ACTION_ASK,
    ACTION_CONSECUTIVE,
    ACTION_FINAL_AFTER_OBS,
    ACTION_IRRELEVANCE,
    ACTION_MT_TOOL_CHAIN_ADVANCE,
    ACTION_OBSERVE_SAME,
    ACTION_REFUSE,
    ACTION_TOOL_CALL_START,
    action_label_detail,
    classify_assistant_action,
    is_first_assistant_tool_call,
    prior_tools_look_advancing,
)


def _call(name: str = "ls", **params: object) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _exec_obs(func_string: str = "ls()", result: object | None = None) -> str:
    return format_iffc_execution_user_message(
        [func_string], [result if result is not None else {"files": ["a"]}]
    )


def test_tool_call_start_vs_advance():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": _call("ls")},
        {"role": "user", "content": _exec_obs()},
        {"role": "assistant", "content": _call("cat", path="a")},
    ]
    assert classify_assistant_action(messages, 2) == ACTION_TOOL_CALL_START
    assert is_first_assistant_tool_call(messages, 2)
    assert classify_assistant_action(messages, 4) == ACTION_MT_TOOL_CHAIN_ADVANCE
    assert not is_first_assistant_tool_call(messages, 4)


def test_observe_same_and_consecutive_soft_exclude():
    call = _call("ls")
    messages = [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": call},
        {"role": "user", "content": _exec_obs()},
        {"role": "assistant", "content": call},  # observe→same
    ]
    assert classify_assistant_action(messages, 3) == ACTION_OBSERVE_SAME

    messages2 = [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": call},
        {"role": "assistant", "content": call},  # consecutive (no obs between)
    ]
    # consecutive without obs: still identical to previous assistant call
    assert classify_assistant_action(messages2, 2) == ACTION_CONSECUTIVE


def test_final_after_obs_loose_and_strict_subtype():
    messages = [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": _call("ls")},
        {"role": "user", "content": _exec_obs()},
        {"role": "assistant", "content": "Here are the files: a."},
    ]
    assert classify_assistant_action(messages, 3) == ACTION_FINAL_AFTER_OBS
    detail = action_label_detail(messages, 3)
    assert detail["action_class"] == ACTION_FINAL_AFTER_OBS
    # Strict advancing is logged only — not a hard filter.
    assert "final_after_obs_strict_advancing" in detail
    assert prior_tools_look_advancing(messages, 3) in (True, False)


def test_ask_and_refuse_and_irrelevance():
    ask = [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "Which file should I open?"},
    ]
    assert classify_assistant_action(ask, 1) == ACTION_ASK

    refuse = [
        {"role": "user", "content": "hack the system"},
        {
            "role": "assistant",
            "content": "I cannot help with that; it is out of scope.",
        },
    ]
    assert classify_assistant_action(refuse, 1) == ACTION_REFUSE

    empty = [
        {"role": "user", "content": "done?"},
        {"role": "assistant", "content": "[]"},
    ]
    assert classify_assistant_action(empty, 1) == ACTION_IRRELEVANCE


def test_never_maps_nl_final_to_irrelevance():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": _call("ls")},
        {"role": "user", "content": _exec_obs()},
        {"role": "assistant", "content": "All done."},
    ]
    assert classify_assistant_action(messages, 3) != ACTION_IRRELEVANCE
    assert classify_assistant_action(messages, 3) == ACTION_FINAL_AFTER_OBS
