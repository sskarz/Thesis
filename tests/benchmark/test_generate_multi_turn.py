import copy
import json
import asyncio

from benchmark import db, generate_multi_turn
from benchmark.iffc import try_parse_iffc_response
from tests.helpers import write_jsonl


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_tool_calls_to_func_strings_preserves_parse_failures():
    tool_calls = [
        {"function": {"name": "weather", "arguments": json.dumps({"city": "Paris"})}},
        {"function": {"name": "lookup", "arguments": "{not valid json"}},
    ]

    result = generate_multi_turn._tool_calls_to_func_strings(tool_calls)

    assert result[0] == "weather(city='Paris')"
    assert "__arg_parse_error__" in result[1]
    assert "lookup(" in result[1]


def test_generate_for_entry_restores_missing_functions_and_collects_timings(
    monkeypatch,
):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "initial_tool",
                                        "arguments": json.dumps({"folder": "docs"}),
                                    },
                                }
                            ]
                        }
                    }
                ],
                "timings": {
                    "prompt_n": 10,
                    "prompt_ms": 50,
                    "predicted_n": 5,
                    "predicted_ms": 25,
                },
            },
            {
                "choices": [
                    {
                        "message": {"content": "turn 1 complete"},
                        "finish_reason": "stop",
                    }
                ],
                "timings": {
                    "prompt_n": 15,
                    "prompt_ms": 30,
                    "predicted_n": 2,
                    "predicted_ms": 10,
                },
            },
            {
                "choices": [
                    {
                        "message": {"content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "timings": {
                    "prompt_n": 20,
                    "prompt_ms": 40,
                    "predicted_n": 3,
                    "predicted_ms": 15,
                },
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
        "id": "mt-1",
        "initial_config": {"cwd": "/tmp"},
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

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_miss_func",
        all_tools,
        execution_model_name="metrics-model",
    )

    assert turn_results == [[["initial_tool(folder='docs')"]], []]
    assert payloads[0]["tools"] == [{"type": "function", "function": {"name": "initial_tool"}}]
    assert payloads[1]["tools"] == [{"type": "function", "function": {"name": "initial_tool"}}]
    assert {tool["function"]["name"] for tool in payloads[2]["tools"]} == {
        "initial_tool",
        "restored_tool",
    }
    assert executed[0]["model_name"] == "metrics-model"
    assert timing == {
        "id": "mt-1",
        "category": "multi_turn_miss_func",
        "turns": 2,
        "api_calls": 3,
        "total_prompt_n": 45,
        "total_prompt_ms": 120.0,
        "total_predicted_n": 10,
        "total_predicted_ms": 50.0,
        "first_prompt_ms": 50.0,
        "flags": "",
    }


def _timings(prompt_n=10, prompt_ms=50, predicted_n=5, predicted_ms=25):
    return {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
    }


def test_generate_for_entry_iffc_parses_json_and_feeds_user_results(monkeypatch):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '[{"function_name": "cd", "parameters": {"folder": "docs"}}]'
                            )
                        }
                    }
                ],
                "timings": _timings(),
            },
            {
                "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
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
        "id": "iffc-1",
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
                "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}},
            },
        }
    ]

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        all_tools,
        fc_mode="iffc",
    )

    assert turn_results == [[["cd(folder='docs')"]]]
    assert "tools" not in payloads[0]
    assert "tool_choice" not in payloads[0]
    assert payloads[0]["messages"][0]["role"] == "system"
    assert "expert in function calling" in payloads[0]["messages"][0]["content"]
    assert "cd" in payloads[0]["messages"][0]["content"]

    followup = payloads[1]["messages"]
    assert followup[-2]["role"] == "assistant"
    assert followup[-1]["role"] == "user"
    assert "tool-result" in followup[-1]["content"]
    assert "[STATE]" not in followup[-1]["content"]
    assert "[GST]" not in json.dumps(payloads)
    assert not any(m.get("role") == "tool" for m in followup)
    assert executed[0]["func_call_list"] == ["cd(folder='docs')"]
    assert timing["id"] == "iffc-1"
    assert timing["api_calls"] == 2
    assert timing["flags"] == ""


def test_generate_for_entry_xlam_parses_wrapped_json_and_omits_native_tools(
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
                                '{"tool_calls": [{"name": "cd", '
                                '"arguments": {"folder": "docs"}}]}'
                            )
                        }
                    }
                ],
                "timings": _timings(),
            },
            {
                "choices": [
                    {
                        "message": {"content": '{"tool_calls": []}'},
                        "finish_reason": "stop",
                    }
                ],
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
        "id": "xlam-1",
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
                "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}},
            },
        }
    ]

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        all_tools,
        fc_mode="xlam",
    )

    assert turn_results == [[["cd(folder='docs')"]]]
    assert "tools" not in payloads[0]
    assert "tool_choice" not in payloads[0]
    assert payloads[0]["messages"][0]["role"] == "system"
    system = payloads[0]["messages"][0]["content"]
    assert "expert in composing functions" in system
    assert '"tool_calls"' in system
    assert "cd" in system
    assert "expert in function calling" not in system

    followup = payloads[1]["messages"]
    assert followup[-2]["role"] == "assistant"
    assert followup[-1]["role"] == "user"
    assert "tool-result" in followup[-1]["content"]
    assert '"tool_calls"' in followup[-1]["content"]
    assert "function_name" not in followup[-1]["content"]
    assert not any(m.get("role") == "tool" for m in followup)
    assert executed[0]["func_call_list"] == ["cd(folder='docs')"]
    assert timing["id"] == "xlam-1"
    assert timing["api_calls"] == 2
    assert timing["flags"] == ""


def test_generate_for_entry_nfc_ignores_xlam_json_in_content(monkeypatch):
    """NFC only executes message.tool_calls — Run 19 collapse stays NFC-only."""
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"tool_calls": [{"name": "cd", '
                                '"arguments": {"folder": "docs"}}]}'
                            )
                        }
                    }
                ],
                "timings": _timings(),
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
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("NFC must not execute content JSON")),
    )

    entry = {
        "id": "nfc-xlam-content",
        "initial_config": {},
        "involved_classes": [],
        "question": [[{"role": "user", "content": "Open docs"}]],
    }
    all_tools = [
        {"type": "function", "function": {"name": "cd"}},
    ]

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        all_tools,
        fc_mode="nfc",
    )

    assert turn_results == [[]]
    assert payloads[0]["tools"] == all_tools
    assert payloads[0]["tool_choice"] == "auto"
    assert timing["api_calls"] == 1
    assert timing["flags"] == ""


def _cd_tool():
    return {
        "type": "function",
        "function": {
            "name": "cd",
            "description": "change directory",
            "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}},
        },
    }


def _two_turn_iffc_payloads(monkeypatch, *, strip_empty_assistant: bool):
    """BFCL two-user-turn IFFC: tool call → [] stop → next user question."""
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '[{"function_name": "cd", "parameters": {"folder": "docs"}}]'
                            )
                        }
                    }
                ],
                "timings": _timings(),
            },
            {
                "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
                "timings": _timings(15, 30, 2, 10),
            },
            {
                "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
                "timings": _timings(8, 12, 1, 4),
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
        lambda **kwargs: (["ok"], None),
    )

    entry = {
        "id": "iffc-strip-hist",
        "initial_config": {},
        "involved_classes": [],
        "question": [
            [{"role": "user", "content": "Open docs"}],
            [{"role": "user", "content": "Now list files"}],
        ],
    }
    generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        [_cd_tool()],
        fc_mode="iffc",
        strip_empty_assistant=strip_empty_assistant,
    )
    return payloads


def _assistant_empty_list_messages(messages: list[dict]) -> list[dict]:
    return [
        m
        for m in messages
        if m.get("role") == "assistant"
        and try_parse_iffc_response(m.get("content")) == []
    ]


def test_generate_for_entry_iffc_keeps_empty_assistant_in_history_by_default(
    monkeypatch,
):
    payloads = _two_turn_iffc_payloads(monkeypatch, strip_empty_assistant=False)
    # Third request is the next BFCL user turn; default IFFC still has [] in history.
    assert len(payloads) == 3
    next_turn = payloads[2]["messages"]
    assert any(m.get("content") == "Now list files" for m in next_turn)
    empties = _assistant_empty_list_messages(next_turn)
    assert empties, "flag off must retain assistant [] before the next user turn"
    idx_empty = next_turn.index(empties[-1])
    idx_next = next(
        i for i, m in enumerate(next_turn) if m.get("content") == "Now list files"
    )
    assert idx_empty < idx_next


def test_generate_for_entry_iffc_strip_empty_assistant_omits_stop_from_next_turn(
    monkeypatch,
):
    payloads = _two_turn_iffc_payloads(monkeypatch, strip_empty_assistant=True)
    assert len(payloads) == 3
    next_turn = payloads[2]["messages"]
    assert any(m.get("content") == "Now list files" for m in next_turn)
    assert _assistant_empty_list_messages(next_turn) == []
    # Tool call from turn 0 is still in history; only the [] stop is gone.
    assert any(
        m.get("role") == "assistant" and "cd" in str(m.get("content"))
        for m in next_turn
    )
    assert "tools" not in payloads[0]


def test_maybe_strip_empty_assistant_is_noop_for_nfc_and_cd():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "[]"},
    ]
    generate_multi_turn._maybe_strip_trailing_empty_assistant(history, "nfc", True)
    assert history[-1]["content"] == "[]"
    generate_multi_turn._maybe_strip_trailing_empty_assistant(history, "cd", True)
    assert history[-1]["content"] == "[]"
    generate_multi_turn._maybe_strip_trailing_empty_assistant(history, "gst", True)
    assert history[-1]["content"] == "[]"
    generate_multi_turn._maybe_strip_trailing_empty_assistant(history, "xlam", True)
    assert history[-1]["content"] == "[]"
    generate_multi_turn._maybe_strip_trailing_empty_assistant(history, "iffc", True)
    assert history[-1]["role"] == "user"
    for mode in ("iffc-state", "iffc-state-bon"):
        state_hist = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[]"},
        ]
        generate_multi_turn._maybe_strip_trailing_empty_assistant(
            state_hist, mode, True
        )
        assert state_hist[-1]["role"] == "user"


def test_generate_for_entry_iffc_retries_then_ends_on_unparseable(monkeypatch):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [{"message": {"content": "Sure, I can help."}}],
                "timings": _timings(),
            },
            {
                "choices": [{"message": {"content": "Still not JSON."}}],
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
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not execute")),
    )

    entry = {
        "id": "iffc-retry",
        "initial_config": {},
        "involved_classes": [],
        "question": [[{"role": "user", "content": "hi"}]],
    }

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        [{"type": "function", "function": {"name": "ls"}}],
        fc_mode="iffc",
    )

    assert turn_results == [[]]
    assert "json" in payloads[1]["messages"][-1]["content"].lower()
    assert "iffc_unparseable:turn_0" in timing["flags"]
    assert timing["api_calls"] == 2


def test_build_chat_payload_nfc_and_gst_keep_tools_iffc_cd_and_iffc_state_omit_them():
    history = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "ls"}}]
    nfc = generate_multi_turn._build_chat_payload(history, tools, "nfc")
    gst = generate_multi_turn._build_chat_payload(history, tools, "gst")
    iffc = generate_multi_turn._build_chat_payload(history, tools, "iffc")
    cd = generate_multi_turn._build_chat_payload(history, tools, "cd")
    iffc_state = generate_multi_turn._build_chat_payload(history, tools, "iffc-state")
    bon = generate_multi_turn._build_chat_payload(history, tools, "iffc-state-bon")
    xlam = generate_multi_turn._build_chat_payload(history, tools, "xlam")
    assert nfc["tools"] == tools
    assert nfc["tool_choice"] == "auto"
    assert nfc["temperature"] == 0.0
    assert gst["tools"] == tools
    assert gst["tool_choice"] == "auto"
    assert gst["temperature"] == 0.0
    assert "tools" not in iffc
    assert "tool_choice" not in iffc
    assert iffc["temperature"] == 0.0
    assert "tools" not in cd
    assert "tool_choice" not in cd
    assert cd["temperature"] == 0.0
    assert "tools" not in iffc_state
    assert "tool_choice" not in iffc_state
    assert iffc_state["temperature"] == 0.0
    assert "tools" not in bon
    assert "tool_choice" not in bon
    assert bon["temperature"] == 0.7
    assert "tools" not in xlam
    assert "tool_choice" not in xlam
    assert xlam["temperature"] == 0.0


def test_generate_for_entry_cd_two_pass_no_native_tools(monkeypatch):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {"message": {"content": '[{"function_name": "cd"}]'}}
                ],
                "timings": _timings(),
            },
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
                "timings": _timings(8, 20, 3, 12),
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
        "id": "cd-1",
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
        fc_mode="cd",
    )

    assert turn_results == [[["cd(folder='docs')"]]]
    assert len(payloads) == 3
    assert "tools" not in payloads[0] and "tool_choice" not in payloads[0]
    assert payloads[0]["grammar"]
    assert "json_schema" not in payloads[0]
    assert "tools" not in payloads[1] and "tool_choice" not in payloads[1]
    assert payloads[1]["json_schema"]["minItems"] == 1
    assert "tools" not in payloads[2]
    assert payloads[2]["grammar"]
    followup = payloads[2]["messages"]
    assert followup[-1]["role"] == "user"
    assert "tool-result" in followup[-1]["content"]
    assert not any(m.get("role") == "tool" for m in followup)
    assert executed[0]["func_call_list"] == ["cd(folder='docs')"]
    assert timing["id"] == "cd-1"
    assert timing["api_calls"] == 3
    assert timing["flags"] == ""


def test_generate_for_entry_iffc_refreshes_system_prompt_when_tools_restored(
    monkeypatch,
):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [{"message": {"content": "[]"}}],
                "timings": _timings(),
            },
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
        "id": "iffc-restore",
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
        fc_mode="iffc",
    )

    assert "tools" not in payloads[0]
    first_sys = payloads[0]["messages"][0]["content"]
    assert "initial_tool" in first_sys
    assert "restored_tool" not in first_sys
    restored_sys = payloads[1]["messages"][0]["content"]
    assert "initial_tool" in restored_sys
    assert "restored_tool" in restored_sys


def test_generate_for_entry_iffc_state_injects_state_keeps_last_two_turns(
    monkeypatch,
):
    payloads: list[dict] = []
    replies = [
        '[{"function_name": "cd", "parameters": {"folder": "docs"}}]',
        "[]",
        '[{"function_name": "echo", "parameters": {"content": "hi", "file_name": "a.txt"}}]',
        "[]",
        '[{"function_name": "pwd", "parameters": {}}]',
        "[]",
    ]
    responses = iter(
        {
            "choices": [{"message": {"content": content}}],
            "timings": _timings(),
        }
        for content in replies
    )

    def fake_post(url, json, timeout):  # noqa: A002
        payloads.append(copy.deepcopy(json))
        return _FakeResponse(next(responses))

    def fake_execute(**kwargs):
        results = []
        for call in kwargs["func_call_list"]:
            if call.startswith("cd"):
                results.append({"current_working_directory": "docs"})
            elif call.startswith("echo"):
                results.append(None)
            elif call.startswith("pwd"):
                results.append({"current_working_directory": "/docs"})
            else:
                results.append("ok")
        return results, None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn,
        "execute_multi_turn_func_call",
        fake_execute,
    )

    entry = {
        "id": "iffc-state-1",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [
            [{"role": "user", "content": "Go to docs"}],
            [{"role": "user", "content": "Write a.txt"}],
            [{"role": "user", "content": "Where am I?"}],
        ],
    }
    all_tools = [
        {"type": "function", "function": {"name": "cd"}},
        {"type": "function", "function": {"name": "echo"}},
        {"type": "function", "function": {"name": "pwd"}},
    ]

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        all_tools,
        fc_mode="iffc-state",
    )

    assert turn_results == [
        [["cd(folder='docs')"]],
        [["echo(content='hi', file_name='a.txt')"]],
        [["pwd()"]],
    ]
    assert all("tools" not in p and "tool_choice" not in p for p in payloads)
    sys0 = payloads[0]["messages"][0]["content"]
    assert payloads[0]["messages"][0]["role"] == "system"
    assert "expert in function calling" in sys0
    assert "[STATE]" in sys0
    assert "[GST]" not in sys0
    assert "Curriculum:" not in sys0

    follow_cd = payloads[1]["messages"]
    dump = follow_cd[-1]["content"]
    assert follow_cd[-1]["role"] == "user"
    assert "Function execution results:" in dump
    assert "[STATE]" in dump
    assert "docs" in dump
    assert not any(m.get("role") == "tool" for m in follow_cd)

    last = payloads[-1]["messages"]
    dumps = [
        m["content"]
        for m in last
        if m.get("role") == "user" and "Function execution results:" in (m.get("content") or "")
    ]
    assert len(dumps) == 2
    assert any("echo" in d for d in dumps)
    assert any("pwd" in d for d in dumps)
    assert not any("cd(folder='docs')" in d for d in dumps)
    assert any(m.get("content") == "Go to docs" for m in last)
    assert "[STATE]" in dumps[-1]
    assert "/docs" in dumps[-1] or "cwd:" in dumps[-1]
    assert timing["id"] == "iffc-state-1"
    assert timing["api_calls"] == 6
    assert timing["flags"] == ""
    assert all(p["temperature"] == 0.0 for p in payloads)


def test_generate_for_entry_gst_sends_nfc_tools_and_curriculum(monkeypatch):
    from benchmark.gst import GST_STEPS, has_gst_scaffold, strip_gst_block

    payloads: list[dict] = []
    gst_block = (
        "[GST]\n"
        "1. get_weather fits\n2. pick it\n3. docs\n4. city required\n"
        "5. SF\n6. draft wrong_tool()\n7. CALL\n"
        "[/GST]"
    )
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": gst_block,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": json.dumps({"city": "SF"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "timings": _timings(),
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "[GST]\n1. done\n2. none\n3. n/a\n4. n/a\n"
                                "5. 72 F\n6. none\n7. ABSTAIN\n[/GST]"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
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
        return ["72 F"], None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn,
        "execute_multi_turn_func_call",
        fake_execute,
    )

    entry = {
        "id": "gst-1",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "Weather in SF?"}]],
    }
    all_tools = [
        {
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
    ]

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        all_tools,
        fc_mode="gst",
    )

    # Checker uses tool_calls, not the GST draft (wrong_tool in step 6).
    assert turn_results == [[["get_weather(city='SF')"]]]
    assert executed[0]["func_call_list"] == ["get_weather(city='SF')"]
    assert "tools" in payloads[0]
    assert payloads[0]["tool_choice"] == "auto"
    assert "grammar" not in payloads[0]
    assert "json_schema" not in payloads[0]
    sys_msg = payloads[0]["messages"][0]
    assert sys_msg["role"] == "system"
    for step in GST_STEPS:
        assert step in sys_msg["content"]
    assert "[GST]" in sys_msg["content"]

    followup = payloads[1]["messages"]
    assert any(m.get("role") == "tool" for m in followup)
    assert not any("[STATE]" in (m.get("content") or "") for m in followup)
    assistant = next(m for m in followup if m.get("tool_calls"))
    assert has_gst_scaffold(assistant["content"])
    assert strip_gst_block(assistant["content"]) == ""
    assert timing["id"] == "gst-1"
    assert timing["api_calls"] == 2
    assert timing["flags"] == ""
    assert all(p["temperature"] == 0.0 for p in payloads)


def test_generate_for_entry_gst_abstain_without_tool_calls(monkeypatch):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "[GST]\n1. none fit\n2. none\n3. n/a\n4. n/a\n"
                                "5. query has no city\n6. none\n7. ABSTAIN\n[/GST]"
                            )
                        }
                    }
                ],
                "timings": _timings(),
            }
        ]
    )

    def fake_post(url, json, timeout):  # noqa: A002
        payloads.append(copy.deepcopy(json))
        return _FakeResponse(next(responses))

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn,
        "execute_multi_turn_func_call",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not execute")),
    )

    entry = {
        "id": "gst-abstain",
        "initial_config": {},
        "involved_classes": [],
        "question": [[{"role": "user", "content": "hello"}]],
    }
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        [{"type": "function", "function": {"name": "get_weather"}}],
        fc_mode="gst",
    )

    assert turn_results == [[]]
    assert payloads[0]["tools"]
    assert payloads[0]["messages"][0]["role"] == "system"
    assert timing["api_calls"] == 1
    assert timing["flags"] == ""


def test_generate_for_entry_nfc_does_not_inject_gst(monkeypatch):
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [{"message": {"content": "done"}}],
                "timings": _timings(),
            }
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
        "id": "nfc-no-gst",
        "initial_config": {},
        "involved_classes": [],
        "question": [[{"role": "user", "content": "hi"}]],
    }
    generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        [{"type": "function", "function": {"name": "ls"}}],
        fc_mode="nfc",
    )
    roles = [m["role"] for m in payloads[0]["messages"]]
    assert roles == ["user"]
    assert "[GST]" not in json.dumps(payloads[0]["messages"])
    assert "[STATE]" not in json.dumps(payloads[0]["messages"])
    assert payloads[0]["temperature"] == 0.0
    assert len(payloads) == 1


def test_generate_for_entry_iffc_state_bon_votes_then_executes_once(
    monkeypatch,
):
    from benchmark.bon import BON_TEMPERATURE

    payloads: list[dict] = []
    # N=3 samples per step: majority cd, then majority ABSTAIN.
    replies = [
        "not json",
        '[{"function_name": "cd", "parameters": {"folder": "docs"}}]',
        '[{"function_name": "cd", "parameters": {"folder": "docs"}}]',
        "[]",
        "[]",
        '[{"function_name": "ls", "parameters": {}}]',
    ]
    responses = iter(
        {
            "choices": [{"message": {"content": content}}],
            "timings": _timings(),
        }
        for content in replies
    )
    executed: list[list[str]] = []

    def fake_post(url, json, timeout):  # noqa: A002
        payloads.append(copy.deepcopy(json))
        return _FakeResponse(next(responses))

    def fake_execute(**kwargs):
        executed.append(list(kwargs["func_call_list"]))
        return [{"current_working_directory": "docs"}], None

    monkeypatch.setattr(generate_multi_turn.requests, "post", fake_post)
    monkeypatch.setattr(
        generate_multi_turn,
        "execute_multi_turn_func_call",
        fake_execute,
    )
    monkeypatch.setattr(generate_multi_turn.config, "BON_N", 3)

    entry = {
        "id": "iffc-state-bon-1",
        "initial_config": {},
        "involved_classes": ["fs"],
        "question": [[{"role": "user", "content": "Go to docs"}]],
    }
    all_tools = [{"type": "function", "function": {"name": "cd"}}]

    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        all_tools,
        fc_mode="iffc-state-bon",
    )

    assert turn_results == [[["cd(folder='docs')"]]]
    assert executed == [["cd(folder='docs')"]]
    assert len(payloads) == 6
    assert all("tools" not in p and "tool_choice" not in p for p in payloads)
    assert all(p["temperature"] == BON_TEMPERATURE for p in payloads)
    sys0 = payloads[0]["messages"][0]["content"]
    assert payloads[0]["messages"][0]["role"] == "system"
    assert "expert in function calling" in sys0
    assert "[STATE]" in sys0
    assert "[GST]" not in sys0
    assert "Curriculum:" not in sys0

    follow = payloads[3]["messages"]
    dump = follow[-1]["content"]
    assert follow[-1]["role"] == "user"
    assert "Function execution results:" in dump
    assert "[STATE]" in dump
    assert "docs" in dump
    assert not any(m.get("role") == "tool" for m in follow)
    # Losing samples must not land in the persistent transcript.
    blob = json.dumps(follow)
    assert blob.count("not json") == 0
    assert timing["id"] == "iffc-state-bon-1"
    assert timing["api_calls"] == 6
    assert timing["flags"] == ""


def test_generate_for_entry_iffc_state_does_not_sample_bon_n(monkeypatch):
    """Greedy iffc-state stays single-sample even when BON_N is raised."""
    payloads: list[dict] = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": '[{"function_name": "cd", "parameters": {"folder": "docs"}}]'
                        }
                    }
                ],
                "timings": _timings(),
            },
            {
                "choices": [{"message": {"content": "[]"}}],
                "timings": _timings(),
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
        lambda **kwargs: ([{"current_working_directory": "docs"}], None),
    )
    monkeypatch.setattr(generate_multi_turn.config, "BON_N", 5)

    entry = {
        "id": "iffc-state-no-bon",
        "initial_config": {},
        "involved_classes": [],
        "question": [[{"role": "user", "content": "Go to docs"}]],
    }
    turn_results, timing = generate_multi_turn.generate_for_entry(
        entry,
        "multi_turn_base",
        [{"type": "function", "function": {"name": "cd"}}],
        fc_mode="iffc-state",
    )
    assert turn_results == [[["cd(folder='docs')"]]]
    assert len(payloads) == 2
    assert timing["api_calls"] == 2
    assert all(p["temperature"] == 0.0 for p in payloads)


def _mt_entry(entry_id: str) -> dict:
    return {
        "id": entry_id,
        "initial_config": {},
        "involved_classes": [],
        "question": [[{"role": "user", "content": "hi"}]],
    }


def _mt_timing(entry_id: str, category: str = "multi_turn_base") -> dict:
    return {
        "id": entry_id,
        "category": category,
        "turns": 1,
        "api_calls": 1,
        "total_prompt_n": 1,
        "total_prompt_ms": 1.0,
        "total_predicted_n": 1,
        "total_predicted_ms": 1.0,
        "first_prompt_ms": 1.0,
        "flags": "",
    }


def test_pending_multi_turn_entries_skips_saved_ids():
    entries = [_mt_entry("a"), _mt_entry("b"), _mt_entry("c")]
    pending = generate_multi_turn.pending_multi_turn_entries(entries, {"b"})
    assert [e["id"] for e in pending] == ["a", "c"]


def test_generate_for_category_skips_saved_and_checkpoints(
    monkeypatch, bfcl_fixture_dirs
):
    data_dir, _ = bfcl_fixture_dirs
    write_jsonl(
        data_dir / "BFCL_v4_multi_turn_base.json",
        [_mt_entry("mt-saved"), _mt_entry("mt-new")],
    )
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b")
    db.insert_multi_turn_entry(
        run_id,
        "multi_turn_base",
        {"id": "mt-saved", "result": [[["old()"]]], "flags": ""},
        _mt_timing("mt-saved"),
    )

    generated: list[str] = []

    def fake_generate_for_entry(entry, category, all_tools, execution_model_name=None, fc_mode=None):
        generated.append(entry["id"])
        return [[["new()"]]], _mt_timing(entry["id"], category)

    monkeypatch.setattr(generate_multi_turn, "generate_for_entry", fake_generate_for_entry)
    monkeypatch.setattr(generate_multi_turn, "_load_func_docs", lambda classes: [])

    results, timings = generate_multi_turn.generate_for_category(
        "multi_turn_base",
        {"file": "BFCL_v4_multi_turn_base.json"},
        fc_mode="cd",
        run_id=run_id,
    )

    assert generated == ["mt-new"]
    assert [r["id"] for r in results] == ["mt-new"]
    assert [t["id"] for t in timings] == ["mt-new"]

    saved_ids = db.get_response_entry_ids(run_id, "multi_turn_base", mode="multi")
    assert saved_ids == {"mt-saved", "mt-new"}
    rows = {r["entry_id"]: r for r in db.get_responses(run_id, category="multi_turn_base")}
    assert json.loads(rows["mt-saved"]["result"]) == [[["old()"]]]
    assert json.loads(rows["mt-new"]["result"]) == [[["new()"]]]
    assert len(db.get_multi_turn_timings(run_id)) == 2


def test_async_category_checkpoints_before_later_failure(
    monkeypatch, bfcl_fixture_dirs
):
    """Completed entries must land in SQLite even if gather() then fails."""
    data_dir, _ = bfcl_fixture_dirs
    write_jsonl(
        data_dir / "BFCL_v4_multi_turn_base.json",
        [_mt_entry("mt-ok"), _mt_entry("mt-fail")],
    )
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b", notes="fc_mode=cd")
    generated: list[str] = []

    async def fake_generate_entry_async(
        session, semaphore, entry, category, all_tools, pbar, fc_mode=None
    ):
        generated.append(entry["id"])
        if entry["id"] == "mt-fail":
            await asyncio.sleep(0.05)
            raise RuntimeError("simulated kill")
        await asyncio.sleep(0.01)
        pbar.update(1)
        return (
            {"id": entry["id"], "result": [[["ok()"]]], "flags": ""},
            _mt_timing(entry["id"], category),
        )

    monkeypatch.setattr(
        generate_multi_turn, "_generate_entry_async", fake_generate_entry_async
    )
    monkeypatch.setattr(generate_multi_turn, "_load_func_docs", lambda classes: [])

    async def run_category():
        sem = asyncio.Semaphore(4)
        return await generate_multi_turn._generate_for_category_async(
            session=None,
            semaphore=sem,
            category="multi_turn_base",
            category_info={"file": "BFCL_v4_multi_turn_base.json"},
            fc_mode="cd",
            run_id=run_id,
        )

    try:
        asyncio.run(run_category())
        raise AssertionError("expected simulated kill")
    except RuntimeError as exc:
        assert "simulated kill" in str(exc)

    assert "mt-ok" in generated
    saved = db.get_response_entry_ids(run_id, "multi_turn_base", mode="multi")
    assert saved == {"mt-ok"}
    timings = db.get_multi_turn_timings(run_id)
    assert [t["entry_id"] for t in timings] == ["mt-ok"]


def test_async_category_skips_already_done(monkeypatch, bfcl_fixture_dirs):
    data_dir, _ = bfcl_fixture_dirs
    write_jsonl(
        data_dir / "BFCL_v4_multi_turn_base.json",
        [_mt_entry("mt-done"), _mt_entry("mt-todo")],
    )
    monkeypatch.setattr(db, "_detect_hardware", lambda: {})
    run_id = db.create_run("qwen2.5-3b", notes="fc_mode=iffc")
    db.insert_multi_turn_entry(
        run_id,
        "multi_turn_base",
        {"id": "mt-done", "result": [[["saved()"]]], "flags": ""},
        _mt_timing("mt-done"),
    )
    generated: list[str] = []

    async def fake_generate_entry_async(
        session, semaphore, entry, category, all_tools, pbar, fc_mode=None
    ):
        generated.append(entry["id"])
        pbar.update(1)
        return (
            {"id": entry["id"], "result": [[["fresh()"]]], "flags": ""},
            _mt_timing(entry["id"], category),
        )

    monkeypatch.setattr(
        generate_multi_turn, "_generate_entry_async", fake_generate_entry_async
    )
    monkeypatch.setattr(generate_multi_turn, "_load_func_docs", lambda classes: [])

    async def run_category():
        sem = asyncio.Semaphore(4)
        return await generate_multi_turn._generate_for_category_async(
            session=None,
            semaphore=sem,
            category="multi_turn_base",
            category_info={"file": "BFCL_v4_multi_turn_base.json"},
            fc_mode="iffc",
            run_id=run_id,
        )

    results, timings = asyncio.run(run_category())
    assert generated == ["mt-todo"]
    assert [r["id"] for r in results] == ["mt-todo"]
    assert [t["id"] for t in timings] == ["mt-todo"]
    assert db.get_response_entry_ids(run_id, "multi_turn_base", mode="multi") == {
        "mt-done",
        "mt-todo",
    }

