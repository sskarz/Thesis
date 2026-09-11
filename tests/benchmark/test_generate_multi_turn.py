import copy
import json

from benchmark import generate_multi_turn


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
    assert not any(m.get("role") == "tool" for m in followup)
    assert executed[0]["func_call_list"] == ["cd(folder='docs')"]
    assert timing["id"] == "iffc-1"
    assert timing["api_calls"] == 2
    assert timing["flags"] == ""


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


def test_build_chat_payload_nfc_keeps_tools_iffc_and_cd_omit_them():
    history = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "ls"}}]
    nfc = generate_multi_turn._build_chat_payload(history, tools, "nfc")
    iffc = generate_multi_turn._build_chat_payload(history, tools, "iffc")
    cd = generate_multi_turn._build_chat_payload(history, tools, "cd")
    assert nfc["tools"] == tools
    assert nfc["tool_choice"] == "auto"
    assert "tools" not in iffc
    assert "tool_choice" not in iffc
    assert "tools" not in cd
    assert "tool_choice" not in cd


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

