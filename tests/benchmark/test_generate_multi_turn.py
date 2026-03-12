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
    }
