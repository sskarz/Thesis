import pytest
import requests

from benchmark import collect_metrics
from benchmark import generate_multi_turn
from benchmark import generate_responses


def _require_live_server():
    try:
        resp = requests.get(
            f"{collect_metrics.config.SERVER_URL}/v1/models",
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on local environment
        pytest.skip(f"Local llama-server unavailable: {exc}")


@pytest.mark.live
def test_live_single_turn_response_is_parseable():
    _require_live_server()

    payload = generate_responses.build_payload(
        {
            "question": [[{"role": "user", "content": "What is the weather in San Francisco?"}]],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
    )

    data = collect_metrics.send_request(payload)
    parsed = generate_responses.parse_tool_calls(data)

    assert data["choices"]
    assert isinstance(parsed, (list, str))
    if isinstance(parsed, list):
        assert parsed


@pytest.mark.live
@pytest.mark.slow
def test_live_multi_turn_loop_returns_structurally_valid_result(monkeypatch):
    _require_live_server()

    monkeypatch.setattr(
        generate_multi_turn,
        "execute_multi_turn_func_call",
        lambda **kwargs: (["72 F"], None),
    )

    entry = {
        "id": "live-mt-1",
        "initial_config": {},
        "involved_classes": [],
        "question": [[{"role": "user", "content": "Call get_weather for San Francisco and then answer briefly."}]],
    }
    tools = [
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
        tools,
    )

    assert isinstance(turn_results, list)
    assert timing["id"] == "live-mt-1"
    assert timing["total_prompt_n"] >= 0
    assert timing["total_predicted_n"] >= 0
    assert timing["api_calls"] >= 1
    if turn_results and turn_results[0]:
        assert all(isinstance(step, list) for step in turn_results[0])
