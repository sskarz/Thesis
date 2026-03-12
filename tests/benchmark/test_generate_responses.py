import json

from benchmark import generate_responses


def test_bfcl_to_openai_tools_converts_types_and_adds_default_parameters():
    tools = generate_responses.bfcl_to_openai_tools(
        [
            {
                "name": "lookup_weather",
                "description": "Weather lookup",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "city": {"type": "String"},
                        "days": {"type": "int"},
                    },
                },
            },
            {
                "name": "ping",
                "description": "No params",
            },
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Weather lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "No params",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def test_parse_tool_calls_handles_openai_and_functiongemma_payloads():
    openai_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup_weather",
                                "arguments": json.dumps({"city": "San Francisco"}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    functiongemma_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<start_function_call>"
                        "call:lookup_weather{city:<escape>San Francisco<escape>}"
                        "<end_function_call>"
                    )
                }
            }
        ]
    }

    assert generate_responses.parse_tool_calls(openai_response) == [
        {"lookup_weather": {"city": "San Francisco"}}
    ]
    assert generate_responses.parse_tool_calls(functiongemma_response) == [
        {"lookup_weather": {"city": "San Francisco"}}
    ]
