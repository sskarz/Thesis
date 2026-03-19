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


def test_convert_types_omits_any_type():
    tools = generate_responses.bfcl_to_openai_tools(
        [
            {
                "name": "flexible",
                "description": "Takes anything",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "data": {"type": "any", "description": "Any value"},
                    },
                },
            }
        ]
    )
    props = tools[0]["function"]["parameters"]["properties"]
    assert "type" not in props["data"]
    assert props["data"]["description"] == "Any value"


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


def test_build_payload_includes_tools_only_when_functions_exist():
    payload_with_tools = generate_responses.build_payload(
        {
            "question": [[{"role": "user", "content": "Weather?"}]],
            "function": [
                {
                    "name": "lookup_weather",
                    "description": "Weather lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        }
    )
    payload_without_tools = generate_responses.build_payload(
        {
            "question": [[{"role": "user", "content": "Hello"}]],
        }
    )

    assert payload_with_tools["model"] == generate_responses.config.MODEL_NAME
    assert payload_with_tools["messages"] == [{"role": "user", "content": "Weather?"}]
    assert payload_with_tools["tool_choice"] == "auto"
    assert payload_with_tools["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Weather lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    assert "tools" not in payload_without_tools
    assert "tool_choice" not in payload_without_tools
