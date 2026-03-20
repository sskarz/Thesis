import json

from benchmark import generate_responses
from benchmark.coerce_types import coerce_value, coerce_tool_call, coerce_tool_calls


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

    # <tool_call> XML blocks (from tool_choice="none" bypass)
    tool_call_xml_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        '<tool_call>\n{"name": "math.factorial", "arguments": {"number": 5}}\n</tool_call>\n'
                        '<tool_call>\n{"name": "math.factorial", "arguments": {"number": 10}}\n</tool_call>'
                    ),
                }
            }
        ]
    }
    assert generate_responses.parse_tool_calls(tool_call_xml_response) == [
        {"math.factorial": {"number": 5}},
        {"math.factorial": {"number": 10}},
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
    # System prompt is prepended when tools are present
    assert payload_with_tools["messages"][0]["role"] == "system"
    assert payload_with_tools["messages"][1] == {"role": "user", "content": "Weather?"}
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
    # No system prompt when no tools
    assert payload_without_tools["messages"] == [{"role": "user", "content": "Hello"}]

    # Parallel categories bypass grammar constraining
    payload_parallel = generate_responses.build_payload(
        {
            "question": [[{"role": "user", "content": "Do two things"}]],
            "function": [
                {
                    "name": "do_thing",
                    "description": "Do a thing",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        category="parallel",
    )
    assert payload_parallel["tool_choice"] == "none"

    payload_parallel_multiple = generate_responses.build_payload(
        {
            "question": [[{"role": "user", "content": "Do two things"}]],
            "function": [
                {
                    "name": "do_thing",
                    "description": "Do a thing",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        category="parallel_multiple",
    )
    assert payload_parallel_multiple["tool_choice"] == "none"


# ---------------------------------------------------------------------------
# Type coercion tests
# ---------------------------------------------------------------------------


def test_coerce_value_string_to_integer():
    assert coerce_value("5", {"type": "integer"}) == 5
    assert coerce_value("3.0", {"type": "integer"}) == 3


def test_coerce_value_string_to_number():
    assert coerce_value("3.14", {"type": "number"}) == 3.14
    assert coerce_value("10", {"type": "number"}) == 10
    assert coerce_value("0", {"type": "float"}) == 0


def test_coerce_value_string_to_array():
    assert coerce_value("[1, 2, 3]", {"type": "array"}) == [1, 2, 3]
    assert coerce_value("['a', 'b']", {"type": "array"}) == ["a", "b"]
    assert coerce_value("[3, 4]", {"type": "tuple"}) == [3, 4]


def test_coerce_value_string_to_object():
    assert coerce_value('{"a": 1}', {"type": "object"}) == {"a": 1}
    assert coerce_value("{'a': 1}", {"type": "dict"}) == {"a": 1}


def test_coerce_value_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "min": {"type": "integer"},
            "max": {"type": "integer"},
        },
    }
    # Already-parsed dict with string leaf values
    assert coerce_value({"min": "100", "max": "200"}, schema) == {
        "min": 100,
        "max": 200,
    }


def test_coerce_value_string_to_boolean():
    assert coerce_value("true", {"type": "boolean"}) is True
    assert coerce_value("false", {"type": "boolean"}) is False


def test_coerce_value_passthrough():
    """Values that already have the correct type are unchanged."""
    assert coerce_value(5, {"type": "integer"}) == 5
    assert coerce_value([1, 2], {"type": "array"}) == [1, 2]
    assert coerce_value("hello", {"type": "string"}) == "hello"
    assert coerce_value("not a number", {"type": "integer"}) == "not a number"


def test_coerce_tool_call_uses_schema():
    func_defs = [
        {
            "name": "calculate",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "number"},
                    "items": {"type": "array"},
                },
            },
        }
    ]
    tc = {"calculate": {"x": "10", "y": "3.14", "items": "[1, 2]"}}
    assert coerce_tool_call(tc, func_defs) == {
        "calculate": {"x": 10, "y": 3.14, "items": [1, 2]}
    }


def test_coerce_tool_calls_list():
    func_defs = [
        {
            "name": "add",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
        }
    ]
    calls = [{"add": {"a": "1", "b": "2"}}]
    assert coerce_tool_calls(calls, func_defs) == [{"add": {"a": 1, "b": 2}}]


def test_parse_tool_calls_applies_coercion_when_func_defs_provided():
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculate",
                                "arguments": json.dumps(
                                    {"x": "5", "y": "3.14"}
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }
    func_defs = [
        {
            "name": "calculate",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "number"},
                },
            },
        }
    ]
    result = generate_responses.parse_tool_calls(response, func_defs=func_defs)
    assert result == [{"calculate": {"x": 5, "y": 3.14}}]


def test_parse_tool_calls_without_func_defs_no_coercion():
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calc",
                                "arguments": json.dumps({"x": "5"}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    result = generate_responses.parse_tool_calls(response)
    assert result == [{"calc": {"x": "5"}}]
