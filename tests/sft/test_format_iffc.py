# SFT formatter / split tests — no HuggingFace downloads, no mlx-lm.

from __future__ import annotations

import json
from benchmark.iffc import (
    IFFC_SYSTEM_PROMPT,
    parse_iffc_response,
)
from sft.format_iffc import (
    dumps_iffc_calls,
    extract_json_array_of_objects,
    format_toolace_row,
    format_xlam_row,
    normalize_function_docs,
    parse_toolace_call_list,
    unroll_conversation,
)


TOOLACE_SYSTEM = (
    "You are an expert in composing functions. Here is a list of functions "
    "in JSON format that you can invoke:\n"
    + json.dumps(
        [
            {
                "name": "Market Trends API",
                "description": "Get market trends",
                "parameters": {
                    "trend_type": {
                        "type": "string",
                        "description": "kind",
                        "required": True,
                    },
                    "country": {"type": "string", "description": "cc", "required": True},
                },
            },
            {
                "name": "Quotes by Keywords",
                "description": "quotes",
                "parameters": {
                    "word": {"type": "string", "description": "keyword", "required": True},
                },
            },
        ]
    )
)


def test_parse_toolace_single_and_parallel_calls():
    one = parse_toolace_call_list(
        '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]'
    )
    assert one == [
        {
            "function_name": "Market Trends API",
            "parameters": {"trend_type": "MARKET_INDEXES", "country": "us"},
        }
    ]
    parallel = parse_toolace_call_list(
        '[Format Number(phone="408-555-1234"), Valistring Phone Number(phone="408-555-1234")]'
    )
    assert [c["function_name"] for c in parallel] == [
        "Format Number",
        "Valistring Phone Number",
    ]
    empty = parse_toolace_call_list("[GetCompetitions()]")
    assert empty == [{"function_name": "GetCompetitions", "parameters": {}}]
    slash = parse_toolace_call_list("[/madlibs-diceware(nphrase=2)]")
    assert slash[0]["function_name"] == "/madlibs-diceware"
    assert slash[0]["parameters"] == {"nphrase": 2}


def test_parse_toolace_nl_and_bool_args():
    assert parse_toolace_call_list("May I know your language code?") == []
    assert parse_toolace_call_list("") == []
    calls = parse_toolace_call_list("[Get NBA Teams(teamStats=True, rosters=True)]")
    assert calls[0]["parameters"] == {"teamStats": True, "rosters": True}


def test_extract_tools_from_system_prompt():
    docs = extract_json_array_of_objects(TOOLACE_SYSTEM)
    assert [d["name"] for d in docs] == ["Market Trends API", "Quotes by Keywords"]


def test_format_toolace_row_is_iffc_json_in_text_not_native_tools():
    row = {
        "row_index": 0,
        "system": TOOLACE_SYSTEM,
        "conversations": [
            {
                "from": "user",
                "value": "Top US market trends?",
            },
            {
                "from": "assistant",
                "value": '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]',
            },
            {
                "from": "tool",
                "value": json.dumps(
                    [{"name": "Market Trends API", "results": {"trends": ["nasdaq"]}}]
                ),
            },
            {
                "from": "assistant",
                "value": "Here are the trends.",
            },
        ],
    }
    rec = format_toolace_row(row, 0)
    assert rec is not None
    assert rec["id"] == "toolace-0"
    assert rec["source"] == "toolace"
    assert rec["is_multi_turn"] is True
    assert rec["chat_format"] == "iffc"
    roles = [m["role"] for m in rec["messages"]]
    assert roles[0] == "system"
    assert rec["messages"][0]["content"].startswith(
        IFFC_SYSTEM_PROMPT.split("{available_functions}")[0][:40]
    )
    assert "function_name" in rec["messages"][0]["content"]
    assert "Market Trends API" in rec["messages"][0]["content"]
    assistant = [m for m in rec["messages"] if m["role"] == "assistant"]
    assert parse_iffc_response(assistant[0]["content"]) == [
        {
            "function_name": "Market Trends API",
            "parameters": {"trend_type": "MARKET_INDEXES", "country": "us"},
        }
    ]
    # Final NL turn becomes IFFC empty list, not free-form text.
    assert parse_iffc_response(assistant[1]["content"]) == []
    assert assistant[1]["content"] == "[]"
    obs = [m for m in rec["messages"] if m["role"] == "user"][1]
    assert "Function execution results:" in obs["content"]
    assert "JSON list" in obs["content"]
    for m in rec["messages"]:
        assert "tool_calls" not in m
        assert m["role"] in {"system", "user", "assistant"}
    assert rec["n_tools_in_gold"] == 1
    assert rec["n_tools_bin"] == "1"


def test_format_xlam_parallel_answers_roundtrip_parser():
    row = {
        "id": 42,
        "query": "Sum multiples and product of primes.",
        "tools": json.dumps(
            [
                {
                    "name": "math_toolkit.sum_of_multiples",
                    "description": "sum",
                    "parameters": {
                        "lower_limit": {"type": "int", "description": "lo", "required": True},
                        "upper_limit": {"type": "int", "description": "hi", "required": True},
                    },
                },
                {
                    "name": "math_toolkit.product_of_primes",
                    "description": "prod",
                    "parameters": {
                        "count": {"type": "int", "description": "n", "required": True},
                    },
                },
            ]
        ),
        "answers": json.dumps(
            [
                {
                    "name": "math_toolkit.sum_of_multiples",
                    "arguments": {"lower_limit": 1, "upper_limit": 1000},
                },
                {
                    "name": "math_toolkit.product_of_primes",
                    "arguments": {"count": 5},
                },
            ]
        ),
    }
    rec = format_xlam_row(row, 0)
    assert rec["id"] == "xlam-42"
    assert rec["source"] == "xlam"
    assert rec["is_multi_turn"] is False
    assert rec["n_tools_in_gold"] == 2
    assert rec["n_tools_bin"] == "ge2"
    assistant = rec["messages"][-1]
    parsed = parse_iffc_response(assistant["content"])
    assert [c["function_name"] for c in parsed] == [
        "math_toolkit.sum_of_multiples",
        "math_toolkit.product_of_primes",
    ]
    assert parsed[1]["parameters"] == {"count": 5}
    assert "tool_calls" not in assistant
    assert rec["messages"][0]["role"] == "system"
    assert "math_toolkit.sum_of_multiples" in rec["messages"][0]["content"]


def test_xlam_nested_type_dict_does_not_raise():
    """xLAM sometimes stores ``parameters["type"]`` as a nested schema dict.

    ``"type" in {"object", "dict"}`` raises TypeError on an unhashable dict;
    formatting must skip that membership check and still emit tools.
    """
    tools = [
        {
            "name": "weird_tool",
            "description": "has nested type dict",
            "parameters": {
                "type": {"type": "string"},
                "foo": {"type": "int", "required": True},
            },
        }
    ]
    openai_tools = normalize_function_docs(tools)
    assert openai_tools
    assert openai_tools[0]["type"] == "function"
    params = openai_tools[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert "foo" in params["properties"]
    assert params["properties"]["foo"]["type"] == "integer"
    assert "foo" in params.get("required", [])

    rec = format_xlam_row(
        {
            "id": "nested-type",
            "query": "Call the weird tool.",
            "tools": tools,
            "answers": [{"name": "weird_tool", "arguments": {"foo": 1}}],
        },
        0,
    )
    assert rec is not None
    assert rec["id"] == "xlam-nested-type"
    assert rec["messages"][0]["role"] == "system"
    assert "weird_tool" in rec["messages"][0]["content"]
    assert rec["n_tools_in_gold"] == 1


def test_dumps_iffc_calls_matches_harness_keys():
    text = dumps_iffc_calls(
        [{"function_name": "cd", "parameters": {"folder": "docs"}}]
    )
    assert parse_iffc_response(text) == [
        {"function_name": "cd", "parameters": {"folder": "docs"}}
    ]


def test_unroll_conversation_one_row_per_assistant_turn():
    rec = {
        "id": "toolace-9",
        "source": "toolace",
        "is_multi_turn": True,
        "n_tools_in_gold": 1,
        "n_tools_bin": "1",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "[]"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": '[{"function_name": "ls", "parameters": {}}]'},
        ],
    }
    rows = unroll_conversation(rec)
    assert len(rows) == 2
    assert rows[0]["id"] == "toolace-9#asst-2"
    assert rows[0]["parent_id"] == "toolace-9"
    assert rows[0]["messages"][-1]["content"] == "[]"
    assert rows[1]["messages"][-1]["role"] == "assistant"
    assert len(rows[1]["messages"]) == 5


def test_malformed_toolace_call_list_is_skipped():
    row = {
        "row_index": 3,
        "system": TOOLACE_SYSTEM,
        "conversations": [
            {"from": "user", "value": "hi"},
            {"from": "assistant", "value": "[NotACall("},
        ],
    }
    assert format_toolace_row(row, 3) is None
