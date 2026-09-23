# APIGen-MT ShareGPT → IFFC converter. No Hub download.

from __future__ import annotations

import json

from benchmark.iffc import IFFC_SYSTEM_PROMPT, parse_iffc_response

from sft.format_apigen_mt import (
    STOP_CLARIFY,
    STOP_FINAL,
    STOP_IRRELEVANCE,
    STOP_TOOL_CALL,
    format_apigen_mt_row,
    label_assistant_content,
    parse_apigen_function_call_value,
)
from sft.format_iffc import format_rows, main as format_main
from sft.config import SOURCE_APIGEN_MT


RETAIL_TOOLS = json.dumps(
    [
        {
            "name": "get_order_details",
            "description": "Look up an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
        {
            "name": "cancel_order",
            "description": "Cancel an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    ]
)


def test_parse_apigen_object_and_parallel_list():
    one = parse_apigen_function_call_value(
        {"name": "get_order_details", "arguments": {"order_id": "A1"}}
    )
    assert one == [
        {"function_name": "get_order_details", "parameters": {"order_id": "A1"}}
    ]
    parallel = parse_apigen_function_call_value(
        json.dumps(
            [
                {"name": "get_order_details", "arguments": {"order_id": "A1"}},
                {"name": "cancel_order", "arguments": {"order_id": "A1", "reason": "dup"}},
            ]
        )
    )
    assert [c["function_name"] for c in parallel] == [
        "get_order_details",
        "cancel_order",
    ]
    already_iffc = parse_apigen_function_call_value(
        {"function_name": "ls", "parameters": {}}
    )
    assert already_iffc == [{"function_name": "ls", "parameters": {}}]
    assert parse_apigen_function_call_value("not-json") is None
    assert parse_apigen_function_call_value("[]") == []


def test_label_assistant_content_taxonomy():
    assert (
        label_assistant_content(
            json.dumps([{"function_name": "ls", "parameters": {}}])
        )
        == STOP_TOOL_CALL
    )
    assert label_assistant_content("[]") == STOP_IRRELEVANCE
    assert label_assistant_content("Which airport would you like?") == STOP_CLARIFY
    assert label_assistant_content("Could you please provide your name") == STOP_CLARIFY
    assert label_assistant_content("Your order has been cancelled.") == STOP_FINAL


def test_format_apigen_tool_obs_gpt_final_is_nl_not_empty_list():
    row = {
        "row_index": 0,
        "system": "You are a retail agent. Keep responses concise.",
        "tools": RETAIL_TOOLS,
        "conversations": [
            {"from": "human", "value": "What's the status of order A1?"},
            {
                "from": "function_call",
                "value": json.dumps(
                    {"name": "get_order_details", "arguments": {"order_id": "A1"}}
                ),
            },
            {
                "from": "observation",
                "value": json.dumps(
                    [{"name": "get_order_details", "results": {"status": "shipped"}}]
                ),
            },
            {
                "from": "gpt",
                "value": "Order A1 has shipped.",
            },
        ],
    }
    rec = format_apigen_mt_row(row, 0)
    assert rec is not None
    assert rec["id"] == "apigen_mt-0"
    assert rec["source"] == SOURCE_APIGEN_MT
    assert rec["is_multi_turn"] is True
    assert rec["chat_format"] == "iffc"
    assert rec["stop_label"] == STOP_FINAL
    assert rec["stop_labels"] == [STOP_TOOL_CALL, STOP_FINAL]
    roles = [m["role"] for m in rec["messages"]]
    assert roles[0] == "system"
    assert rec["messages"][0]["content"].startswith(
        IFFC_SYSTEM_PROMPT.split("{available_functions}")[0][:40]
    )
    assert "get_order_details" in rec["messages"][0]["content"]
    assert "Domain policy:" in rec["messages"][0]["content"]
    assert "retail agent" in rec["messages"][0]["content"]
    assistants = [m for m in rec["messages"] if m["role"] == "assistant"]
    assert parse_iffc_response(assistants[0]["content"]) == [
        {"function_name": "get_order_details", "parameters": {"order_id": "A1"}}
    ]
    assert assistants[1]["content"] == "Order A1 has shipped."
    assert assistants[1]["content"] != "[]"
    obs = [m for m in rec["messages"] if m["role"] == "user"][1]
    assert "Function execution results:" in obs["content"]
    for m in rec["messages"]:
        assert "tool_calls" not in m
    assert rec["n_tools_in_gold"] == 1


def test_format_apigen_clarify_only_keeps_question_text():
    row = {
        "id": "clarify-1",
        "tools": RETAIL_TOOLS,
        "conversations": [
            {"from": "human", "value": "Cancel my order."},
            {
                "from": "gpt",
                "value": "Which order_id should I cancel?",
            },
        ],
    }
    rec = format_apigen_mt_row(row, 7)
    assert rec is not None
    assert rec["id"] == "apigen_mt-clarify-1"
    assert rec["stop_label"] == STOP_CLARIFY
    gold = rec["messages"][-1]
    assert gold["role"] == "assistant"
    assert gold["content"] == "Which order_id should I cancel?"
    assert parse_iffc_response(gold["content"]) == []
    assert gold["content"] != "[]"
    assert rec["n_tools_in_gold"] == 0


def test_format_apigen_parallel_function_calls():
    row = {
        "row_index": 2,
        "tools": RETAIL_TOOLS,
        "conversations": [
            {"from": "human", "value": "Look up and cancel A1."},
            {
                "from": "function_call",
                "value": [
                    {"name": "get_order_details", "arguments": {"order_id": "A1"}},
                    {
                        "name": "cancel_order",
                        "arguments": {"order_id": "A1", "reason": "changed mind"},
                    },
                ],
            },
        ],
    }
    rec = format_apigen_mt_row(row, 2)
    assert rec is not None
    parsed = parse_iffc_response(rec["messages"][-1]["content"])
    assert [c["function_name"] for c in parsed] == [
        "get_order_details",
        "cancel_order",
    ]
    assert rec["n_tools_in_gold"] == 2
    assert rec["n_tools_bin"] == "ge2"
    assert rec["stop_label"] == STOP_TOOL_CALL


def test_format_apigen_rejects_unparseable_and_no_assistant():
    bad = {
        "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "function_call", "value": "not a call"},
        ],
        "tools": RETAIL_TOOLS,
    }
    assert format_apigen_mt_row(bad, 0) is None
    no_asst = {
        "conversations": [{"from": "human", "value": "hi"}],
        "tools": RETAIL_TOOLS,
    }
    assert format_apigen_mt_row(no_asst, 1) is None
    unknown_role = {
        "conversations": [
            {"from": "narrator", "value": "once upon a time"},
            {"from": "gpt", "value": "ok"},
        ],
        "tools": RETAIL_TOOLS,
    }
    assert format_apigen_mt_row(unknown_role, 2) is None


def test_format_apigen_rejects_native_tool_calls_in_turn():
    row = {
        "tools": RETAIL_TOOLS,
        "conversations": [
            {"from": "human", "value": "hi"},
            {
                "from": "gpt",
                "value": "calling",
                "tool_calls": [{"id": "1", "function": {"name": "x"}}],
            },
        ],
    }
    assert format_apigen_mt_row(row, 0) is None


def test_format_rows_dispatches_apigen_source():
    rows = [
        {
            "row_index": 0,
            "tools": RETAIL_TOOLS,
            "conversations": [
                {"from": "human", "value": "status?"},
                {
                    "from": "function_call",
                    "value": {"name": "get_order_details", "arguments": {"order_id": "Z"}},
                },
            ],
        }
    ]
    recs, skipped = format_rows(rows, SOURCE_APIGEN_MT)
    assert skipped == 0
    assert recs[0]["source"] == SOURCE_APIGEN_MT


def test_format_iffc_cli_apigen_mt_only(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "formatted"
    raw.mkdir()
    (raw / "apigen_mt.jsonl").write_text(
        json.dumps(
            {
                "row_index": 0,
                "tools": RETAIL_TOOLS,
                "conversations": [
                    {"from": "human", "value": "hi"},
                    {
                        "from": "function_call",
                        "value": {"name": "get_order_details", "arguments": {"order_id": "1"}},
                    },
                ],
            }
        )
        + "\n"
    )
    assert (
        format_main(
            [
                "--raw-dir",
                str(raw),
                "--out-dir",
                str(out),
                "--apigen-mt-only",
            ]
        )
        == 0
    )
    lines = (out / "apigen_mt.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["source"] == "apigen_mt"
    assert rec["messages"][-1]["role"] == "assistant"
