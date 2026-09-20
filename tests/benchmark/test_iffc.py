import json

import pytest

from benchmark import config
from benchmark.iffc import (
    IFFC_SYSTEM_PROMPT,
    build_iffc_system_prompt,
    format_available_functions,
    format_iffc_execution_user_message,
    iffc_calls_to_func_strings,
    is_empty_iffc_assistant_message,
    parse_iffc_response,
    strip_trailing_empty_assistant,
    try_parse_iffc_response,
)
from benchmark.smoke_iffc import run_offline_smoke


def test_system_prompt_is_appendix_b_with_placeholder():
    assert "{available_functions}" in IFFC_SYSTEM_PROMPT
    assert "function_name" in IFFC_SYSTEM_PROMPT
    assert "parameters" in IFFC_SYSTEM_PROMPT
    assert "expert in function calling" in IFFC_SYSTEM_PROMPT
    filled = build_iffc_system_prompt(
        [{"name": "mv", "description": "move a file", "parameters": {"type": "object"}}]
    )
    assert "{available_functions}" not in filled
    assert '"name": "mv"' in filled


def test_format_available_functions_unwraps_openai_tools_and_bfcl_docs():
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "cd",
                "description": "change directory",
                "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}},
            },
        }
    ]
    bfcl_docs = [{"name": "ls", "description": "list files"}]
    mixed = format_available_functions(openai_tools + bfcl_docs)
    parsed = json.loads(mixed)
    assert [item["name"] for item in parsed] == ["cd", "ls"]


def test_parse_empty_list():
    assert parse_iffc_response("[]") == []
    assert parse_iffc_response("  []  ") == []
    assert try_parse_iffc_response("[]") == []


def test_strip_trailing_empty_assistant_pops_only_empty_list_stop():
    history = [
        {"role": "user", "content": "Open docs"},
        {
            "role": "assistant",
            "content": '[{"function_name": "cd", "parameters": {"folder": "docs"}}]',
        },
        {"role": "user", "content": "Function execution results:\n- cd -> ok"},
        {"role": "assistant", "content": "[]"},
    ]
    assert is_empty_iffc_assistant_message(history[-1])
    assert not is_empty_iffc_assistant_message(history[1])
    assert strip_trailing_empty_assistant(history) == 1
    assert history[-1]["role"] == "user"
    assert "execution results" in history[-1]["content"]
    assert not any(
        m.get("role") == "assistant" and try_parse_iffc_response(m.get("content")) == []
        for m in history
    )
    # Second call is a no-op; a real call list is not treated as empty.
    assert strip_trailing_empty_assistant(history) == 0
    fenced = [{"role": "assistant", "content": "```json\n[]\n```"}]
    assert strip_trailing_empty_assistant(fenced) == 1
    unparseable = [{"role": "assistant", "content": "Sure, I can help."}]
    assert strip_trailing_empty_assistant(unparseable) == 0
    assert unparseable[0]["content"] == "Sure, I can help."


def test_parse_one_call():
    text = (
        '[{"function_name": "mv", "parameters": '
        '{"source": "/tmp/a", "destination": "/tmp/b"}}]'
    )
    assert parse_iffc_response(text) == [
        {
            "function_name": "mv",
            "parameters": {"source": "/tmp/a", "destination": "/tmp/b"},
        }
    ]


def test_parse_multi_call():
    text = (
        '[{"function_name": "cd", "parameters": {"folder": "docs"}},'
        ' {"function_name": "ls", "parameters": {}}]'
    )
    calls = parse_iffc_response(text)
    assert [c["function_name"] for c in calls] == ["cd", "ls"]
    assert calls[0]["parameters"] == {"folder": "docs"}
    assert calls[1]["parameters"] == {}


def test_parse_fenced_json():
    text = (
        "Here you go:\n"
        "```json\n"
        '[{"function_name": "ping", "parameters": {"host": "localhost"}}]\n'
        "```\n"
    )
    assert parse_iffc_response(text) == [
        {"function_name": "ping", "parameters": {"host": "localhost"}}
    ]


def test_parse_slightly_noisy_text():
    text = (
        "I will call the function now.\n"
        '[{"function_name": "ls", "parameters": {"path": "."}}]\n'
        "That should list the files."
    )
    assert parse_iffc_response(text) == [
        {"function_name": "ls", "parameters": {"path": "."}}
    ]


def test_parse_strips_think_wrappers_and_accepts_name_alias():
    text = (
        "<think>planning</think>\n"
        '[{"name": "cd", "arguments": {"folder": "src"}}]'
    )
    assert parse_iffc_response(text) == [
        {"function_name": "cd", "parameters": {"folder": "src"}}
    ]


def test_parse_unparseable_returns_empty_list_and_try_returns_none():
    assert try_parse_iffc_response("Sure, I can help with that.") is None
    assert parse_iffc_response("Sure, I can help with that.") == []


def test_iffc_calls_to_func_strings_matches_nfc_repr_convention():
    calls = [
        {"function_name": "mv", "parameters": {"source": "a", "destination": "b"}},
        {"function_name": "count", "parameters": {"n": 5, "ok": True}},
    ]
    assert iffc_calls_to_func_strings(calls) == [
        "mv(source='a', destination='b')",
        "count(n=5, ok=True)",
    ]


def test_format_execution_user_message_is_user_role_instruction():
    msg = format_iffc_execution_user_message(
        ["cd(folder='docs')"],
        ["ok"],
    )
    assert "cd(folder='docs') -> ok" in msg
    assert "JSON list" in msg


def test_resolve_multi_turn_categories_filters_and_rejects_unknown():
    subset = config.resolve_multi_turn_categories(["multi_turn_base"])
    assert list(subset) == ["multi_turn_base"]
    comma = config.resolve_multi_turn_categories(["multi_turn_base,multi_turn_miss_func"])
    assert list(comma) == ["multi_turn_base", "multi_turn_miss_func"]
    with pytest.raises(ValueError, match="Unknown multi-turn"):
        config.resolve_multi_turn_categories(["not_a_category"])


def test_offline_smoke_script_passes():
    run_offline_smoke()
