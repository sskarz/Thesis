import json

import pytest

from benchmark import config
from benchmark.gst import (
    GST_CLOSE,
    GST_MAX_GUIDANCE_LINES,
    GST_OPEN,
    GST_STEPS,
    GST_SYSTEM_PROMPT,
    build_gst_system_prompt,
    ensure_gst_system_message,
    extract_gst_block,
    gst_payload_has_curriculum,
    gst_payload_has_nfc_tools,
    has_gst_scaffold,
    parse_gst_steps,
    strip_gst_block,
)
from benchmark.smoke_gst import run_offline_smoke


def test_system_prompt_encodes_detail_template_and_wrapper():
    prompt = build_gst_system_prompt()
    assert prompt == GST_SYSTEM_PROMPT
    assert GST_OPEN in prompt
    assert GST_CLOSE in prompt
    assert GST_MAX_GUIDANCE_LINES == 8
    assert f"max {GST_MAX_GUIDANCE_LINES} lines" in prompt
    assert len(GST_STEPS) == 7
    for step in GST_STEPS:
        assert step in prompt
    assert "ABSTAIN" in prompt
    assert "tool_calls" in prompt
    assert "observations" in prompt.lower() or "prior tool" in prompt.lower()
    # Tool schemas stay on the NFC channel — do not dump them into GST PE.
    assert "{available_functions}" not in prompt
    assert "expert in function calling" not in prompt


def test_ensure_gst_system_message_inserts_once_and_preserves_other_system():
    history: list[dict] = [{"role": "user", "content": "hi"}]
    ensure_gst_system_message(history)
    ensure_gst_system_message(history)
    assert history[0]["role"] == "system"
    assert history[0]["content"] == GST_SYSTEM_PROMPT
    assert [m["role"] for m in history] == ["system", "user"]

    with_bfcl_system = [{"role": "system", "content": "BFCL system"}, {"role": "user", "content": "q"}]
    ensure_gst_system_message(with_bfcl_system)
    assert with_bfcl_system[0]["content"] == GST_SYSTEM_PROMPT
    assert with_bfcl_system[1]["content"] == "BFCL system"


def test_extract_and_strip_gst_wrapper():
    text = (
        f"{GST_OPEN}\n"
        "1. weather\n2. pick get_weather\n3. docs\n4. city required\n"
        "5. SF\n6. draft\n7. ok\n"
        f"{GST_CLOSE}\n"
        "ignore me"
    )
    block = extract_gst_block(text)
    assert block is not None
    assert "get_weather" in block
    assert GST_OPEN not in block
    stripped = strip_gst_block(text)
    assert "ignore me" in stripped
    assert GST_OPEN not in stripped
    assert "get_weather" not in stripped


def test_parse_gst_steps_accepts_unlabeled_numbered_list():
    unlabeled = (
        "1. identify get_weather\n"
        "2. pick it\n"
        "3. examine docs\n"
        "4. city is required string\n"
        "5. city=SF\n"
        "6. get_weather(city='SF')\n"
        "7. ABSTAIN — missing nothing, already answered"
    )
    steps = parse_gst_steps(unlabeled)
    assert set(steps) == set(range(1, 8))
    assert "ABSTAIN" in steps[7]
    assert has_gst_scaffold(unlabeled)
    assert extract_gst_block(unlabeled) is not None
    assert not has_gst_scaffold("I will call the weather tool.")
    assert extract_gst_block("I will call the weather tool.") is None


def test_gst_payload_helpers_require_nfc_tools_and_curriculum():
    history = [{"role": "user", "content": "hi"}]
    ensure_gst_system_message(history)
    tools = [{"type": "function", "function": {"name": "ls"}}]
    good = {
        "messages": history,
        "tools": tools,
        "tool_choice": "auto",
    }
    assert gst_payload_has_nfc_tools(good)
    assert gst_payload_has_curriculum(good)
    assert not gst_payload_has_nfc_tools({"messages": history})
    assert not gst_payload_has_curriculum({"messages": [{"role": "user", "content": "hi"}]})


def test_set_fc_mode_accepts_gst_and_rejects_unknown():
    config.set_fc_mode("gst")
    assert config.FC_MODE == config.FC_GST
    assert config.FC_GST in config.FC_MODES
    assert config.FC_GST not in config.FC_JSON_TEXT_MODES
    config.set_fc_mode("nfc")
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("guided")


def test_offline_smoke_script_passes():
    run_offline_smoke()


def test_strip_gst_does_not_eat_tool_call_json():
    """GST drafts must not be treated as the executable call list."""
    content = (
        f"{GST_OPEN}\n"
        '6. draft [{"function_name": "wrong_tool", "parameters": {}}]\n'
        "7. CALL\n"
        f"{GST_CLOSE}"
    )
    leftover = strip_gst_block(content)
    assert leftover == ""
    # Native tool_calls (tested in generate_multi_turn) remain the action.
    native = [
        {
            "id": "c1",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "SF"}),
            },
        }
    ]
    assert native[0]["function"]["name"] == "get_weather"
