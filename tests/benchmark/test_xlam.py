"""xLAM JSON-in-text adapter (prompt + fc-mode wiring; parser lives in iffc)."""

import pytest

from benchmark import config
from benchmark.xlam import (
    XLAM_SYSTEM_PROMPT,
    build_xlam_system_prompt,
    ensure_xlam_system_message,
    format_xlam_execution_user_message,
)


def test_system_prompt_asks_for_tool_calls_json_not_iffc_list():
    assert "{available_functions}" in XLAM_SYSTEM_PROMPT
    assert '"tool_calls"' in XLAM_SYSTEM_PROMPT
    assert '"name"' in XLAM_SYSTEM_PROMPT
    assert '"arguments"' in XLAM_SYSTEM_PROMPT
    assert "expert in composing functions" in XLAM_SYSTEM_PROMPT
    assert "function_name" not in XLAM_SYSTEM_PROMPT
    assert "refuse to answer" not in XLAM_SYSTEM_PROMPT.lower()
    filled = build_xlam_system_prompt(
        [{"name": "mv", "description": "move a file", "parameters": {"type": "object"}}]
    )
    assert "{available_functions}" not in filled
    assert '"name": "mv"' in filled
    assert "expert in function calling" not in filled


def test_ensure_xlam_system_message_inserts_once_and_refreshes_on_tool_change():
    history: list[dict] = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "ls"}}]
    ensure_xlam_system_message(history, tools)
    ensure_xlam_system_message(history, tools)
    assert history[0]["role"] == "system"
    assert history[0]["content"] == build_xlam_system_prompt(tools)
    assert [m["role"] for m in history] == ["system", "user"]

    more = tools + [{"type": "function", "function": {"name": "cd"}}]
    ensure_xlam_system_message(history, more)
    assert '"name": "cd"' in history[0]["content"]
    assert [m["role"] for m in history] == ["system", "user"]


def test_format_execution_user_message_asks_for_xlam_json():
    msg = format_xlam_execution_user_message(
        ["cd(folder='docs')"],
        ["ok"],
    )
    assert "cd(folder='docs') -> ok" in msg
    assert '"tool_calls"' in msg
    assert "function_name" not in msg


def test_set_fc_mode_accepts_xlam_as_json_text_not_nfc():
    config.set_fc_mode("xlam")
    assert config.FC_MODE == config.FC_XLAM
    assert config.FC_XLAM in config.FC_MODES
    assert config.FC_XLAM in config.FC_JSON_TEXT_MODES
    assert config.FC_XLAM not in config.FC_IFFC_STATE_MODES
    assert config.FC_NFC not in config.FC_JSON_TEXT_MODES
    config.set_fc_mode("nfc")
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("xlam-nfc")
