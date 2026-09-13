import json

import pytest

from benchmark import config
from benchmark.iffc import IFFC_SYSTEM_PROMPT, format_iffc_execution_user_message
from benchmark.iffc_state import (
    KEEP_RECENT_TURNS,
    STATE_ADDENDUM,
    STATE_CLOSE,
    STATE_OPEN,
    StateTracker,
    attach_state_block,
    build_iffc_state_system_prompt,
    compress_older_dumps,
    ensure_iffc_state_system_message,
    iffc_state_payload_has_state,
    iffc_state_payload_omits_tools,
    is_execution_dump,
    is_user_turn_start,
    parse_func_string,
    record_and_format,
)
from benchmark.smoke_iffc_state import run_offline_smoke


def test_system_prompt_is_iffc_plus_state_not_gst():
    prompt = build_iffc_state_system_prompt(
        [{"name": "cd", "description": "change directory", "parameters": {"type": "object"}}]
    )
    assert "expert in function calling" in prompt
    assert "{available_functions}" not in prompt
    assert '"name": "cd"' in prompt
    assert STATE_ADDENDUM in prompt
    assert STATE_OPEN in prompt
    assert "[GST]" not in prompt
    assert "Curriculum:" not in prompt
    assert IFFC_SYSTEM_PROMPT.split("{available_functions}")[0] in prompt


def test_ensure_iffc_state_system_message_inserts_once_and_refreshes():
    history: list[dict] = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "cd"}}]
    ensure_iffc_state_system_message(history, tools)
    ensure_iffc_state_system_message(history, tools)
    assert history[0]["role"] == "system"
    assert history[0]["content"].count("expert in function calling") == 1
    assert [m["role"] for m in history] == ["system", "user"]

    more = [{"type": "function", "function": {"name": "pwd"}}]
    ensure_iffc_state_system_message(history, tools + more)
    assert "pwd" in history[0]["content"]
    assert "cd" in history[0]["content"]


def test_parse_func_string_kwargs():
    name, args = parse_func_string("cd(folder='docs')")
    assert name == "cd"
    assert args == {"folder": "docs"}
    name, args = parse_func_string("echo(content='hello', file_name='notes.txt')")
    assert name == "echo"
    assert args["file_name"] == "notes.txt"
    name, args = parse_func_string("ls()")
    assert name == "ls"
    assert args == {}


def test_state_construction_from_gfs_and_api_observations():
    tracker = StateTracker()
    tracker.update(
        ["cd(folder='docs')"],
        [{"current_working_directory": "docs"}],
    )
    tracker.update(
        ["ls()"],
        [{"current_directory_content": ["notes.txt", "todo.md"]}],
    )
    tracker.update(
        ["echo(content='hi', file_name='notes.txt')"],
        [None],
    )
    tracker.update(
        ["startEngine(ignitionMode='START')"],
        [{"engineOn": True, "fuelLevel": 12.5, "headlightStatus": "off"}],
    )
    tracker.update(
        ["fillFuelTank(fuelAmount=3)"],
        [{"fuelLevel": 15.5}],
    )
    block = tracker.render()
    assert STATE_OPEN in block and STATE_CLOSE in block
    assert "cwd:" in block
    assert "docs" in block
    assert "notes.txt" in block
    assert "engineOn=True" in block or "engineOn=true" in block
    assert "fuelLevel=15.5" in block
    assert "headlightStatus=off" in block
    assert "[GST]" not in block

    tracker.update(
        ["cd(folder='missing')"],
        [{"error": "cd: missing: No such file or directory"}],
    )
    err_block = tracker.render()
    assert "error:" in err_block
    assert "docs" in err_block  # cwd unchanged on error


def test_state_ignores_model_draft_and_uses_execution_only():
    tracker = StateTracker()
    tracker.update(
        ["cd(folder='docs')"],
        [{"current_working_directory": "docs"}],
    )
    # A GST-like draft must not become STATE; only exec results do.
    assert "Identify which function" not in tracker.render()
    assert tracker.cwd.endswith("docs")


def test_attach_state_keeps_iffc_execution_shape():
    tracker = StateTracker()
    tracker.update(["pwd()"], [{"current_working_directory": "/root"}])
    base = format_iffc_execution_user_message(["pwd()"], [{"current_working_directory": "/root"}])
    attached = attach_state_block(base, tracker)
    assert attached.startswith("Function execution results:")
    assert "pwd()" in attached
    assert STATE_OPEN in attached
    assert "Respond with a JSON list" in attached
    assert attached.index(STATE_OPEN) < attached.index("Respond with a JSON list")

    recorded = record_and_format(
        StateTracker(),
        ["ls()"],
        [{"current_directory_content": ["a.txt"]}],
    )
    assert "Function execution results:" in recorded
    assert STATE_OPEN in recorded
    assert "a.txt" in recorded


def test_compress_older_dumps_keeps_last_two_turns():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q0"},
        {"role": "assistant", "content": "[]"},
        {"role": "user", "content": "Function execution results:\n- cd() -> dump0"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "[]"},
        {"role": "user", "content": "Function execution results:\n- echo() -> dump1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "[]"},
        {"role": "user", "content": "Function execution results:\n- pwd() -> dump2"},
    ]
    compress_older_dumps(history, keep_recent=KEEP_RECENT_TURNS)
    blob = json.dumps(history)
    assert "Q0" in blob and "dump0" not in blob
    assert "dump1" in blob and "dump2" in blob
    assert history[0]["content"] == "sys"
    assert any(m.get("content") == "Q0" for m in history)
    dumps = [m for m in history if is_execution_dump(m)]
    assert len(dumps) == 2


def test_turn_start_ignores_dumps_and_retries():
    assert is_user_turn_start({"role": "user", "content": "Open docs"})
    assert not is_user_turn_start(
        {"role": "user", "content": "Function execution results:\n- x -> y"}
    )
    assert not is_user_turn_start({"role": "assistant", "content": "[]"})
    assert is_execution_dump(
        {"role": "user", "content": f"{STATE_OPEN}\ncwd: /\n{STATE_CLOSE}"}
    )


def test_payload_helpers_require_no_native_tools_and_state():
    history = [{"role": "user", "content": "hi"}]
    ensure_iffc_state_system_message(
        history, [{"type": "function", "function": {"name": "ls"}}]
    )
    history.append(
        {
            "role": "user",
            "content": "Function execution results:\n- ls() -> ok\n\n[STATE]\ncwd: /\n[/STATE]",
        }
    )
    good = {"messages": history, "temperature": 0.0}
    assert iffc_state_payload_omits_tools(good)
    assert iffc_state_payload_has_state(good)
    nfc = {"messages": history, "tools": [{}], "tool_choice": "auto"}
    assert not iffc_state_payload_omits_tools(nfc)


def test_set_fc_mode_accepts_iffc_state_and_rejects_unknown():
    config.set_fc_mode("iffc-state")
    assert config.FC_MODE == config.FC_IFFC_STATE
    assert config.FC_IFFC_STATE in config.FC_MODES
    assert config.FC_IFFC_STATE in config.FC_JSON_TEXT_MODES
    assert config.FC_GST not in config.FC_JSON_TEXT_MODES
    config.set_fc_mode("nfc")
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("gst-state")


def test_offline_smoke_script_passes():
    run_offline_smoke()
