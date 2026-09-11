from pathlib import Path

import pytest

from vfs_shell.bfcl_fs import to_bfcl_func_string
from vfs_shell.bfcl_fs_eval import (
    flatten_recorded_emissions,
    format_gfs_observation,
    generate_shell_for_entry,
    interpret_shell_step,
    load_jsonl,
    score_shell_results,
)
from vfs_shell.models import ScriptedModel
from vfs_shell.parser import parse_command
from vfs_shell.slice import filter_slice
from vfs_shell.types import ToolCall

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_cli_dry_run_fixture(capsys):
    from run_bfcl_fs import main

    rc = main([
        "--slice",
        "fs-only",
        "--dry-run",
        "--data",
        str(FIXTURES / "slice_entries.jsonl"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n=1" in out
    assert "fs_toy_0" in out
    assert "fs_mixed_0" not in out


def test_slice_filter_on_fixture():
    entries = load_jsonl(FIXTURES / "slice_entries.jsonl")
    assert [e["id"] for e in filter_slice(entries, "fs-only")] == ["fs_toy_0"]
    assert [e["id"] for e in filter_slice(entries, "any-fs")] == [
        "fs_toy_0",
        "fs_mixed_0",
    ]


def test_interpret_maps_and_ends_turn():
    action, funcs, err = interpret_shell_step("ls -a")
    assert action == "execute" and funcs == ["ls(a=True)"] and err is None
    action, funcs, err = interpret_shell_step("done")
    assert action == "end_turn"
    action, funcs, err = interpret_shell_step("All set, the file is moved.")
    assert action == "end_turn"
    action, funcs, err = interpret_shell_step("mkdir -p archive")
    assert action == "repair" and err and "-p" in err
    action, funcs, err = interpret_shell_step("cat foo | grep x")
    assert action == "repair"


def test_maps_gfs_tail_touch():
    call = parse_command("tail -n 20 log.txt")
    assert isinstance(call, ToolCall)
    assert to_bfcl_func_string(call) == "tail(file_name='log.txt', lines=20)"
    call = parse_command("touch notes.txt")
    assert isinstance(call, ToolCall)
    assert to_bfcl_func_string(call) == "touch(file_name='notes.txt')"


def test_recorded_loop_shape_with_mock_execute():
    executed: list[list[str]] = []

    def fake_execute(func_strings, entry, category, exec_model_name):
        executed.append(list(func_strings))
        return ["ok"] * len(func_strings)

    entry = {
        "id": "fs_toy_0",
        "involved_classes": ["GorillaFileSystem"],
        "initial_config": {},
        "question": [
            [{"role": "user", "content": "list then stop"}],
        ],
    }
    model = ScriptedModel(flatten_recorded_emissions([["ls -a"]]))
    seen: list[str] = []

    def generate(messages):
        if messages:
            seen.append(messages[-1].get("content") or "")
        return model.generate(messages)

    turns, timing = generate_shell_for_entry(
        entry, "multi_turn_base", generate, fake_execute
    )
    assert turns == [[["ls(a=True)"]]]
    assert executed == [["ls(a=True)"]]
    assert timing["shell_emissions"] == [[["ls -a"]]]
    assert any("ls -a ->" in c for c in seen)
    assert not any("ls(a=True)" in c for c in seen)


def test_observation_uses_shell_not_bfcl_form():
    obs = format_gfs_observation(["ls(a=True)"], ["file.txt\nnotes"])
    assert "- ls -a ->" in obs
    assert "ls(a=True)" not in obs
    obs = format_gfs_observation(
        ["echo(content='buy milk', file_name='todo.txt')"],
        ["ok"],
    )
    assert "- echo buy milk > todo.txt ->" in obs


def test_real_fs_slice_counts():
    bfcl_eval = pytest.importorskip("bfcl_eval")
    from pathlib import Path

    data = Path(bfcl_eval.__file__).parent / "data" / "BFCL_v4_multi_turn_base.json"
    entries = load_jsonl(data)
    assert len(filter_slice(entries, "fs-only")) == 13
    assert len(filter_slice(entries, "any-fs")) == 50


def test_recorded_oracle_passes_checker_on_one_fs_entry():
    pytest.importorskip("bfcl_eval")
    from pathlib import Path

    import bfcl_eval

    from vfs_shell.bfcl_fs_eval import bfcl_execute, load_jsonl_map

    data = Path(bfcl_eval.__file__).parent / "data" / "BFCL_v4_multi_turn_base.json"
    answers_path = (
        Path(bfcl_eval.__file__).parent
        / "data"
        / "possible_answer"
        / "BFCL_v4_multi_turn_base.json"
    )
    entries = {e["id"]: e for e in load_jsonl(data)}
    entry = entries["multi_turn_base_1"]
    rec = load_jsonl_map(FIXTURES / "recorded_multi_turn_base_1.jsonl")[entry["id"]]
    model = ScriptedModel(flatten_recorded_emissions(rec["emissions"]))
    turn_results, timing = generate_shell_for_entry(
        entry,
        "multi_turn_base",
        model.generate,
        bfcl_execute,
        exec_model_name="shell_fs_oracle_test",
    )
    assert timing["shell_emissions"][0] == [["ls -a"]]
    assert timing["shell_emissions"][1] == [["cd workspace"], ["mv log.txt archive"]]
    summary = score_shell_results(
        category="multi_turn_base",
        test_entries={entry["id"]: entry},
        answers=load_jsonl_map(answers_path),
        results=[{"id": entry["id"], "result": turn_results}],
        model_name="shell_fs_oracle_test",
    )
    assert summary["total"] == 1
    assert summary["correct"] == 1, summary["errors"]
