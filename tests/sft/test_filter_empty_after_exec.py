# Drop non-terminal IFFC [] after tool-result observations.

from __future__ import annotations

import json
from pathlib import Path

from benchmark.iffc import format_iffc_execution_user_message
from sft.filter_empty_after_exec import (
    KIND_EMPTY,
    KIND_EMPTY_AFTER_EXEC_NONTERMINAL,
    KIND_EMPTY_AFTER_EXEC_TERMINAL,
    KIND_NONEMPTY,
    classify_assistant_turn,
    is_iffc_execution_observation,
    main as filter_main,
    should_drop_assistant_turn,
    unroll_and_filter,
)
from sft.format_iffc import format_toolace_row, unroll_conversation
from sft.train_mlx import prepare_mlx_data


def _call(name: str, **params: object) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _exec_obs(func_string: str, result: object) -> str:
    return format_iffc_execution_user_message([func_string], [result])


def _record(
    example_id: str,
    messages: list[dict],
    *,
    source: str = "toolace",
    multi: bool = True,
    n_tools: int = 1,
) -> dict:
    return {
        "id": example_id,
        "source": source,
        "is_multi_turn": multi,
        "n_tools_in_gold": n_tools,
        "n_tools_bin": "ge2" if n_tools >= 2 else "1",
        "chat_format": "iffc",
        "messages": messages,
    }


def _mid_trajectory_empty_after_exec() -> dict:
    """Tool result → [] → later non-empty call (the Run 12 failure mode)."""
    return _record(
        "toolace-mid",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list then cd"},
            {"role": "assistant", "content": _call("ls")},
            {"role": "user", "content": _exec_obs("ls()", {"files": ["docs"]})},
            {"role": "assistant", "content": "[]"},
            {"role": "user", "content": "now enter docs"},
            {
                "role": "assistant",
                "content": _call("cd", folder="docs"),
            },
        ],
        n_tools=2,
    )


def _terminal_empty_after_exec() -> dict:
    return _record(
        "toolace-terminal",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": _call("ls")},
            {"role": "user", "content": _exec_obs("ls()", {"files": ["a"]})},
            {"role": "assistant", "content": "[]"},
        ],
    )


def _continue_after_exec() -> dict:
    return _record(
        "toolace-continue",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list then cd"},
            {"role": "assistant", "content": _call("ls")},
            {"role": "user", "content": _exec_obs("ls()", {"files": ["docs"]})},
            {"role": "assistant", "content": _call("cd", folder="docs")},
        ],
        n_tools=2,
    )


def _clarifying_empty_then_call() -> dict:
    return _record(
        "toolace-clarify",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "can you help?"},
            {"role": "assistant", "content": "[]"},
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": _call("ls")},
        ],
    )


def _xlam_single_call() -> dict:
    return _record(
        "xlam-1",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "sum 1 to 10"},
            {"role": "assistant", "content": _call("sum", lo=1, hi=10)},
        ],
        source="xlam",
        multi=False,
        n_tools=1,
    )


def _assistant_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def test_execution_observation_matches_iffc_formatter():
    obs = _exec_obs("pwd()", {"cwd": "/root"})
    assert "Function execution results:" in obs
    assert is_iffc_execution_observation(obs)
    assert not is_iffc_execution_observation("list the files please")
    assert not is_iffc_execution_observation("can you help?")


def test_mid_trajectory_empty_after_exec_is_dropped():
    rec = _mid_trajectory_empty_after_exec()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    # user query → ls; exec obs → []; user follow-up → cd
    assert classify_assistant_turn(messages, idxs[0]) == KIND_NONEMPTY
    assert classify_assistant_turn(messages, idxs[1]) == KIND_EMPTY_AFTER_EXEC_NONTERMINAL
    assert classify_assistant_turn(messages, idxs[2]) == KIND_NONEMPTY
    assert should_drop_assistant_turn(messages, idxs[1])
    assert not should_drop_assistant_turn(messages, idxs[0])
    assert not should_drop_assistant_turn(messages, idxs[2])

    raw = unroll_conversation(rec)
    assert len(raw) == 3
    assert raw[1]["messages"][-1]["content"] == "[]"

    kept, stats = unroll_and_filter([rec])
    assert [row["messages"][-1]["content"] for row in kept] == [
        _call("ls"),
        _call("cd", folder="docs"),
    ]
    assert stats.dropped == 1
    assert stats.nonterminal_empty_after_exec_before == 1
    assert stats.nonterminal_empty_after_exec_after == 0
    assert stats.empty_after_exec_after == 0


def test_true_terminal_empty_after_exec_is_kept():
    rec = _terminal_empty_after_exec()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert classify_assistant_turn(messages, idxs[-1]) == KIND_EMPTY_AFTER_EXEC_TERMINAL
    assert not should_drop_assistant_turn(messages, idxs[-1])

    kept, stats = unroll_and_filter([rec])
    assert len(kept) == 2
    assert kept[-1]["messages"][-1]["content"] == "[]"
    assert stats.dropped == 0
    assert stats.empty_after_exec_before == 1
    assert stats.empty_after_exec_after == 1
    assert stats.nonterminal_empty_after_exec_before == 0


def test_nonempty_call_after_exec_is_kept():
    rec = _continue_after_exec()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert classify_assistant_turn(messages, idxs[-1]) == KIND_NONEMPTY
    assert not should_drop_assistant_turn(messages, idxs[-1])

    kept, stats = unroll_and_filter([rec])
    assert len(kept) == 2
    assert json.loads(kept[-1]["messages"][-1]["content"])[0]["function_name"] == "cd"
    assert stats.dropped == 0
    assert stats.empty_targets_before == 0


def test_clarifying_user_empty_turn_is_kept():
    rec = _clarifying_empty_then_call()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert classify_assistant_turn(messages, idxs[0]) == KIND_EMPTY
    assert not should_drop_assistant_turn(messages, idxs[0])

    kept, stats = unroll_and_filter([rec])
    assert len(kept) == 2
    assert kept[0]["messages"][-1]["content"] == "[]"
    assert stats.dropped == 0
    assert stats.empty_targets_after == 1
    assert stats.empty_after_exec_before == 0


def test_terminal_empty_after_exec_then_later_empty_is_kept():
    """Later [] is not a tool call, so the after-exec [] is still terminal."""
    rec = _record(
        "toolace-thanks",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": _call("ls")},
            {"role": "user", "content": _exec_obs("ls()", {"files": ["a"]})},
            {"role": "assistant", "content": "[]"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "[]"},
        ],
    )
    kept, stats = unroll_and_filter([rec])
    assert len(kept) == 3
    assert stats.dropped == 0
    assert stats.empty_after_exec_after == 1


def test_xlam_rows_pass_through_unchanged():
    rec = _xlam_single_call()
    raw = unroll_conversation(rec)
    kept, stats = unroll_and_filter([rec])
    assert [row["messages"] for row in kept] == [row["messages"] for row in raw]
    assert stats.dropped == 0
    assert stats.by_source_before == {"xlam": 1}
    assert stats.by_source_after == {"xlam": 1}
    assert stats.empty_targets_before == 0


def test_stats_by_source_before_after_and_opt_out():
    records = [
        _mid_trajectory_empty_after_exec(),
        _terminal_empty_after_exec(),
        _xlam_single_call(),
    ]
    kept, stats = unroll_and_filter(records)
    d = stats.as_dict()
    assert d["total_before"] == 3 + 2 + 1  # mid 3, terminal 2, xlam 1
    assert d["dropped"] == 1
    assert d["total_after"] == d["total_before"] - 1
    assert d["dropped_by_source"] == {"toolace": 1}
    assert d["empty_after_exec_by_source_before"]["toolace"] == 2  # mid + terminal
    assert d["empty_after_exec_by_source_after"]["toolace"] == 1  # terminal only
    assert d["empty_targets_by_source_before"].get("xlam", 0) == 0
    assert d["nonterminal_empty_after_exec_after"] == 0
    assert len(kept) == d["total_after"]

    kept_v1, stats_v1 = unroll_and_filter(
        records, drop_nonterminal_empty_after_exec=False
    )
    assert len(kept_v1) == d["total_before"]
    assert stats_v1.dropped == 0
    assert stats_v1.nonterminal_empty_after_exec_after == 1


def test_prepare_mlx_data_drops_mid_trajectory_empty_after_exec(
    tmp_path: Path, capsys
):
    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx"
    rec = _mid_trajectory_empty_after_exec()
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    prepare_mlx_data(splits_dir=splits, mlx_dir=mlx)
    train_rows = [
        json.loads(line) for line in (mlx / "train.jsonl").read_text().splitlines() if line
    ]
    assert len(train_rows) == 2
    assert all(row["messages"][-1]["content"] != "[]" for row in train_rows)
    logged = capsys.readouterr().out
    assert "empty-after-exec filter (train)" in logged
    assert "non-terminal 1 → 0" in logged
    assert "toolace:" in logged

    mlx_v1 = tmp_path / "mlx-v1"
    prepare_mlx_data(
        splits_dir=splits,
        mlx_dir=mlx_v1,
        drop_nonterminal_empty_after_exec=False,
    )
    v1_rows = [
        json.loads(line)
        for line in (mlx_v1 / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert len(v1_rows) == 3


def test_filter_cli_writes_unrolled_jsonl_and_prints_stats(tmp_path: Path, capsys):
    src = tmp_path / "train.jsonl"
    out = tmp_path / "unrolled.jsonl"
    src.write_text(json.dumps(_mid_trajectory_empty_after_exec()) + "\n")
    assert filter_main(["--input", str(src), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 2
    logged = capsys.readouterr().out
    assert "dropped 1" in logged
    assert "wrote 2" in logged


def test_formatted_toolace_nl_after_tool_is_terminal_keep_unless_later_call():
    """NL after a tool observation becomes ``[]``; drop only if a later call exists."""
    system = (
        "You are an expert in composing functions. Here is a list of functions "
        "in JSON format that you can invoke:\n"
        + json.dumps(
            [
                {
                    "name": "ls",
                    "description": "list",
                    "parameters": {},
                },
                {
                    "name": "cd",
                    "description": "chdir",
                    "parameters": {
                        "folder": {"type": "string", "required": True},
                    },
                },
            ]
        )
    )
    terminal_row = {
        "row_index": 0,
        "system": system,
        "conversations": [
            {"from": "user", "value": "list files"},
            {"from": "assistant", "value": "[ls()]"},
            {
                "from": "tool",
                "value": json.dumps([{"name": "ls", "results": {"files": ["docs"]}}]),
            },
            {"from": "assistant", "value": "Here are the files."},
        ],
    }
    terminal = format_toolace_row(terminal_row, 0)
    assert terminal is not None
    kept, stats = unroll_and_filter([terminal])
    assert any(row["messages"][-1]["content"] == "[]" for row in kept)
    assert stats.dropped == 0
    assert stats.empty_after_exec_after == 1

    mid_row = {
        "row_index": 1,
        "system": system,
        "conversations": terminal_row["conversations"]
        + [
            {"from": "user", "value": "enter docs"},
            {"from": "assistant", "value": '[cd(folder="docs")]'},
        ],
    }
    mid = format_toolace_row(mid_row, 1)
    assert mid is not None
    kept_mid, stats_mid = unroll_and_filter([mid])
    assert stats_mid.dropped == 1
    assert all(row["messages"][-1]["content"] != "[]" for row in kept_mid)
    names = [
        json.loads(row["messages"][-1]["content"])[0]["function_name"]
        for row in kept_mid
    ]
    assert names == ["ls", "cd"]
