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
    RECIPE_V1,
    RECIPE_V2,
    RECIPE_V3,
    RECIPE_V4,
    RECIPE_V5,
    classify_assistant_turn,
    infer_recipe,
    inject_recovery_stop,
    is_clarifying_empty_after_user,
    is_first_assistant_empty_target,
    is_iffc_execution_observation,
    is_recovery_after_empty,
    main as filter_main,
    recipe_drop_flags,
    recipe_v4_drop_flags,
    recipe_v5_flags,
    resolve_recipe,
    should_drop_assistant_turn,
    unroll_and_filter,
)
from sft.format_iffc import format_toolace_row, unroll_conversation
from sft.split_data import bfcl_leak_reasons
from sft.train_mlx import prepare_mlx_data

# v2 ablation: drop non-terminal empty-after-exec only; no synthetic recovery.
_V2 = dict(
    drop_nonterminal_empty_after_exec=True,
    drop_terminal_empty_after_exec=False,
    synthesize_recovery_stop=False,
)
# v1 ablation: keep every assistant turn.
_V1 = dict(
    drop_nonterminal_empty_after_exec=False,
    drop_terminal_empty_after_exec=False,
    synthesize_recovery_stop=False,
)
# v3 ablation: empty-after-exec + recovery; keep first-turn / clarifying [].
_V3 = dict(
    drop_nonterminal_empty_after_exec=True,
    drop_terminal_empty_after_exec=True,
    drop_first_assistant_empty=False,
    drop_clarifying_empty_after_user=False,
    synthesize_recovery_stop=True,
)
_V3_DROP = dict(
    drop_first_assistant_empty=False,
    drop_clarifying_empty_after_user=False,
)


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
    assert stats.recipe == RECIPE_V5
    assert stats.nonterminal_empty_after_exec_before == 1
    assert stats.nonterminal_empty_after_exec_after == 0
    assert stats.empty_after_exec_after == 0
    assert stats.recovery_after_empty_after == 1
    assert kept[-1].get("recovery_after_empty") is True
    assert json.loads(kept[-1]["messages"][-1]["content"])[0]["function_name"] == "cd"


def test_true_terminal_empty_after_exec_is_kept_on_v2_flag():
    rec = _terminal_empty_after_exec()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert classify_assistant_turn(messages, idxs[-1]) == KIND_EMPTY_AFTER_EXEC_TERMINAL
    assert not should_drop_assistant_turn(
        messages, idxs[-1], drop_terminal_empty_after_exec=False
    )
    assert should_drop_assistant_turn(messages, idxs[-1])  # v4/v3 default

    kept_v2, stats_v2 = unroll_and_filter([rec], **_V2)
    assert len(kept_v2) == 2
    assert kept_v2[-1]["messages"][-1]["content"] == "[]"
    assert stats_v2.dropped == 0
    assert stats_v2.empty_after_exec_before == 1
    assert stats_v2.empty_after_exec_after == 1
    assert stats_v2.nonterminal_empty_after_exec_before == 0
    assert stats_v2.recipe == RECIPE_V2


def test_v3_drops_all_empty_after_exec_including_terminal():
    rec = _terminal_empty_after_exec()
    kept, stats = unroll_and_filter([rec])
    assert [row["messages"][-1]["content"] for row in kept] == [_call("ls")]
    assert stats.dropped == 1
    assert stats.empty_after_exec_before == 1
    assert stats.empty_after_exec_after == 0
    assert stats.terminal_empty_after_exec_after == 0
    assert all(row["messages"][-1]["content"] != "[]" for row in kept)


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


def test_clarifying_user_empty_turn_is_kept_on_v3():
    rec = _clarifying_empty_then_call()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert classify_assistant_turn(messages, idxs[0]) == KIND_EMPTY
    assert is_first_assistant_empty_target(messages, idxs[0])
    assert is_clarifying_empty_after_user(messages, idxs[0])
    assert not should_drop_assistant_turn(messages, idxs[0], **_V3_DROP)
    assert should_drop_assistant_turn(messages, idxs[0])  # v4 default

    kept, stats = unroll_and_filter([rec], **_V3)
    assert len(kept) == 2
    assert kept[0]["messages"][-1]["content"] == "[]"
    assert stats.dropped == 0
    assert stats.empty_targets_after == 1
    assert stats.empty_after_exec_before == 0
    assert stats.recipe == RECIPE_V3
    assert stats.first_turn_empty_after == 1
    assert stats.clarifying_empty_after == 1


def test_terminal_empty_after_exec_then_later_empty_drops_after_exec_keeps_clarify_on_v3():
    """v3 drops the after-exec []; later user→[] clarifying turn is kept."""
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
    kept, stats = unroll_and_filter([rec], **_V3)
    assert [row["messages"][-1]["content"] for row in kept] == [
        _call("ls"),
        "[]",
    ]
    assert stats.dropped == 1
    assert stats.empty_after_exec_after == 0
    assert stats.empty_targets_after == 1
    assert stats.clarifying_empty_after == 1
    assert stats.first_turn_empty_after == 0
    assert stats.recipe == RECIPE_V3

    kept_v2, stats_v2 = unroll_and_filter([rec], **_V2)
    assert len(kept_v2) == 3
    assert stats_v2.dropped == 0
    assert stats_v2.empty_after_exec_after == 1


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
    assert d["recipe"] == RECIPE_V5
    assert d["total_before"] == 3 + 2 + 1  # mid 3, terminal 2, xlam 1
    assert d["dropped"] == 2  # mid [] + terminal []
    assert d["total_after"] == d["total_before"] - 2
    assert d["dropped_by_source"] == {"toolace": 2}
    assert d["empty_after_exec_by_source_before"]["toolace"] == 2  # mid + terminal
    assert d["empty_after_exec_by_source_after"].get("toolace", 0) == 0
    assert d["empty_targets_by_source_before"].get("xlam", 0) == 0
    assert d["nonterminal_empty_after_exec_after"] == 0
    assert d["terminal_empty_after_exec_after"] == 0
    assert d["recovery_after_empty_after"] == 1  # mid-trajectory cd after []
    assert len(kept) == d["total_after"]

    kept_v2, stats_v2 = unroll_and_filter(records, **_V2)
    assert stats_v2.recipe == RECIPE_V2
    assert stats_v2.dropped == 1  # mid only
    assert stats_v2.empty_after_exec_after == 1  # terminal kept
    assert stats_v2.terminal_empty_after_exec_after == 1

    kept_v1, stats_v1 = unroll_and_filter(records, **_V1)
    assert len(kept_v1) == d["total_before"]
    assert stats_v1.dropped == 0
    assert stats_v1.nonterminal_empty_after_exec_after == 1
    assert stats_v1.recipe == RECIPE_V1


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
    assert "recipe=v5" in logged
    assert "non-terminal 1 → 0" in logged
    assert "recovery_after_empty" in logged
    assert "toolace:" in logged

    mlx_v1 = tmp_path / "mlx-v1"
    prepare_mlx_data(
        splits_dir=splits,
        mlx_dir=mlx_v1,
        drop_nonterminal_empty_after_exec=False,
        drop_terminal_empty_after_exec=False,
        synthesize_recovery_stop=False,
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
    kept_v2, stats_v2 = unroll_and_filter([terminal], **_V2)
    assert any(row["messages"][-1]["content"] == "[]" for row in kept_v2)
    assert stats_v2.dropped == 0
    assert stats_v2.empty_after_exec_after == 1

    kept_v3, stats_v3 = unroll_and_filter([terminal])
    assert stats_v3.dropped == 1
    assert stats_v3.empty_after_exec_after == 0
    assert all(row["messages"][-1]["content"] != "[]" for row in kept_v3)

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
    assert stats_mid.recovery_after_empty_after == 1
    assert kept_mid[-1].get("recovery_after_empty") is True


def _exec_then_new_user_then_call() -> dict:
    """Tool results jump straight to a new user question (no assistant [] yet)."""
    return _record(
        "toolace-gap",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list then cd"},
            {"role": "assistant", "content": _call("ls")},
            {"role": "user", "content": _exec_obs("ls()", {"files": ["docs"]})},
            {"role": "user", "content": "now enter docs"},
            {
                "role": "assistant",
                "content": _call("cd", folder="docs"),
            },
        ],
        n_tools=2,
    )


def test_recovery_row_kept_with_nonempty_target():
    rec = _mid_trajectory_empty_after_exec()
    kept, stats = unroll_and_filter([rec])
    recovery = [row for row in kept if row.get("recovery_after_empty")]
    assert len(recovery) == 1
    gold = recovery[0]["messages"][-1]["content"]
    assert gold != "[]"
    assert json.loads(gold)[0]["function_name"] == "cd"
    # History still shows the prior stop, then the new user question.
    assert any(m["role"] == "assistant" and m["content"] == "[]" for m in recovery[0]["messages"][:-1])
    assert recovery[0]["messages"][-2]["role"] == "user"
    assert "Function execution results" not in recovery[0]["messages"][-2]["content"]
    assert stats.recovery_after_empty_before == 1
    assert stats.recovery_after_empty_after == 1
    assert stats.synthetic_recovery_inserted == 0


def test_synthetic_recovery_inserts_stop_in_history_not_as_target():
    rec = _exec_then_new_user_then_call()
    raw = unroll_conversation(rec)
    assert len(raw) == 2
    assert all(row["messages"][-1]["content"] != "[]" for row in raw)
    assert not is_recovery_after_empty(raw[-1]["messages"])

    injected, did = inject_recovery_stop(list(raw[-1]["messages"]))
    assert did
    assert injected[-1]["content"] == _call("cd", folder="docs")
    assert is_recovery_after_empty(injected)
    # Inserted immediately after the exec observation, before the new user.
    exec_idx = next(
        i
        for i, m in enumerate(injected)
        if m["role"] == "user" and is_iffc_execution_observation(m["content"])
    )
    assert injected[exec_idx + 1] == {"role": "assistant", "content": "[]"}
    assert injected[exec_idx + 2]["role"] == "user"
    assert injected[exec_idx + 2]["content"] == "now enter docs"

    kept, stats = unroll_and_filter([rec])
    assert stats.dropped == 0
    assert stats.empty_after_exec_after == 0
    assert stats.synthetic_recovery_inserted == 1
    assert stats.recovery_after_empty_before == 0
    assert stats.recovery_after_empty_after == 1
    targets = [row["messages"][-1]["content"] for row in kept]
    assert "[]" not in targets
    recovery = next(row for row in kept if row.get("recovery_after_empty"))
    assert recovery.get("synthetic_recovery_stop") is True
    assert json.loads(recovery["messages"][-1]["content"])[0]["function_name"] == "cd"
    assert any(
        m["role"] == "assistant" and m["content"] == "[]"
        for m in recovery["messages"][:-1]
    )

    kept_off, stats_off = unroll_and_filter(
        [rec], synthesize_recovery_stop=False
    )
    assert stats_off.synthetic_recovery_inserted == 0
    assert stats_off.recovery_after_empty_after == 0
    assert not any(row.get("synthetic_recovery_stop") for row in kept_off)


def test_synthetic_recovery_does_not_touch_continue_after_exec():
    rec = _continue_after_exec()
    kept, stats = unroll_and_filter([rec])
    assert stats.synthetic_recovery_inserted == 0
    assert stats.recovery_after_empty_after == 0
    assert stats.dropped == 0
    assert [row["messages"][-1]["content"] for row in kept] == [
        _call("ls"),
        _call("cd", folder="docs"),
    ]


def test_clarifying_empty_then_call_counts_as_recovery():
    rec = _clarifying_empty_then_call()
    kept, stats = unroll_and_filter([rec])
    assert stats.empty_after_exec_before == 0
    assert stats.recovery_after_empty_after == 1
    recovery = next(row for row in kept if row.get("recovery_after_empty"))
    assert json.loads(recovery["messages"][-1]["content"])[0]["function_name"] == "ls"
    # v4 drops the [] *target*; the [] remains in the recovery row's history.
    assert stats.dropped == 1
    assert stats.dropped_first_turn_empty == 1
    assert stats.dropped_clarifying_empty == 1
    assert stats.first_turn_empty_after == 0
    assert stats.clarifying_empty_after == 0
    assert all(row["messages"][-1]["content"] != "[]" for row in kept)
    assert any(
        m["role"] == "assistant" and m["content"] == "[]"
        for m in recovery["messages"][:-1]
    )

    kept_v3, stats_v3 = unroll_and_filter([rec], **_V3)
    assert stats_v3.dropped == 0
    assert stats_v3.recovery_after_empty_after == 1
    assert stats_v3.first_turn_empty_after == 1


def test_recipe_flags_v1_v2_v3_v4_v5():
    assert resolve_recipe() == RECIPE_V5
    assert resolve_recipe(keep_terminal_empty_after_exec=True) == RECIPE_V2
    assert resolve_recipe(keep_nonterminal_empty_after_exec=True) == RECIPE_V1
    assert (
        resolve_recipe(
            keep_nonterminal_empty_after_exec=True,
            keep_terminal_empty_after_exec=True,
        )
        == RECIPE_V1
    )
    assert (
        resolve_recipe(
            keep_first_assistant_empty=True,
            keep_clarifying_empty_after_user=True,
        )
        == RECIPE_V3
    )
    assert resolve_recipe(keep_first_assistant_empty=True) == RECIPE_V5
    assert resolve_recipe(keep_clarifying_empty_after_user=True) == RECIPE_V5
    assert (
        resolve_recipe(
            keep_underspecified_turn0=True,
            keep_state_inconsistent=True,
            no_synthetic_complete_turn0=True,
        )
        == RECIPE_V4
    )
    assert resolve_recipe(recipe=RECIPE_V4) == RECIPE_V4
    assert recipe_drop_flags(RECIPE_V5) == (True, True)
    assert recipe_drop_flags(RECIPE_V4) == (True, True)
    assert recipe_drop_flags(RECIPE_V3) == (True, True)
    assert recipe_drop_flags(RECIPE_V2) == (True, False)
    assert recipe_drop_flags(RECIPE_V1) == (False, False)
    assert recipe_v4_drop_flags(RECIPE_V5) == (True, True)
    assert recipe_v4_drop_flags(RECIPE_V4) == (True, True)
    assert recipe_v4_drop_flags(RECIPE_V3) == (False, False)
    assert recipe_v4_drop_flags(RECIPE_V2) == (False, False)
    assert recipe_v5_flags(RECIPE_V5) == (True, True, True, True)
    assert recipe_v5_flags(RECIPE_V4) == (False, False, False, False)
    assert infer_recipe(
        drop_nonterminal_empty_after_exec=True,
        drop_terminal_empty_after_exec=True,
    ) == RECIPE_V5
    assert infer_recipe(
        drop_nonterminal_empty_after_exec=True,
        drop_terminal_empty_after_exec=True,
        drop_first_assistant_empty=False,
        drop_clarifying_empty_after_user=False,
    ) == RECIPE_V3
    assert infer_recipe(
        drop_nonterminal_empty_after_exec=True,
        drop_terminal_empty_after_exec=True,
        drop_underspecified_turn0=False,
        drop_state_inconsistent=False,
        synthesize_complete_turn0=False,
    ) == RECIPE_V4


def test_filter_cli_v2_flag_keeps_terminal(tmp_path: Path, capsys):
    src = tmp_path / "train.jsonl"
    out = tmp_path / "unrolled.jsonl"
    src.write_text(json.dumps(_terminal_empty_after_exec()) + "\n")
    assert (
        filter_main(
            [
                "--input",
                str(src),
                "--out",
                str(out),
                "--keep-terminal-empty-after-exec",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 2
    assert rows[-1]["messages"][-1]["content"] == "[]"
    logged = capsys.readouterr().out
    assert "recipe=v2" in logged
    assert "synthetic_recovery=False" in logged


def test_filter_cli_v5_default_drops_terminal(tmp_path: Path, capsys):
    src = tmp_path / "train.jsonl"
    out = tmp_path / "unrolled.jsonl"
    src.write_text(json.dumps(_terminal_empty_after_exec()) + "\n")
    assert filter_main(["--input", str(src), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == _call("ls")
    logged = capsys.readouterr().out
    assert "recipe=v5" in logged
    assert "dropped 1" in logged
    assert "drop_first_assistant_empty=True" in logged
    assert "drop_clarifying_empty_after_user=True" in logged
    assert "drop_underspecified_turn0=True" in logged
    assert "drop_state_inconsistent=True" in logged


def test_unroll_rejects_bfcl_id_leakage():
    leak = _record(
        "multi_turn_base_0",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": _call("ls")},
        ],
    )
    assert bfcl_leak_reasons(leak)
    try:
        unroll_and_filter([leak])
    except ValueError as exc:
        assert "BFCL" in str(exc)
    else:
        raise AssertionError("expected BFCL leak to raise")
    kept, stats = unroll_and_filter([leak], check_bfcl_leak=False)
    assert stats.total_after == 1
    assert kept[0]["id"].startswith("multi_turn_base_0")


def test_synthetic_recovery_does_not_inject_bfcl_prompts():
    rec = _exec_then_new_user_then_call()
    kept, _stats = unroll_and_filter([rec])
    recovery = next(row for row in kept if row.get("synthetic_recovery_stop"))
    assert not bfcl_leak_reasons(recovery)
    assert "multi_turn_base" not in recovery["id"]
    assert "gorillafilesystem" not in json.dumps(recovery["messages"]).lower()
    for msg in recovery["messages"]:
        assert "tool_calls" not in msg


def test_train_mlx_prepare_only_defaults_to_v5(tmp_path: Path, capsys):
    from sft.train_mlx import main as train_main

    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx"
    rec = _terminal_empty_after_exec()
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    assert train_main(
        [
            "--prepare-only",
            "--splits-dir",
            str(splits),
            "--mlx-dir",
            str(mlx),
        ]
    ) == 0
    rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == _call("ls")
    logged = capsys.readouterr().out
    assert "recipe=v5" in logged
    assert "drop_terminal_empty_after_exec=True" in logged
    assert "drop_first_assistant_empty=True" in logged
    assert "drop_clarifying_empty_after_user=True" in logged
    assert "drop_underspecified_turn0=True" in logged


def test_train_mlx_keep_terminal_flag_is_v2(tmp_path: Path, capsys):
    from sft.train_mlx import main as train_main

    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx"
    rec = _terminal_empty_after_exec()
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    assert train_main(
        [
            "--prepare-only",
            "--keep-terminal-empty-after-exec",
            "--splits-dir",
            str(splits),
            "--mlx-dir",
            str(mlx),
        ]
    ) == 0
    rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert len(rows) == 2
    assert rows[-1]["messages"][-1]["content"] == "[]"
    logged = capsys.readouterr().out
    assert "recipe=v2" in logged


def _first_assistant_empty_after_system() -> dict:
    """Turn-0 [] whose previous message is not a user (not clarifying)."""
    return _record(
        "toolace-first-sys",
        [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "[]"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": _call("ls")},
        ],
    )


def _later_clarifying_empty() -> dict:
    """Later user→[] chitchat; first assistant is a nonempty call."""
    return _record(
        "toolace-later-chitchat",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": _call("ls")},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "[]"},
        ],
    )


def test_v4_drops_first_assistant_empty_target():
    rec = _clarifying_empty_then_call()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert is_first_assistant_empty_target(messages, idxs[0])
    assert not is_first_assistant_empty_target(messages, idxs[1])
    kept, stats = unroll_and_filter([rec])
    targets = [row["messages"][-1]["content"] for row in kept]
    assert "[]" not in targets
    assert targets == [_call("ls")]
    assert stats.recipe == RECIPE_V5
    assert stats.first_turn_empty_before == 1
    assert stats.first_turn_empty_after == 0
    assert stats.dropped_first_turn_empty == 1
    assert stats.empty_after_exec_after == 0
    assert stats.recovery_after_empty_after == 1


def test_v4_drops_later_clarifying_empty_after_user():
    rec = _later_clarifying_empty()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert not is_first_assistant_empty_target(messages, idxs[-1])
    assert is_clarifying_empty_after_user(messages, idxs[-1])
    assert should_drop_assistant_turn(messages, idxs[-1])
    assert not should_drop_assistant_turn(messages, idxs[-1], **_V3_DROP)

    kept, stats = unroll_and_filter([rec])
    assert [row["messages"][-1]["content"] for row in kept] == [_call("ls")]
    assert stats.dropped == 1
    assert stats.dropped_clarifying_empty == 1
    assert stats.dropped_first_turn_empty == 0
    assert stats.clarifying_empty_before == 1
    assert stats.clarifying_empty_after == 0
    assert stats.first_turn_empty_before == 0
    assert stats.empty_after_exec_after == 0

    kept_v3, stats_v3 = unroll_and_filter([rec], **_V3)
    assert [row["messages"][-1]["content"] for row in kept_v3] == [
        _call("ls"),
        "[]",
    ]
    assert stats_v3.dropped == 0
    assert stats_v3.clarifying_empty_after == 1


def test_v4_first_turn_flag_alone_does_not_keep_clarifying_overlap():
    """First-turn [] after a user is also clarifying; one keep-flag is not v3."""
    rec = _clarifying_empty_then_call()
    kept_first, stats_first = unroll_and_filter(
        [rec], drop_first_assistant_empty=False
    )
    assert stats_first.recipe == RECIPE_V5
    assert all(row["messages"][-1]["content"] != "[]" for row in kept_first)
    assert stats_first.dropped_clarifying_empty == 1
    assert stats_first.recovery_after_empty_after == 1

    kept_clarify, stats_clarify = unroll_and_filter(
        [rec], drop_clarifying_empty_after_user=False
    )
    assert stats_clarify.recipe == RECIPE_V5
    assert all(row["messages"][-1]["content"] != "[]" for row in kept_clarify)
    assert stats_clarify.dropped_first_turn_empty == 1


def test_v4_keep_first_only_keeps_non_clarifying_first_empty():
    rec = _first_assistant_empty_after_system()
    messages = rec["messages"]
    idxs = _assistant_indices(messages)
    assert is_first_assistant_empty_target(messages, idxs[0])
    assert not is_clarifying_empty_after_user(messages, idxs[0])

    kept_v4, stats_v4 = unroll_and_filter([rec])
    assert stats_v4.dropped_first_turn_empty == 1
    assert stats_v4.dropped_clarifying_empty == 0
    assert all(row["messages"][-1]["content"] != "[]" for row in kept_v4)
    assert stats_v4.recovery_after_empty_after == 1

    kept, stats = unroll_and_filter([rec], drop_first_assistant_empty=False)
    assert any(row["messages"][-1]["content"] == "[]" for row in kept)
    assert stats.dropped_first_turn_empty == 0
    assert stats.first_turn_empty_after == 1
    assert stats.recovery_after_empty_after == 1


def test_v4_does_not_weaken_recovery_or_empty_after_exec():
    records = [
        _mid_trajectory_empty_after_exec(),
        _terminal_empty_after_exec(),
        _clarifying_empty_then_call(),
        _exec_then_new_user_then_call(),
        _later_clarifying_empty(),
    ]
    kept, stats = unroll_and_filter(records)
    assert stats.empty_after_exec_after == 0
    assert stats.first_turn_empty_after == 0
    assert stats.clarifying_empty_after == 0
    assert stats.recovery_after_empty_after >= 1
    assert stats.synthetic_recovery_inserted == 1
    assert all(row["messages"][-1]["content"] != "[]" for row in kept)
    # Mid-trajectory cd after [], clarifying→ls, and synthetic gap recovery.
    assert stats.recovery_after_empty_after == 3
    d = stats.as_dict()
    assert d["dropped_first_turn_empty"] >= 1
    assert d["dropped_clarifying_empty"] >= 1
    assert d["empty_after_exec_after"] == 0


def test_v4_prepare_logs_first_turn_and_clarifying_counts(tmp_path: Path, capsys):
    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx"
    recs = [
        _clarifying_empty_then_call(),
        _later_clarifying_empty(),
        _mid_trajectory_empty_after_exec(),
    ]
    splits.mkdir()
    (splits / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs)
    )
    (splits / "valid.jsonl").write_text(json.dumps(recs[0]) + "\n")
    prepare_mlx_data(splits_dir=splits, mlx_dir=mlx)
    train_rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert all(row["messages"][-1]["content"] != "[]" for row in train_rows)
    logged = capsys.readouterr().out
    assert "recipe=v5" in logged
    assert "first_turn_empty:" in logged
    assert "clarifying_empty:" in logged
    assert "recovery_after_empty:" in logged
    assert "empty-after-exec:" in logged


def test_filter_cli_v3_ablation_keeps_first_and_clarifying(tmp_path: Path, capsys):
    src = tmp_path / "train.jsonl"
    out = tmp_path / "unrolled.jsonl"
    src.write_text(json.dumps(_clarifying_empty_then_call()) + "\n")
    assert (
        filter_main(
            [
                "--input",
                str(src),
                "--out",
                str(out),
                "--keep-first-assistant-empty",
                "--keep-clarifying-empty-after-user",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 2
    assert rows[0]["messages"][-1]["content"] == "[]"
    logged = capsys.readouterr().out
    assert "recipe=v3" in logged
    assert "drop_first_assistant_empty=False" in logged
    assert "drop_clarifying_empty_after_user=False" in logged
    assert "synthetic_recovery=True" in logged


def test_train_mlx_v3_ablation_flags(tmp_path: Path, capsys):
    from sft.train_mlx import main as train_main

    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx"
    rec = _clarifying_empty_then_call()
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    assert (
        train_main(
            [
                "--prepare-only",
                "--keep-first-assistant-empty",
                "--keep-clarifying-empty-after-user",
                "--splits-dir",
                str(splits),
                "--mlx-dir",
                str(mlx),
            ]
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert len(rows) == 2
    assert rows[0]["messages"][-1]["content"] == "[]"
    logged = capsys.readouterr().out
    assert "recipe=v3" in logged
    assert "drop_first_assistant_empty=False" in logged
