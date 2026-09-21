# SFT v5: turn-0 completeness + env-state consistency (on top of v4).

from __future__ import annotations

import json
from pathlib import Path

from benchmark.iffc import build_iffc_system_prompt, format_iffc_execution_user_message
from sft.curriculum_v5 import (
    independently_knowable_later_calls,
    is_incomplete_turn0,
    is_state_consistent_continue,
    is_state_inconsistent_continue,
    is_underspecified_turn0,
    plan_turn0,
    tools_mentioned_in_text,
)
from sft.filter_empty_after_exec import (
    RECIPE_V4,
    RECIPE_V5,
    compare_recipe_unroll,
    flags_for_recipe,
    main as filter_main,
    unroll_and_filter,
)
from sft.format_iffc import dumps_iffc_calls, format_xlam_row, parse_iffc_response


def _call(name: str, **params: object) -> str:
    return dumps_iffc_calls([{"function_name": name, "parameters": params}])


def _exec_obs(func_string: str, result: object) -> str:
    return format_iffc_execution_user_message([func_string], [result])


def _gfs_system() -> str:
    return build_iffc_system_prompt(
        [
            {"name": "ls", "description": "list", "parameters": {"type": "object", "properties": {}}},
            {
                "name": "cd",
                "description": "chdir",
                "parameters": {
                    "type": "object",
                    "properties": {"folder": {"type": "string"}},
                    "required": ["folder"],
                },
            },
            {
                "name": "cp",
                "description": "copy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                    },
                },
            },
        ]
    )


def _record(example_id: str, messages: list[dict], *, source: str = "toolace") -> dict:
    return {
        "id": example_id,
        "source": source,
        "is_multi_turn": source == "toolace",
        "n_tools_in_gold": 1,
        "n_tools_bin": "1",
        "chat_format": "iffc",
        "messages": messages,
    }


def _incomplete_first_episode() -> dict:
    """ls then cp(a.txt → backup); user already named both paths (under-call)."""
    return _record(
        "toolace-incomplete",
        [
            {"role": "system", "content": _gfs_system()},
            {
                "role": "user",
                "content": "list files then copy a.txt to backup",
            },
            {"role": "assistant", "content": _call("ls")},
            {
                "role": "user",
                "content": _exec_obs("ls()", {"files": ["a.txt", "b.txt", "backup"]}),
            },
            {
                "role": "assistant",
                "content": _call("cp", src="a.txt", dst="backup"),
            },
        ],
    )


def _state_inconsistent_cd() -> dict:
    """User asked to enter docs; listing does not contain docs."""
    return _record(
        "toolace-mismatch",
        [
            {"role": "system", "content": _gfs_system()},
            {"role": "user", "content": "list files then enter docs"},
            {"role": "assistant", "content": _call("ls")},
            {
                "role": "user",
                "content": _exec_obs("ls()", {"files": ["src", "lib"]}),
            },
            {"role": "assistant", "content": _call("cd", folder="docs")},
        ],
    )


def _state_consistent_cd() -> dict:
    """User asked to enter docs; listing confirms docs."""
    return _record(
        "toolace-consistent",
        [
            {"role": "system", "content": _gfs_system()},
            {"role": "user", "content": "list files then enter docs"},
            {"role": "assistant", "content": _call("ls")},
            {
                "role": "user",
                "content": _exec_obs("ls()", {"files": ["docs", "src"]}),
            },
            {"role": "assistant", "content": _call("cd", folder="docs")},
        ],
    )


def _xlam_underspecified() -> dict:
    rec = format_xlam_row(
        {
            "id": "under-1",
            "query": "Sum the multiples and take the product of primes.",
            "tools": [
                {
                    "name": "math_toolkit.sum_of_multiples",
                    "description": "sum",
                    "parameters": {
                        "lower_limit": {"type": "int", "required": True},
                        "upper_limit": {"type": "int", "required": True},
                    },
                },
                {
                    "name": "math_toolkit.product_of_primes",
                    "description": "prod",
                    "parameters": {"count": {"type": "int", "required": True}},
                },
            ],
            "answers": [
                {
                    "name": "math_toolkit.sum_of_multiples",
                    "arguments": {"lower_limit": 1, "upper_limit": 10},
                }
            ],
        },
        0,
    )
    assert rec is not None
    return rec


def _xlam_complete_parallel() -> dict:
    rec = format_xlam_row(
        {
            "id": "par-1",
            "query": "Sum the multiples and take the product of primes.",
            "tools": [
                {
                    "name": "math_toolkit.sum_of_multiples",
                    "description": "sum",
                    "parameters": {
                        "lower_limit": {"type": "int", "required": True},
                        "upper_limit": {"type": "int", "required": True},
                    },
                },
                {
                    "name": "math_toolkit.product_of_primes",
                    "description": "prod",
                    "parameters": {"count": {"type": "int", "required": True}},
                },
            ],
            "answers": [
                {
                    "name": "math_toolkit.sum_of_multiples",
                    "arguments": {"lower_limit": 1, "upper_limit": 10},
                },
                {
                    "name": "math_toolkit.product_of_primes",
                    "arguments": {"count": 5},
                },
            ],
        },
        0,
    )
    assert rec is not None
    return rec


def _recovery_after_empty() -> dict:
    return _record(
        "toolace-recover",
        [
            {"role": "system", "content": _gfs_system()},
            {"role": "user", "content": "can you help?"},
            {"role": "assistant", "content": "[]"},
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": _call("ls")},
        ],
    )


def test_tools_mentioned_matches_distinctive_tokens():
    names = [
        "math_toolkit.sum_of_multiples",
        "math_toolkit.product_of_primes",
    ]
    hit = tools_mentioned_in_text(
        "Sum the multiples and take the product of primes.", names
    )
    assert set(hit) == set(names)


def test_xlam_underspecified_single_call_is_dropped_on_v5():
    rec = _xlam_underspecified()
    idx = len(rec["messages"]) - 1
    assert is_underspecified_turn0(rec["messages"], idx)
    kept_v5, stats_v5 = unroll_and_filter([rec])
    assert stats_v5.recipe == RECIPE_V5
    assert stats_v5.dropped_underspecified_turn0 == 1
    assert kept_v5 == []

    kept_v4, stats_v4 = unroll_and_filter([rec], **flags_for_recipe(RECIPE_V4))
    assert stats_v4.recipe == RECIPE_V4
    assert stats_v4.dropped_underspecified_turn0 == 0
    assert len(kept_v4) == 1
    assert json.loads(kept_v4[0]["messages"][-1]["content"])[0][
        "function_name"
    ] == "math_toolkit.sum_of_multiples"


def test_xlam_parallel_gold_is_kept():
    rec = _xlam_complete_parallel()
    idx = len(rec["messages"]) - 1
    assert not is_underspecified_turn0(rec["messages"], idx)
    kept, stats = unroll_and_filter([rec])
    assert stats.dropped_underspecified_turn0 == 0
    assert len(kept) == 1
    names = [
        c["function_name"]
        for c in parse_iffc_response(kept[0]["messages"][-1]["content"])
    ]
    assert names == [
        "math_toolkit.sum_of_multiples",
        "math_toolkit.product_of_primes",
    ]


def test_incomplete_turn0_is_rewritten_to_complete_call_set():
    rec = _incomplete_first_episode()
    messages = rec["messages"]
    first = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    extras = independently_knowable_later_calls(messages, first)
    assert [c["function_name"] for c in extras] == ["cp"]
    assert is_incomplete_turn0(messages, first)
    plan = plan_turn0(messages)
    assert plan.incomplete
    assert plan.merged_gold is not None

    kept, stats = unroll_and_filter([rec])
    assert stats.recipe == RECIPE_V5
    assert stats.synthetic_complete_turn0 == 1
    assert stats.dropped_absorbed_followup == 1
    assert stats.empty_after_exec_after == 0
    golds = [parse_iffc_response(row["messages"][-1]["content"]) for row in kept]
    assert len(golds) == 1
    assert [c["function_name"] for c in golds[0]] == ["ls", "cp"]
    assert kept[0].get("synthetic_complete_turn0") is True
    # Original sequential cp is not a training target (absorbed).
    assert all(
        [c["function_name"] for c in parse_iffc_response(row["messages"][-1]["content"])]
        != ["cp"]
        for row in kept
    )


def test_no_synthetic_complete_drops_incomplete_instead_of_rewriting():
    rec = _incomplete_first_episode()
    kept, stats = unroll_and_filter(
        [rec],
        recipe=RECIPE_V5,
        synthesize_complete_turn0=False,
        drop_underspecified_turn0=True,
        drop_state_inconsistent=True,
        drop_absorbed_independent_followup=False,
    )
    assert stats.synthetic_complete_turn0 == 0
    assert stats.dropped_incomplete_turn0 == 1
    names = [
        parse_iffc_response(row["messages"][-1]["content"])[0]["function_name"]
        for row in kept
    ]
    assert names == ["cp"]  # first-turn ls dropped; sequential cp kept


def test_state_inconsistent_continue_is_dropped_not_merged():
    rec = _state_inconsistent_cd()
    messages = rec["messages"]
    cd_idx = next(
        i
        for i, m in enumerate(messages)
        if m["role"] == "assistant" and "cd" in str(m.get("content"))
    )
    assert is_state_inconsistent_continue(messages, cd_idx)
    assert not independently_knowable_later_calls(messages, 2)
    kept, stats = unroll_and_filter([rec])
    assert stats.dropped_state_inconsistent == 1
    assert stats.synthetic_complete_turn0 == 0
    assert [parse_iffc_response(r["messages"][-1]["content"])[0]["function_name"] for r in kept] == [
        "ls"
    ]


def test_state_consistent_continue_is_kept_and_merged_when_user_named_it():
    rec = _state_consistent_cd()
    messages = rec["messages"]
    cd_idx = next(
        i
        for i, m in enumerate(messages)
        if m["role"] == "assistant" and "cd" in str(m.get("content"))
    )
    assert is_state_consistent_continue(messages, cd_idx)
    assert not is_state_inconsistent_continue(messages, cd_idx)
    extras = independently_knowable_later_calls(messages, 2)
    assert extras and extras[0]["function_name"] == "cd"

    kept, stats = unroll_and_filter([rec])
    assert stats.dropped_state_inconsistent == 0
    assert stats.synthetic_complete_turn0 == 1
    assert stats.dropped_absorbed_followup == 1
    gold = parse_iffc_response(kept[0]["messages"][-1]["content"])
    assert [c["function_name"] for c in gold] == ["ls", "cd"]


def test_v5_does_not_weaken_recovery_after_empty():
    rec = _recovery_after_empty()
    kept, stats = unroll_and_filter([rec])
    assert stats.empty_after_exec_after == 0
    assert stats.first_turn_empty_after == 0
    assert stats.recovery_after_empty_after == 1
    assert stats.dropped_underspecified_turn0 == 0
    recovery = next(row for row in kept if row.get("recovery_after_empty"))
    assert parse_iffc_response(recovery["messages"][-1]["content"])[0][
        "function_name"
    ] == "ls"
    assert any(
        m["role"] == "assistant" and m["content"] == "[]"
        for m in recovery["messages"][:-1]
    )


def test_v5_vs_v4_compare_counts():
    records = [
        _xlam_underspecified(),
        _incomplete_first_episode(),
        _state_inconsistent_cd(),
        _recovery_after_empty(),
    ]
    delta = compare_recipe_unroll(records, recipe=RECIPE_V5, baseline=RECIPE_V4)
    assert delta["recipe"] == RECIPE_V5
    assert delta["baseline"] == RECIPE_V4
    assert delta["dropped_underspecified_turn0"] == 1
    assert delta["dropped_state_inconsistent"] == 1
    assert delta["synthetic_complete_turn0"] == 1
    assert delta["recovery_after_empty_after"] == delta["baseline_recovery_after_empty_after"]
    assert delta["empty_after_exec_after"] == 0
    assert delta["baseline_empty_after_exec_after"] == 0
    # v5 drops underspecified xLAM + mismatched cd + absorbed follow-up,
    # and rewrites incomplete first-turn (no extra row).
    assert delta["total_after"] < delta["baseline_total_after"]


def test_v4_ablation_keeps_underspecified_and_mismatch():
    records = [_xlam_underspecified(), _state_inconsistent_cd()]
    kept, stats = unroll_and_filter(records, **flags_for_recipe(RECIPE_V4))
    assert stats.recipe == RECIPE_V4
    assert stats.dropped_underspecified_turn0 == 0
    assert stats.dropped_state_inconsistent == 0
    assert stats.synthetic_complete_turn0 == 0
    names = [
        parse_iffc_response(row["messages"][-1]["content"])[0]["function_name"]
        for row in kept
        if parse_iffc_response(row["messages"][-1]["content"])
    ]
    assert "math_toolkit.sum_of_multiples" in names
    assert "cd" in names


def test_filter_cli_recipe_v5_and_compare(tmp_path: Path, capsys):
    src = tmp_path / "train.jsonl"
    out = tmp_path / "unrolled.jsonl"
    recs = [_xlam_underspecified(), _state_inconsistent_cd(), _recovery_after_empty()]
    src.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert (
        filter_main(
            [
                "--input",
                str(src),
                "--out",
                str(out),
                "--recipe",
                "v5",
                "--compare-recipe",
                "v4",
            ]
        )
        == 0
    )
    logged = capsys.readouterr().out
    assert "recipe=v5" in logged
    assert "recipe compare" in logged
    assert "underspecified dropped" in logged
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert all("multi_turn_base" not in row["id"] for row in rows)


def test_train_mlx_prepare_v5_compare_log(tmp_path: Path, capsys):
    from sft.train_mlx import main as train_main

    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx"
    recs = [_incomplete_first_episode(), _recovery_after_empty()]
    splits.mkdir()
    (splits / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (splits / "valid.jsonl").write_text(json.dumps(recs[1]) + "\n")
    assert (
        train_main(
            [
                "--prepare-only",
                "--splits-dir",
                str(splits),
                "--mlx-dir",
                str(mlx),
                "--compare-recipe",
                "v4",
            ]
        )
        == 0
    )
    logged = capsys.readouterr().out
    assert "recipe=v5" in logged
    assert "synthetic_complete_turn0" in logged
    assert "recipe compare (train)" in logged
    train_rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert any(row.get("synthetic_complete_turn0") for row in train_rows)
    assert any(row.get("recovery_after_empty") for row in train_rows)
