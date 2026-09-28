# SFT v8 / v8.1 lever 1: observe→same-call drop, ~35% non-tool cap, loop filter.

from __future__ import annotations

import json
from pathlib import Path

from benchmark.iffc import format_iffc_execution_user_message
from sft.build_mix_v7 import MixAbort
from sft.cap_nontool_v7 import cap_kwargs_for_recipe, cap_nontool_unrolls
from sft.config import (
    SEED_V8,
    SOURCE_APIGEN_MT,
    SPLITS_DIR,
    SPLITS_DIR_V7,
    V8_EXPERIMENT,
    V8_LOOP_MIN_RUN,
    V8_MIN_APIGEN_FRAC,
    V8_NONTOOL_MAX,
    V8_NONTOOL_MIN,
    V8_NONTOOL_TARGET,
    V8_RECIPE_BUMP,
)
from sft.drop_identical_call import (
    REASON_OBSERVE_SAME,
    canonical_tool_calls,
    count_observe_same_call_turns,
    drop_identical_call_unrolls,
    drop_reason_for_gold,
    is_identical_to_previous_assistant_call,
    is_observe_same_call,
    max_identical_call_run,
    max_observe_same_call_run,
)
from sft.filter_empty_after_exec import (
    RECIPE_V4,
    RECIPE_V7,
    RECIPE_V8,
    flags_for_recipe,
    recipe_v4_drop_flags,
    recipe_v5_flags,
    resolve_recipe,
    unroll_and_filter,
)
from sft.format_apigen_mt import STOP_TOOL_CALL, label_assistant_content
from sft.mix_stats_v7 import gate_reasons, scan_unrolled
from sft.mix_stats_v8 import collect_stats as collect_stats_v8
from sft.student_filter_v8 import (
    LoopFilterAbort,
    filter_loop_trajectories,
    is_loop_trajectory,
)


def _call(name: str = "ls", **params: object) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _exec_obs(func_string: str = "ls()", result: object | None = None) -> str:
    return format_iffc_execution_user_message(
        [func_string], [result if result is not None else {"files": ["a"]}]
    )


def _record(
    example_id: str,
    messages: list[dict],
    *,
    source: str = SOURCE_APIGEN_MT,
    n_tools: int = 1,
) -> dict:
    return {
        "id": example_id,
        "source": source,
        "is_multi_turn": True,
        "n_tools_in_gold": n_tools,
        "n_tools_bin": "ge2" if n_tools >= 2 else "1",
        "chat_format": "iffc",
        "messages": messages,
    }


def _loop_convo(
    example_id: str,
    *,
    n_repeat: int = 3,
    source: str = SOURCE_APIGEN_MT,
    name: str = "ls",
) -> dict:
    """Observe → re-emit the same call ``n_repeat`` times (Run 21 shape)."""
    call = _call(name)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": call},
    ]
    for _ in range(n_repeat - 1):
        messages.append({"role": "user", "content": _exec_obs()})
        messages.append({"role": "assistant", "content": call})
    return _record(example_id, messages, source=source, n_tools=1)


def _bfcl_spam_convo(
    example_id: str,
    *,
    n_cycles: int = 3,
    source: str = SOURCE_APIGEN_MT,
) -> dict:
    """BFCL-style ls() / displayCarStatus A-B-A-B observe→re-emit spam.

    Adjacent assistant calls differ, so consecutive-identical is 1, but later
    turns re-emit already-executed calls after the tool result.
    """
    ls_call = _call("ls")
    status_call = _call("displayCarStatus", carId="A")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "inspect the garage and the car"},
        {"role": "assistant", "content": ls_call},
        {"role": "user", "content": _exec_obs("ls()", {"files": ["carA"]})},
        {"role": "assistant", "content": status_call},
        {
            "role": "user",
            "content": _exec_obs(
                'displayCarStatus(carId="A")', {"fuel": 10, "status": "idle"}
            ),
        },
    ]
    extra_pairs = max(n_cycles - 1, 0)
    for _ in range(extra_pairs):
        messages.append({"role": "assistant", "content": ls_call})
        messages.append(
            {"role": "user", "content": _exec_obs("ls()", {"files": ["carA"]})}
        )
        messages.append({"role": "assistant", "content": status_call})
        messages.append(
            {
                "role": "user",
                "content": _exec_obs(
                    'displayCarStatus(carId="A")', {"fuel": 10, "status": "idle"}
                ),
            }
        )
    if messages[-1]["role"] == "user":
        messages.pop()
    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    return _record(example_id, messages, source=source, n_tools=n_asst)


def _unique_then_nontool(
    example_id: str,
    *,
    n_tools: int = 20,
    n_clarify: int = 10,
    n_final: int = 10,
    n_irr: int = 5,
    extra_identical: int = 0,
    source: str = SOURCE_APIGEN_MT,
) -> dict:
    messages = [{"role": "system", "content": "sys"}]
    for t in range(n_tools):
        messages.append({"role": "user", "content": f"do {t}"})
        messages.append({"role": "assistant", "content": _call("ls", i=t)})
    for _ in range(extra_identical):
        messages.append({"role": "user", "content": _exec_obs("ls()", {"ok": True})})
        messages.append({"role": "assistant", "content": _call("ls", i=n_tools - 1)})
    for c in range(n_clarify):
        messages.append({"role": "user", "content": f"need more {c}"})
        messages.append({"role": "assistant", "content": f"Which folder {c}?"})
    for i in range(n_irr):
        messages.append({"role": "user", "content": f"chitchat {i}"})
        messages.append({"role": "assistant", "content": "[]"})
    for f in range(n_final):
        messages.append({"role": "user", "content": f"thanks {f}"})
        messages.append({"role": "assistant", "content": f"Done with the task {f}."})
    return _record(
        example_id, messages, source=source, n_tools=n_tools + extra_identical
    )


def _xlam_mid(i: int) -> dict:
    return {
        "id": f"xlam-{i}",
        "source": "xlam",
        "is_multi_turn": False,
        "n_tools_in_gold": 1,
        "n_tools_bin": "1",
        "chat_format": "iffc",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "cd"},
            {"role": "assistant", "content": _call("cd", folder="docs")},
        ],
    }


def test_canonical_tool_calls_ignores_json_key_order():
    a = json.dumps([{"function_name": "ls", "parameters": {"b": 2, "a": 1}}])
    b = json.dumps([{"function_name": "ls", "parameters": {"a": 1, "b": 2}}])
    assert canonical_tool_calls(a) == canonical_tool_calls(b)
    assert canonical_tool_calls("Which folder?") is None
    assert canonical_tool_calls("[]") is None


def test_identical_call_detects_observe_reemit_not_first_turn():
    rec = _loop_convo("apigen_mt-loop", n_repeat=3)
    messages = rec["messages"]
    asst_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    assert is_identical_to_previous_assistant_call(messages, asst_idxs[0]) is False
    assert is_identical_to_previous_assistant_call(messages, asst_idxs[1]) is True
    assert is_identical_to_previous_assistant_call(messages, asst_idxs[2]) is True
    assert max_identical_call_run(messages) == 3
    assert is_observe_same_call(messages, asst_idxs[0]) is False
    assert is_observe_same_call(messages, asst_idxs[1]) is True
    assert count_observe_same_call_turns(messages) == 2


def test_identical_call_does_not_drop_different_calls_or_nl():
    messages = [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": _call("ls")},
        {"role": "user", "content": _exec_obs()},
        {"role": "assistant", "content": _call("cd", folder="docs")},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "All set."},
        {"role": "user", "content": "and again"},
        {"role": "assistant", "content": "All set."},
    ]
    assert is_identical_to_previous_assistant_call(messages, 3) is False
    assert is_observe_same_call(messages, 3) is False
    assert is_identical_to_previous_assistant_call(messages, 7) is False
    assert is_observe_same_call(messages, 7) is False
    assert max_identical_call_run(messages) == 1


def test_bfcl_spam_consecutive_identical_is_one_observe_same_catches_reemits():
    rec = _bfcl_spam_convo("apigen_mt-spam", n_cycles=3)
    messages = rec["messages"]
    asst_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    assert len(asst_idxs) == 6  # (ls, status) × 3; trailing user obs popped
    assert max_identical_call_run(messages) == 1
    assert is_observe_same_call(messages, asst_idxs[0]) is False
    assert is_identical_to_previous_assistant_call(messages, asst_idxs[0]) is False
    assert is_observe_same_call(messages, asst_idxs[1]) is False
    assert is_identical_to_previous_assistant_call(messages, asst_idxs[1]) is False
    assert is_observe_same_call(messages, asst_idxs[2]) is True
    assert is_identical_to_previous_assistant_call(messages, asst_idxs[2]) is False
    assert is_observe_same_call(messages, asst_idxs[3]) is True
    assert is_identical_to_previous_assistant_call(messages, asst_idxs[3]) is False
    assert drop_reason_for_gold(messages, asst_idxs[2]) == REASON_OBSERVE_SAME
    assert count_observe_same_call_turns(messages) == 4
    assert max_observe_same_call_run(messages) == 4


def test_drop_identical_call_unrolls_keeps_first_drops_repeats():
    rec = _loop_convo("apigen_mt-drop", n_repeat=4)
    rows, _filt = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V8)
    )
    tool_rows = [
        r
        for r in rows
        if label_assistant_content(r["messages"][-1]["content"]) == STOP_TOOL_CALL
    ]
    assert len(tool_rows) == 4
    kept, stats = drop_identical_call_unrolls(rows)
    kept_tools = [
        r
        for r in kept
        if label_assistant_content(r["messages"][-1]["content"]) == STOP_TOOL_CALL
    ]
    assert len(kept_tools) == 1
    assert stats["dropped"] == 3
    assert stats["dropped_observe_same_call"] == 3
    assert stats["dropped_by_source"][SOURCE_APIGEN_MT] == 3
    golds = [r["messages"][-1]["content"] for r in kept_tools]
    assert golds == [_call("ls")]


def test_drop_bfcl_spam_keeps_first_ls_and_status_drops_reemits():
    rec = _bfcl_spam_convo("apigen_mt-spam-drop", n_cycles=4)
    rows, _filt = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V8)
    )
    tool_rows = [
        r
        for r in rows
        if label_assistant_content(r["messages"][-1]["content"]) == STOP_TOOL_CALL
    ]
    assert len(tool_rows) == 8
    kept, stats = drop_identical_call_unrolls(rows)
    kept_tools = [
        r
        for r in kept
        if label_assistant_content(r["messages"][-1]["content"]) == STOP_TOOL_CALL
    ]
    golds = [r["messages"][-1]["content"] for r in kept_tools]
    assert golds == [_call("ls"), _call("displayCarStatus", carId="A")]
    assert stats["dropped"] == 6
    assert stats["dropped_observe_same_call"] == 6
    assert stats["dropped_consecutive"] == 0
    assert stats["dropped_by_reason"][REASON_OBSERVE_SAME] == 6


def test_observe_same_call_keeps_new_call_and_new_user_question():
    messages = [
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": _call("ls")},
        {"role": "user", "content": _exec_obs("ls()", {"files": ["docs"]})},
        {"role": "assistant", "content": _call("cd", folder="docs")},
        {"role": "user", "content": "list again please"},
        {"role": "assistant", "content": _call("ls")},
    ]
    assert is_observe_same_call(messages, 5) is False
    assert is_identical_to_previous_assistant_call(messages, 5) is False
    assert drop_reason_for_gold(messages, 5) is None


def test_v8_recipe_is_v4_empty_no_v5_synth():
    assert resolve_recipe(recipe=RECIPE_V8) == RECIPE_V8
    assert recipe_v4_drop_flags(RECIPE_V8) == (True, True)
    assert recipe_v5_flags(RECIPE_V8) == (False, False, False, False)
    assert recipe_v5_flags(RECIPE_V8) == recipe_v5_flags(RECIPE_V4)
    assert recipe_v5_flags(RECIPE_V8) == recipe_v5_flags(RECIPE_V7)
    assert V8_RECIPE_BUMP == "v8.1"


def test_v8_unroll_matches_v7_empty_policy_on_filler():
    filler = [
        _record(
            "toolace-mid",
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "list then cd"},
                {"role": "assistant", "content": _call("ls")},
                {"role": "user", "content": _exec_obs("ls()", {"files": ["docs"]})},
                {"role": "assistant", "content": "[]"},
                {"role": "user", "content": "now enter docs"},
                {"role": "assistant", "content": _call("cd", folder="docs")},
            ],
            source="toolace",
            n_tools=2,
        ),
        _record(
            "toolace-terminal",
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "list files"},
                {"role": "assistant", "content": _call("ls")},
                {"role": "user", "content": _exec_obs()},
                {"role": "assistant", "content": "[]"},
            ],
            source="toolace",
        ),
        _record(
            "toolace-clarify",
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "can you help?"},
                {"role": "assistant", "content": "[]"},
                {"role": "user", "content": "list the files"},
                {"role": "assistant", "content": _call("ls")},
            ],
            source="toolace",
        ),
    ]
    kept_v8, stats_v8 = unroll_and_filter(
        filler, check_bfcl_leak=False, **flags_for_recipe(RECIPE_V8)
    )
    kept_v7, stats_v7 = unroll_and_filter(
        filler, check_bfcl_leak=False, **flags_for_recipe(RECIPE_V7)
    )
    assert stats_v8.recipe == RECIPE_V8
    assert stats_v8.synthetic_complete_turn0 == 0
    assert stats_v8.empty_after_exec_after == 0
    assert stats_v7.empty_after_exec_after == 0
    assert [r["id"] for r in kept_v8] == [r["id"] for r in kept_v7]
    assert all(row["messages"][-1]["content"] != "[]" for row in kept_v8)


def test_v8_nontool_cap_kwargs_seed_48_band_035():
    kwargs = cap_kwargs_for_recipe(RECIPE_V8)
    assert kwargs["seed"] == SEED_V8 == 48
    assert kwargs["target"] == V8_NONTOOL_TARGET == 0.35
    assert kwargs["min_frac"] == V8_NONTOOL_MIN == 0.25
    assert kwargs["max_frac"] == V8_NONTOOL_MAX == 0.35
    rec = _unique_then_nontool("apigen_mt-cap", extra_identical=5)
    rows, _ = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V8)
    )
    dropped, idrop = drop_identical_call_unrolls(rows)
    assert idrop["dropped"] == 5
    kept, stats = cap_nontool_unrolls(dropped, **kwargs)
    after = scan_unrolled(kept)
    assert stats["seed"] == 48
    assert stats["capped"] is True
    assert V8_NONTOOL_MIN <= after["pct_nontool"] <= V8_NONTOOL_MAX
    tools = [
        r
        for r in kept
        if label_assistant_content(r["messages"][-1]["content"]) == STOP_TOOL_CALL
    ]
    assert len(tools) == 20


def test_mix_stats_v8_gate_requires_capped_nontool_band(tmp_path: Path):
    from sft.mix_stats_v8 import main as stats_main

    rec = _unique_then_nontool("apigen_mt-gate", extra_identical=4)
    splits = tmp_path / "splits"
    mlx_raw = tmp_path / "mlx-raw"
    mlx_v8 = tmp_path / "mlx-v8"
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "corpus_test.jsonl").write_text("")

    raw, _ = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V8)
    )
    mlx_raw.mkdir()
    (mlx_raw / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    (mlx_raw / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw[:2]))

    dropped, _ = drop_identical_call_unrolls(raw)
    capped, _ = cap_nontool_unrolls(dropped, **cap_kwargs_for_recipe(RECIPE_V8))
    mlx_v8.mkdir()
    (mlx_v8 / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in capped))
    (mlx_v8 / "valid.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in capped[:2])
    )

    assert stats_main(["--splits", str(splits), "--gate"]) == 0
    assert (
        stats_main(["--splits", str(splits), "--mlx-dir", str(mlx_raw), "--gate"]) == 1
    )
    assert (
        stats_main(["--splits", str(splits), "--mlx-dir", str(mlx_v8), "--gate"]) == 0
    )

    projected = collect_stats_v8([rec])
    assert projected["recipe_bump"] == "v8.1"
    assert V8_NONTOOL_MIN <= projected["unrolled"]["pct_nontool"] <= V8_NONTOOL_MAX
    assert projected["identical_call_drop"]["dropped"] == 4
    assert projected["identical_call_drop"]["dropped_observe_same_call"] == 4
    assert gate_reasons(projected, require_unrolled_nontool=True) == []


def test_mix_stats_v8_projects_bfcl_spam_observe_drop():
    rec = _bfcl_spam_convo("apigen_mt-spam-stats", n_cycles=4)
    projected = collect_stats_v8([rec])
    idrop = projected["identical_call_drop"]
    assert idrop["dropped"] == 6
    assert idrop["dropped_observe_same_call"] == 6
    assert idrop["dropped_consecutive"] == 0


def test_loop_filter_drops_long_runs_keeps_short_retry():
    long_loop = _loop_convo("apigen_mt-long", n_repeat=5)
    retry = _loop_convo("apigen_mt-retry", n_repeat=2)
    unique = _unique_then_nontool(
        "apigen_mt-ok", n_tools=2, n_clarify=0, n_final=1, n_irr=0
    )
    assert is_loop_trajectory(long_loop, min_run=V8_LOOP_MIN_RUN) is True
    assert is_loop_trajectory(retry, min_run=V8_LOOP_MIN_RUN) is False
    kept, stats = filter_loop_trajectories(
        [long_loop, retry, unique], min_run=V8_LOOP_MIN_RUN
    )
    ids = {r["id"] for r in kept}
    assert "apigen_mt-long" not in ids
    assert "apigen_mt-retry" in ids
    assert "apigen_mt-ok" in ids
    assert stats["dropped"] == 1
    assert stats["pct_apigen_mt_after"] >= V8_MIN_APIGEN_FRAC


def test_loop_filter_drops_bfcl_spam_keeps_short_ab_cycle():
    long_spam = _bfcl_spam_convo("apigen_mt-spam-long", n_cycles=4)
    short_spam = _bfcl_spam_convo("apigen_mt-spam-short", n_cycles=2)
    assert max_identical_call_run(long_spam["messages"]) == 1
    assert count_observe_same_call_turns(long_spam["messages"]) >= V8_LOOP_MIN_RUN
    assert is_loop_trajectory(long_spam, min_run=V8_LOOP_MIN_RUN) is True
    assert is_loop_trajectory(short_spam, min_run=V8_LOOP_MIN_RUN) is False
    kept, stats = filter_loop_trajectories(
        [long_spam, short_spam], min_run=V8_LOOP_MIN_RUN
    )
    ids = {r["id"] for r in kept}
    assert "apigen_mt-spam-long" not in ids
    assert "apigen_mt-spam-short" in ids
    assert stats["dropped_by_kind"].get("observe_same_call") == 1


def test_loop_filter_aborts_when_apigen_floor_would_fail():
    loops = [_loop_convo(f"apigen_mt-loop{i}", n_repeat=4) for i in range(6)]
    filler = [_xlam_mid(i) for i in range(8)]
    try:
        filter_loop_trajectories(loops + filler, min_run=3, min_apigen_frac=0.50)
        raise AssertionError("expected LoopFilterAbort")
    except LoopFilterAbort as exc:
        assert "APIGen-MT floor" in str(exc)
        assert "splits-v8" in str(exc)


def test_build_mix_v8_writes_splits_seed_48_refuses_v1_and_v7(tmp_path: Path):
    from sft.build_mix_v8 import build_v8_mix

    apigen = [
        _unique_then_nontool(
            f"apigen_mt-{i}", n_tools=3, n_clarify=0, n_final=1, n_irr=0
        )
        for i in range(12)
    ]
    xlam = [_xlam_mid(i) for i in range(4)]
    out = tmp_path / "splits-v8"
    manifest = build_v8_mix(apigen=apigen, toolace=[], xlam=xlam, out_dir=out, seed=48)
    assert manifest["experiment"] == V8_EXPERIMENT
    assert manifest["recipe_bump"] == "v8.1"
    assert manifest["seed"] == 48
    assert "identical_call_drop" in manifest
    assert "observe→same-call" in manifest["identical_call_drop"]
    assert "0.35" in str(manifest["targets"]["unrolled_pct_nontool"])
    assert "ToolACE-R-lite" in manifest["student_filter"]
    assert (out / "train.jsonl").is_file()
    assert (out / "MANIFEST.json").is_file()

    try:
        build_v8_mix(apigen=apigen, toolace=[], xlam=xlam, out_dir=SPLITS_DIR, seed=48)
        raise AssertionError("should refuse frozen v1 splits")
    except MixAbort as exc:
        assert "frozen v1" in str(exc).lower() or "Refusing" in str(exc)

    try:
        build_v8_mix(
            apigen=apigen, toolace=[], xlam=xlam, out_dir=SPLITS_DIR_V7, seed=48
        )
        raise AssertionError("should refuse v7 splits")
    except MixAbort as exc:
        assert "v7" in str(exc).lower() or "Refusing" in str(exc)


def test_build_mix_v8_aborts_on_loop_filter_floor(tmp_path: Path):
    from sft.build_mix_v8 import build_v8_mix

    apigen = [_loop_convo(f"apigen_mt-{i}", n_repeat=4) for i in range(12)]
    xlam = [_xlam_mid(i) for i in range(10)]
    try:
        build_v8_mix(
            apigen=apigen,
            toolace=[],
            xlam=xlam,
            out_dir=tmp_path / "out",
            seed=48,
        )
        raise AssertionError("expected MixAbort from loop filter")
    except MixAbort as exc:
        assert "APIGen-MT floor" in str(exc)


def test_prepare_mlx_v8_drops_identical_and_caps_nontool(tmp_path: Path, capsys):
    from sft.train_mlx import prepare_mlx_data

    class _TestTokenizer:
        def apply_chat_template(
            self, messages, *, add_generation_prompt=False, return_dict=False
        ):
            length = sum(len(str(m.get("content", ""))) + 2 for m in messages)
            if add_generation_prompt:
                length += 2
            return list(range(length))

    rec = _unique_then_nontool("apigen_mt-prep", extra_identical=6)
    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx-v8"
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    prepare_mlx_data(
        splits_dir=splits,
        mlx_dir=mlx,
        recipe=RECIPE_V8,
        budget_tokenizer=_TestTokenizer(),
    )
    train_rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    stats = scan_unrolled(train_rows)
    assert V8_NONTOOL_MIN <= stats["pct_nontool"] <= V8_NONTOOL_MAX
    tools = [
        r
        for r in train_rows
        if label_assistant_content(r["messages"][-1]["content"]) == STOP_TOOL_CALL
    ]
    assert len(tools) == 20
    logged = capsys.readouterr().out
    assert "v8 identical-call drop" in logged
    assert "dropped_observe_same_call" in logged
    assert "v8 non-tool unroll cap" in logged
    assert "token budget (v8 train)" in logged
    assert "v8_nontool_cap=on" in logged
    assert "identical_call_drop=" in logged
    assert "synthesize_complete_turn0=False" in logged


def test_prepare_mlx_v8_drops_bfcl_spam_reemits(tmp_path: Path):
    from sft.train_mlx import prepare_mlx_data

    class _TestTokenizer:
        def apply_chat_template(
            self, messages, *, add_generation_prompt=False, return_dict=False
        ):
            length = sum(len(str(m.get("content", ""))) + 2 for m in messages)
            if add_generation_prompt:
                length += 2
            return list(range(length))

    rec = _bfcl_spam_convo("apigen_mt-prep-spam", n_cycles=4)
    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx-v8"
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    prepare_mlx_data(
        splits_dir=splits,
        mlx_dir=mlx,
        recipe=RECIPE_V8,
        budget_tokenizer=_TestTokenizer(),
    )
    train_rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    golds = [r["messages"][-1]["content"] for r in train_rows]
    assert _call("ls") in golds
    assert _call("displayCarStatus", carId="A") in golds
    assert golds.count(_call("ls")) == 1
    assert golds.count(_call("displayCarStatus", carId="A")) == 1


def test_build_mix_v8_cli_subprocess_nonzero(tmp_path: Path):
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sft.build_mix_v8",
            "--formatted-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
        cwd="/workspace",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "APIGen-MT" in (proc.stderr + proc.stdout)
