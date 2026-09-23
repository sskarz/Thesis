# Cap non-tool unrolls for SFT v7 (keep all tool_calls; ~15% non-tool).

from __future__ import annotations

import json
import sys
from pathlib import Path

from sft.cap_nontool_v7 import (
    cap_nontool_unrolls,
    format_cap_report,
    target_nontool_keep_count,
    unroll_gold,
)
from sft.config import (
    SOURCE_APIGEN_MT,
    V7_NONTOOL_MAX,
    V7_NONTOOL_MIN,
    V7_NONTOOL_TARGET,
)
from sft.filter_empty_after_exec import RECIPE_V7, flags_for_recipe, unroll_and_filter
from sft.format_apigen_mt import (
    STOP_CLARIFY,
    STOP_FINAL,
    STOP_IRRELEVANCE,
    STOP_TOOL_CALL,
    label_assistant_content,
)
from sft.mix_stats_v7 import collect_stats, gate_reasons, scan_unrolled


def _call(i: int) -> str:
    return json.dumps([{"function_name": "ls", "parameters": {"i": i}}])


def _mt_convo(
    example_id: str,
    *,
    n_tools: int = 4,
    n_clarify: int = 6,
    n_final: int = 6,
    n_irr: int = 2,
    source: str = SOURCE_APIGEN_MT,
) -> dict:
    """Synthetic APIGen-style traj: many tool calls then many NL assistant turns."""
    messages = [{"role": "system", "content": "sys"}]
    for t in range(n_tools):
        messages.append({"role": "user", "content": f"do {t}"})
        messages.append({"role": "assistant", "content": _call(t)})
    for c in range(n_clarify):
        messages.append({"role": "user", "content": f"need more {c}"})
        messages.append({"role": "assistant", "content": f"Which folder {c}?"})
    for i in range(n_irr):
        messages.append({"role": "user", "content": f"chitchat {i}"})
        messages.append({"role": "assistant", "content": "[]"})
    for f in range(n_final):
        messages.append({"role": "user", "content": f"thanks {f}"})
        messages.append({"role": "assistant", "content": f"Done with the task {f}."})
    n_gold_tools = n_tools
    if n_final:
        stop = STOP_FINAL
    elif n_irr:
        stop = STOP_IRRELEVANCE
    elif n_clarify:
        stop = STOP_CLARIFY
    else:
        stop = STOP_TOOL_CALL
    return {
        "id": example_id,
        "source": source,
        "is_multi_turn": True,
        "n_tools_in_gold": n_gold_tools,
        "n_tools_bin": "ge2" if n_gold_tools >= 2 else "1",
        "chat_format": "iffc",
        "stop_label": stop,
        "messages": messages,
    }


def test_target_keep_count_lands_in_band():
    n_tool, n_non = 520, 480  # ~48% non-tool (the observed gate failure)
    k = target_nontool_keep_count(n_tool, n_non)
    pct = k / (n_tool + k)
    assert V7_NONTOOL_MIN <= pct <= V7_NONTOOL_MAX
    assert abs(pct - V7_NONTOOL_TARGET) < 0.02


def test_cap_synthetic_mt_pct_nontool_in_band_keeps_all_tool_calls():
    rec = _mt_convo("apigen_mt-mt0", n_tools=20, n_clarify=10, n_final=10, n_irr=5)
    rows, _filt = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V7)
    )
    before = scan_unrolled(rows)
    assert before["pct_nontool"] > V7_NONTOOL_MAX

    tool_golds = [
        unroll_gold(r)
        for r in rows
        if label_assistant_content(unroll_gold(r)) == STOP_TOOL_CALL
    ]
    assert len(tool_golds) == 20

    kept, stats = cap_nontool_unrolls(rows)
    after = scan_unrolled(kept)
    assert stats["capped"] is True
    assert stats["seed"] == 47
    assert V7_NONTOOL_MIN <= after["pct_nontool"] <= V7_NONTOOL_MAX
    assert stats["n_tool"] == 20
    kept_tools = [
        unroll_gold(r)
        for r in kept
        if label_assistant_content(unroll_gold(r)) == STOP_TOOL_CALL
    ]
    assert kept_tools == tool_golds

    labels_after = set(after["stop_label"])
    assert STOP_TOOL_CALL in labels_after
    assert STOP_CLARIFY in labels_after
    assert STOP_FINAL in labels_after
    assert STOP_IRRELEVANCE in labels_after
    finals = [
        unroll_gold(r)
        for r in kept
        if label_assistant_content(unroll_gold(r)) == STOP_FINAL
    ]
    assert finals
    assert all(g != "[]" and "Done with the task" in g for g in finals)
    assert "splits-v7" in format_cap_report(stats)


def test_cap_is_deterministic_seed_47():
    rec = _mt_convo("apigen_mt-det")
    rows, _ = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V7)
    )
    a, sa = cap_nontool_unrolls(rows, seed=47)
    b, sb = cap_nontool_unrolls(rows, seed=47)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert sa["n_after"] == sb["n_after"]


def test_cap_noop_when_already_in_band():
    rec = _mt_convo("apigen_mt-ok", n_tools=17, n_clarify=1, n_final=1, n_irr=1)
    rows, _ = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V7)
    )
    before = scan_unrolled(rows)
    assert V7_NONTOOL_MIN <= before["pct_nontool"] <= V7_NONTOOL_MAX
    kept, stats = cap_nontool_unrolls(rows)
    assert stats["capped"] is False
    assert len(kept) == len(rows)


def test_conversation_gate_ignores_uncapped_pct_nontool():
    recs = [_mt_convo(f"apigen_mt-{i}") for i in range(6)] + [
        {
            "id": f"xlam-{i}",
            "source": "xlam",
            "is_multi_turn": False,
            "n_tools_in_gold": 1,
            "n_tools_bin": "1",
            "chat_format": "iffc",
            "stop_label": STOP_TOOL_CALL,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "cd"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        [{"function_name": "cd", "parameters": {"folder": "docs"}}]
                    ),
                },
            ],
        }
        for i in range(2)
    ]
    uncapped, _ = unroll_and_filter(
        recs, check_bfcl_leak=False, **flags_for_recipe(RECIPE_V7)
    )
    uncapped_stats = {
        "conversations": collect_stats(recs, cap_nontool=False)["conversations"],
        "unrolled": scan_unrolled(uncapped),
        "recipe": RECIPE_V7,
    }
    assert uncapped_stats["unrolled"]["pct_nontool"] > V7_NONTOOL_MAX
    assert gate_reasons(uncapped_stats, require_unrolled_nontool=False) == []
    assert any(
        "pct_nontool" in r
        for r in gate_reasons(uncapped_stats, require_unrolled_nontool=True)
    )

    capped_stats = collect_stats(recs)  # projected cap
    assert V7_NONTOOL_MIN <= capped_stats["unrolled"]["pct_nontool"] <= V7_NONTOOL_MAX
    assert gate_reasons(capped_stats, require_unrolled_nontool=True) == []


def test_build_mix_v7_writes_high_unrolled_nontool_conversations(tmp_path: Path):
    from sft.build_mix_v7 import MixAbort, build_v7_mix, main as mix_main

    apigen = [_mt_convo(f"apigen_mt-{i}") for i in range(12)]
    xlam = [
        {
            "id": f"xlam-{i}",
            "source": "xlam",
            "is_multi_turn": False,
            "n_tools_in_gold": 1,
            "n_tools_bin": "1",
            "chat_format": "iffc",
            "stop_label": STOP_TOOL_CALL,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "cd"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        [{"function_name": "cd", "parameters": {"folder": "docs"}}]
                    ),
                },
            ],
        }
        for i in range(4)
    ]
    out = tmp_path / "splits-v7"
    manifest = build_v7_mix(apigen=apigen, toolace=[], xlam=xlam, out_dir=out, seed=47)
    assert (out / "train.jsonl").is_file()
    # Splits still store the full MT traj (many NL turns), not the capped unrolls.
    train_line = (out / "train.jsonl").read_text().splitlines()[0]
    train_rec = json.loads(train_line)
    n_asst = sum(1 for m in train_rec["messages"] if m["role"] == "assistant")
    assert n_asst > 5
    assert "nontool_unroll_cap" in manifest
    assert "mlx-v7" in manifest["nontool_unroll_cap"]

    assert issubclass(MixAbort, Exception)
    assert not issubclass(MixAbort, SystemExit)
    formatted = tmp_path / "formatted"
    formatted.mkdir()
    rc = mix_main(["--formatted-dir", str(formatted), "--out", str(tmp_path / "out2")])
    assert rc == 1


def test_prepare_mlx_v7_writes_capped_rows(tmp_path: Path, capsys):
    from sft.train_mlx import prepare_mlx_data

    class _TestTokenizer:
        def apply_chat_template(
            self, messages, *, add_generation_prompt=False, return_dict=False
        ):
            length = sum(len(str(m.get("content", ""))) + 2 for m in messages)
            if add_generation_prompt:
                length += 2
            return list(range(length))

    rec = _mt_convo("apigen_mt-prep", n_tools=20, n_clarify=10, n_final=10, n_irr=5)
    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx-v7"
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    prepare_mlx_data(
        splits_dir=splits,
        mlx_dir=mlx,
        recipe=RECIPE_V7,
        budget_tokenizer=_TestTokenizer(),
    )
    train_rows = [
        json.loads(line)
        for line in (mlx / "train.jsonl").read_text().splitlines()
        if line
    ]
    stats = scan_unrolled(train_rows)
    assert V7_NONTOOL_MIN <= stats["pct_nontool"] <= V7_NONTOOL_MAX
    tools = [
        r
        for r in train_rows
        if label_assistant_content(unroll_gold(r)) == STOP_TOOL_CALL
    ]
    assert len(tools) == 20
    logged = capsys.readouterr().out
    assert "v7 non-tool unroll cap" in logged
    assert "token budget (v7 train)" in logged
    assert "v7_nontool_cap=on" in logged


def test_mix_stats_mlx_dir_gate_requires_capped_nontool(tmp_path: Path):
    from sft.mix_stats_v7 import main as stats_main

    rec = _mt_convo("apigen_mt-gate", n_tools=20, n_clarify=10, n_final=10, n_irr=5)
    splits = tmp_path / "splits"
    mlx_raw = tmp_path / "mlx-raw"
    mlx_cap = tmp_path / "mlx-cap"
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "corpus_test.jsonl").write_text("")

    raw, _ = unroll_and_filter(
        [rec], check_bfcl_leak=False, **flags_for_recipe(RECIPE_V7)
    )
    mlx_raw.mkdir()
    (mlx_raw / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    (mlx_raw / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw[:2]))

    capped, _ = cap_nontool_unrolls(raw)
    mlx_cap.mkdir()
    (mlx_cap / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in capped))
    (mlx_cap / "valid.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in capped[:2])
    )

    assert stats_main(["--splits", str(splits), "--gate"]) == 0
    assert (
        stats_main(["--splits", str(splits), "--mlx-dir", str(mlx_raw), "--gate"]) == 1
    )
    assert (
        stats_main(["--splits", str(splits), "--mlx-dir", str(mlx_cap), "--gate"]) == 0
    )


def test_build_mix_cli_subprocess_nonzero(tmp_path: Path):
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sft.build_mix_v7",
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
