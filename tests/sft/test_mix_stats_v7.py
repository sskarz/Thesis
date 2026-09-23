# Light v7 mix-stats labeling + download CLI flags. No Hub download.

from __future__ import annotations

import json
from pathlib import Path

from sft.config import APIGEN_MT_HF_ID, SOURCE_APIGEN_MT
from sft.download_datasets import main as download_main
from sft.format_apigen_mt import (
    STOP_CLARIFY,
    STOP_FINAL,
    STOP_IRRELEVANCE,
    STOP_TOOL_CALL,
    label_assistant_content,
)
from sft.mix_stats_v7 import (
    collect_stats,
    conversation_stop_label,
    gate_reasons,
    scan_conversations,
    scan_unrolled,
)
from sft.student_filter_v7 import main as student_main


def _rec(example_id: str, source: str, assistant: str, user: str = "hello") -> dict:
    return {
        "id": example_id,
        "source": source,
        "is_multi_turn": source == SOURCE_APIGEN_MT,
        "n_tools_in_gold": 1 if "function_name" in assistant else 0,
        "n_tools_bin": "1",
        "chat_format": "iffc",
        "stop_label": label_assistant_content(assistant),
        "messages": [
            {"role": "system", "content": "You are an expert in function calling."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_label_keywords_and_question_mark():
    assert label_assistant_content("Need more information.") == STOP_CLARIFY
    assert label_assistant_content("please provide the id") == STOP_CLARIFY
    assert label_assistant_content("All set.") == STOP_FINAL
    assert label_assistant_content(" [] ") == STOP_IRRELEVANCE


def test_scan_conversations_pct_apigen_and_stop_labels():
    recs = [
        _rec(
            "apigen_mt-0",
            SOURCE_APIGEN_MT,
            json.dumps([{"function_name": "ls", "parameters": {}}]),
        ),
        _rec("apigen_mt-1", SOURCE_APIGEN_MT, "Which folder?"),
        _rec("xlam-0", "xlam", json.dumps([{"function_name": "cd", "parameters": {}}])),
    ]
    stats = scan_conversations(recs)
    assert stats["n"] == 3
    assert stats["pct_apigen_mt"] == 0.6667
    assert stats["stop_label"][STOP_TOOL_CALL] == 2
    assert stats["stop_label"][STOP_CLARIFY] == 1
    assert conversation_stop_label(recs[1]) == STOP_CLARIFY


def test_scan_unrolled_nontool_and_empty_after_exec():
    from benchmark.iffc import format_iffc_execution_user_message

    tool = json.dumps([{"function_name": "ls", "parameters": {}}])
    exec_user = format_iffc_execution_user_message(["ls()"], [{"files": ["a"]}])
    rows = [
        {
            "id": "a#1",
            "source": SOURCE_APIGEN_MT,
            "messages": [
                {"role": "user", "content": "list"},
                {"role": "assistant", "content": tool},
            ],
        },
        {
            "id": "a#2",
            "source": SOURCE_APIGEN_MT,
            "messages": [
                {"role": "user", "content": "list"},
                {"role": "assistant", "content": tool},
                {"role": "user", "content": exec_user},
                {"role": "assistant", "content": "Listed the files."},
            ],
        },
        {
            "id": "t#empty",
            "source": "toolace",
            "messages": [
                {"role": "user", "content": exec_user},
                {"role": "assistant", "content": "[]"},
            ],
        },
    ]
    stats = scan_unrolled(rows)
    assert stats["n"] == 3
    assert stats["pct_apigen_mt"] == 0.6667
    assert stats["pct_nontool"] > 0
    assert stats["empty_after_exec"] == 1
    assert stats["stop_label"][STOP_FINAL] == 1
    assert stats["stop_label"][STOP_IRRELEVANCE] == 1


def test_collect_stats_gate_reasons_low_apigen():
    recs = [
        _rec("xlam-0", "xlam", json.dumps([{"function_name": "ls", "parameters": {}}])),
        _rec("xlam-1", "xlam", json.dumps([{"function_name": "cd", "parameters": {}}])),
    ]
    stats = collect_stats(recs)
    reasons = gate_reasons(stats)
    assert any("pct_apigen_mt" in r for r in reasons)


def test_download_apigen_mt_only_mocked(tmp_path: Path, monkeypatch):
    from sft import download_datasets as dd

    captured: dict = {}

    def fake_download(hf_id, out_path, **kwargs):
        captured["hf_id"] = hf_id
        captured["out"] = out_path
        captured["gated"] = kwargs.get("gated")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n")
        return 1

    monkeypatch.setattr(dd, "download_hf_split", fake_download)
    assert download_main(["--apigen-mt-only", "--out-dir", str(tmp_path)]) == 0
    assert captured["hf_id"] == APIGEN_MT_HF_ID
    assert captured["gated"] is True
    assert captured["out"].name == "apigen_mt.jsonl"
    assert (tmp_path / "apigen_mt.jsonl").is_file()


def test_student_filter_v7_stub_prints_todo(capsys):
    assert student_main(["--pools", "filler"]) == 0
    logged = capsys.readouterr().out
    assert "TODO" in logged
    assert "Phase 3" in logged


def _apigen_multi_tool_then_final(i: int, n_tools: int = 4) -> dict:
    messages = [{"role": "system", "content": "sys"}]
    for t in range(n_tools):
        messages.append({"role": "user", "content": f"do {t}"})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    [{"function_name": "ls", "parameters": {"i": t}}]
                ),
            }
        )
    messages.append({"role": "user", "content": "thanks"})
    messages.append({"role": "assistant", "content": "Done with the task."})
    return {
        "id": f"apigen_mt-{i}",
        "source": SOURCE_APIGEN_MT,
        "is_multi_turn": True,
        "n_tools_in_gold": n_tools,
        "n_tools_bin": "ge2" if n_tools >= 2 else "1",
        "chat_format": "iffc",
        "stop_label": STOP_FINAL,
        "messages": messages,
    }


def _xlam_mid(i: int) -> dict:
    return _rec(
        f"xlam-{i}",
        "xlam",
        json.dumps([{"function_name": "cd", "parameters": {"folder": "docs"}}]),
    )


def test_build_mix_v7_writes_splits_and_refuses_v1_dir(tmp_path: Path):
    from sft.build_mix_v7 import MixAbort, build_v7_mix
    from sft.config import SPLITS_DIR, V7_EXPERIMENT

    apigen = [_apigen_multi_tool_then_final(i) for i in range(12)]
    xlam = [_xlam_mid(i) for i in range(4)]
    out = tmp_path / "splits-v7"
    manifest = build_v7_mix(
        apigen=apigen, toolace=[], xlam=xlam, out_dir=out, seed=47
    )
    assert manifest["experiment"] == V7_EXPERIMENT
    assert manifest["seed"] == 47
    assert "CC-BY-NC" in manifest["license_note"]
    assert (out / "train.jsonl").is_file()
    assert (out / "MANIFEST.json").is_file()
    train_n = manifest["counts"]["train"]["n"]
    assert train_n >= 1
    assert "apigen_mt" in manifest["counts"]["train"]["by_source"]

    try:
        build_v7_mix(
            apigen=apigen, toolace=[], xlam=xlam, out_dir=SPLITS_DIR, seed=47
        )
        raise AssertionError("should refuse frozen v1 splits")
    except MixAbort as exc:
        assert "frozen v1" in str(exc).lower() or "Refusing" in str(exc)


def test_build_mix_v7_aborts_without_apigen():
    from sft.build_mix_v7 import MixAbort, assemble_v7_pool

    try:
        assemble_v7_pool([], [], [])
        raise AssertionError("expected MixAbort")
    except MixAbort as exc:
        assert "APIGen-MT" in str(exc)


