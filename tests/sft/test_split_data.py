from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft.format_iffc import n_tools_bin
from sft.split_data import (
    assert_no_bfcl,
    bfcl_leak_reasons,
    build_pool,
    sample_xlam_rows,
    split_counts,
    stratum_key,
    stratified_split,
    write_splits,
)
from sft.train_mlx import build_lora_command, prepare_mlx_data


def _rec(
    example_id: str,
    source: str,
    *,
    multi: bool,
    n_tools: int,
    user: str = "hello",
) -> dict:
    return {
        "id": example_id,
        "source": source,
        "is_multi_turn": multi,
        "n_tools_in_gold": n_tools,
        "n_tools_bin": n_tools_bin(n_tools),
        "chat_format": "iffc",
        "messages": [
            {"role": "system", "content": "You are an expert in function calling."},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": "[]"
                if n_tools == 0
                else json.dumps(
                    [
                        {
                            "function_name": "f",
                            "parameters": {"i": i},
                        }
                        for i in range(n_tools)
                    ]
                ),
            },
        ],
    }


def _balanced_pool(n_per_cell: int = 40) -> list[dict]:
    rows: list[dict] = []
    k = 0
    for source, multi in (("toolace", True), ("xlam", False)):
        for n_tools in (1, 2):
            for _ in range(n_per_cell):
                rows.append(
                    _rec(f"{source}-{k}", source, multi=multi, n_tools=n_tools)
                )
                k += 1
    return rows


def test_sample_xlam_oversamples_parallel():
    rows = [_rec(f"xlam-{i}", "xlam", multi=False, n_tools=1) for i in range(80)]
    rows += [_rec(f"xlam-p-{i}", "xlam", multi=False, n_tools=2) for i in range(20)]
    sampled = sample_xlam_rows(rows, n=30, seed=42)
    assert len(sampled) == 30
    n_parallel = sum(1 for r in sampled if r["n_tools_in_gold"] >= 2)
    assert n_parallel == 20  # all parallel kept
    again = sample_xlam_rows(rows, n=30, seed=42)
    assert [r["id"] for r in again] == [r["id"] for r in sampled]


def test_stratified_split_is_deterministic_disjoint_and_roughly_balanced():
    pool = _balanced_pool(40)
    a = stratified_split(pool, seed=42)
    b = stratified_split(pool, seed=42)
    for name in ("train", "valid", "corpus_test"):
        assert [r["id"] for r in a[name]] == [r["id"] for r in b[name]]

    ids = {
        name: {r["id"] for r in a[name]}
        for name in ("train", "valid", "corpus_test")
    }
    assert not ids["train"] & ids["valid"]
    assert not ids["train"] & ids["corpus_test"]
    assert not ids["valid"] & ids["corpus_test"]
    assert ids["train"] | ids["valid"] | ids["corpus_test"] == {r["id"] for r in pool}

    n = len(pool)
    assert len(a["train"]) == split_counts(n)[0]
    # Per-stratum ~90/5/5 (int truncation → train gets remainder).
    from collections import defaultdict

    by_stratum = defaultdict(lambda: {"train": 0, "valid": 0, "corpus_test": 0})
    for name, rows in a.items():
        for rec in rows:
            by_stratum[stratum_key(rec)][name] += 1
    for key, counts in by_stratum.items():
        total = sum(counts.values())
        assert total == 40, key
        assert counts["train"] == 36
        assert counts["valid"] == 2
        assert counts["corpus_test"] == 2


def test_no_bfcl_ids_in_splits():
    pool = _balanced_pool(5)
    splits = stratified_split(pool, seed=42)
    flat = [r for rows in splits.values() for r in rows]
    assert_no_bfcl(flat)
    leak = _rec("multi_turn_base_12", "toolace", multi=True, n_tools=1)
    assert bfcl_leak_reasons(leak)
    with pytest.raises(ValueError, match="BFCL"):
        assert_no_bfcl(flat + [leak])


def test_bfcl_content_is_rejected_and_native_tool_calls_too():
    content = _rec(
        "toolace-x",
        "toolace",
        multi=True,
        n_tools=1,
        user="please solve multi_turn_base_99",
    )
    assert "message_content" in bfcl_leak_reasons(content)
    native = _rec("xlam-1", "xlam", multi=False, n_tools=1)
    native["messages"].append(
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}
    )
    assert "native_tool_calls" in bfcl_leak_reasons(native)


def test_write_splits_manifest_checksums_and_seed(tmp_path: Path):
    pool = _balanced_pool(20)
    splits = stratified_split(pool, seed=42)
    manifest = write_splits(
        splits,
        out_dir=tmp_path,
        seed=42,
        xlam_pool=80,
        xlam_sampled=40,
        toolace_n=40,
    )
    assert manifest["seed"] == 42
    assert manifest["experiment"] == "sft-v1"
    assert manifest["chat_format"] == "iffc"
    assert "multi_turn_base" in manifest["thesis_test"]
    assert set(manifest["checksums_sha256"]) == {
        "train.jsonl",
        "valid.jsonl",
        "corpus_test.jsonl",
    }
    for name in ("train.jsonl", "valid.jsonl", "corpus_test.jsonl"):
        digest = manifest["checksums_sha256"][name]
        assert len(digest) == 64
        assert (tmp_path / name).is_file()

    again = write_splits(splits, out_dir=tmp_path / "copy", seed=42)
    assert again["checksums_sha256"]["train.jsonl"] == manifest["checksums_sha256"]["train.jsonl"]


def test_build_pool_keeps_all_toolace_and_caps_xlam():
    toolace = [_rec(f"toolace-{i}", "toolace", multi=True, n_tools=1) for i in range(10)]
    xlam = [_rec(f"xlam-{i}", "xlam", multi=False, n_tools=1 if i < 8 else 2) for i in range(10)]
    pool, n_sampled = build_pool(toolace, xlam, xlam_n=5, seed=42)
    assert n_sampled == 5
    assert sum(1 for r in pool if r["source"] == "toolace") == 10
    assert sum(1 for r in pool if r["source"] == "xlam") == 5
    # Oversample: both parallel xLAM rows should be preferred.
    xlam_ids = {r["id"] for r in pool if r["source"] == "xlam"}
    assert "xlam-8" in xlam_ids
    assert "xlam-9" in xlam_ids


def test_prepare_mlx_data_unrolls_and_smoke_slices(tmp_path: Path):
    splits = tmp_path / "splits"
    mlx = tmp_path / "mlx"
    rec = _rec("toolace-0", "toolace", multi=True, n_tools=1)
    rec["messages"] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "[]"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": '[{"function_name": "ls", "parameters": {}}]'},
    ]
    splits.mkdir()
    (splits / "train.jsonl").write_text(json.dumps(rec) + "\n")
    (splits / "valid.jsonl").write_text(json.dumps(rec) + "\n")
    prepare_mlx_data(splits_dir=splits, mlx_dir=mlx, smoke=True, smoke_train_rows=1, smoke_valid_rows=1)
    train_lines = (mlx / "train.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == 1
    row = json.loads(train_lines[0])
    assert row["messages"][-1]["role"] == "assistant"
    cmd = build_lora_command(
        model="sft/models/gemma-4-e2b-it-mlx",
        data_dir=mlx,
        adapter_path=tmp_path / "adapters",
        config_path=tmp_path / "lora_config.yaml",
        iters=100,
    )
    assert "--mask-prompt" in cmd
    assert "100" in cmd
    assert "-m" in cmd and "mlx_lm.lora" in cmd
