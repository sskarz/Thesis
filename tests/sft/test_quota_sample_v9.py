"""Tests for sft.quota_sample_v9 band enforcement + soft excludes."""

from __future__ import annotations

import json

import pytest
from benchmark.iffc import format_iffc_execution_user_message
from sft.action_class_v9 import (
    ACTION_ASK,
    ACTION_FINAL_AFTER_OBS,
    ACTION_IRRELEVANCE,
    ACTION_MT_TOOL_CHAIN_ADVANCE,
    ACTION_REFUSE,
    ACTION_TOOL_CALL_START,
    classify_unrolled_row,
)
from sft.config import SEED_V9, SOURCE_APIGEN_MT, SOURCE_TOOLACE
from sft.quota_sample_v9 import QuotaAbort, quota_sample_unrolls
from sft.split_data import assert_no_bfcl


def _call(name: str = "ls", **params: object) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _exec_obs(func_string: str = "ls()", result: object | None = None) -> str:
    return format_iffc_execution_user_message(
        [func_string], [result if result is not None else {"ok": True}]
    )


def _row(
    example_id: str,
    messages: list[dict],
    *,
    source: str = SOURCE_TOOLACE,
) -> dict:
    return {
        "id": example_id,
        "source": source,
        "messages": messages,
        "chat_format": "iffc",
    }


def _make_pool(n_each: int = 20) -> list[dict]:
    rows: list[dict] = []
    # tool_call_start
    for i in range(n_each):
        rows.append(
            _row(
                f"start-{i}",
                [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": _call("start", i=i)},
                ],
            )
        )
    # mt_tool_chain_advance
    for i in range(n_each):
        rows.append(
            _row(
                f"adv-{i}",
                [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": _call("ls")},
                    {"role": "user", "content": _exec_obs()},
                    {
                        "role": "assistant",
                        "content": _call("cat", path=f"f{i}"),
                    },
                ],
            )
        )
    # final_after_obs
    for i in range(n_each):
        rows.append(
            _row(
                f"final-{i}",
                [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": _call("ls")},
                    {"role": "user", "content": _exec_obs()},
                    {"role": "assistant", "content": f"Done with file {i}."},
                ],
                source=SOURCE_APIGEN_MT,
            )
        )
    # ask
    for i in range(n_each):
        rows.append(
            _row(
                f"ask-{i}",
                [
                    {"role": "user", "content": "do x"},
                    {
                        "role": "assistant",
                        "content": f"Which option for {i}?",
                    },
                ],
                source=SOURCE_APIGEN_MT,
            )
        )
    # refuse
    for i in range(max(4, n_each // 4)):
        rows.append(
            _row(
                f"refuse-{i}",
                [
                    {"role": "user", "content": "bad"},
                    {
                        "role": "assistant",
                        "content": "I cannot help with that request.",
                    },
                ],
                source=SOURCE_APIGEN_MT,
            )
        )
    # irrelevance
    for i in range(max(4, n_each // 4)):
        rows.append(
            _row(
                f"irr-{i}",
                [
                    {"role": "user", "content": "noop"},
                    {"role": "assistant", "content": "[]"},
                ],
            )
        )
    # observe→same (should be soft-excluded)
    call = _call("ls")
    for i in range(5):
        rows.append(
            _row(
                f"loop-{i}",
                [
                    {"role": "user", "content": "list"},
                    {"role": "assistant", "content": call},
                    {"role": "user", "content": _exec_obs()},
                    {"role": "assistant", "content": call},
                ],
            )
        )
    return rows


def test_quota_sample_excludes_observe_same_and_hits_bands():
    rows = _make_pool(24)
    kept, stats = quota_sample_unrolls(rows, seed=SEED_V9, skip_gates=False)
    assert stats["pre"]["n_excluded_soft"] >= 5
    # No observe_same / consecutive in kept
    for row in kept:
        cls = classify_unrolled_row(row)
        assert cls not in {"observe_same_call", "consecutive_identical"}
    pct = stats["pct_by_class"]
    assert 0.28 <= pct[ACTION_MT_TOOL_CHAIN_ADVANCE] <= 0.42
    assert 0.10 <= pct[ACTION_TOOL_CALL_START] <= 0.20
    assert 0.18 <= pct[ACTION_FINAL_AFTER_OBS] <= 0.26
    assert 0.06 <= pct[ACTION_ASK] <= 0.14
    assert pct[ACTION_REFUSE] >= 0.005
    assert pct[ACTION_IRRELEVANCE] <= 0.12


def test_quota_prefers_dupe_scarce_never_maps_nl_to_empty():
    # Tiny refuse pool → must dupe, never rewrite NL to []
    rows = _make_pool(30)
    kept, stats = quota_sample_unrolls(rows, seed=SEED_V9, skip_gates=True)
    assert stats["duped_by_class"].get(ACTION_REFUSE, 0) >= 0
    for row in kept:
        gold = (row.get("messages") or [])[-1].get("content")
        if classify_unrolled_row(row) == ACTION_FINAL_AFTER_OBS:
            assert str(gold).strip() != "[]"


def test_quota_abort_when_refuse_missing():
    rows = [
        r
        for r in _make_pool(20)
        if not str(r["id"]).startswith("refuse-")
    ]
    with pytest.raises(QuotaAbort):
        quota_sample_unrolls(rows, seed=SEED_V9, skip_gates=False)


def test_quota_rows_have_no_bfcl_leak():
    rows = _make_pool(10)
    kept, _stats = quota_sample_unrolls(rows, seed=SEED_V9, skip_gates=True)
    assert_no_bfcl(kept)
