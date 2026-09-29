"""Regression and invariant tests for exact-prefix Stage B pair mining."""

from __future__ import annotations

import json

import pytest

from benchmark.iffc import format_iffc_execution_user_message
from sft.drop_identical_call import canonical_tool_calls
from sft.pref_pairs_antiloop_v9 import (
    canonical_response,
    mine_advance_vs_loop_candidates,
    mine_splits,
    validate_pair,
)


def _call(name: str, **params: object) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _obs(call: str, result: object) -> str:
    return format_iffc_execution_user_message([call], [result])


def _record(record_id: str = "source-parent-1") -> dict:
    first = _call("ls", path="/docs")
    advance = _call("read_file", path="/docs/a.txt")
    return {
        "id": record_id,
        "source": "toolace",
        "messages": [
            {"role": "user", "content": "Read the first file in /docs"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": _obs("ls(path='/docs')", {"files": ["a.txt"]})},
            {"role": "assistant", "content": advance},
            {"role": "user", "content": _obs("read_file(path='/docs/a.txt')", {"text": "hello"})},
            # A later loop that the old miner incorrectly borrowed as the reject
            # for the earlier prompt.
            {"role": "assistant", "content": advance},
        ],
    }


def test_reject_is_constructed_at_same_prefix_not_borrowed_from_later_turn():
    record = _record()
    pairs = mine_advance_vs_loop_candidates([record])
    assert len(pairs) == 1
    pair = pairs[0]

    preceding = record["messages"][1]["content"]
    later_loop = record["messages"][5]["content"]
    assert canonical_tool_calls(pair["rejected"]["content"]) == canonical_tool_calls(preceding)
    assert canonical_tool_calls(pair["rejected"]["content"]) != canonical_tool_calls(later_loop)
    assert pair["prompt"] == record["messages"][:3]


def test_pair_has_byte_identical_shared_prompts_and_distinct_responses():
    pair = mine_advance_vs_loop_candidates([_record()])[0]
    prompt_bytes = json.dumps(
        pair["prompt"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    # Both branches use the one shared prompt; hashes prove the exact bytes.
    assert pair["chosen_prompt_sha256"] == pair["rejected_prompt_sha256"]
    assert pair["prompt_sha256"] == pair["chosen_prompt_sha256"]
    assert prompt_bytes
    assert canonical_response(pair["chosen"]["content"]) != canonical_response(
        pair["rejected"]["content"]
    )
    assert validate_pair(pair) == []


def test_final_after_successful_observation_is_allowed():
    record = _record("final-parent")
    record["messages"] = record["messages"][:3] + [
        {"role": "assistant", "content": "The first file is a.txt."}
    ]
    pair = mine_advance_vs_loop_candidates([record])[0]
    assert pair["chosen_action"] == "final_after_obs"
    assert canonical_tool_calls(pair["rejected"]["content"]) == canonical_tool_calls(
        record["messages"][1]["content"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["messages"][2].update(
            content=_obs("ls(path='/docs')", {"error": "timeout; retry"})
        ),
        lambda r: r.update(metadata={"non_deterministic": True}),
        lambda r: r.update(retry_valid=True),
    ],
)
def test_retry_valid_or_ambiguous_cases_are_excluded(mutation):
    record = _record()
    mutation(record)
    assert mine_advance_vs_loop_candidates([record]) == []


def test_nondeterministic_tool_name_is_excluded():
    record = _record()
    record["messages"][1]["content"] = _call("get_current_weather", city="NYC")
    assert mine_advance_vs_loop_candidates([record]) == []


def test_bfcl_looking_ids_are_excluded_and_reported():
    bfcl = _record("multi_turn_base_42")
    clean = _record("source-parent-clean")
    mined, report = mine_splits(
        {"train": [bfcl, clean], "valid": [], "corpus_test": []}
    )
    assert {p["parent_id"] for p in mined["train"]} == {"source-parent-clean"}
    assert report["exclusions"]["train"]["excluded_bfcl"] == 1
    assert report["gate_report"]["zero_bfcl_leak"] is True


def test_parent_must_not_cross_source_splits():
    with pytest.raises(ValueError, match="crosses"):
        mine_splits(
            {
                "train": [_record("same-parent")],
                "valid": [_record("same-parent")],
                "corpus_test": [],
            }
        )


def test_dedupe_and_parent_cap_apply_after_pair_expansion():
    record = _record("capped-parent")
    duplicate = json.loads(json.dumps(record))
    pairs = mine_advance_vs_loop_candidates(
        [record, duplicate], max_per_parent=1, max_per_tool=10
    )
    assert len(pairs) == 1
