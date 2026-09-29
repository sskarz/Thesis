"""Dependency-free tests for the Stage B DPO objective scaffold."""

from __future__ import annotations

import json
import math

import pytest

from benchmark.iffc import format_iffc_execution_user_message
from sft.pref_pairs_antiloop_v9 import mine_advance_vs_loop_candidates
from sft.train_dpo_mlx import (
    ContinuationTruncationError,
    DPOConfig,
    assert_lora_only_updates,
    check_numeric_fixture,
    continuation_logprob,
    dpo_loss,
    load_reference_cache,
    persist_reference_cache,
    prefreeze_reference_cache,
    readiness_report,
    synthetic_overfit_check,
    tokenize_preference_pair,
    validate_policy_initializer,
)


def _call(name: str, **params: object) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def _pair() -> dict:
    first = _call("ls", path="/docs")
    advance = _call("read_file", path="/docs/a.txt")
    record = {
        "id": "dpo-fixture-parent",
        "source": "toolace",
        "messages": [
            {"role": "user", "content": "Read the first file in /docs"},
            {"role": "assistant", "content": first},
            {
                "role": "user",
                "content": format_iffc_execution_user_message(
                    ["ls(path='/docs')"], [{"files": ["a.txt"]}]
                ),
            },
            {"role": "assistant", "content": advance},
        ],
    }
    return mine_advance_vs_loop_candidates([record])[0]


class FakeTokenizer:
    def __init__(self, prompt: list[int], chosen: list[int], rejected: list[int]):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected

    def tokenize_prompt(self, prompt):
        del prompt
        return self.prompt

    def tokenize_response(self, response):
        return self.chosen if "read_file" in response["content"] else self.rejected


class FakeReference:
    def __init__(self):
        self.calls: list[bool] = []

    def token_logprobs(
        self, input_ids, *, adapter_enabled: bool
    ) -> list[float]:
        self.calls.append(adapter_enabled)
        if adapter_enabled:
            raise ValueError("adapters on")
        return [-0.25] * len(input_ids)


def test_hand_computed_equal_margin_loss_is_log_two():
    assert dpo_loss(-2.0, -3.0, -4.0, -5.0, beta=0.1) == pytest.approx(
        math.log(2.0), abs=1e-12
    )


def test_better_policy_preference_margin_lowers_loss():
    baseline = dpo_loss(-2.0, -3.0, -2.0, -3.0, beta=0.1)
    improved = dpo_loss(-1.0, -3.0, -2.0, -3.0, beta=0.1)
    assert improved < baseline


def test_only_assistant_continuation_tokens_are_summed():
    assert continuation_logprob(
        [-100.0, -100.0, -0.25, -0.5], [False, False, True, True]
    ) == pytest.approx(-0.75)
    with pytest.raises(ValueError, match="selects no assistant"):
        continuation_logprob([-1.0], [False])


def test_locked_recipe_uses_untouched_base_and_expected_lora():
    config = DPOConfig()
    assert config.model == config.reference_model == "google/gemma-4-e2b-it"
    assert config.lora_rank == 16
    assert config.lora_scale == 32.0
    assert config.num_layers == 16
    assert config.beta == 0.1
    assert config.learning_rate == 1e-6
    assert config.continuation_only is True
    assert config.precompute_reference_logprobs is True


def test_dependency_free_numeric_fixture():
    report = check_numeric_fixture()
    assert report["equal_margin_loss"] == pytest.approx(math.log(2.0))
    assert report["improved_margin_loss"] < report["equal_margin_loss"]


def test_tokenize_masks_prompt_and_preserves_both_continuations():
    pair = _pair()
    tokenized = tokenize_preference_pair(
        pair,
        FakeTokenizer(
            prompt=list(range(10)), chosen=[101, 102, 103], rejected=[201, 202]
        ),
        max_seq_length=8,
    )
    assert tokenized.chosen.input_ids == (5, 6, 7, 8, 9, 101, 102, 103)
    assert tokenized.rejected.input_ids == (5, 6, 7, 8, 9, 201, 202)
    assert tokenized.chosen.continuation_mask == (
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    )
    assert tokenized.rejected.continuation_mask[-2:] == (True, True)
    assert tokenized.chosen.prompt_tokens_dropped == 5
    assert tokenized.rejected.prompt_tokens_dropped == 5


def test_tokenize_kills_if_either_continuation_would_be_truncated():
    with pytest.raises(ContinuationTruncationError, match="chosen continuation"):
        tokenize_preference_pair(
            _pair(),
            FakeTokenizer(prompt=[1], chosen=list(range(9)), rejected=[2]),
            max_seq_length=8,
        )


def test_reference_cache_roundtrip_invariance_and_adapters_on_refused(tmp_path):
    pair = _pair()
    tokenizer = FakeTokenizer(prompt=[1, 2], chosen=[3, 4], rejected=[5])
    scorer = FakeReference()
    rows = prefreeze_reference_cache([pair], tokenizer, scorer)
    assert scorer.calls == [False, False, False, False]
    assert rows[0]["adapter_enabled"] is False
    assert rows[0]["reference_chosen_logprob"] == pytest.approx(-0.5)
    assert rows[0]["reference_rejected_logprob"] == pytest.approx(-0.25)

    path = tmp_path / "reference.jsonl"
    assert persist_reference_cache(path, rows) == 1
    assert load_reference_cache(path) == rows
    unsafe = dict(rows[0], adapter_enabled=True)
    path.write_text(json.dumps(unsafe) + "\n")
    with pytest.raises(ValueError, match="adapters disabled"):
        load_reference_cache(path)
    with pytest.raises(ValueError, match="refuses adapters-on"):
        prefreeze_reference_cache(
            [pair], tokenizer, scorer, adapter_enabled=True
        )


def test_reference_cache_invariance_failure_is_kill_gate():
    class DriftingReference(FakeReference):
        def token_logprobs(self, input_ids, *, adapter_enabled):
            values = super().token_logprobs(
                input_ids, adapter_enabled=adapter_enabled
            )
            return [value + len(self.calls) * 1e-3 for value in values]

    with pytest.raises(AssertionError, match="invariance"):
        prefreeze_reference_cache(
            [_pair()],
            FakeTokenizer(prompt=[1], chosen=[2], rejected=[3]),
            DriftingReference(),
        )


def test_lora_only_update_assert_rejects_base_mutation():
    before = {"base.weight": 3.0, "lora.a": 0.0}
    assert_lora_only_updates(before, {"base.weight": 3.0, "lora.a": 0.1}, {"lora.a"})
    with pytest.raises(AssertionError, match="base parameter changed"):
        assert_lora_only_updates(
            before, {"base.weight": 2.9, "lora.a": 0.1}, {"lora.a"}
        )


def test_synthetic_overfit_improves_margin_and_lowers_loss():
    report = synthetic_overfit_check(steps=8)
    assert report["final_margin"] > report["initial_margin"]
    assert report["final_loss"] < report["initial_loss"]
    assert report["base_unchanged"] is True


@pytest.mark.parametrize(
    "path",
    [
        "sft/adapters-v9-smoke",
        "/tmp/work/sft/adapters-v9-smoke91/checkpoint",
        "sft/adapters-v9-smoke92",
    ],
)
def test_failed_smoke_adapters_are_forbidden_initializers(path):
    with pytest.raises(ValueError, match="forbidden policy initializer"):
        validate_policy_initializer(path)
    validate_policy_initializer(None)


def test_readiness_exposes_only_guarded_100_update_smoke():
    report = readiness_report()
    assert report["tokenize_mask_truncation_kill"] == "PASS"
    assert report["reference_cache_hash_invariance"] == "PASS"
    assert report["lora_only_update_assert"] == "PASS"
    assert report["synthetic_overfit"] == "PASS"
    assert report["pair_audit"].startswith("READY")
    assert report["100_update_smoke"].startswith("READY")
    assert report["smoke_verdict"] == "GO for exactly one guarded 100-update smoke"


def test_smoke_row_order_is_deterministic_and_round_robins_strata():
    from sft.run_dpo_smoke_mlx import stratified_row_order

    rows = [
        {"source": "b", "chosen_action": "x"},
        {"source": "a", "chosen_action": "y"},
        {"source": "a", "chosen_action": "y"},
        {"source": "b", "chosen_action": "x"},
    ]
    order = stratified_row_order(rows, seed=7)
    assert order == stratified_row_order(rows, seed=7)
    assert {tuple(rows[index].values()) for index in order[:2]} == {
        ("a", "y"),
        ("b", "x"),
    }
