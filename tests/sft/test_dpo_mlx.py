"""Dependency-free tests for the Stage B DPO objective scaffold."""

from __future__ import annotations

import math

import pytest

from sft.train_dpo_mlx import (
    DPOConfig,
    check_numeric_fixture,
    continuation_logprob,
    dpo_loss,
)


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
