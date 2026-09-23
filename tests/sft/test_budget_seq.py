from __future__ import annotations

import json

import pytest

from sft.budget_seq import (
    DEFAULT_MODEL_DIR,
    FUNCTIONS_END,
    FUNCTIONS_START,
    POLICY_MARKER,
    BudgetGateError,
    budget_messages,
    budget_unrolled_rows,
    compact_system_content,
    load_tokenizer,
)


class CharTokenizer:
    """Small deterministic stand-in for chat-template length accounting."""

    def apply_chat_template(
        self, messages, *, add_generation_prompt=False, return_dict=False
    ):
        assert return_dict is False
        length = sum(len(str(message.get("content", ""))) + 2 for message in messages)
        if add_generation_prompt:
            length += 2
        return list(range(length))


def _system(schema: list[dict], policy: str = "") -> str:
    text = (
        "Contract. "
        + FUNCTIONS_START
        + json.dumps(schema, indent=2)
        + FUNCTIONS_END
        + " answer with JSON."
    )
    if policy:
        text += POLICY_MARKER + policy
    return text


def test_compact_system_reduces_length_without_changing_schema_fields():
    schema = [
        {
            "name": "lookup_order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "enum": ["A", "B"],
                    }
                },
                "required": ["order_id"],
            },
        }
    ]
    original = _system(schema, "policy " * 200)
    compacted = compact_system_content(original, policy_max_chars=64)

    assert len(compacted) < len(original)
    schema_text = compacted.split(FUNCTIONS_START, 1)[1].split(FUNCTIONS_END, 1)[0]
    assert json.loads(schema_text) == schema
    assert POLICY_MARKER in compacted
    assert len(compacted.split(POLICY_MARKER, 1)[1]) <= 64


def test_message_trim_keeps_system_and_final_assistant_target():
    messages = [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "old question " * 8},
        {"role": "assistant", "content": "old call " * 8},
        {"role": "user", "content": "Function execution results: old result"},
        {"role": "user", "content": "new question " * 4},
        {"role": "assistant", "content": "FINAL_TARGET"},
    ]
    budgeted, stats = budget_messages(messages, CharTokenizer(), max_seq_length=90)

    assert budgeted[0] == messages[0]
    assert budgeted[-1] == messages[-1]
    assert budgeted[0]["role"] == "system"
    assert budgeted[-1]["role"] == "assistant"
    assert "Function execution results: old result" not in {
        message["content"] for message in budgeted
    }
    assert stats["turns_dropped"] >= 1
    assert "trim_history" in stats["actions"]
    assert stats["after"]["offset"] < 90
    assert stats["after"]["target_tokens"] > 0


def test_irreducible_overlong_row_triggers_aggregated_hard_gate():
    row = {
        "id": "too-long",
        "source": "apigen_mt",
        "messages": [
            {"role": "system", "content": "x" * 200},
            {"role": "user", "content": "drop me"},
            {"role": "assistant", "content": "target"},
        ],
    }

    with pytest.raises(BudgetGateError, match="too-long") as exc:
        budget_unrolled_rows(
            [row], max_seq_length=100, tokenizer=CharTokenizer(), hard_gate=True
        )

    assert exc.value.failures[0]["source"] == "apigen_mt"
    assert exc.value.failures[0]["offset"] >= 100


def test_budget_report_counts_actions_and_sources():
    row = {
        "id": "compact-me",
        "source": "apigen_mt",
        "messages": [
            {"role": "system", "content": _system([{"name": "f"}])},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "answer"},
        ],
    }
    budgeted, report = budget_unrolled_rows(
        [row], max_seq_length=1000, tokenizer=CharTokenizer()
    )

    assert budgeted[0]["id"] == row["id"]
    assert report["empty_mask_after"] == 0
    assert report["rows_compacted"] == 1
    assert report["per_source"]["apigen_mt"]["rows"] == 1
    assert report["per_source"]["apigen_mt"]["rows_compacted"] == 1


@pytest.mark.skipif(not DEFAULT_MODEL_DIR.exists(), reason="local MLX tokenizer absent")
def test_real_local_tokenizer_matches_positive_target_accounting():
    tokenizer = load_tokenizer(DEFAULT_MODEL_DIR)
    messages = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello!"},
    ]
    _budgeted, stats = budget_messages(messages, tokenizer, max_seq_length=8192)

    assert stats["after"]["full_len"] > stats["after"]["offset"]
    assert stats["after"]["target_tokens"] > 0
    assert stats["gate_failed"] is False
