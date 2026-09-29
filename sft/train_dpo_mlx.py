"""Reference-anchored MLX DPO scaffold for Stage B (no training entrypoint).

The future policy and frozen reference must both start from the untouched
``google/gemma-4-e2b-it`` base. Failed v9/v9.1/v9.2 smoke adapters are not
valid initializers. The intended trainable state is LoRA only (rank 16,
alpha/scale 32, 16 layers); beta defaults to 0.1 and learning rate to 1e-6.

This scaffold defines and tests the assistant-continuation-only DPO objective,
fixture validation, and the frozen-reference cache record. It deliberately
does not expose ``--run`` and never imports MLX during dry-run/check-fixtures.
A real Mac implementation still needs reviewed model/tokenizer wiring,
prompt/target token masking, reference-logprob precomputation with adapters
disabled, cache validation, optimizer/update isolation, and checkpoint I/O.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sft.config import HF_INSTRUCT_ID, LORA_NUM_LAYERS, LORA_RANK, LORA_SCALE
from sft.io import read_jsonl
from sft.pref_pairs_antiloop_v9 import validate_pair

DEFAULT_BETA = 0.1
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_CONFIG = Path(__file__).resolve().parent / "dpo_config.yaml"


@dataclass(frozen=True)
class DPOConfig:
    model: str = HF_INSTRUCT_ID
    reference_model: str = HF_INSTRUCT_ID
    beta: float = DEFAULT_BETA
    learning_rate: float = DEFAULT_LEARNING_RATE
    lora_rank: int = LORA_RANK
    lora_scale: float = LORA_SCALE
    num_layers: int = LORA_NUM_LAYERS
    batch_size: int = 1
    max_seq_length: int = 8192
    continuation_only: bool = True
    precompute_reference_logprobs: bool = True


def continuation_logprob(
    token_logprobs: Sequence[float], continuation_mask: Sequence[bool]
) -> float:
    """Sum log-probabilities only where the assistant continuation mask is true."""
    if len(token_logprobs) != len(continuation_mask):
        raise ValueError("token_logprobs and continuation_mask lengths differ")
    if not any(continuation_mask):
        raise ValueError("continuation mask selects no assistant tokens")
    return float(
        sum(value for value, include in zip(token_logprobs, continuation_mask) if include)
    )


def dpo_logit(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    *,
    beta: float = DEFAULT_BETA,
) -> float:
    """Return beta-scaled policy-vs-reference preference log-odds."""
    if beta <= 0 or not all(
        math.isfinite(value)
        for value in (
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
        )
    ):
        raise ValueError("beta must be positive and all log-probabilities finite")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    return beta * (policy_margin - reference_margin)


def dpo_loss(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    *,
    beta: float = DEFAULT_BETA,
) -> float:
    """Stable scalar ``-log(sigmoid(dpo_logit))`` for a single pair."""
    z = dpo_logit(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )
    return max(-z, 0.0) + math.log1p(math.exp(-abs(z)))


def reference_cache_record(
    pair: Mapping[str, object], chosen_logprob: float, rejected_logprob: float
) -> dict[str, object]:
    """Create a cache row keyed by the immutable prompt/response hashes."""
    if validate_pair(pair):
        raise ValueError("cannot cache an invalid preference pair")
    if not math.isfinite(chosen_logprob) or not math.isfinite(rejected_logprob):
        raise ValueError("reference log-probabilities must be finite")
    return {
        "prompt_sha256": pair["prompt_sha256"],
        "chosen_sha256": _response_hash(pair["chosen"]),
        "rejected_sha256": _response_hash(pair["rejected"]),
        "reference_chosen_logprob": chosen_logprob,
        "reference_rejected_logprob": rejected_logprob,
        "reference_model": HF_INSTRUCT_ID,
        "adapter_enabled": False,
        "continuation_only": True,
    }


def _response_hash(response: object) -> str:
    import hashlib

    payload = json.dumps(
        response, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def check_pair_fixtures(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    checked = invalid = 0
    for row in rows:
        checked += 1
        invalid += bool(validate_pair(row))
    if invalid:
        raise ValueError(f"{invalid}/{checked} preference fixtures failed invariants")
    return {"checked": checked, "invalid": invalid}


def check_numeric_fixture() -> dict[str, float]:
    """Run dependency-free hand-computed loss and continuation-mask checks."""
    equal_loss = dpo_loss(-2.0, -3.0, -2.0, -3.0, beta=0.1)
    expected = math.log(2.0)
    if not math.isclose(equal_loss, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"equal-margin DPO loss {equal_loss} != ln(2)")
    improved_loss = dpo_loss(-1.0, -3.0, -2.0, -3.0, beta=0.1)
    if not improved_loss < equal_loss:
        raise AssertionError("improved policy margin did not lower DPO loss")
    masked = continuation_logprob([-9.0, -8.0, -0.2, -0.3], [False, False, True, True])
    if not math.isclose(masked, -0.5, abs_tol=1e-12):
        raise AssertionError("prompt tokens leaked into continuation log-probability")
    return {"equal_margin_loss": equal_loss, "improved_margin_loss": improved_loss, "masked_logprob": masked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the MLX DPO scaffold; no GPU training is implemented."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print locked config and remaining implementation gates.")
    parser.add_argument("--check-fixtures", type=Path, default=None, metavar="JSONL", help="Validate pair JSONL and the hand-computed loss fixture.")
    args = parser.parse_args(argv)
    if not args.dry_run and args.check_fixtures is None:
        parser.error("pass --dry-run or --check-fixtures; this scaffold cannot train")

    output: dict[str, object] = {
        "training": False,
        "config_file": str(DEFAULT_CONFIG),
        "config": asdict(DPOConfig()),
        "numeric_fixture": check_numeric_fixture(),
        "reference_cache": "precompute with base adapters disabled; implementation follow-up required",
        "remaining_gates": [
            "reviewed MLX tokenization and assistant continuation masks",
            "frozen reference cache reproducibility",
            "LoRA-only parameter update assertion",
            "tiny synthetic overfit before any 100-update Mac smoke",
        ],
    }
    if args.check_fixtures is not None:
        output["pair_fixtures"] = check_pair_fixtures(read_jsonl(args.check_fixtures))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
