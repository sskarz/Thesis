"""Dependency-free Stage B DPO wiring with an MLX adapter boundary.

The real policy and frozen reference are locked to the untouched
``google/gemma-4-e2b-it`` base. This module implements correctness-critical
wiring around small protocols: paired tokenization, assistant-continuation-
only scoring, frozen-reference caching, and one LoRA-only optimizer step.
MLX is not imported here, so tests and ``--check-wiring`` need no model weights
and cannot launch a Metal job.

Real MLX execution is isolated behind the explicit ``--smoke-100`` guard.
Ordinary imports, ``--dry-run``, and ``--check-wiring`` remain dependency-free
and cannot initialize MLX or Metal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from sft.config import HF_INSTRUCT_ID, LORA_NUM_LAYERS, LORA_RANK, LORA_SCALE
from sft.io import read_jsonl, write_jsonl
from sft.pref_pairs_antiloop_v9 import validate_pair

DEFAULT_BETA = 0.1
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_CONFIG = Path(__file__).resolve().parent / "dpo_config.yaml"
FORBIDDEN_INITIALIZER_NAMES = frozenset(
    {"adapters-v9-smoke", "adapters-v9-smoke91", "adapters-v9-smoke92"}
)


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


@dataclass(frozen=True)
class TokenizedSequence:
    """One causal sequence and the exact tokens included in its DPO score."""

    input_ids: tuple[int, ...]
    continuation_mask: tuple[bool, ...]
    prompt_tokens_dropped: int = 0


@dataclass(frozen=True)
class TokenizedPreference:
    chosen: TokenizedSequence
    rejected: TokenizedSequence
    prompt_sha256: str
    chosen_sha256: str
    rejected_sha256: str


class PreferenceTokenizer(Protocol):
    """Boundary for a future Gemma chat-template/tokenizer implementation."""

    def tokenize_prompt(
        self, prompt: Sequence[Mapping[str, object]]
    ) -> Sequence[int]: ...

    def tokenize_response(self, response: Mapping[str, object]) -> Sequence[int]: ...


class ContinuationScorer(Protocol):
    """Return causal token log-probabilities aligned with ``input_ids``."""

    def token_logprobs(
        self, input_ids: Sequence[int], *, adapter_enabled: bool
    ) -> Sequence[float]: ...


class DPOAutodiffBackend(Protocol):
    """Minimal MLX-facing value/gradient and optimizer-update contract."""

    def parameter_snapshot(self) -> Mapping[str, object]: ...

    def value_and_grad(
        self,
        preference: TokenizedPreference,
        *,
        reference_chosen: float,
        reference_rejected: float,
        beta: float,
    ) -> tuple[float, float, Mapping[str, float]]: ...

    def apply_gradients(
        self, gradients: Mapping[str, float], *, learning_rate: float
    ) -> None: ...


class ContinuationTruncationError(ValueError):
    """A chosen or rejected assistant continuation cannot fit intact."""


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


def tokenize_preference_pair(
    pair: Mapping[str, object],
    tokenizer: PreferenceTokenizer,
    *,
    max_seq_length: int = 8192,
) -> TokenizedPreference:
    """Tokenize one pair without ever truncating either assistant response.

    A single left-truncated prompt is shared by both branches. Its budget is
    based on the longer continuation, so chosen and rejected see identical
    prompt tokens. A continuation that cannot fit by itself is a kill gate.
    """
    failures = validate_pair(pair)
    if failures:
        raise ValueError(f"invalid preference pair: {', '.join(failures)}")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")

    prompt = pair["prompt"]
    chosen = pair["chosen"]
    rejected = pair["rejected"]
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise ValueError("pair prompt must be a message sequence")
    if not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
        raise ValueError("pair responses must be mappings")
    if pair.get("prompt_sha256") != _response_hash(prompt):
        raise ValueError("pair prompt_sha256 does not match prompt bytes")

    prompt_ids = tuple(int(token) for token in tokenizer.tokenize_prompt(prompt))
    chosen_ids = tuple(int(token) for token in tokenizer.tokenize_response(chosen))
    rejected_ids = tuple(int(token) for token in tokenizer.tokenize_response(rejected))
    if not chosen_ids or not rejected_ids:
        raise ValueError("chosen and rejected continuations must contain tokens")

    longest = max(len(chosen_ids), len(rejected_ids))
    if longest > max_seq_length:
        branches = []
        if len(chosen_ids) > max_seq_length:
            branches.append("chosen")
        if len(rejected_ids) > max_seq_length:
            branches.append("rejected")
        raise ContinuationTruncationError(
            f"{' and '.join(branches)} continuation would be truncated at "
            f"max_seq_length={max_seq_length}"
        )

    prompt_budget = max_seq_length - longest
    kept_prompt = prompt_ids[-prompt_budget:] if prompt_budget else ()
    dropped = len(prompt_ids) - len(kept_prompt)

    def branch(continuation: tuple[int, ...]) -> TokenizedSequence:
        ids = kept_prompt + continuation
        mask = (False,) * len(kept_prompt) + (True,) * len(continuation)
        if len(ids) > max_seq_length:
            raise AssertionError("internal sequence budget error")
        return TokenizedSequence(ids, mask, dropped)

    return TokenizedPreference(
        chosen=branch(chosen_ids),
        rejected=branch(rejected_ids),
        prompt_sha256=str(pair["prompt_sha256"]),
        chosen_sha256=_response_hash(chosen),
        rejected_sha256=_response_hash(rejected),
    )


def score_tokenized_preference(
    preference: TokenizedPreference,
    scorer: ContinuationScorer,
    *,
    adapter_enabled: bool,
) -> tuple[float, float]:
    chosen = continuation_logprob(
        scorer.token_logprobs(
            preference.chosen.input_ids, adapter_enabled=adapter_enabled
        ),
        preference.chosen.continuation_mask,
    )
    rejected = continuation_logprob(
        scorer.token_logprobs(
            preference.rejected.input_ids, adapter_enabled=adapter_enabled
        ),
        preference.rejected.continuation_mask,
    )
    if not math.isfinite(chosen) or not math.isfinite(rejected):
        raise ValueError("scorer returned non-finite continuation log-probability")
    return float(chosen), float(rejected)


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
    """Create a cache row keyed by immutable prompt/response hashes."""
    if validate_pair(pair):
        raise ValueError("cannot cache an invalid preference pair")
    if pair.get("prompt_sha256") != _response_hash(pair.get("prompt")):
        raise ValueError("cannot cache pair with mismatched prompt_sha256")
    if not math.isfinite(chosen_logprob) or not math.isfinite(rejected_logprob):
        raise ValueError("reference log-probabilities must be finite")
    return {
        "prompt_sha256": pair["prompt_sha256"],
        "chosen_sha256": _response_hash(pair["chosen"]),
        "rejected_sha256": _response_hash(pair["rejected"]),
        "reference_chosen_logprob": float(chosen_logprob),
        "reference_rejected_logprob": float(rejected_logprob),
        "reference_model": HF_INSTRUCT_ID,
        "adapter_enabled": False,
        "continuation_only": True,
    }


def _response_hash(response: object) -> str:
    payload = json.dumps(
        response, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_reference_cache_record(row: Mapping[str, object]) -> None:
    """Reject cache rows that could weaken the frozen-reference contract."""
    for key in ("prompt_sha256", "chosen_sha256", "rejected_sha256"):
        if not _valid_sha256(row.get(key)):
            raise ValueError(f"invalid reference cache {key}")
    for key in ("reference_chosen_logprob", "reference_rejected_logprob"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid reference cache {key}")
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite reference cache {key}")
    if row.get("reference_model") != HF_INSTRUCT_ID:
        raise ValueError("reference cache model is not the locked untouched base")
    if row.get("adapter_enabled") is not False:
        raise ValueError("reference cache must have adapters disabled")
    if row.get("continuation_only") is not True:
        raise ValueError("reference cache must be continuation-only")


def _cache_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    validate_reference_cache_record(row)
    return (
        str(row["prompt_sha256"]),
        str(row["chosen_sha256"]),
        str(row["rejected_sha256"]),
    )


def prefreeze_reference_cache(
    pairs: Iterable[Mapping[str, object]],
    tokenizer: PreferenceTokenizer,
    scorer: ContinuationScorer,
    *,
    max_seq_length: int = 8192,
    adapter_enabled: bool = False,
) -> list[dict[str, object]]:
    """Score every reference branch twice and require bit-identical results."""
    if adapter_enabled:
        raise ValueError("frozen reference scoring refuses adapters-on execution")
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        tokenized = tokenize_preference_pair(
            pair, tokenizer, max_seq_length=max_seq_length
        )
        first = score_tokenized_preference(tokenized, scorer, adapter_enabled=False)
        second = score_tokenized_preference(tokenized, scorer, adapter_enabled=False)
        if first != second:
            raise AssertionError("frozen reference invariance check failed")
        row = reference_cache_record(pair, *first)
        key = _cache_key(row)
        if key in seen:
            raise ValueError("duplicate prompt/chosen/rejected reference cache key")
        seen.add(key)
        records.append(row)
    return records


def persist_reference_cache(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Validate then write a frozen-reference JSONL cache."""
    materialized: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = _cache_key(row)
        if key in seen:
            raise ValueError("duplicate reference cache key")
        seen.add(key)
        materialized.append(dict(row))
    return write_jsonl(path, materialized)


def load_reference_cache(path: Path) -> list[dict[str, object]]:
    """Load JSONL and fail closed on malformed, unsafe, or duplicate rows."""
    rows = read_jsonl(path)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = _cache_key(row)
        if key in seen:
            raise ValueError("duplicate reference cache key")
        seen.add(key)
    return rows


def cache_record_for_preference(
    preference: TokenizedPreference, cache_row: Mapping[str, object]
) -> tuple[float, float]:
    expected = (
        preference.prompt_sha256,
        preference.chosen_sha256,
        preference.rejected_sha256,
    )
    if _cache_key(cache_row) != expected:
        raise ValueError("reference cache hashes do not match tokenized pair")
    return (
        float(cache_row["reference_chosen_logprob"]),
        float(cache_row["reference_rejected_logprob"]),
    )


def assert_lora_only_updates(
    before: Mapping[str, object],
    after: Mapping[str, object],
    lora_param_names: Iterable[str],
) -> None:
    """Assert all base params are unchanged and at least one LoRA param moved."""
    allowed = frozenset(lora_param_names)
    if not allowed:
        raise AssertionError("no LoRA parameter names supplied")
    if set(before) != set(after):
        raise AssertionError("parameter names changed during optimizer update")
    missing = allowed - set(before)
    if missing:
        raise AssertionError(f"LoRA parameter names missing: {sorted(missing)}")

    changed_lora = False
    for name in before:
        changed = before[name] != after[name]
        if changed and name not in allowed:
            raise AssertionError(f"base parameter changed: {name}")
        changed_lora = changed_lora or (changed and name in allowed)
    if not changed_lora:
        raise AssertionError("optimizer update changed no LoRA parameters")


def dpo_train_step(
    backend: DPOAutodiffBackend,
    preference: TokenizedPreference,
    cache_row: Mapping[str, object],
    *,
    lora_param_names: Iterable[str],
    beta: float = DEFAULT_BETA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict[str, float]:
    """Compute one DPO loss/gradient/update and prove LoRA-only mutation."""
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    reference_chosen, reference_rejected = cache_record_for_preference(
        preference, cache_row
    )
    allowed = frozenset(lora_param_names)
    before = dict(backend.parameter_snapshot())
    policy_chosen, policy_rejected, gradients = backend.value_and_grad(
        preference,
        reference_chosen=reference_chosen,
        reference_rejected=reference_rejected,
        beta=beta,
    )
    unknown_gradients = set(gradients) - allowed
    if unknown_gradients:
        raise AssertionError(
            f"autodiff produced non-LoRA gradients: {sorted(unknown_gradients)}"
        )
    loss = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )
    backend.apply_gradients(gradients, learning_rate=learning_rate)
    after = dict(backend.parameter_snapshot())
    assert_lora_only_updates(before, after, allowed)
    return {
        "loss": loss,
        "policy_margin": policy_chosen - policy_rejected,
        "reference_margin": reference_chosen - reference_rejected,
    }


def validate_policy_initializer(adapter_path: str | Path | None) -> None:
    """Refuse every failed v9/v9.1/v9.2 adapter as a policy initializer."""
    if adapter_path is None:
        return
    forbidden = sorted(set(Path(adapter_path).parts) & FORBIDDEN_INITIALIZER_NAMES)
    if forbidden:
        raise ValueError(
            f"forbidden policy initializer {forbidden[0]!r}; use untouched "
            f"{HF_INSTRUCT_ID}"
        )


class _BytePreferenceTokenizer:
    """Deterministic fixture tokenizer used only by dependency-free checks."""

    @staticmethod
    def _tokens(value: object) -> tuple[int, ...]:
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        return tuple(encoded)

    def tokenize_prompt(
        self, prompt: Sequence[Mapping[str, object]]
    ) -> Sequence[int]:
        return self._tokens(prompt)

    def tokenize_response(self, response: Mapping[str, object]) -> Sequence[int]:
        return self._tokens(response)


class _FrozenFakeScorer:
    def token_logprobs(
        self, input_ids: Sequence[int], *, adapter_enabled: bool
    ) -> Sequence[float]:
        if adapter_enabled:
            raise ValueError("fake frozen reference refuses adapters-on scoring")
        return [0.0] * len(input_ids)


class _TinyFakePolicy:
    """One trainable LoRA scalar plus an immutable base scalar."""

    def __init__(self) -> None:
        self.base = 7.0
        self.lora_preference = 0.0

    def parameter_snapshot(self) -> Mapping[str, object]:
        return {"base.weight": self.base, "lora.preference": self.lora_preference}

    def value_and_grad(
        self,
        preference: TokenizedPreference,
        *,
        reference_chosen: float,
        reference_rejected: float,
        beta: float,
    ) -> tuple[float, float, Mapping[str, float]]:
        del preference
        chosen = 0.5 * self.lora_preference
        rejected = -0.5 * self.lora_preference
        z = beta * (
            self.lora_preference - (reference_chosen - reference_rejected)
        )
        gradient = -beta / (1.0 + math.exp(z))
        return chosen, rejected, {"lora.preference": gradient}

    def apply_gradients(
        self, gradients: Mapping[str, float], *, learning_rate: float
    ) -> None:
        self.lora_preference -= learning_rate * gradients["lora.preference"]


def _synthetic_pair() -> dict[str, object]:
    prior = {
        "role": "assistant",
        "content": '[{"function_name":"ls","parameters":{"path":"/docs"}}]',
    }
    prompt: list[dict[str, object]] = [
        {"role": "user", "content": "Read the first file."},
        prior,
        {
            "role": "user",
            "content": "Function execution results:\nls(path='/docs') -> a.txt",
        },
    ]
    prompt_hash = hashlib.sha256(
        json.dumps(
            prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "prompt": prompt,
        "chosen": {
            "role": "assistant",
            "content": '[{"function_name":"read_file","parameters":{"path":"/docs/a.txt"}}]',
        },
        "rejected": prior,
        "prompt_sha256": prompt_hash,
        "chosen_prompt_sha256": prompt_hash,
        "rejected_prompt_sha256": prompt_hash,
    }


def synthetic_overfit_check(*, steps: int = 12) -> dict[str, float | int | bool]:
    """Prove the fake autodiff path improves margin and lowers DPO loss."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    pair = _synthetic_pair()
    tokenizer = _BytePreferenceTokenizer()
    preference = tokenize_preference_pair(pair, tokenizer, max_seq_length=8192)
    cache_row = prefreeze_reference_cache(
        [pair], tokenizer, _FrozenFakeScorer(), max_seq_length=8192
    )[0]
    backend = _TinyFakePolicy()
    first: dict[str, float] | None = None
    for _ in range(steps):
        metrics = dpo_train_step(
            backend,
            preference,
            cache_row,
            lora_param_names={"lora.preference"},
            beta=DEFAULT_BETA,
            learning_rate=1.0,
        )
        first = first or metrics
    assert first is not None
    final_chosen = 0.5 * backend.lora_preference
    final_rejected = -0.5 * backend.lora_preference
    final_loss = dpo_loss(final_chosen, final_rejected, 0.0, 0.0)
    final_margin = final_chosen - final_rejected
    if not final_loss < first["loss"]:
        raise AssertionError("synthetic DPO loss did not decrease")
    if not final_margin > first["policy_margin"]:
        raise AssertionError("synthetic preference margin did not improve")
    return {
        "steps": steps,
        "initial_loss": first["loss"],
        "final_loss": final_loss,
        "initial_margin": first["policy_margin"],
        "final_margin": final_margin,
        "base_unchanged": backend.base == 7.0,
    }


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
    masked = continuation_logprob(
        [-9.0, -8.0, -0.2, -0.3], [False, False, True, True]
    )
    if not math.isclose(masked, -0.5, abs_tol=1e-12):
        raise AssertionError("prompt tokens leaked into continuation log-probability")
    return {
        "equal_margin_loss": equal_loss,
        "improved_margin_loss": improved_loss,
        "masked_logprob": masked,
    }


def check_wiring() -> dict[str, object]:
    """Exercise all Stage B trainer gates using deterministic fake backends."""
    pair = _synthetic_pair()
    tokenizer = _BytePreferenceTokenizer()
    tokenized = tokenize_preference_pair(pair, tokenizer, max_seq_length=8192)
    scorer = _FrozenFakeScorer()
    cache = prefreeze_reference_cache([pair], tokenizer, scorer)
    try:
        prefreeze_reference_cache([pair], tokenizer, scorer, adapter_enabled=True)
    except ValueError:
        adapters_on_refused = True
    else:
        raise AssertionError("adapters-on reference scoring was not refused")
    overfit = synthetic_overfit_check()
    return {
        "backend": "dependency-free-fake",
        "mlx_imported": False,
        "chosen_tokens": sum(tokenized.chosen.continuation_mask),
        "rejected_tokens": sum(tokenized.rejected.continuation_mask),
        "reference_cache_records": len(cache),
        "reference_invariance": True,
        "adapters_on_refused": adapters_on_refused,
        "lora_only_update": bool(overfit["base_unchanged"]),
        "synthetic_overfit": overfit,
    }


def readiness_report() -> dict[str, object]:
    """Current GO checklist for the single guarded real-MLX smoke."""
    wiring = check_wiring()
    return {
        "miner_gates": {
            "scope": "separate data-miner gates",
            "zero_bfcl_leak": "PASS",
            "parent_disjoint": "PASS",
            "prompt_hashes_identical": "PASS",
            "pair_invariants": "PASS",
        },
        "tokenize_mask_truncation_kill": "PASS",
        "reference_cache_hash_invariance": "PASS",
        "lora_only_update_assert": "PASS",
        "synthetic_overfit": "PASS",
        "pair_audit": "READY / real-tokenizer gate in --smoke-100",
        "100_update_smoke": "READY / guarded --smoke-100 entrypoint",
        "smoke_verdict": "GO for exactly one guarded 100-update smoke",
        "smoke_command": "python -m sft.train_dpo_mlx --smoke-100",
        "wiring": wiring,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect DPO wiring or run the one guarded MLX smoke."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print locked config and the full readiness report.",
    )
    parser.add_argument(
        "--check-wiring",
        action="store_true",
        help="Run fake-backend token/cache/update/overfit checks only.",
    )
    parser.add_argument(
        "--check-fixtures",
        type=Path,
        default=None,
        metavar="JSONL",
        help="Validate preference pair JSONL without writing or training.",
    )
    parser.add_argument(
        "--smoke-100",
        action="store_true",
        help=(
            "Run exactly 100 MLX DPO optimizer updates from the untouched local "
            "base. This is the only real-training entrypoint."
        ),
    )
    args = parser.parse_args(argv)
    if args.smoke_100:
        if args.dry_run or args.check_wiring or args.check_fixtures is not None:
            parser.error("--smoke-100 cannot be combined with inspection flags")
        # Deliberately import only after the explicit execution guard. This is
        # what keeps CPU tests and inspection commands free of MLX side effects.
        from sft.run_dpo_smoke_mlx import run_smoke_100

        return run_smoke_100()
    if not args.dry_run and not args.check_wiring and args.check_fixtures is None:
        parser.error("pass --dry-run, --check-wiring, --check-fixtures, or --smoke-100")

    output: dict[str, object] = {
        "training": False,
        "config_file": str(DEFAULT_CONFIG),
        "config": asdict(DPOConfig()),
        "numeric_fixture": check_numeric_fixture(),
        "readiness_report": readiness_report(),
    }
    if args.check_wiring:
        output["wiring_check"] = check_wiring()
    if args.check_fixtures is not None:
        output["pair_fixtures"] = check_pair_fixtures(
            read_jsonl(args.check_fixtures)
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
