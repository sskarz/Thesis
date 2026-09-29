"""One guarded, local-only, 100-update MLX DPO smoke.

This module intentionally imports MLX only inside :func:`run_smoke_100`.
Do not turn it into a general trainer: the experiment contract is one fixed
smoke from the untouched local base, with no adapter resume path.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sft.config import MLX_MODEL_DIR, SFT_ROOT
from sft.io import read_jsonl, write_jsonl
from sft.train_dpo_mlx import (
    DEFAULT_BETA,
    DEFAULT_LEARNING_RATE,
    FORBIDDEN_INITIALIZER_NAMES,
    _response_hash,
    validate_policy_initializer,
)

UPDATES = 100
SUBSET_SIZE = 128
MAX_SEQ_LENGTH = 8192
SEED = 57
LORA_RANK = 16
LORA_SCALE = 32.0
LORA_LAYERS = 16
PAIR_PATH = SFT_ROOT / "data" / "pref-v9" / "train.jsonl"
ADAPTER_PATH = SFT_ROOT / "adapters-dpo-smoke100"
LOG_PATH = Path("/tmp/sft_logs/dpo_smoke100.log")
STATUS_PATH = Path("/tmp/sft_logs/dpo_smoke100_STATUS.json")


@dataclass(frozen=True)
class SmokePair:
    key: tuple[str, str, str]
    source: str
    chosen_action: str
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    continuation_start: int


def stratified_row_order(
    rows: Sequence[Mapping[str, object]], *, seed: int = SEED
) -> list[int]:
    """Return deterministic round-robin order across source/action strata."""
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (str(row.get("source", "unknown")), str(row.get("chosen_action", "unknown")))
        buckets[key].append(index)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    keys = sorted(buckets)
    ordered: list[int] = []
    cursor = 0
    while True:
        added = False
        for key in keys:
            values = buckets[key]
            if cursor < len(values):
                ordered.append(values[cursor])
                added = True
        if not added:
            return ordered
        cursor += 1


def encode_pair(
    row: Mapping[str, object], tokenizer: Any, *, max_seq_length: int = MAX_SEQ_LENGTH
) -> tuple[SmokePair | None, str | None]:
    """Apply the real chat template and reject every would-be truncation."""
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(prompt, list) or not isinstance(chosen, dict) or not isinstance(rejected, dict):
        raise ValueError("malformed preference row")
    prompt_ids = tuple(
        int(x)
        for x in tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
    )

    branches: dict[str, tuple[int, ...]] = {}
    for name, response in (("chosen", chosen), ("rejected", rejected)):
        full = tuple(
            int(x)
            for x in tokenizer.apply_chat_template(
                prompt + [response], tokenize=True, add_generation_prompt=False
            )
        )
        if full[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"real chat template changed the shared prompt for {name}")
        if len(full) <= len(prompt_ids):
            raise ValueError(f"{name} continuation tokenized empty")
        if len(full) > max_seq_length:
            return None, f"{name}_would_truncate"
        branches[name] = full

    key = (
        str(row["prompt_sha256"]),
        _response_hash(chosen),
        _response_hash(rejected),
    )
    return (
        SmokePair(
            key=key,
            source=str(row.get("source", "unknown")),
            chosen_action=str(row.get("chosen_action", "unknown")),
            chosen_ids=branches["chosen"],
            rejected_ids=branches["rejected"],
            continuation_start=len(prompt_ids),
        ),
        None,
    )


def select_tokenized_subset(
    rows: Sequence[Mapping[str, object]], tokenizer: Any, *, limit: int = SUBSET_SIZE
) -> tuple[list[SmokePair], dict[str, int]]:
    selected: list[SmokePair] = []
    drops: dict[str, int] = defaultdict(int)
    for index in stratified_row_order(rows):
        pair, reason = encode_pair(rows[index], tokenizer)
        if pair is None:
            drops[str(reason)] += 1
            continue
        selected.append(pair)
        if len(selected) == limit:
            break
    return selected, dict(sorted(drops.items()))


def _sha256_files(paths: Sequence[Path]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        output[path.name] = digest.hexdigest()
    return output


def _write_status(payload: Mapping[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parameter_signatures(model: Any, excluded: set[str], mx: Any, tree_flatten: Any):
    """Sample every frozen tensor so runtime base mutation is a kill gate."""
    pending: list[tuple[str, tuple[int, ...], str, Any]] = []
    for name, value in tree_flatten(model.parameters()):
        if name in excluded:
            continue
        shape = tuple(int(dimension) for dimension in value.shape)
        # Some tied Gemma tensors have more than 2^31 logical elements. MLX
        # uses 32-bit reshape dimensions, so flattening those tensors merely
        # to sample them overflows before training starts. Index each axis
        # directly instead; this also avoids materializing a giant flat view.
        locations = {
            tuple(0 for _ in shape),
            tuple(dimension // 2 for dimension in shape),
            tuple(dimension - 1 for dimension in shape),
        }
        sample = mx.stack([value[location] for location in sorted(locations)]).astype(
            mx.float32
        )
        pending.append((name, shape, str(value.dtype), sample))
    mx.eval([sample for _, _, _, sample in pending])
    return {
        name: (shape, dtype, tuple(float(value) for value in sample.tolist()))
        for name, shape, dtype, sample in pending
    }


def run_smoke_100() -> int:
    started = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("w", buffering=1)

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    status: dict[str, object] = {
        "schema": "dpo-smoke100-v1",
        "training": True,
        "updates_requested": UPDATES,
        "updates_completed": 0,
        "model_dir": str(MLX_MODEL_DIR),
        "pair_path": str(PAIR_PATH),
        "adapter_path": str(ADAPTER_PATH),
        "log_path": str(LOG_PATH),
        "PASS": False,
    }
    try:
        validate_policy_initializer(None)
        if any(name in str(MLX_MODEL_DIR) for name in FORBIDDEN_INITIALIZER_NAMES):
            raise ValueError("forbidden adapter appeared in model initializer")
        if not MLX_MODEL_DIR.is_dir() or not (MLX_MODEL_DIR / "config.json").is_file():
            raise FileNotFoundError(f"local MLX model missing: {MLX_MODEL_DIR}")
        if not PAIR_PATH.is_file():
            raise FileNotFoundError(f"preference train split missing: {PAIR_PATH}")
        if ADAPTER_PATH.exists():
            raise FileExistsError(f"refusing to overwrite smoke adapter: {ADAPTER_PATH}")

        # MLX imports are below every CLI and path guard by design.
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers

        log(f"loading untouched local base only: {MLX_MODEL_DIR}")
        model_files = sorted(MLX_MODEL_DIR.glob("*.safetensors"))
        base_files_before = _sha256_files(model_files)
        model, tokenizer = load(
            str(MLX_MODEL_DIR), tokenizer_config={"trust_remote_code": True}
        )
        model.eval()

        rows = read_jsonl(PAIR_PATH)
        selected, truncation_drops = select_tokenized_subset(rows, tokenizer)
        if len(selected) < 32:
            raise RuntimeError(
                f"kill: real-tokenizer subset has {len(selected)} pairs, need >=32"
            )
        status.update(
            {
                "input_pairs": len(rows),
                "subset_pairs": len(selected),
                "truncation_drops_before_subset_full": truncation_drops,
                "strata": dict(
                    sorted(
                        {
                            f"{source}/{action}": sum(
                                p.source == source and p.chosen_action == action
                                for p in selected
                            )
                            for source, action in {
                                (p.source, p.chosen_action) for p in selected
                            }
                        }.items()
                    )
                ),
            }
        )
        log(
            f"real-tokenizer subset={len(selected)} max_seq={MAX_SEQ_LENGTH} "
            f"drops={truncation_drops}"
        )

        def continuation_logprob(target_model: Any, ids: tuple[int, ...], start: int):
            inputs = mx.array(ids[:-1], dtype=mx.int32)[None, :]
            targets = mx.array(ids[start:], dtype=mx.int32)[None, :]
            logits = target_model(inputs)[:, start - 1 :, :]
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            selected_log_probs = mx.take_along_axis(
                log_probs, targets[..., None], axis=-1
            ).squeeze(-1)
            return selected_log_probs.astype(mx.float32).sum()

        log("prefreezing frozen-base continuation logprobs")
        reference: dict[tuple[str, str, str], tuple[float, float]] = {}
        cache_rows: list[dict[str, object]] = []
        for index, pair in enumerate(selected, start=1):
            chosen = continuation_logprob(model, pair.chosen_ids, pair.continuation_start)
            rejected = continuation_logprob(model, pair.rejected_ids, pair.continuation_start)
            mx.eval(chosen, rejected)
            values = (float(chosen.item()), float(rejected.item()))
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError("kill: non-finite frozen reference logprob")
            reference[pair.key] = values
            cache_rows.append(
                {
                    "prompt_sha256": pair.key[0],
                    "chosen_sha256": pair.key[1],
                    "rejected_sha256": pair.key[2],
                    "reference_chosen_logprob": values[0],
                    "reference_rejected_logprob": values[1],
                    "reference_model": str(MLX_MODEL_DIR),
                    "adapter_enabled": False,
                    "continuation_only": True,
                }
            )
            mx.clear_cache()
            if index % 16 == 0 or index == len(selected):
                log(f"reference {index}/{len(selected)}")

        # Re-score one branch before adapters exist as a deterministic frozen
        # reference guard without doubling the entire prefreeze cost.
        check = continuation_logprob(
            model, selected[0].chosen_ids, selected[0].continuation_start
        )
        mx.eval(check)
        if float(check.item()) != reference[selected[0].key][0]:
            raise AssertionError("frozen reference invariance check failed")
        mx.clear_cache()

        model.freeze()
        linear_to_lora_layers(
            model,
            LORA_LAYERS,
            {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0},
        )
        grad_checkpoint(model.layers[0])
        trainable = dict(tree_flatten(model.trainable_parameters()))
        if not trainable or any("lora_" not in name for name in trainable):
            raise AssertionError(
                f"kill: non-LoRA trainables: {[n for n in trainable if 'lora_' not in n]}"
            )
        lora_before = {name: mx.array(value) + 0 for name, value in trainable.items()}
        mx.eval(lora_before)
        base_runtime_before = _parameter_signatures(
            model, set(trainable), mx, tree_flatten
        )
        log(
            f"LoRA trainables={sum(value.size for value in trainable.values())} "
            f"tensors={len(trainable)} rank={LORA_RANK} scale={LORA_SCALE} "
            f"layers={LORA_LAYERS}"
        )

        def dpo_objective(model, pair: SmokePair, ref_chosen: float, ref_rejected: float):
            policy_chosen = continuation_logprob(
                model, pair.chosen_ids, pair.continuation_start
            )
            policy_rejected = continuation_logprob(
                model, pair.rejected_ids, pair.continuation_start
            )
            margin = policy_chosen - policy_rejected
            z = DEFAULT_BETA * (margin - (ref_chosen - ref_rejected))
            loss = mx.logaddexp(mx.array(0.0, dtype=mx.float32), -z)
            return loss, margin

        loss_and_grad = nn.value_and_grad(model, dpo_objective)
        optimizer = optim.Adam(learning_rate=DEFAULT_LEARNING_RATE)
        rng = random.Random(SEED)
        update_order = list(range(len(selected)))
        rng.shuffle(update_order)
        losses: list[float] = []
        margins: list[float] = []
        model.train()
        log(
            f"starting exactly {UPDATES} updates batch=1 lr={DEFAULT_LEARNING_RATE} "
            f"beta={DEFAULT_BETA} continuation_only=true"
        )
        for update in range(1, UPDATES + 1):
            pair = selected[update_order[(update - 1) % len(update_order)]]
            ref_chosen, ref_rejected = reference[pair.key]
            (loss, margin), gradients = loss_and_grad(
                model, pair, ref_chosen, ref_rejected
            )
            optimizer.update(model, gradients)
            mx.eval(loss, margin, model.state, optimizer.state)
            loss_value = float(loss.item())
            margin_value = float(margin.item())
            if not math.isfinite(loss_value) or not math.isfinite(margin_value):
                raise FloatingPointError(f"kill: non-finite metric at update {update}")
            losses.append(loss_value)
            margins.append(margin_value)
            status["updates_completed"] = update
            mx.clear_cache()
            if update == 1 or update % 10 == 0:
                window_loss = sum(losses[-10:]) / len(losses[-10:])
                window_margin = sum(margins[-10:]) / len(margins[-10:])
                log(
                    f"update={update:03d} loss={loss_value:.6f} "
                    f"margin={margin_value:.6f} loss_avg10={window_loss:.6f} "
                    f"margin_avg10={window_margin:.6f}"
                )

        if status["updates_completed"] != UPDATES:
            raise AssertionError("did not execute exactly 100 optimizer updates")
        lora_after = dict(tree_flatten(model.trainable_parameters()))
        changed = []
        for name, before in lora_before.items():
            delta = mx.max(mx.abs(lora_after[name] - before))
            mx.eval(delta)
            if float(delta.item()) != 0.0:
                changed.append(name)
        if not changed:
            raise AssertionError("kill: no LoRA parameter changed")

        base_runtime_after = _parameter_signatures(
            model, set(lora_after), mx, tree_flatten
        )
        if base_runtime_after != base_runtime_before:
            raise AssertionError("kill: frozen runtime base parameter mutated")

        base_files_after = _sha256_files(model_files)
        if base_files_after != base_files_before:
            raise AssertionError("kill: untouched base weight files mutated")

        ADAPTER_PATH.mkdir(parents=True, exist_ok=False)
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(ADAPTER_PATH / "adapters.safetensors"), adapter_weights)
        adapter_config = {
            "model": str(MLX_MODEL_DIR),
            "fine_tune_type": "lora",
            "num_layers": LORA_LAYERS,
            "batch_size": 1,
            "iters": UPDATES,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_parameters": {
                "rank": LORA_RANK,
                "scale": LORA_SCALE,
                "dropout": 0.0,
            },
            "dpo_beta": DEFAULT_BETA,
            "continuation_only": True,
            "policy_init": "untouched_local_base",
            "reference_frozen": True,
        }
        (ADAPTER_PATH / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
        )
        write_jsonl(ADAPTER_PATH / "reference_cache.jsonl", cache_rows)

        status.update(
            {
                "training": False,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_mean_first10": sum(losses[:10]) / 10,
                "loss_mean_last10": sum(losses[-10:]) / 10,
                "margin_first": margins[0],
                "margin_last": margins[-1],
                "margin_mean_first10": sum(margins[:10]) / 10,
                "margin_mean_last10": sum(margins[-10:]) / 10,
                "loss_curve": losses,
                "margin_curve": margins,
                "lora_tensors_changed": len(changed),
                "lora_only_trainables": True,
                "runtime_base_parameters_unchanged": True,
                "base_weight_files_unchanged": True,
                "forbidden_adapter_init": False,
                "elapsed_s": time.time() - started,
                "PASS": True,
            }
        )
        _write_status(status)
        log(
            f"PASS updates=100 adapter={ADAPTER_PATH} "
            f"loss_first10={status['loss_mean_first10']:.6f} "
            f"loss_last10={status['loss_mean_last10']:.6f}"
        )
        return 0
    except Exception as error:
        status.update(
            {
                "training": False,
                "elapsed_s": time.time() - started,
                "error": f"{type(error).__name__}: {error}",
                "PASS": False,
            }
        )
        _write_status(status)
        log(f"FAIL {status['error']}")
        return 1
    finally:
        log_handle.close()
