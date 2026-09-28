"""Assemble the SFT v9 (lever2) conversation mix → ``sft/data/splits-v9/``.

Conversation-level pool (CONFIRM locked):

* ToolACE **45–55%** (primary MT body)
* xLAM **15–20%** (turn-0 / parallel start-call density)
* APIGen-MT **20–30%** (quota filler for ask / final_after_obs / refuse /
  verified advance) with hard ceiling **≤35%** and floor **≥15%**
* Abort if ToolACE + xLAM < 55%

**Does not** import ``filter_loop_trajectories`` / teacher observe→same traj
drop (v8 / v8.1 retired as primary). Unrolled action-class quotas apply later
via ``quota_sample_v9`` at prepare → mlx-v9.

Writes ``train.jsonl`` / ``valid.jsonl`` / ``corpus_test.jsonl`` plus
``MANIFEST.json`` with ``experiment: sft-v9``. Seed **49**. Never overwrites
frozen v1 ``splits/``, ``splits-v7``, or ``splits-v8``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from sft.build_mix_v7 import MixAbort, _count_by, _sorted_shuffle, _take
from sft.config import (
    APIGEN_MT_CITATION,
    APIGEN_MT_HF_ID,
    APIGEN_MT_LICENSE,
    CHAT_FORMAT,
    CORPUS_TEST_FRAC,
    FORMATTED_APIGEN_MT,
    FORMATTED_TOOLACE,
    FORMATTED_XLAM,
    SEED_V9,
    SOURCE_APIGEN_MT,
    SOURCE_TOOLACE,
    SOURCE_XLAM,
    SPLITS_DIR,
    SPLITS_DIR_V7,
    SPLITS_DIR_V8,
    SPLITS_DIR_V9,
    THESIS_TEST,
    TRAIN_FRAC,
    VAL_FRAC,
    V9_ABORT_IRRELEVANCE_FRAC,
    V9_EXPERIMENT,
    V9_LEVER,
    V9_MAX_APIGEN_FRAC,
    V9_MAX_IRRELEVANCE_CONV_FRAC,
    V9_MAX_SEQ_LENGTH,
    V9_MAX_TOOLACE_FRAC,
    V9_MAX_XLAM_FRAC,
    V9_MIN_APIGEN_FRAC,
    V9_MIN_TOOLACE_FRAC,
    V9_MIN_TOOLACE_XLAM_FRAC,
    V9_MIN_XLAM_FRAC,
    V9_PROBE_IDS,
    V9_PROBE_IDS_VC_SIDECAR,
    V9_TARGET_APIGEN_FRAC,
    V9_TARGET_TOOLACE_FRAC,
    V9_TARGET_XLAM_FRAC,
)
from sft.filter_empty_after_exec import RECIPE_V9
from sft.format_apigen_mt import STOP_IRRELEVANCE
from sft.io import read_jsonl, sha256_file, write_jsonl
from sft.mix_stats_v7 import conversation_stop_label
from sft.mix_stats_v9 import collect_stats, format_report, gate_reasons
from sft.split_data import assert_no_bfcl, stratified_split


def assemble_v9_pool(
    apigen: list[dict],
    toolace: list[dict],
    xlam: list[dict],
    *,
    seed: int = SEED_V9,
) -> list[dict]:
    """Build the conversation pool toward ToolACE-primary v9 % targets."""
    rng = random.Random(seed)
    toolace = _sorted_shuffle(list(toolace), rng)
    xlam = _sorted_shuffle(list(xlam), rng)
    apigen = _sorted_shuffle(list(apigen), rng)

    if not toolace:
        raise MixAbort(
            "No ToolACE formatted rows. Run:\n"
            "  uv run python -m sft.download_datasets\n"
            "  uv run python -m sft.format_iffc --toolace"
        )
    if not apigen:
        raise MixAbort(
            "No APIGen-MT formatted rows (needed as quota filler for ask / "
            f"final_after_obs / refuse). Accept Hub gate on {APIGEN_MT_HF_ID} "
            f"({APIGEN_MT_LICENSE}), then:\n"
            "  uv run python -m sft.download_datasets --apigen-mt-only\n"
            "  uv run python -m sft.format_iffc --apigen-mt"
        )

    # Size the pool from ToolACE target share.
    n_toolace_avail = len(toolace)
    # Choose total so target ToolACE share is hit with as many ToolACE as fit.
    total = max(
        n_toolace_avail,
        int(round(n_toolace_avail / V9_TARGET_TOOLACE_FRAC)),
    )
    n_t = min(n_toolace_avail, int(round(total * V9_TARGET_TOOLACE_FRAC)))
    n_x = int(round(total * V9_TARGET_XLAM_FRAC))
    n_a = int(round(total * V9_TARGET_APIGEN_FRAC))
    # Clamp APIGen to [min, max] of total.
    n_a = max(
        int(round(total * V9_MIN_APIGEN_FRAC)),
        min(n_a, int(total * V9_MAX_APIGEN_FRAC), len(apigen)),
    )
    n_x = max(0, min(n_x, len(xlam)))
    # Recompute total from selected counts so shares stay coherent.
    selected = _take(toolace, n_t) + _take(xlam, n_x) + _take(apigen, n_a)
    # If ToolACE+xLAM would fall below floor, add more ToolACE/xLAM.
    n_sel = len(selected)
    tx = sum(
        1
        for r in selected
        if str(r.get("source")) in {SOURCE_TOOLACE, SOURCE_XLAM}
    )
    if n_sel and (tx / n_sel) < V9_MIN_TOOLACE_XLAM_FRAC:
        need = int(
            math_ceil_tx(n_sel, tx, V9_MIN_TOOLACE_XLAM_FRAC)
        )
        seen = {str(r.get("id")) for r in selected}
        extra_t = [
            r for r in toolace if str(r.get("id")) not in seen
        ]
        extra_x = [
            r for r in xlam if str(r.get("id")) not in seen
        ]
        add = _take(extra_t, need) + _take(extra_x, max(0, need - len(extra_t)))
        selected.extend(add)

    # Cap APIGen if somehow over ceiling after extras.
    n_sel = len(selected)
    n_apigen = sum(
        1 for r in selected if str(r.get("source")) == SOURCE_APIGEN_MT
    )
    if n_sel and (n_apigen / n_sel) > V9_MAX_APIGEN_FRAC:
        keep_a = int(n_sel * V9_MAX_APIGEN_FRAC)
        apigen_rows = [
            r for r in selected if str(r.get("source")) == SOURCE_APIGEN_MT
        ]
        other = [
            r for r in selected if str(r.get("source")) != SOURCE_APIGEN_MT
        ]
        selected = other + _take(apigen_rows, keep_a)

    # Cap conversation-level irrelevance_[] (almost all ToolACE terminal []).
    # Matches V9_MAX_IRRELEVANCE_CONV_FRAC / abort 0.12 — v7-style downsample;
    # prefer dropping ToolACE/xLAM irrelevance before APIGen.
    n_sel = len(selected)
    irr_rows = [
        r for r in selected if conversation_stop_label(r) == STOP_IRRELEVANCE
    ]
    non_irr = [
        r for r in selected if conversation_stop_label(r) != STOP_IRRELEVANCE
    ]
    if n_sel and irr_rows:
        # keep_irr / (n_non + keep_irr) <= max  => keep_irr <= max/(1-max) * n_non
        # Aim slightly under max so stratified train/valid stay ≤ max.
        max_frac = min(V9_MAX_IRRELEVANCE_CONV_FRAC, 0.098)
        keep_irr = int(max_frac / (1.0 - max_frac) * len(non_irr) + 1e-12)
        if len(irr_rows) > keep_irr:
            filler_irr = [
                r
                for r in irr_rows
                if str(r.get("source")) in {SOURCE_TOOLACE, SOURCE_XLAM}
            ]
            apigen_irr = [
                r
                for r in irr_rows
                if str(r.get("source")) == SOURCE_APIGEN_MT
            ]
            # Always keep APIGen irrelevance (scarce); fill remaining from filler.
            keep_filler = max(0, keep_irr - len(apigen_irr))
            selected = non_irr + list(apigen_irr) + _take(filler_irr, keep_filler)

    # Enforce individual ToolACE / xLAM bands (do not widen). Drop overweight
    # source rows after irrelevance/APIGen caps; never expand bands for convenience.
    selected = _clamp_source_max(selected, SOURCE_XLAM, V9_MAX_XLAM_FRAC)
    selected = _clamp_source_max(selected, SOURCE_TOOLACE, V9_MAX_TOOLACE_FRAC)
    selected = _clamp_source_max(selected, SOURCE_APIGEN_MT, V9_MAX_APIGEN_FRAC)
    # Re-check floors after clamps (add back from leftovers if needed).
    selected = _ensure_source_min(
        selected, SOURCE_TOOLACE, toolace, V9_MIN_TOOLACE_FRAC
    )
    selected = _ensure_source_min(
        selected, SOURCE_XLAM, xlam, V9_MIN_XLAM_FRAC
    )
    selected = _ensure_source_min(
        selected, SOURCE_APIGEN_MT, apigen, V9_MIN_APIGEN_FRAC
    )
    # Final max clamp (mins may have nudged a peer over max).
    selected = _clamp_source_max(selected, SOURCE_XLAM, V9_MAX_XLAM_FRAC)
    selected = _clamp_source_max(selected, SOURCE_TOOLACE, V9_MAX_TOOLACE_FRAC)
    selected = _clamp_source_max(selected, SOURCE_APIGEN_MT, V9_MAX_APIGEN_FRAC)

    selected.sort(key=lambda r: str(r.get("id", "")))
    return selected


def _clamp_source_max(
    selected: list[dict], source: str, max_frac: float
) -> list[dict]:
    """Downsample ``source`` so its share is ≤ ``max_frac``."""
    if max_frac <= 0 or max_frac >= 1:
        return selected
    src_rows = [r for r in selected if str(r.get("source")) == source]
    other = [r for r in selected if str(r.get("source")) != source]
    if not src_rows:
        return selected
    # keep / (n_other + keep) <= max  => keep <= max/(1-max) * n_other
    keep_n = int(max_frac / (1.0 - max_frac) * len(other) + 1e-12)
    # Also never exceed floor(max_frac * total) style; prefer slightly under.
    keep_n = min(keep_n, int(max_frac * (len(other) + keep_n) + 1e-12))
    if len(src_rows) <= keep_n:
        return selected
    return other + _take(src_rows, keep_n)


def _ensure_source_min(
    selected: list[dict],
    source: str,
    pool: list[dict],
    min_frac: float,
) -> list[dict]:
    """Add unused ``source`` rows from ``pool`` until share ≥ ``min_frac``."""
    if min_frac <= 0:
        return selected
    seen = {str(r.get("id")) for r in selected}
    src_n = sum(1 for r in selected if str(r.get("source")) == source)
    n = len(selected)
    if n and (src_n / n) >= min_frac:
        return selected
    extras = [r for r in pool if str(r.get("id")) not in seen]
    # (src_n + k) / (n + k) >= min => k >= (min*n - src_n)/(1-min)
    import math

    need = min_frac * n - src_n
    if need <= 0:
        return selected
    k = int(math.ceil(need / (1.0 - min_frac) - 1e-12))
    if k <= 0:
        return selected
    selected = list(selected) + _take(extras, k)
    return selected


def math_ceil_tx(n_sel: int, tx: int, floor: float) -> int:
    """How many extra ToolACE/xLAM rows to add so tx/n ≥ floor."""
    # (tx + k) / (n_sel + k) >= floor  =>  tx + k >= floor * n_sel + floor * k
    # => k * (1 - floor) >= floor * n_sel - tx
    import math

    need = floor * n_sel - tx
    if need <= 0:
        return 0
    return int(math.ceil(need / (1.0 - floor) - 1e-12))


def build_v9_manifest(
    splits: dict[str, list[dict]],
    paths: dict[str, Path],
    *,
    seed: int,
    pool_counts: dict[str, int],
    mix_stats: dict[str, Any],
) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for name, rows in splits.items():
        counts[name] = {
            "n": len(rows),
            "by_source": _count_by(rows, "source"),
            "by_bucket": _count_by(rows, "bucket"),
            "by_stop_label": _count_by(rows, "stop_label"),
            "by_is_multi_turn": _count_by(rows, "is_multi_turn"),
            "by_n_tools_bin": _count_by(rows, "n_tools_bin"),
        }
    checksums = {
        paths[name].name: sha256_file(paths[name])
        for name in ("train", "valid", "corpus_test")
        if name in paths and paths[name].exists()
    }
    return {
        "experiment": V9_EXPERIMENT,
        "lever": V9_LEVER,
        "seed": seed,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "corpus_test_frac": CORPUS_TEST_FRAC,
        "chat_format": CHAT_FORMAT,
        "thesis_test": THESIS_TEST,
        "split_unit": "conversation/example id (never by turn)",
        "max_seq_length": V9_MAX_SEQ_LENGTH,
        "loss_mask": "assistant_only_via_mask_prompt_unroll",
        "apigen_mt_hf_id": APIGEN_MT_HF_ID,
        "apigen_mt_license": APIGEN_MT_LICENSE,
        "apigen_mt_citation": APIGEN_MT_CITATION,
        "license_note": (
            f"{APIGEN_MT_HF_ID} is {APIGEN_MT_LICENSE} and Hub-gated. "
            "Cite Prabhakar et al. (APIGen-MT, arXiv:2504.03601). "
            "Academic/thesis use only — not for competing with OpenAI. "
            "Do not commit raw dataset files."
        ),
        "pool_counts": pool_counts,
        "targets": {
            "pct_toolace": V9_TARGET_TOOLACE_FRAC,
            "pct_toolace_band": [V9_MIN_TOOLACE_FRAC, V9_MAX_TOOLACE_FRAC],
            "pct_xlam": V9_TARGET_XLAM_FRAC,
            "pct_xlam_band": [V9_MIN_XLAM_FRAC, V9_MAX_XLAM_FRAC],
            "pct_apigen_mt": V9_TARGET_APIGEN_FRAC,
            "pct_apigen_mt_band": [V9_MIN_APIGEN_FRAC, V9_MAX_APIGEN_FRAC],
            "min_pct_toolace_xlam": V9_MIN_TOOLACE_XLAM_FRAC,
            "max_irrelevance_conv": V9_MAX_IRRELEVANCE_CONV_FRAC,
            "abort_irrelevance_conv": V9_ABORT_IRRELEVANCE_FRAC,
            "action_class_quotas": (
                "deferred to sft.quota_sample_v9 at unroll → mlx-v9 "
                "(not a splits-v9 conversation gate)"
            ),
        },
        "student_filter": (
            "retired as primary — do not port filter_loop_trajectories / "
            "observe→same teacher traj drop into v9. Soft-exclude those "
            "shapes only inside quota_sample_v9."
        ),
        "identical_call_policy": (
            "Reuse drop_identical_call predicates as quota sampler "
            "exclusion + logging only. Not a conversation-level traj delete."
        ),
        "stage_b_pref_pairs": (
            "stub only — sft.pref_pairs_antiloop_v9 --dry-run; no DPO/ORPO "
            "in this scaffold PR"
        ),
        "probe_ids": list(V9_PROBE_IDS),
        "probe_ids_vc_sidecar": list(V9_PROBE_IDS_VC_SIDECAR),
        "probe_note": (
            "Formal gate = probe_ids (10). VC sidecar is diagnostic only; "
            "never fold into /10 denominators."
        ),
        "eval_note": (
            "Later BFCL under plain iffc regardless of stop-repeat harness. "
            "Do not require stop-repeat to pass smoke for v9."
        ),
        "standing_constraints": [
            "no BoN",
            "no v7 resume",
            "no full LoRA on failed v8 mixes",
            "no teacher observe→same as primary",
            "no v5 synth complete-turn0",
        ],
        "mix_stats": mix_stats,
        "counts": counts,
        "checksums_sha256": checksums,
        "files": {name: str(path) for name, path in paths.items()},
        "unroll_recipe": RECIPE_V9,
    }


def write_v9_splits(
    splits: dict[str, list[dict]],
    *,
    out_dir: Path,
    seed: int,
    pool_counts: dict[str, int],
    mix_stats: dict[str, Any],
) -> dict[str, Any]:
    resolved = out_dir.resolve()
    if resolved == SPLITS_DIR.resolve():
        raise MixAbort(
            f"Refusing to overwrite frozen v1 splits at {SPLITS_DIR}. "
            f"v9 writes to {SPLITS_DIR_V9}."
        )
    if resolved == SPLITS_DIR_V7.resolve():
        raise MixAbort(
            f"Refusing to overwrite v7 splits at {SPLITS_DIR_V7}. "
            f"v9 writes to {SPLITS_DIR_V9}."
        )
    if resolved == SPLITS_DIR_V8.resolve():
        raise MixAbort(
            f"Refusing to overwrite v8 splits at {SPLITS_DIR_V8}. "
            f"v9 writes to {SPLITS_DIR_V9}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.jsonl",
        "valid": out_dir / "valid.jsonl",
        "corpus_test": out_dir / "corpus_test.jsonl",
        "manifest": out_dir / "MANIFEST.json",
    }
    all_rows = [r for rows in splits.values() for r in rows]
    assert_no_bfcl(all_rows)
    ids = [r["id"] for r in all_rows]
    if len(ids) != len(set(ids)):
        raise MixAbort("duplicate example ids in v9 splits")

    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["valid"], splits["valid"])
    write_jsonl(paths["corpus_test"], splits["corpus_test"])
    manifest = build_v9_manifest(
        splits,
        paths,
        seed=seed,
        pool_counts=pool_counts,
        mix_stats=mix_stats,
    )
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def build_v9_mix(
    *,
    apigen: list[dict],
    toolace: list[dict],
    xlam: list[dict],
    out_dir: Path,
    seed: int = SEED_V9,
    skip_gates: bool = False,
) -> dict[str, Any]:
    # Explicit: NO filter_loop_trajectories import / call.
    pool = assemble_v9_pool(apigen, toolace, xlam, seed=seed)
    stats = collect_stats(pool)
    reasons = gate_reasons(stats, require_unrolled_actions=False)
    if reasons and not skip_gates:
        raise MixAbort(
            "v9 mix gates failed:\n  - "
            + "\n  - ".join(reasons)
            + "\n"
            + format_report(stats, require_unrolled_actions=False)
        )

    splits = stratified_split(pool, seed=seed)
    train_stats = collect_stats(splits["train"])
    train_reasons = gate_reasons(train_stats, require_unrolled_actions=False)
    if train_reasons and not skip_gates:
        raise MixAbort(
            "v9 train-split gates failed:\n  - "
            + "\n  - ".join(train_reasons)
            + "\n"
            + format_report(train_stats, require_unrolled_actions=False)
        )

    pool_counts = {
        SOURCE_APIGEN_MT: len(apigen),
        SOURCE_TOOLACE: len(toolace),
        SOURCE_XLAM: len(xlam),
        "pool_selected": len(pool),
        "loop_filter_dropped": 0,  # intentionally unused in v9
    }
    mix_stats = {
        "pool": stats,
        "train": train_stats,
        "gate_reasons_pool": reasons,
        "gate_reasons_train": train_reasons,
        "unroll_recipe": RECIPE_V9,
    }
    manifest = write_v9_splits(
        splits,
        out_dir=out_dir,
        seed=seed,
        pool_counts=pool_counts,
        mix_stats=mix_stats,
    )
    print(format_report(train_stats, require_unrolled_actions=False))
    print(
        "v9 splits: "
        f"train={len(splits['train'])} "
        f"valid={len(splits['valid'])} "
        f"corpus_test={len(splits['corpus_test'])} → {out_dir}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble SFT v9 (lever2) mix into sft/data/splits-v9/ (seed 49). "
            "Scaffold only — do not prepare/LoRA until gates reviewed."
        )
    )
    parser.add_argument("--formatted-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR_V9)
    parser.add_argument("--seed", type=int, default=SEED_V9)
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help=(
            "Write splits even if conversation-level source-share gates fail "
            "(debug only)."
        ),
    )
    args = parser.parse_args(argv)

    apigen_path = FORMATTED_APIGEN_MT
    toolace_path = FORMATTED_TOOLACE
    xlam_path = FORMATTED_XLAM
    if args.formatted_dir:
        apigen_path = args.formatted_dir / "apigen_mt.jsonl"
        toolace_path = args.formatted_dir / "toolace.jsonl"
        xlam_path = args.formatted_dir / "xlam.jsonl"

    apigen = read_jsonl(apigen_path) if apigen_path.exists() else []
    toolace = read_jsonl(toolace_path) if toolace_path.exists() else []
    xlam = read_jsonl(xlam_path) if xlam_path.exists() else []
    try:
        build_v9_mix(
            apigen=apigen,
            toolace=toolace,
            xlam=xlam,
            out_dir=args.out,
            seed=args.seed,
            skip_gates=args.skip_gates,
        )
    except MixAbort as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
