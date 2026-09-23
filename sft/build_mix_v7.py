"""Assemble the SFT v7 conversation mix → ``sft/data/splits-v7/``.

Target (conversation-level starting point):

* ≥55% APIGen-MT (abort if ``pct_apigen_mt < 50``)
* ~15% mid-diff function-calling filler (xLAM / nonempty ToolACE)
* ~10% clarify
* ≤10% irrelevance_[] (abort if conversation share > 0.12)
* ~5% final-answer NL (not ``[]``)

Writes ``train.jsonl`` / ``valid.jsonl`` / ``corpus_test.jsonl`` plus
``MANIFEST.json`` with ``experiment: sft-v7``. RNG seed **47**. Never overwrites
frozen v1 ``sft/data/splits/``.

Full conversations are stored in splits-v7 (including APIGen clarify/final
turns). Unrolled ``pct_nontool`` is **not** a splits gate — ``sft.cap_nontool_v7``
downsamples non-tool assistant rows at unroll → mlx-v7 time. Gate that band
with ``mix_stats_v7 --mlx-dir ... --gate`` after prepare.

Student-aware filter (Phase 3) is a stub — this builder samples the formatted
pools directly. Hard abort (nonzero exit) if conversation mix gates fail.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sft.config import (
    APIGEN_MT_CITATION,
    APIGEN_MT_HF_ID,
    APIGEN_MT_LICENSE,
    CHAT_FORMAT,
    CORPUS_TEST_FRAC,
    FORMATTED_APIGEN_MT,
    FORMATTED_TOOLACE,
    FORMATTED_XLAM,
    SEED_V7,
    SOURCE_APIGEN_MT,
    SOURCE_TOOLACE,
    SOURCE_XLAM,
    SPLITS_DIR,
    SPLITS_DIR_V7,
    THESIS_TEST,
    TRAIN_FRAC,
    VAL_FRAC,
    V7_ABORT_IRRELEVANCE_FRAC,
    V7_EXPERIMENT,
    V7_MAX_IRRELEVANCE_FRAC,
    V7_MIN_APIGEN_FRAC,
    V7_TARGET_APIGEN_FRAC,
    V7_TARGET_CLARIFY_FRAC,
    V7_TARGET_FINAL_FRAC,
    V7_TARGET_IRRELEVANCE_FRAC,
    V7_TARGET_MID_FC_FRAC,
)
from sft.io import read_jsonl, sha256_file, write_jsonl
from sft.format_apigen_mt import (
    STOP_CLARIFY,
    STOP_FINAL,
    STOP_IRRELEVANCE,
)
from sft.mix_stats_v7 import (
    collect_stats,
    conversation_stop_label,
    format_report,
    gate_reasons,
)
from sft.split_data import assert_no_bfcl, stratified_split

BUCKET_APIGEN = "apigen_mt"
BUCKET_MID_FC = "mid_fc"
BUCKET_CLARIFY = "clarify"
BUCKET_IRRELEVANCE = "irrelevance_[]"
BUCKET_FINAL = "final_answer"


class MixAbort(Exception):
    """Hard abort: conversation mix gates failed. ``main`` exits 1."""


def _sorted_shuffle(rows: list[dict], rng: random.Random) -> list[dict]:
    items = sorted(rows, key=lambda r: str(r.get("id", "")))
    rng.shuffle(items)
    return items


def conversation_bucket(record: dict) -> str:
    """Primary mix bucket for sampling (APIGen source wins over taxonomy)."""
    if str(record.get("source") or "") == SOURCE_APIGEN_MT:
        return BUCKET_APIGEN
    label = conversation_stop_label(record)
    if label == STOP_CLARIFY:
        return BUCKET_CLARIFY
    if label == STOP_FINAL:
        return BUCKET_FINAL
    if label == STOP_IRRELEVANCE:
        return BUCKET_IRRELEVANCE
    return BUCKET_MID_FC


def _take(rows: list[dict], n: int) -> list[dict]:
    if n <= 0:
        return []
    return rows[: min(n, len(rows))]


def assemble_v7_pool(
    apigen: list[dict],
    toolace: list[dict],
    xlam: list[dict],
    *,
    seed: int = SEED_V7,
    extra_mid_fc: int = 0,
) -> list[dict]:
    """Build the conversation pool toward v7 % targets. All APIGen-MT kept."""
    rng = random.Random(seed)
    apigen = _sorted_shuffle(list(apigen), rng)
    if not apigen:
        raise MixAbort(
            "No APIGen-MT formatted rows. Accept the Hub gate on "
            f"{APIGEN_MT_HF_ID} ({APIGEN_MT_LICENSE}), then:\n"
            "  uv run python -m sft.download_datasets --apigen-mt-only\n"
            "  uv run python -m sft.format_iffc --apigen-mt"
        )

    n_a = len(apigen)
    total = max(n_a, int(round(n_a / V7_TARGET_APIGEN_FRAC)))
    filler = list(toolace) + list(xlam)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in filler:
        buckets[conversation_bucket(rec)].append(rec)
    for key in list(buckets):
        buckets[key] = _sorted_shuffle(buckets[key], rng)

    n_b = int(round(total * V7_TARGET_MID_FC_FRAC)) + extra_mid_fc
    n_c = int(round(total * V7_TARGET_CLARIFY_FRAC))
    n_d = int(round(total * V7_TARGET_IRRELEVANCE_FRAC))
    n_d = min(n_d, int(total * V7_MAX_IRRELEVANCE_FRAC))
    n_e = int(round(total * V7_TARGET_FINAL_FRAC))

    selected = list(apigen)
    selected += _take(buckets[BUCKET_MID_FC], n_b)
    selected += _take(buckets[BUCKET_CLARIFY], n_c)
    selected += _take(buckets[BUCKET_IRRELEVANCE], n_d)
    selected += _take(buckets[BUCKET_FINAL], n_e)

    # Prefer xLAM for leftover mid-FC slots when ToolACE was classified elsewhere.
    seen = {str(r.get("id")) for r in selected}
    leftover_xlam = [
        r
        for r in _sorted_shuffle(
            [r for r in xlam if conversation_bucket(r) == BUCKET_MID_FC], rng
        )
        if str(r.get("id")) not in seen
    ]
    need = n_b - sum(1 for r in selected if conversation_bucket(r) == BUCKET_MID_FC)
    if need > 0:
        extra = _take(leftover_xlam, need)
        selected.extend(extra)

    selected.sort(key=lambda r: str(r.get("id", "")))
    return selected


def _count_by(records: list[dict], field: str) -> dict[str, int]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for rec in records:
        if field == "bucket":
            counter[conversation_bucket(rec)] += 1
        elif field == "stop_label":
            counter[conversation_stop_label(rec)] += 1
        else:
            counter[str(rec.get(field))] += 1
    return dict(sorted(counter.items()))


def build_v7_manifest(
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
        "experiment": V7_EXPERIMENT,
        "seed": seed,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "corpus_test_frac": CORPUS_TEST_FRAC,
        "chat_format": CHAT_FORMAT,
        "thesis_test": THESIS_TEST,
        "split_unit": "conversation/example id (never by turn)",
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
            "pct_apigen_mt": V7_TARGET_APIGEN_FRAC,
            "min_pct_apigen_mt": V7_MIN_APIGEN_FRAC,
            "mid_fc": V7_TARGET_MID_FC_FRAC,
            "clarify": V7_TARGET_CLARIFY_FRAC,
            "irrelevance_[]": V7_TARGET_IRRELEVANCE_FRAC,
            "max_irrelevance_[]": V7_MAX_IRRELEVANCE_FRAC,
            "abort_irrelevance_[]": V7_ABORT_IRRELEVANCE_FRAC,
            "final_answer": V7_TARGET_FINAL_FRAC,
            "unrolled_pct_nontool": (
                "deferred to sft.cap_nontool_v7 at unroll → mlx-v7 "
                "(not a splits-v7 conversation gate)"
            ),
        },
        "student_filter": (
            "stub — sft.student_filter_v7 Phase 3 not applied in this PR"
        ),
        "nontool_unroll_cap": (
            "splits-v7 stores full conversations. "
            "sft.cap_nontool_v7 (seed 47) downsamples clarify/final/"
            "irrelevance_[] assistant unrolls after empty-after-exec so "
            "mlx-v7 pct_nontool is in [0.10, 0.20] (target ~0.15). "
            "Keep all tool_call unrolls. Never map NL finals to []. "
            "mix_stats_v7 --gate --mlx-dir evaluates the capped rows."
        ),
        "mix_stats": mix_stats,
        "counts": counts,
        "checksums_sha256": checksums,
        "files": {name: str(path) for name, path in paths.items()},
    }


def write_v7_splits(
    splits: dict[str, list[dict]],
    *,
    out_dir: Path,
    seed: int,
    pool_counts: dict[str, int],
    mix_stats: dict[str, Any],
) -> dict[str, Any]:
    if out_dir.resolve() == SPLITS_DIR.resolve():
        raise MixAbort(
            f"Refusing to overwrite frozen v1 splits at {SPLITS_DIR}. "
            f"v7 writes to {SPLITS_DIR_V7}."
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
        raise MixAbort("duplicate example ids in v7 splits")

    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["valid"], splits["valid"])
    write_jsonl(paths["corpus_test"], splits["corpus_test"])
    manifest = build_v7_manifest(
        splits, paths, seed=seed, pool_counts=pool_counts, mix_stats=mix_stats
    )
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def _try_assemble(
    apigen: list[dict],
    toolace: list[dict],
    xlam: list[dict],
    *,
    seed: int,
    extra_mid_fc: int,
) -> tuple[list[dict], dict[str, Any], list[str]]:
    pool = assemble_v7_pool(
        apigen, toolace, xlam, seed=seed, extra_mid_fc=extra_mid_fc
    )
    stats = collect_stats(pool)
    return pool, stats, gate_reasons(stats, require_unrolled_nontool=False)


def build_v7_mix(
    *,
    apigen: list[dict],
    toolace: list[dict],
    xlam: list[dict],
    out_dir: Path,
    seed: int = SEED_V7,
    skip_gates: bool = False,
) -> dict[str, Any]:
    pool, stats, reasons = _try_assemble(
        apigen, toolace, xlam, seed=seed, extra_mid_fc=0
    )
    if reasons and not skip_gates:
        raise MixAbort(
            "v7 mix gates failed:\n  - "
            + "\n  - ".join(reasons)
            + "\n"
            + format_report(stats, require_unrolled_nontool=False)
        )

    splits = stratified_split(pool, seed=seed)
    train_stats = collect_stats(splits["train"])
    train_reasons = gate_reasons(train_stats, require_unrolled_nontool=False)
    if train_reasons and not skip_gates:
        raise MixAbort(
            "v7 train-split gates failed:\n  - "
            + "\n  - ".join(train_reasons)
            + "\n"
            + format_report(train_stats, require_unrolled_nontool=False)
        )

    pool_counts = {
        SOURCE_APIGEN_MT: len(apigen),
        SOURCE_TOOLACE: len(toolace),
        SOURCE_XLAM: len(xlam),
        "pool_selected": len(pool),
    }
    mix_stats = {
        "pool": stats,
        "train": train_stats,
        "gate_reasons_pool": reasons,
        "gate_reasons_train": train_reasons,
    }
    manifest = write_v7_splits(
        splits,
        out_dir=out_dir,
        seed=seed,
        pool_counts=pool_counts,
        mix_stats=mix_stats,
    )
    print(format_report(train_stats, require_unrolled_nontool=False))
    print(
        "v7 splits: "
        f"train={len(splits['train'])} "
        f"valid={len(splits['valid'])} "
        f"corpus_test={len(splits['corpus_test'])} → {out_dir}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble SFT v7 mix into sft/data/splits-v7/ (seed 47)."
    )
    parser.add_argument("--formatted-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR_V7)
    parser.add_argument("--seed", type=int, default=SEED_V7)
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help="Write splits even if conversation-level taxonomy gates fail (debug only).",
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
        build_v7_mix(
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
