"""Assemble the SFT v8 conversation mix → ``sft/data/splits-v8/``.

Conversation-level starting point is the same as v7 (APIGen-MT core ≥50%)
but **seed 48**, a new experiment id, and ToolACE-R-lite loop-trajectory
filter before split. v8.1 keeps this module / dirs / ``--recipe v8``:

* Drop teacher trajs with ≥3 consecutive identical assistant tool-calls
  **or** ≥3 observe→same-call re-emits (A-B-A-B ``ls`` / ``displayCarStatus``)
* Log drops by source
* Abort if remaining ``pct_apigen_mt`` would fall below 50%

Writes ``train.jsonl`` / ``valid.jsonl`` / ``corpus_test.jsonl`` plus
``MANIFEST.json`` with ``experiment: sft-v8``. Never overwrites frozen v1
``sft/data/splits/`` or ``splits-v7``.

Unrolled ``pct_nontool`` (~0.35) and identical-call / observe→same-call
drop are **not** splits gates — they apply at unroll → mlx-v8 time. Gate
that band with ``mix_stats_v8 --mlx-dir ... --gate`` after prepare.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sft.build_mix_v7 import (
    MixAbort,
    _count_by,
    assemble_v7_pool,
)
from sft.config import (
    APIGEN_MT_CITATION,
    APIGEN_MT_HF_ID,
    APIGEN_MT_LICENSE,
    CHAT_FORMAT,
    CORPUS_TEST_FRAC,
    FORMATTED_APIGEN_MT,
    FORMATTED_TOOLACE,
    FORMATTED_XLAM,
    SEED_V8,
    SOURCE_APIGEN_MT,
    SOURCE_TOOLACE,
    SOURCE_XLAM,
    SPLITS_DIR,
    SPLITS_DIR_V7,
    SPLITS_DIR_V8,
    THESIS_TEST,
    TRAIN_FRAC,
    VAL_FRAC,
    V8_ABORT_IRRELEVANCE_FRAC,
    V8_EXPERIMENT,
    V8_LOOP_MIN_RUN,
    V8_MAX_IRRELEVANCE_FRAC,
    V8_MIN_APIGEN_FRAC,
    V8_NONTOOL_MAX,
    V8_NONTOOL_MIN,
    V8_NONTOOL_TARGET,
    V8_RECIPE_BUMP,
    V8_TARGET_APIGEN_FRAC,
    V8_TARGET_CLARIFY_FRAC,
    V8_TARGET_FINAL_FRAC,
    V8_TARGET_IRRELEVANCE_FRAC,
    V8_TARGET_MID_FC_FRAC,
)
from sft.filter_empty_after_exec import RECIPE_V8
from sft.io import read_jsonl, sha256_file, write_jsonl
from sft.mix_stats_v8 import collect_stats, format_report, gate_reasons
from sft.split_data import assert_no_bfcl, stratified_split
from sft.student_filter_v8 import (
    LoopFilterAbort,
    filter_loop_trajectories,
    format_loop_filter_report,
)


def build_v8_manifest(
    splits: dict[str, list[dict]],
    paths: dict[str, Path],
    *,
    seed: int,
    pool_counts: dict[str, int],
    mix_stats: dict[str, Any],
    loop_filter: dict[str, Any] | None,
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
        "experiment": V8_EXPERIMENT,
        "recipe_bump": V8_RECIPE_BUMP,
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
            "pct_apigen_mt": V8_TARGET_APIGEN_FRAC,
            "min_pct_apigen_mt": V8_MIN_APIGEN_FRAC,
            "mid_fc": V8_TARGET_MID_FC_FRAC,
            "clarify": V8_TARGET_CLARIFY_FRAC,
            "irrelevance_[]": V8_TARGET_IRRELEVANCE_FRAC,
            "max_irrelevance_[]": V8_MAX_IRRELEVANCE_FRAC,
            "abort_irrelevance_[]": V8_ABORT_IRRELEVANCE_FRAC,
            "final_answer": V8_TARGET_FINAL_FRAC,
            "unrolled_pct_nontool": (
                f"deferred to unroll → mlx-v8: v8.1 identical-call + "
                f"observe→same-call drop then cap_nontool_v7 (seed {seed}) "
                f"so pct_nontool is in [{V8_NONTOOL_MIN}, {V8_NONTOOL_MAX}] "
                f"(target ~{V8_NONTOOL_TARGET}). "
                "Not a splits-v8 conversation gate."
            ),
        },
        "student_filter": (
            "ToolACE-R-lite (v8.1): drop teacher loop trajs with "
            f">={V8_LOOP_MIN_RUN} consecutive identical assistant tool-calls "
            f"**or** >={V8_LOOP_MIN_RUN} observe→same-call re-emits "
            "before split; abort if APIGen-MT floor would be violated. "
            "No MLX Jaccard drafts (that remains student_filter_v7 Phase 3)."
        ),
        "loop_filter": loop_filter,
        "identical_call_drop": (
            "sft.drop_identical_call at unroll → mlx-v8 (v8.1): drop gold "
            "assistant turns that equal the previous assistant call "
            "(consecutive) **or** re-emit already-executed calls after a "
            "tool-result observation (observe→same-call; ls/displayCarStatus "
            "A-B-A-B). Consecutive-only was too light (~32 rows). "
            "Does not rewrite NL finals or reintroduce v5 complete-turn0."
        ),
        "nontool_unroll_cap": (
            "splits-v8 stores full conversations. After v8.1 identical-call "
            "+ observe→same-call drop, "
            f"sft.cap_nontool_v7 (seed {seed}) downsamples clarify/final/"
            "irrelevance_[] assistant unrolls so mlx-v8 pct_nontool is in "
            f"[{V8_NONTOOL_MIN}, {V8_NONTOOL_MAX}] "
            f"(target ~{V8_NONTOOL_TARGET}). "
            "Keep remaining tool_call unrolls. Never map NL finals to []. "
            "mix_stats_v8 --gate --mlx-dir evaluates the post-processed rows."
        ),
        "mix_stats": mix_stats,
        "counts": counts,
        "checksums_sha256": checksums,
        "files": {name: str(path) for name, path in paths.items()},
    }


def write_v8_splits(
    splits: dict[str, list[dict]],
    *,
    out_dir: Path,
    seed: int,
    pool_counts: dict[str, int],
    mix_stats: dict[str, Any],
    loop_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved = out_dir.resolve()
    if resolved == SPLITS_DIR.resolve():
        raise MixAbort(
            f"Refusing to overwrite frozen v1 splits at {SPLITS_DIR}. "
            f"v8 writes to {SPLITS_DIR_V8}."
        )
    if resolved == SPLITS_DIR_V7.resolve():
        raise MixAbort(
            f"Refusing to overwrite v7 splits at {SPLITS_DIR_V7}. "
            f"v8 writes to {SPLITS_DIR_V8}."
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
        raise MixAbort("duplicate example ids in v8 splits")

    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["valid"], splits["valid"])
    write_jsonl(paths["corpus_test"], splits["corpus_test"])
    manifest = build_v8_manifest(
        splits,
        paths,
        seed=seed,
        pool_counts=pool_counts,
        mix_stats=mix_stats,
        loop_filter=loop_filter,
    )
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def build_v8_mix(
    *,
    apigen: list[dict],
    toolace: list[dict],
    xlam: list[dict],
    out_dir: Path,
    seed: int = SEED_V8,
    skip_gates: bool = False,
    loop_min_run: int = V8_LOOP_MIN_RUN,
) -> dict[str, Any]:
    pool = assemble_v7_pool(apigen, toolace, xlam, seed=seed)
    try:
        pool, loop_stats = filter_loop_trajectories(
            pool,
            min_run=loop_min_run,
            min_apigen_frac=V8_MIN_APIGEN_FRAC,
            abort_if_apigen_floor_violated=not skip_gates,
        )
    except LoopFilterAbort as exc:
        raise MixAbort(str(exc)) from exc
    print(format_loop_filter_report(loop_stats))

    stats = collect_stats(pool)
    reasons = gate_reasons(stats, require_unrolled_nontool=False)
    if reasons and not skip_gates:
        raise MixAbort(
            "v8 mix gates failed:\n  - "
            + "\n  - ".join(reasons)
            + "\n"
            + format_report(stats, require_unrolled_nontool=False)
        )

    splits = stratified_split(pool, seed=seed)
    train_stats = collect_stats(splits["train"])
    train_reasons = gate_reasons(train_stats, require_unrolled_nontool=False)
    if train_reasons and not skip_gates:
        raise MixAbort(
            "v8 train-split gates failed:\n  - "
            + "\n  - ".join(train_reasons)
            + "\n"
            + format_report(train_stats, require_unrolled_nontool=False)
        )

    pool_counts = {
        SOURCE_APIGEN_MT: len(apigen),
        SOURCE_TOOLACE: len(toolace),
        SOURCE_XLAM: len(xlam),
        "pool_selected": len(pool),
        "loop_filter_dropped": loop_stats.get("dropped"),
    }
    mix_stats = {
        "pool": stats,
        "train": train_stats,
        "gate_reasons_pool": reasons,
        "gate_reasons_train": train_reasons,
        "unroll_recipe": RECIPE_V8,
    }
    manifest = write_v8_splits(
        splits,
        out_dir=out_dir,
        seed=seed,
        pool_counts=pool_counts,
        mix_stats=mix_stats,
        loop_filter=loop_stats,
    )
    print(format_report(train_stats, require_unrolled_nontool=False))
    print(
        "v8 splits: "
        f"train={len(splits['train'])} "
        f"valid={len(splits['valid'])} "
        f"corpus_test={len(splits['corpus_test'])} → {out_dir}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble SFT v8 mix into sft/data/splits-v8/ (seed 48)."
    )
    parser.add_argument("--formatted-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR_V8)
    parser.add_argument("--seed", type=int, default=SEED_V8)
    parser.add_argument(
            "--loop-min-run",
            type=int,
            default=V8_LOOP_MIN_RUN,
            help=(
                "Drop conversations with this many consecutive identical "
                "calls or observe→same-call re-emits."
            ),
    )
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help=(
            "Write splits even if conversation-level taxonomy / APIGen-floor "
            "gates fail (debug only)."
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
        build_v8_mix(
            apigen=apigen,
            toolace=toolace,
            xlam=xlam,
            out_dir=args.out,
            seed=args.seed,
            skip_gates=args.skip_gates,
            loop_min_run=args.loop_min_run,
        )
    except MixAbort as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
