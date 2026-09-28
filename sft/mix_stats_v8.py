"""Stop-taxonomy / source stats for the SFT v8 mix.

Same conversation gates as v7 (APIGen-MT ≥50%, irrelevance caps) plus the
v8.1 unroll pipeline: identical-call **and** observe→same-call drop, then
non-tool cap to ``[0.25, 0.35]`` (target 0.35, seed 48). Recipe flag stays
``v8``; dirs stay ``splits-v8`` / ``mlx-v8``.

CLI::

    uv run python -m sft.build_mix_v8 --out sft/data/splits-v8
    uv run python -m sft.mix_stats_v8 --splits sft/data/splits-v8 --gate
    uv run python -m sft.train_mlx --prepare-only --recipe v8 \\
      --splits-dir sft/data/splits-v8 --mlx-dir sft/data/mlx-v8
    uv run python -m sft.mix_stats_v8 --splits sft/data/splits-v8 \\
        --mlx-dir sft/data/mlx-v8 --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sft.config import (
    SPLITS_DIR_V8,
    V8_ABORT_IRRELEVANCE_FRAC,
    V8_MIN_APIGEN_FRAC,
    V8_NONTOOL_MAX,
    V8_NONTOOL_MIN,
    V8_RECIPE_BUMP,
)
from sft.filter_empty_after_exec import RECIPE_V8
from sft.io import read_jsonl
from sft.mix_stats_v7 import (
    collect_stats as collect_stats_base,
    format_report as format_report_base,
    gate_reasons,
)


def collect_stats(
    conversations: list[dict],
    *,
    unrolled: list[dict] | None = None,
    unroll_recipe: str = RECIPE_V8,
    cap_nontool: bool | None = None,
    drop_identical: bool | None = None,
):
    """v8.1 projected unroll: observe→same-call drop then non-tool cap (~0.35)."""
    stats = collect_stats_base(
        conversations,
        unrolled=unrolled,
        unroll_recipe=unroll_recipe,
        cap_nontool=cap_nontool,
        drop_identical=drop_identical,
        min_apigen_frac=V8_MIN_APIGEN_FRAC,
        abort_irrelevance_frac=V8_ABORT_IRRELEVANCE_FRAC,
        nontool_min=V8_NONTOOL_MIN,
        nontool_max=V8_NONTOOL_MAX,
    )
    stats["recipe_bump"] = V8_RECIPE_BUMP
    return stats


def format_report(
    stats: dict,
    *,
    require_unrolled_nontool: bool = False,
) -> str:
    body = format_report_base(
        stats, require_unrolled_nontool=require_unrolled_nontool
    )
    bump = stats.get("recipe_bump") or V8_RECIPE_BUMP
    return (
        f"{body}\n"
        f"recipe_bump={bump} (flag remains --recipe v8; dirs splits-v8/mlx-v8; "
        "v8.1 observe→same-call drop + non-tool target 0.35)"
    )


def _load_split_dir(path: Path) -> list[dict]:
    rows: list[dict] = []
    for name in ("train.jsonl", "valid.jsonl", "corpus_test.jsonl"):
        f = path / name
        if f.exists():
            rows.extend(read_jsonl(f))
    if not rows and path.is_file():
        rows = read_jsonl(path)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print v8.1 stop-taxonomy / source / identical-call / "
            "observe→same-call / non-tool stats."
        )
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=SPLITS_DIR_V8,
        help="Conversation JSONL dir (train/valid/corpus_test) or a JSONL file.",
    )
    parser.add_argument(
        "--mlx-dir",
        type=Path,
        default=None,
        help=(
            "Optional already-unrolled mlx JSONL dir (skips in-memory unroll). "
            "Must be the v8 prepare output (v8.1 observe→same-call drop + "
            "~35% cap). With --gate, also requires unrolled pct_nontool in "
            "[0.25, 0.35]."
        ),
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Exit 1 if hard mix constraints fail. Without --mlx-dir this is "
            "conversation-level only (pct_apigen, irrelevance). Pass --mlx-dir "
            "to gate the post-processed unroll the trainer will see."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Dump stats as JSON.")
    args = parser.parse_args(argv)

    conversations = _load_split_dir(args.splits)
    if not conversations:
        parser.error(f"no JSONL rows under {args.splits}")

    unrolled = None
    if args.mlx_dir is not None:
        unrolled = []
        for name in ("train.jsonl", "valid.jsonl"):
            f = args.mlx_dir / name
            if f.exists():
                unrolled.extend(read_jsonl(f))
    require_unrolled = args.mlx_dir is not None
    stats = collect_stats(conversations, unrolled=unrolled)
    print(format_report(stats, require_unrolled_nontool=require_unrolled))
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.gate:
        reasons = gate_reasons(stats, require_unrolled_nontool=require_unrolled)
        if reasons:
            print("mix_stats_v8: abort", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
