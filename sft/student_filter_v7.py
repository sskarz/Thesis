"""Student-aware mid-difficulty filter for SFT v7 filler pools.

**Not implemented in this scaffolding PR.** Phase 3 of the v7 plan:

1. Draft the first nonempty tool-call prefix with base Gemma MLX
   (``sft/models/gemma-4-e2b-it-mlx``, greedy ``max-tokens 128``).
2. Parse draft + gold with ``benchmark.iffc.parse_iffc_response``.
3. Difficulty ``D = 1 - Jaccard(set(gold_names), set(draft_names))``.
4. Keep mid band ``0.15 ≤ D ≤ 0.85`` on ToolACE/xLAM filler; for APIGen-MT
   only drop ``D≈0`` extreme easy or unparseable so the core stays ≥50%.
5. Cache drafts at ``sft/data/cache/student_filter_v7.jsonl``.

Do **not** iterate more than one filter→train cycle in v7. This module is a
stub so ``python -m sft.student_filter_v7 --help`` exists; the MLX loop lands
in a follow-up after Hub Accept + mix smoke.

See the SFT v7 implementation plan (2026-09-23), Phase 3.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SFT v7 student-aware mid-difficulty filter (Phase 3 stub). "
            "The MLX Jaccard loop is not implemented in this PR."
        )
    )
    parser.add_argument(
        "--pools",
        default="filler",
        help="Which pools to score (filler | apigen | all). Stub only.",
    )
    parser.add_argument(
        "--band",
        default="0.15,0.85",
        help="Keep band lo,hi for filler Jaccard difficulty D.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Calibration subsample size (e.g. 400). Stub only.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Score the full filler pool (Mac wall ~hours). Stub only.",
    )
    args = parser.parse_args(argv)
    print(
        "TODO: student_filter_v7 Phase 3 — MLX greedy draft Jaccard filter "
        f"(pools={args.pools!r} band={args.band!r} sample={args.sample} "
        f"full={args.full}). See the v7 implementation plan. "
        "No rows were filtered."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
