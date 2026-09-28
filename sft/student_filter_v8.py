"""ToolACE-R-lite loop-trajectory filter for SFT v8 (no MLX student drafts).

The v7 ``student_filter_v7`` stub is the full Jaccard mid-difficulty loop
(Phase 3; needs Mac MLX). v8 lever 1 only needs a **teacher-side** filter:
drop conversations whose gold already contains a long identical-call run
(observe→re-emit the same tool) **before** unroll.

* Detect consecutive assistant tool-call turns with the same canonical
  IFFC call set (user / exec observations in between still count).
* Default ``min_run`` is ``V8_LOOP_MIN_RUN`` (3): a single retry is left
  for the unroll-time identical-call drop; 20× ``ls`` loops are removed
  from the mix.
* Log drops by source. **Abort** if applying the filter would put
  conversation ``pct_apigen_mt`` below ``V8_MIN_APIGEN_FRAC`` — do not
  silently write a mix that misses the APIGen-MT floor.

This is **not** the MLX Jaccard student draft. Do not iterate filter→train
cycles in this PR.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Iterable

from sft.config import (
    SOURCE_APIGEN_MT,
    V8_LOOP_MIN_RUN,
    V8_MIN_APIGEN_FRAC,
)
from sft.drop_identical_call import max_identical_call_run


class LoopFilterAbort(Exception):
    """Applying the loop filter would violate the APIGen-MT conversation floor."""


def is_loop_trajectory(
    record: dict,
    *,
    min_run: int = V8_LOOP_MIN_RUN,
) -> bool:
    """True when gold has ``min_run``+ consecutive identical assistant calls."""
    if min_run <= 1:
        return max_identical_call_run(record.get("messages") or []) >= 1
    return max_identical_call_run(record.get("messages") or []) >= min_run


def filter_loop_trajectories(
    records: Iterable[dict],
    *,
    min_run: int = V8_LOOP_MIN_RUN,
    min_apigen_frac: float = V8_MIN_APIGEN_FRAC,
    abort_if_apigen_floor_violated: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Drop long identical-call loop trajs; abort if APIGen floor would fail.

    When ``abort_if_apigen_floor_violated`` is true (default) this raises
    ``LoopFilterAbort`` instead of returning a pool with
    ``pct_apigen_mt < min_apigen_frac``. Callers (``build_mix_v8``) treat that
    as a hard mix abort.
    """
    items = list(records)
    n_apigen_before = sum(
        1 for rec in items if str(rec.get("source") or "") == SOURCE_APIGEN_MT
    )
    dropped: list[dict] = []
    kept: list[dict] = []
    dropped_by_source: Counter[str] = Counter()
    dropped_run_lengths: Counter[int] = Counter()
    for rec in items:
        run = max_identical_call_run(rec.get("messages") or [])
        if run >= min_run:
            dropped.append(rec)
            source = str(rec.get("source") or "unknown")
            dropped_by_source[source] += 1
            dropped_run_lengths[run] += 1
            continue
        kept.append(rec)

    n = len(kept)
    n_apigen = sum(
        1 for rec in kept if str(rec.get("source") or "") == SOURCE_APIGEN_MT
    )
    pct_apigen = (n_apigen / n) if n else 0.0
    pct_apigen_before = (n_apigen_before / len(items)) if items else 0.0
    stats: dict[str, Any] = {
        "min_run": min_run,
        "min_apigen_frac": min_apigen_frac,
        "n_before": len(items),
        "n_after": n,
        "dropped": len(dropped),
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
        "dropped_run_lengths": dict(sorted(dropped_run_lengths.items())),
        "n_apigen_before": n_apigen_before,
        "n_apigen_after": n_apigen,
        "pct_apigen_mt_before": round(pct_apigen_before, 4),
        "pct_apigen_mt_after": round(pct_apigen, 4),
        "aborted": False,
        "note": (
            "ToolACE-R-lite (v8): drop teacher loop trajs with "
            f">={min_run} consecutive identical assistant tool-calls. "
            "No MLX student drafts. Unroll-time identical-call drop still "
            "removes remaining pairwise repeats."
        ),
    }
    if abort_if_apigen_floor_violated and pct_apigen < min_apigen_frac:
        stats["aborted"] = True
        raise LoopFilterAbort(
            "v8 loop filter would violate APIGen-MT floor: "
            f"pct_apigen_mt={pct_apigen:.3f} < {min_apigen_frac} "
            f"(dropped {len(dropped)} loop trajs; "
            f"apigen {n_apigen_before}→{n_apigen}). "
            f"dropped_by_source={dict(dropped_by_source)}. "
            "Do not write splits-v8."
        )
    return kept, stats


def format_loop_filter_report(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "v8 loop-trajectory filter: (not applied)"
    return "\n".join(
        [
            "v8 ToolACE-R-lite loop filter "
            f"(min_run={stats.get('min_run')} "
            f"dropped {stats.get('dropped')} of {stats.get('n_before')} "
            f"→ {stats.get('n_after')}):",
            f"  pct_apigen_mt: {stats.get('pct_apigen_mt_before'):.4f} → "
            f"{stats.get('pct_apigen_mt_after'):.4f} "
            f"(floor {stats.get('min_apigen_frac')})",
            f"  dropped_by_source: {stats.get('dropped_by_source')}",
            f"  dropped_run_lengths: {stats.get('dropped_run_lengths')}",
            f"  note: {stats.get('note')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SFT v8 ToolACE-R-lite: drop teacher identical-call loop "
            "trajectories (no MLX Jaccard drafts)."
        )
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=V8_LOOP_MIN_RUN,
        help="Drop conversations with this many consecutive identical calls.",
    )
    parser.add_argument(
        "--min-apigen-frac",
        type=float,
        default=V8_MIN_APIGEN_FRAC,
        help="Abort if remaining pct_apigen_mt would fall below this floor.",
    )
    args = parser.parse_args(argv)
    print(
        "v8 student_filter_v8 is applied inside sft.build_mix_v8 "
        f"(min_run={args.min_run}, min_apigen_frac={args.min_apigen_frac}). "
        "Pass conversations through build_mix_v8; this CLI is a help stub."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
