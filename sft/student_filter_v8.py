"""ToolACE-R-lite loop-trajectory filter for SFT v8 (no MLX student drafts).

The v7 ``student_filter_v7`` stub is the full Jaccard mid-difficulty loop
(Phase 3; needs Mac MLX). v8 lever 1 only needs a **teacher-side** filter:
drop conversations whose gold already contains a long identical-call run
(observe→re-emit the same tool) **before** unroll.

v8.1 (recipe flag still ``v8``):

* Consecutive assistant tool-call turns with the same canonical IFFC call
  set (user / exec observations in between still count) — original v8.
* **Also** BFCL-style A-B-A-B: count assistant turns that re-emit an
  already-executed call after a tool result (``ls()`` /
  ``displayCarStatus`` spam). Consecutive-only missed those loops.

* Default ``min_run`` is ``V8_LOOP_MIN_RUN`` (3): a single retry is left
  for the unroll-time identical-call drop; 20× ``ls`` / A-B-A-B loops are
  removed from the mix.
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
from sft.drop_identical_call import (
    count_observe_same_call_turns,
    max_identical_call_run,
    max_observe_same_call_run,
)


class LoopFilterAbort(Exception):
    """Applying the loop filter would violate the APIGen-MT conversation floor."""


def is_loop_trajectory(
    record: dict,
    *,
    min_run: int = V8_LOOP_MIN_RUN,
) -> bool:
    """True when gold has a long consecutive identical **or** observe→same-call loop."""
    messages = record.get("messages") or []
    if min_run <= 1:
        return (
            max_identical_call_run(messages) >= 1
            or count_observe_same_call_turns(messages) >= 1
        )
    return (
        max_identical_call_run(messages) >= min_run
        or count_observe_same_call_turns(messages) >= min_run
        or max_observe_same_call_run(messages) >= min_run
    )


def filter_loop_trajectories(
    records: Iterable[dict],
    *,
    min_run: int = V8_LOOP_MIN_RUN,
    min_apigen_frac: float = V8_MIN_APIGEN_FRAC,
    abort_if_apigen_floor_violated: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Drop long identical-call / A-B-A-B loop trajs; abort if APIGen floor would fail.

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
    dropped_observe_counts: Counter[int] = Counter()
    dropped_by_kind: Counter[str] = Counter()
    for rec in items:
        messages = rec.get("messages") or []
        run = max_identical_call_run(messages)
        n_observe = count_observe_same_call_turns(messages)
        observe_run = max_observe_same_call_run(messages)
        if not is_loop_trajectory(rec, min_run=min_run):
            kept.append(rec)
            continue
        dropped.append(rec)
        source = str(rec.get("source") or "unknown")
        dropped_by_source[source] += 1
        dropped_run_lengths[run] += 1
        dropped_observe_counts[n_observe] += 1
        consecutive_loop = run >= min_run if min_run > 1 else run >= 1
        observe_loop = (
            n_observe >= min_run or observe_run >= min_run
            if min_run > 1
            else n_observe >= 1
        )
        if consecutive_loop and observe_loop:
            dropped_by_kind["both"] += 1
        elif consecutive_loop:
            dropped_by_kind["consecutive_identical"] += 1
        else:
            dropped_by_kind["observe_same_call"] += 1

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
        "dropped_by_kind": dict(sorted(dropped_by_kind.items())),
        "dropped_run_lengths": dict(sorted(dropped_run_lengths.items())),
        "dropped_observe_same_call_counts": dict(
            sorted(dropped_observe_counts.items())
        ),
        "n_apigen_before": n_apigen_before,
        "n_apigen_after": n_apigen,
        "pct_apigen_mt_before": round(pct_apigen_before, 4),
        "pct_apigen_mt_after": round(pct_apigen, 4),
        "aborted": False,
        "note": (
            "ToolACE-R-lite (v8.1): drop teacher loop trajs with "
            f">={min_run} consecutive identical assistant tool-calls **or** "
            f">={min_run} observe→same-call re-emits (ls/displayCarStatus "
            "A-B-A-B). No MLX student drafts. Unroll-time drop still removes "
            "remaining pairwise / observe→same-call repeats."
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
            f"  dropped_by_kind: {stats.get('dropped_by_kind')}",
            f"  dropped_run_lengths: {stats.get('dropped_run_lengths')}",
            f"  dropped_observe_same_call_counts: "
            f"{stats.get('dropped_observe_same_call_counts')}",
            f"  note: {stats.get('note')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SFT v8 ToolACE-R-lite: drop teacher identical-call / "
            "observe→same-call loop trajectories (no MLX Jaccard drafts)."
        )
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=V8_LOOP_MIN_RUN,
        help=(
            "Drop conversations with this many consecutive identical calls "
            "or observe→same-call re-emits."
        ),
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
