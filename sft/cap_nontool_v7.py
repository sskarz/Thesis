"""Cap non-tool assistant unrolls for SFT recipe v7.

APIGen-MT multi-turn trajs contain many ``gpt`` clarify / final assistant
turns. Conversation-level mix can still be ≥50% APIGen, but **unrolling every
assistant turn** over-weights non-tool targets (~48% observed). The plan wants
~12–15% non-tool **after unroll**.

This module downsamples non-tool unrolled rows so ``pct_nontool`` lands in
``[V7_NONTOOL_MIN, V7_NONTOOL_MAX]`` (target ``V7_NONTOOL_TARGET`` ≈ 0.15).

**What this does not do**

* It does **not** delete APIGen conversations from ``splits-v7``. Full
  transcripts stay in the conversation-level mix.
* It does **not** rewrite NL finals to ``[]``.
* It does **not** drop ``tool_call`` unrolled rows.
* The cap applies at **unroll → mlx-v7** time (``train_mlx --prepare``,
  ``mix_stats_v7`` projected unroll, ``filter_empty_after_exec --recipe v7``
  writes). ``mix_stats_v7 --gate --mlx-dir`` must see these same capped rows.

Call **after** the normal v7 empty-after-exec filter (and after the v8
identical-call drop when recipe is v8). RNG seed **47** (v7) or **48** (v8).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

from sft.config import (
    SEED_V7,
    SEED_V8,
    V7_NONTOOL_MAX,
    V7_NONTOOL_MIN,
    V7_NONTOOL_TARGET,
    V8_NONTOOL_MAX,
    V8_NONTOOL_MIN,
    V8_NONTOOL_TARGET,
)
from sft.format_apigen_mt import (
    NONTOOL_LABELS,
    STOP_CLARIFY,
    STOP_FINAL,
    STOP_IRRELEVANCE,
    STOP_TOOL_CALL,
    label_assistant_content,
)

NONTOOL_CAP_LABELS = (STOP_CLARIFY, STOP_FINAL, STOP_IRRELEVANCE)

_CAP_NOTE = (
    "Full conversations remain in splits-v7; this cap applies at "
    "unroll → mlx-v7 time (prepare / mix_stats projected unroll)."
)
_CAP_NOTE_V8 = (
    "Full conversations remain in splits-v8; this cap applies at "
    "unroll → mlx-v8 time (prepare / mix_stats projected unroll, after "
    "identical-call drop)."
)


def cap_kwargs_for_recipe(recipe: str) -> dict[str, Any]:
    """Seed / band / labels for ``cap_nontool_unrolls`` under a mix recipe."""
    if recipe == "v8":
        return {
            "seed": SEED_V8,
            "target": V8_NONTOOL_TARGET,
            "min_frac": V8_NONTOOL_MIN,
            "max_frac": V8_NONTOOL_MAX,
            "experiment": "v8",
            "note": _CAP_NOTE_V8,
        }
    return {
        "seed": SEED_V7,
        "target": V7_NONTOOL_TARGET,
        "min_frac": V7_NONTOOL_MIN,
        "max_frac": V7_NONTOOL_MAX,
        "experiment": "v7",
        "note": _CAP_NOTE,
    }


def unroll_gold(row: dict) -> str:
    messages = row.get("messages") or []
    if not messages:
        return ""
    return str(messages[-1].get("content") or "")


def target_nontool_keep_count(
    n_tool: int,
    n_nontool: int,
    *,
    target: float = V7_NONTOOL_TARGET,
    min_frac: float = V7_NONTOOL_MIN,
    max_frac: float = V7_NONTOOL_MAX,
) -> int:
    """How many non-tool rows to keep so pct lands in ``[min_frac, max_frac]``.

    Solves ``k / (n_tool + k) ≈ target``. If the band is unreachable (tiny
    ``n_tool``), returns the count closest to ``target`` without adding rows.
    """
    if n_nontool <= 0:
        return 0
    if n_tool <= 0:
        return n_nontool
    n = n_tool + n_nontool
    current = n_nontool / n
    if current <= max_frac:
        return n_nontool

    target = min(max(target, min_frac), max_frac)
    if target <= 0:
        return 0
    if target >= 1:
        return n_nontool

    k = int(round(target * n_tool / (1.0 - target)))
    k_lo = math.ceil(min_frac * n_tool / (1.0 - min_frac) - 1e-12)
    k_hi = math.floor(max_frac * n_tool / (1.0 - max_frac) + 1e-12)
    k_lo = max(0, k_lo)
    k_hi = max(0, k_hi)
    if k_lo <= k_hi:
        k = min(max(k, k_lo), k_hi)
    k = min(max(k, 0), n_nontool)
    return k


def _allocate_by_label(available: dict[str, int], k: int) -> dict[str, int]:
    """Keep a balanced mix of clarify / final / irrelevance among kept non-tool.

    Reserve one of each nonempty label when ``k`` allows, then largest-remainder
    on the leftover so we do not drop an entire stop type.
    """
    labels = [lab for lab in NONTOOL_CAP_LABELS if available.get(lab, 0) > 0]
    out = {lab: 0 for lab in NONTOOL_CAP_LABELS}
    if not labels or k <= 0:
        return out
    total_avail = sum(available[lab] for lab in labels)
    k = min(k, total_avail)

    remaining_avail = {lab: available[lab] for lab in labels}
    remaining_k = k
    if k >= len(labels):
        for lab in labels:
            out[lab] = 1
            remaining_avail[lab] -= 1
            remaining_k -= 1
    else:
        # Prefer diversity: one each from the largest pools, stable by name.
        order = sorted(labels, key=lambda lab: (-available[lab], lab))
        for lab in order[:k]:
            out[lab] = 1
        return out

    rem_total = sum(remaining_avail.values())
    if remaining_k <= 0 or rem_total <= 0:
        return out

    raw = {lab: remaining_k * remaining_avail[lab] / rem_total for lab in labels}
    floors = {lab: min(remaining_avail[lab], math.floor(raw[lab])) for lab in labels}
    leftover = remaining_k - sum(floors.values())
    remainders = sorted(
        labels,
        key=lambda lab: (
            raw[lab] - math.floor(raw[lab]),
            remaining_avail[lab],
            lab,
        ),
        reverse=True,
    )
    extra = {lab: 0 for lab in labels}
    for lab in remainders:
        if leftover <= 0:
            break
        room = remaining_avail[lab] - floors[lab]
        if room <= 0:
            continue
        extra[lab] += 1
        leftover -= 1
    if leftover > 0:
        for lab in remainders:
            while leftover > 0 and floors[lab] + extra[lab] < remaining_avail[lab]:
                extra[lab] += 1
                leftover -= 1
    for lab in labels:
        out[lab] += floors[lab] + extra[lab]
    return out


def cap_nontool_unrolls(
    rows: Iterable[dict],
    *,
    seed: int = SEED_V7,
    target: float = V7_NONTOOL_TARGET,
    min_frac: float = V7_NONTOOL_MIN,
    max_frac: float = V7_NONTOOL_MAX,
    experiment: str = "v7",
    note: str | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Keep every ``tool_call`` unroll; downsample non-tool to the target band.

    Deterministic: sort by ``id`` within each label, then ``random.Random(seed)``.
    Original relative order of kept rows is preserved.
    """
    items = list(rows)
    labeled: list[tuple[int, dict, str]] = []
    for i, row in enumerate(items):
        labeled.append((i, row, label_assistant_content(unroll_gold(row))))

    tool_idxs = [i for i, _row, lab in labeled if lab == STOP_TOOL_CALL]
    nontool = [(i, row, lab) for i, row, lab in labeled if lab in NONTOOL_LABELS]
    other = [(i, row) for i, row, lab in labeled if lab not in NONTOOL_LABELS and lab != STOP_TOOL_CALL]

    n_tool = len(tool_idxs)
    n_non = len(nontool)
    n = n_tool + n_non
    current = (n_non / n) if n else 0.0
    before_labels = Counter(lab for _i, _row, lab in labeled)

    k = target_nontool_keep_count(
        n_tool, n_non, target=target, min_frac=min_frac, max_frac=max_frac
    )
    capped = k < n_non

    keep_non_idxs: set[int] = {i for i, _row, _lab in nontool}
    kept_by_label = Counter(lab for _i, _row, lab in nontool)
    if capped:
        available = Counter(lab for _i, _row, lab in nontool)
        alloc = _allocate_by_label(dict(available), k)
        rng = random.Random(seed)
        keep_non_idxs = set()
        kept_by_label = Counter()
        by_label: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for i, row, lab in nontool:
            by_label[lab].append((i, row))
        for lab in NONTOOL_CAP_LABELS:
            pool = by_label.get(lab) or []
            pool = sorted(pool, key=lambda item: str(item[1].get("id", "")))
            rng.shuffle(pool)
            take = alloc.get(lab, 0)
            for i, _row in pool[:take]:
                keep_non_idxs.add(i)
                kept_by_label[lab] += 1

    keep_idxs = set(tool_idxs) | keep_non_idxs | {i for i, _row in other}
    kept = [row for i, row in enumerate(items) if i in keep_idxs]
    n_kept_non = len(keep_non_idxs)
    n_after = len(kept)
    pct_after = (n_kept_non / n_after) if n_after else 0.0
    after_labels = Counter(
        label_assistant_content(unroll_gold(row)) for row in kept
    )

    stats: dict[str, Any] = {
        "seed": seed,
        "target": target,
        "min_frac": min_frac,
        "max_frac": max_frac,
        "capped": capped,
        "n_before": len(items),
        "n_after": n_after,
        "n_tool": n_tool,
        "n_nontool_before": n_non,
        "n_nontool_after": n_kept_non,
        "pct_nontool_before": round(current, 4),
        "pct_nontool_after": round(pct_after, 4),
        "by_label_before": dict(sorted(before_labels.items())),
        "by_label_after": dict(sorted(after_labels.items())),
        "nontool_kept_by_label": dict(sorted(kept_by_label.items())),
        "dropped_nontool": n_non - n_kept_non,
        "experiment": experiment,
        "note": note if note is not None else _CAP_NOTE,
    }
    return kept, stats


def format_cap_report(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "v7 non-tool unroll cap: (not applied)"
    experiment = stats.get("experiment") or "v7"
    return "\n".join(
        [
            f"{experiment} non-tool unroll cap "
            f"(seed={stats.get('seed')} target={stats.get('target')} "
            f"band=[{stats.get('min_frac')}, {stats.get('max_frac')}] "
            f"capped={stats.get('capped')}):",
            f"  unrolled rows: {stats.get('n_before')} → {stats.get('n_after')} "
            f"(kept all {stats.get('n_tool')} tool_call; non-tool "
            f"{stats.get('n_nontool_before')} → {stats.get('n_nontool_after')}; "
            f"dropped {stats.get('dropped_nontool')})",
            f"  pct_nontool: {stats.get('pct_nontool_before'):.4f} → "
            f"{stats.get('pct_nontool_after'):.4f}",
            f"  by_label before: {stats.get('by_label_before')}",
            f"  by_label after: {stats.get('by_label_after')}",
            f"  note: {stats.get('note')}",
        ]
    )
