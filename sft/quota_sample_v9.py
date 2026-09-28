"""Unroll-time action-class quota sampling for SFT v9 (lever2).

Replaces v7/v8 ``cap_nontool_*`` for recipe v9. After the v4 empty/recovery
filter (and **without** a teacher observe→same traj drop), stratify unrolled
assistant rows by ``action_class_v9`` and keep/dupe toward target shares.

Hard rules (locked):

* Soft-exclude ``observe_same_call`` / ``consecutive_identical`` from keep
* Prefer **duplication** of scarce classes over synthesizing NL (no v5 turn0)
* When duplicating ``final_after_obs``, prefer rows whose prior tools look
  advancing (strict subtype); never hard-filter loose finals
* Never map NL finals → ``[]``
* Abort if ``irrelevance_[]`` share > 0.12 or ``refuse`` ≈ 0 after sampling
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

from sft.action_class_v9 import (
    ACTION_ASK,
    ACTION_FINAL_AFTER_OBS,
    ACTION_IRRELEVANCE,
    ACTION_MT_TOOL_CHAIN_ADVANCE,
    ACTION_OTHER_NL,
    ACTION_OTHER_TOOL,
    ACTION_REFUSE,
    ACTION_TOOL_CALL_START,
    EXCLUDE_FROM_KEEP,
    QUOTA_CLASSES,
    action_label_detail,
    classify_unrolled_row,
)
from sft.config import (
    SEED_V9,
    V9_ABORT_IRRELEVANCE_UNROLL_FRAC,
    V9_ABORT_REFUSE_MIN,
    V9_ACTION_ASK,
    V9_ACTION_FINAL_AFTER_OBS,
    V9_ACTION_IRRELEVANCE,
    V9_ACTION_MT_TOOL_CHAIN_ADVANCE,
    V9_ACTION_REFUSE,
    V9_ACTION_TOOL_CALL_START,
)

# Target midpoints + hard bands from config tuples (lo, hi, hard_lo, hard_hi).
_BANDS: dict[str, tuple[float, float, float, float]] = {
    ACTION_MT_TOOL_CHAIN_ADVANCE: V9_ACTION_MT_TOOL_CHAIN_ADVANCE,
    ACTION_TOOL_CALL_START: V9_ACTION_TOOL_CALL_START,
    ACTION_FINAL_AFTER_OBS: V9_ACTION_FINAL_AFTER_OBS,
    ACTION_ASK: V9_ACTION_ASK,
    ACTION_REFUSE: V9_ACTION_REFUSE,
    ACTION_IRRELEVANCE: V9_ACTION_IRRELEVANCE,
}


class QuotaAbort(Exception):
    """Hard abort: action-class quotas unreachable or refuse≈0 / irrelevance>12%."""


def _target_share(band: tuple[float, float, float, float]) -> float:
    return (band[0] + band[1]) / 2.0


def _midpoint_targets() -> dict[str, float]:
    return {cls: _target_share(_BANDS[cls]) for cls in QUOTA_CLASSES}


def classify_rows(
    rows: Iterable[dict],
) -> tuple[list[tuple[dict, str, dict]], dict[str, Any]]:
    """Label each unrolled row; return (items, pre_stats)."""
    items: list[tuple[dict, str, dict]] = []
    counts: Counter[str] = Counter()
    excluded = 0
    strict_final = 0
    for row in rows:
        messages = list(row.get("messages") or [])
        detail = action_label_detail(messages)
        cls = detail["action_class"]
        counts[cls] += 1
        if cls in EXCLUDE_FROM_KEEP:
            excluded += 1
            continue
        if detail.get("final_after_obs_strict_advancing"):
            strict_final += 1
        items.append((row, cls, detail))
    n = sum(counts.values())
    pre = {
        "n_input": n,
        "n_eligible": len(items),
        "n_excluded_soft": excluded,
        "by_class_input": dict(sorted(counts.items())),
        "n_final_after_obs_strict": strict_final,
    }
    return items, pre


def _allocate_counts(n_total: int, targets: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of ``n_total`` slots across quota classes."""
    if n_total <= 0:
        return {cls: 0 for cls in QUOTA_CLASSES}
    # Renormalize targets over QUOTA_CLASSES only.
    s = sum(targets[cls] for cls in QUOTA_CLASSES)
    if s <= 0:
        raise QuotaAbort("action-class target shares sum to 0")
    raw = {cls: n_total * (targets[cls] / s) for cls in QUOTA_CLASSES}
    floors = {cls: int(math.floor(raw[cls])) for cls in QUOTA_CLASSES}
    leftover = n_total - sum(floors.values())
    remainders = sorted(
        QUOTA_CLASSES,
        key=lambda cls: (raw[cls] - floors[cls], targets[cls], cls),
        reverse=True,
    )
    out = dict(floors)
    for cls in remainders:
        if leftover <= 0:
            break
        out[cls] += 1
        leftover -= 1
    return out


def _dupe_key(row: dict, detail: dict) -> tuple:
    """Sort key: prefer strict advancing finals when duplicating."""
    strict = 1 if detail.get("final_after_obs_strict_advancing") else 0
    return (-strict, str(row.get("id", "")))


def quota_sample_unrolls(
    rows: Iterable[dict],
    *,
    seed: int = SEED_V9,
    n_target: int | None = None,
    skip_gates: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    """Keep/dupe unrolled rows toward v9 action-class target shares.

    ``n_target`` defaults to the count of eligible (non-excluded) input rows so
    the sampler redistributes without inventing a larger corpus. Scarce classes
    are duplicated; oversupplied classes are downsampled. Soft-excluded loop
    shapes are logged and never kept.
    """
    eligible, pre = classify_rows(rows)
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row, cls, detail in eligible:
        by_class[cls].append((row, detail))

    for cls in list(by_class):
        pool = by_class[cls]
        pool = sorted(pool, key=lambda item: _dupe_key(item[0], item[1]))
        # Stable shuffle after preference sort: shuffle within equal keys.
        # Prefer strict finals first (already sorted), then shuffle tails.
        strict = [(r, d) for r, d in pool if d.get("final_after_obs_strict_advancing")]
        loose = [(r, d) for r, d in pool if not d.get("final_after_obs_strict_advancing")]
        rng.shuffle(loose)
        # Keep strict block first for final_after_obs duplication preference.
        if cls == ACTION_FINAL_AFTER_OBS:
            by_class[cls] = strict + loose
        else:
            combined = strict + loose
            rng.shuffle(combined)
            by_class[cls] = combined

    n_eligible = len(eligible)
    if n_target is None:
        n_target = n_eligible
    n_target = max(0, int(n_target))

    targets = _midpoint_targets()
    want = _allocate_counts(n_target, targets)

    kept: list[dict] = []
    kept_by: Counter[str] = Counter()
    duped_by: Counter[str] = Counter()
    dropped_by: Counter[str] = Counter()
    other_kept = 0

    # First pass: fill quota classes (downsample or dupe).
    for cls in QUOTA_CLASSES:
        pool = by_class.get(cls) or []
        need = want.get(cls, 0)
        if need <= 0:
            dropped_by[cls] += len(pool)
            continue
        if not pool:
            # Cannot fill — leave short; gates may abort.
            continue
        if len(pool) >= need:
            take = pool[:need]
            dropped_by[cls] += len(pool) - need
            for row, _d in take:
                kept.append(row)
                kept_by[cls] += 1
        else:
            for row, _d in pool:
                kept.append(row)
                kept_by[cls] += 1
            # Duplicate scarce rows (prefer already-ordered strict finals).
            i = 0
            while kept_by[cls] < need:
                row, _d = pool[i % len(pool)]
                # Mark duplicates with a suffix id for uniqueness in logs only;
                # training uses messages; keep a copy so JSONL ids stay unique.
                dup = dict(row)
                base_id = str(row.get("id", ""))
                dup_i = kept_by[cls] - len(pool) + 1
                dup["id"] = f"{base_id}#dup{dup_i}"
                dup["quota_dup"] = True
                dup["quota_dup_of"] = base_id
                kept.append(dup)
                kept_by[cls] += 1
                duped_by[cls] += 1
                i += 1

    # Residual other_tool / other_nl: do not upsample; optionally keep a few
    # to avoid emptying the bag when quotas undershoot n_target.
    residual_budget = max(0, n_target - len(kept))
    residual_pool: list[dict] = []
    for cls in (ACTION_OTHER_TOOL, ACTION_OTHER_NL):
        for row, _d in by_class.get(cls) or []:
            residual_pool.append(row)
    rng.shuffle(residual_pool)
    for row in residual_pool[:residual_budget]:
        kept.append(row)
        other_kept += 1
        kept_by[classify_unrolled_row(row)] += 1

    # Preserve a deterministic order: sort by id.
    kept.sort(key=lambda r: str(r.get("id", "")))

    n_after = len(kept)
    pct = {
        cls: round((kept_by[cls] / n_after) if n_after else 0.0, 4)
        for cls in QUOTA_CLASSES
    }
    # Also count residual labels present after.
    after_all: Counter[str] = Counter()
    for row in kept:
        after_all[classify_unrolled_row(row)] += 1

    stats: dict[str, Any] = {
        "seed": seed,
        "n_target": n_target,
        "n_before_eligible": n_eligible,
        "n_after": n_after,
        "pre": pre,
        "want": want,
        "kept_by_class": dict(sorted(kept_by.items())),
        "duped_by_class": dict(sorted(duped_by.items())),
        "dropped_by_class": dict(sorted(dropped_by.items())),
        "pct_by_class": pct,
        "by_class_after": dict(sorted(after_all.items())),
        "other_kept": other_kept,
        "targets": {cls: round(targets[cls], 4) for cls in QUOTA_CLASSES},
        "bands": {
            cls: {
                "target": [band[0], band[1]],
                "hard": [band[2], band[3]],
            }
            for cls, band in _BANDS.items()
        },
        "note": (
            "v9 lever2 action-class quota sample (assistant gold only). "
            "Soft-excludes observe→same / consecutive identical. "
            "Dupes scarce classes; never maps NL→[]. No teacher loop traj filter."
        ),
    }

    reasons = gate_quota_stats(stats)
    stats["gate_reasons"] = reasons
    if reasons and not skip_gates:
        raise QuotaAbort(
            "v9 action-class quota gates failed:\n  - " + "\n  - ".join(reasons)
        )
    return kept, stats


def gate_quota_stats(stats: dict[str, Any]) -> list[str]:
    """Return hard-gate failure reasons for post-sample action-class shares."""
    reasons: list[str] = []
    n_after = int(stats.get("n_after") or 0)
    if n_after <= 0:
        return ["n_after == 0"]
    pct = stats.get("pct_by_class") or {}
    after = stats.get("by_class_after") or {}

    irr = float(pct.get(ACTION_IRRELEVANCE, 0.0))
    if irr > V9_ABORT_IRRELEVANCE_UNROLL_FRAC:
        reasons.append(
            f"pct irrelevance_[]={irr:.4f} > abort "
            f"{V9_ABORT_IRRELEVANCE_UNROLL_FRAC}"
        )

    refuse_n = int(after.get(ACTION_REFUSE, 0))
    refuse_pct = float(pct.get(ACTION_REFUSE, 0.0))
    if refuse_n == 0 or refuse_pct < V9_ABORT_REFUSE_MIN:
        reasons.append(
            f"refuse≈0 after quota sample (n={refuse_n}, pct={refuse_pct:.4f}); "
            f"abort threshold {V9_ABORT_REFUSE_MIN}"
        )

    for cls in QUOTA_CLASSES:
        if cls == ACTION_REFUSE:
            # Soft floor 2% OK; only ≈0 aborts (handled above). Hard hi still.
            _t_lo, _t_hi, _h_lo, h_hi = _BANDS[cls]
            p = float(pct.get(cls, 0.0))
            if p > h_hi:
                reasons.append(f"{cls} pct={p:.4f} > hard_hi={h_hi}")
            continue
        _t_lo, _t_hi, h_lo, h_hi = _BANDS[cls]
        p = float(pct.get(cls, 0.0))
        if p < h_lo or p > h_hi:
            reasons.append(
                f"{cls} pct={p:.4f} outside hard band [{h_lo}, {h_hi}]"
            )
    return reasons


def format_quota_report(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "v9 action-class quota sample: (not applied)"
    lines = [
        "v9 action-class quota sample "
        f"(seed={stats.get('seed')} n_target={stats.get('n_target')} "
        f"eligible={stats.get('n_before_eligible')} → after={stats.get('n_after')}):",
        f"  soft-excluded (observe/consecutive): "
        f"{(stats.get('pre') or {}).get('n_excluded_soft')}",
        f"  want: {stats.get('want')}",
        f"  kept: {stats.get('kept_by_class')}",
        f"  duped: {stats.get('duped_by_class')}",
        f"  dropped: {stats.get('dropped_by_class')}",
        f"  pct: {stats.get('pct_by_class')}",
        f"  note: {stats.get('note')}",
    ]
    reasons = stats.get("gate_reasons") or []
    if reasons:
        lines.append("  gate_reasons:")
        lines.extend(f"    - {r}" for r in reasons)
    return "\n".join(lines)
