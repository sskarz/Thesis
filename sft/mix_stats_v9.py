"""Conversation + unrolled action-class stats for SFT v9 (lever2).

Conversation gates:

* ``0.15 ≤ pct_apigen_mt ≤ 0.35``
* ``pct_toolace + pct_xlam ≥ 0.55``
* irrelevance conversation share ≤ 0.10 (abort > 0.12)
* no BFCL leak (checked at write time)

Unrolled gates (``--mlx-dir`` / projected quota sample):

* action-class shares inside hard bands
* ``empty_after_exec = 0``; ``first_assistant_empty = 0``
* ``observe_same_call`` gold ≤ 1% of tool rows
* ``pct_irrelevance_[] ≤ 0.10`` (abort > 0.12)
* refuse ≈ 0 aborts

CLI::

    uv run python -m sft.build_mix_v9 --out sft/data/splits-v9
    uv run python -m sft.mix_stats_v9 --splits sft/data/splits-v9 --gate
    uv run python -m sft.train_mlx --prepare-only --recipe v9 \\
      --splits-dir sft/data/splits-v9 --mlx-dir sft/data/mlx-v9
    uv run python -m sft.mix_stats_v9 --splits sft/data/splits-v9 \\
        --mlx-dir sft/data/mlx-v9 --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from sft.action_class_v9 import (
    ACTION_IRRELEVANCE,
    ACTION_OBSERVE_SAME,
    ACTION_REFUSE,
    QUOTA_CLASSES,
    classify_unrolled_row,
)
from sft.config import (
    SOURCE_APIGEN_MT,
    SOURCE_TOOLACE,
    SOURCE_XLAM,
    SPLITS_DIR_V9,
    V9_ABORT_IRRELEVANCE_FRAC,
    V9_ABORT_IRRELEVANCE_UNROLL_FRAC,
    V9_ABORT_REFUSE_MIN,
    V9_ACTION_ASK,
    V9_ACTION_FINAL_AFTER_OBS,
    V9_ACTION_IRRELEVANCE,
    V9_ACTION_MT_TOOL_CHAIN_ADVANCE,
    V9_ACTION_REFUSE,
    V9_ACTION_TOOL_CALL_START,
    V9_MAX_APIGEN_FRAC,
    V9_MAX_IRRELEVANCE_CONV_FRAC,
    V9_MAX_OBSERVE_SAME_OF_TOOL,
    V9_MAX_TOOLACE_FRAC,
    V9_MAX_XLAM_FRAC,
    V9_MIN_APIGEN_FRAC,
    V9_MIN_TOOLACE_FRAC,
    V9_MIN_TOOLACE_XLAM_FRAC,
    V9_MIN_XLAM_FRAC,
)
from sft.filter_empty_after_exec import (
    RECIPE_V9,
    classify_assistant_turn,
    flags_for_recipe,
    is_first_assistant_empty_target,
    unroll_and_filter,
)
from sft.format_apigen_mt import STOP_IRRELEVANCE
from sft.io import read_jsonl
from sft.mix_stats_v7 import (
    conversation_stop_label as _conv_stop,
    scan_conversations as scan_conversations_base,
)
from sft.quota_sample_v9 import (
    format_quota_report,
    gate_quota_stats,
    quota_sample_unrolls,
)

_BANDS = {
    "mt_tool_chain_advance": V9_ACTION_MT_TOOL_CHAIN_ADVANCE,
    "tool_call_start": V9_ACTION_TOOL_CALL_START,
    "final_after_obs": V9_ACTION_FINAL_AFTER_OBS,
    "ask": V9_ACTION_ASK,
    "refuse": V9_ACTION_REFUSE,
    "irrelevance_[]": V9_ACTION_IRRELEVANCE,
}


def _pct(n: int, total: int) -> float:
    return (n / total) if total else 0.0


def scan_conversations(records: Iterable[dict]) -> dict[str, Any]:
    recs = list(records)
    base = scan_conversations_base(recs)
    # Re-scan sources for toolace/xlam shares (base only tracks apigen).
    by_source = Counter(str(r.get("source") or "unknown") for r in recs)
    n = len(recs)
    n_toolace = by_source.get(SOURCE_TOOLACE, 0)
    n_xlam = by_source.get(SOURCE_XLAM, 0)
    n_apigen = by_source.get(SOURCE_APIGEN_MT, 0)
    irr = 0
    for rec in recs:
        if _conv_stop(rec) == STOP_IRRELEVANCE:
            irr += 1
    base["pct_toolace"] = round(_pct(n_toolace, n), 4)
    base["pct_xlam"] = round(_pct(n_xlam, n), 4)
    base["pct_toolace_xlam"] = round(_pct(n_toolace + n_xlam, n), 4)
    base["pct_apigen_mt"] = round(_pct(n_apigen, n), 4)
    base["pct_irrelevance_conv"] = round(_pct(irr, n), 4)
    base["by_source"] = dict(sorted(by_source.items()))
    return base


def scan_unrolled_actions(rows: Iterable[dict]) -> dict[str, Any]:
    items = list(rows)
    by_class: Counter[str] = Counter()
    empty_after_exec = 0
    first_assistant_empty = 0
    n_tool = 0
    n_observe_same = 0
    for row in items:
        messages = list(row.get("messages") or [])
        cls = classify_unrolled_row(row)
        by_class[cls] += 1
        if cls == ACTION_OBSERVE_SAME:
            n_observe_same += 1
        if messages:
            kind = classify_assistant_turn(messages, len(messages) - 1)
            if kind and "empty_after_exec" in str(kind):
                empty_after_exec += 1
            if is_first_assistant_empty_target(messages, len(messages) - 1):
                first_assistant_empty += 1
            gold = messages[-1].get("content") if messages else ""
            from sft.drop_identical_call import canonical_tool_calls

            if canonical_tool_calls(gold) is not None:
                n_tool += 1
    n = len(items)
    pct = {cls: round(_pct(by_class[cls], n), 4) for cls in QUOTA_CLASSES}
    return {
        "n": n,
        "by_action_class": dict(sorted(by_class.items())),
        "pct_action_class": pct,
        "empty_after_exec": empty_after_exec,
        "first_assistant_empty": first_assistant_empty,
        "n_tool_rows": n_tool,
        "n_observe_same_call": n_observe_same,
        "pct_observe_same_of_tool": round(
            _pct(n_observe_same, n_tool), 4
        ),
        "pct_irrelevance_[]": pct.get(ACTION_IRRELEVANCE, 0.0),
        "pct_refuse": pct.get(ACTION_REFUSE, 0.0),
    }


def collect_stats(
    conversations: list[dict],
    *,
    unrolled: list[dict] | None = None,
    unroll_recipe: str = RECIPE_V9,
    apply_quota: bool | None = None,
    seed: int = 49,
) -> dict[str, Any]:
    """Conversation stats plus optional projected / provided unrolled actions."""
    stats: dict[str, Any] = {"conversation": scan_conversations(conversations)}
    stats["experiment"] = "sft-v9"
    stats["lever"] = "lever2"

    if unrolled is not None:
        stats["unrolled"] = scan_unrolled_actions(unrolled)
        stats["quota"] = {
            "applied": False,
            "note": "stats from provided mlx rows (post-prepare)",
        }
        return stats

    # Projected unroll: v9 empty/recovery then quota sample (no teacher loop drop).
    if apply_quota is None:
        apply_quota = True
    flags = flags_for_recipe(unroll_recipe)
    rows, _filt = unroll_and_filter(conversations, check_bfcl_leak=False, **flags)
    quota_stats = None
    if apply_quota:
        try:
            rows, quota_stats = quota_sample_unrolls(rows, seed=seed, skip_gates=True)
        except Exception as exc:  # pragma: no cover - surfaced via gate_reasons
            quota_stats = {"error": str(exc)}
    stats["unrolled"] = scan_unrolled_actions(rows)
    stats["quota"] = quota_stats
    return stats


def gate_reasons(
    stats: dict[str, Any],
    *,
    require_unrolled_actions: bool = False,
) -> list[str]:
    reasons: list[str] = []
    conv = stats.get("conversation") or stats
    # Allow either nested or flat (when only conversation scan passed).
    if "pct_apigen_mt" not in conv and "conversation" in stats:
        conv = stats["conversation"]

    pct_a = float(conv.get("pct_apigen_mt") or 0.0)
    if pct_a < V9_MIN_APIGEN_FRAC or pct_a > V9_MAX_APIGEN_FRAC:
        reasons.append(
            f"pct_apigen_mt={pct_a:.4f} outside [{V9_MIN_APIGEN_FRAC}, "
            f"{V9_MAX_APIGEN_FRAC}]"
        )
    pct_t = float(conv.get("pct_toolace") or 0.0)
    if pct_t < V9_MIN_TOOLACE_FRAC or pct_t > V9_MAX_TOOLACE_FRAC:
        reasons.append(
            f"pct_toolace={pct_t:.4f} outside [{V9_MIN_TOOLACE_FRAC}, "
            f"{V9_MAX_TOOLACE_FRAC}]"
        )
    pct_x = float(conv.get("pct_xlam") or 0.0)
    if pct_x < V9_MIN_XLAM_FRAC or pct_x > V9_MAX_XLAM_FRAC:
        reasons.append(
            f"pct_xlam={pct_x:.4f} outside [{V9_MIN_XLAM_FRAC}, "
            f"{V9_MAX_XLAM_FRAC}]"
        )
    pct_tx = float(conv.get("pct_toolace_xlam") or 0.0)
    if pct_tx < V9_MIN_TOOLACE_XLAM_FRAC:
        reasons.append(
            f"pct_toolace+xlam={pct_tx:.4f} < {V9_MIN_TOOLACE_XLAM_FRAC}"
        )
    pct_irr_c = float(conv.get("pct_irrelevance_conv") or 0.0)
    if pct_irr_c > V9_ABORT_IRRELEVANCE_FRAC:
        reasons.append(
            f"pct_irrelevance_conv={pct_irr_c:.4f} > abort "
            f"{V9_ABORT_IRRELEVANCE_FRAC}"
        )
    elif pct_irr_c > V9_MAX_IRRELEVANCE_CONV_FRAC:
        reasons.append(
            f"pct_irrelevance_conv={pct_irr_c:.4f} > max "
            f"{V9_MAX_IRRELEVANCE_CONV_FRAC}"
        )

    if not require_unrolled_actions:
        return reasons

    unrolled = stats.get("unrolled") or {}
    if not unrolled:
        reasons.append("unrolled action stats missing")
        return reasons
    if int(unrolled.get("empty_after_exec") or 0) != 0:
        reasons.append(
            f"empty_after_exec={unrolled.get('empty_after_exec')} (want 0)"
        )
    if int(unrolled.get("first_assistant_empty") or 0) != 0:
        reasons.append(
            f"first_assistant_empty={unrolled.get('first_assistant_empty')} (want 0)"
        )
    obs = float(unrolled.get("pct_observe_same_of_tool") or 0.0)
    if obs > V9_MAX_OBSERVE_SAME_OF_TOOL:
        reasons.append(
            f"observe_same_call/tool={obs:.4f} > {V9_MAX_OBSERVE_SAME_OF_TOOL}"
        )
    irr_u = float(unrolled.get("pct_irrelevance_[]") or 0.0)
    if irr_u > V9_ABORT_IRRELEVANCE_UNROLL_FRAC:
        reasons.append(
            f"pct_irrelevance_[]={irr_u:.4f} > abort "
            f"{V9_ABORT_IRRELEVANCE_UNROLL_FRAC}"
        )
    elif irr_u > V9_ACTION_IRRELEVANCE[3]:
        reasons.append(
            f"pct_irrelevance_[]={irr_u:.4f} > hard_hi "
            f"{V9_ACTION_IRRELEVANCE[3]}"
        )
    refuse = float(unrolled.get("pct_refuse") or 0.0)
    refuse_n = int((unrolled.get("by_action_class") or {}).get(ACTION_REFUSE, 0))
    if refuse_n == 0 or refuse < V9_ABORT_REFUSE_MIN:
        reasons.append(
            f"refuse≈0 (n={refuse_n}, pct={refuse:.4f})"
        )

    pct_actions = unrolled.get("pct_action_class") or {}
    for cls, band in _BANDS.items():
        if cls == ACTION_REFUSE:
            _t_lo, _t_hi, _h_lo, h_hi = band
            p = float(pct_actions.get(cls, 0.0))
            if p > h_hi:
                reasons.append(f"{cls} pct={p:.4f} > hard_hi={h_hi}")
            continue
        _t_lo, _t_hi, h_lo, h_hi = band
        p = float(pct_actions.get(cls, 0.0))
        if p < h_lo or p > h_hi:
            reasons.append(
                f"{cls} pct={p:.4f} outside hard band [{h_lo}, {h_hi}]"
            )

    quota = stats.get("quota")
    if isinstance(quota, dict) and quota.get("pct_by_class"):
        reasons.extend(gate_quota_stats(quota))
    return reasons


def format_report(
    stats: dict[str, Any],
    *,
    require_unrolled_actions: bool = False,
) -> str:
    conv = stats.get("conversation") or stats
    lines = [
        "sft-v9 / lever2 mix_stats:",
        f"  n={conv.get('n')} by_source={conv.get('by_source')}",
        f"  pct_toolace={conv.get('pct_toolace')} pct_xlam={conv.get('pct_xlam')} "
        f"pct_toolace_xlam={conv.get('pct_toolace_xlam')} "
        f"pct_apigen_mt={conv.get('pct_apigen_mt')}",
        f"  pct_irrelevance_conv={conv.get('pct_irrelevance_conv')}",
        f"  stop_label={conv.get('stop_label')}",
    ]
    unrolled = stats.get("unrolled")
    if unrolled:
        lines.append(
            f"  unrolled n={unrolled.get('n')} "
            f"pct_action={unrolled.get('pct_action_class')}"
        )
        lines.append(
            f"  empty_after_exec={unrolled.get('empty_after_exec')} "
            f"first_assistant_empty={unrolled.get('first_assistant_empty')} "
            f"observe_same/tool={unrolled.get('pct_observe_same_of_tool')}"
        )
    if stats.get("quota"):
        lines.append(format_quota_report(stats["quota"]))
    reasons = gate_reasons(stats, require_unrolled_actions=require_unrolled_actions)
    if reasons:
        lines.append("  gate_reasons:")
        lines.extend(f"    - {r}" for r in reasons)
    else:
        lines.append("  gates: PASS")
    return "\n".join(lines)


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
        description="Print v9 lever2 source / action-class / quota stats."
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=SPLITS_DIR_V9,
        help="Conversation JSONL dir (train/valid/corpus_test) or a JSONL file.",
    )
    parser.add_argument(
        "--mlx-dir",
        type=Path,
        default=None,
        help=(
            "Optional already-unrolled mlx JSONL dir (skips projected quota). "
            "With --gate, requires action-class hard bands."
        ),
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Exit 1 if hard mix constraints fail. Without --mlx-dir this is "
            "conversation-level only. Pass --mlx-dir to gate post-quota rows."
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
    print(format_report(stats, require_unrolled_actions=require_unrolled))
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.gate:
        reasons = gate_reasons(stats, require_unrolled_actions=require_unrolled)
        if reasons:
            print("mix_stats_v9: abort", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
