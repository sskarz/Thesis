"""Stop-taxonomy / source stats for the SFT v7 mix.

Deterministic labels (no LLM):

* ``tool_call``: assistant content parses to ≥1 IFFC call
* ``irrelevance_[]``: content.strip() == ``[]``
* ``clarify``: nonempty non-JSON text with ``?`` or ask keywords, no tool call
* ``final_answer``: other nonempty non-JSON assistant text

CLI::

    uv run python -m sft.mix_stats_v7 --splits sft/data/splits-v7 --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from benchmark.iffc import parse_iffc_response

from sft.config import (
    SOURCE_APIGEN_MT,
    SPLITS_DIR_V7,
    V7_ABORT_IRRELEVANCE_FRAC,
    V7_MIN_APIGEN_FRAC,
    V7_NONTOOL_MAX,
    V7_NONTOOL_MIN,
)
from sft.filter_empty_after_exec import (
    RECIPE_V7,
    classify_assistant_turn,
    flags_for_recipe,
    is_first_assistant_empty_target,
    unroll_and_filter,
)
from sft.format_apigen_mt import (
    NONTOOL_LABELS,
    STOP_IRRELEVANCE,
    STOP_LABELS,
    label_assistant_content,
)
from sft.io import read_jsonl


def last_assistant_content(record: dict) -> str:
    messages = record.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def conversation_stop_label(record: dict) -> str:
    existing = record.get("stop_label")
    if existing in STOP_LABELS:
        return str(existing)
    return label_assistant_content(last_assistant_content(record))


def _pct(n: int, total: int) -> float:
    return (n / total) if total else 0.0


def _plain(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def scan_conversations(records: Iterable[dict]) -> dict[str, Any]:
    recs = list(records)
    stop = Counter()
    by_source = Counter()
    n_apigen = 0
    assistant_turns = 0
    tool_calls = 0
    for rec in recs:
        source = str(rec.get("source") or "unknown")
        by_source[source] += 1
        if source == SOURCE_APIGEN_MT:
            n_apigen += 1
        stop[conversation_stop_label(rec)] += 1
        for msg in rec.get("messages") or []:
            if msg.get("role") != "assistant":
                continue
            assistant_turns += 1
            tool_calls += len(parse_iffc_response(str(msg.get("content") or "")))
    n = len(recs)
    return {
        "n": n,
        "by_source": _plain(by_source),
        "stop_label": _plain(stop),
        "stop_label_pct": {k: round(_pct(stop[k], n), 4) for k in STOP_LABELS},
        "pct_apigen_mt": round(_pct(n_apigen, n), 4),
        "mean_assistant_turns": round(assistant_turns / n, 4) if n else 0.0,
        "mean_tool_calls": round(tool_calls / n, 4) if n else 0.0,
    }


def scan_unrolled(rows: Iterable[dict]) -> dict[str, Any]:
    items = list(rows)
    stop = Counter()
    by_source = Counter()
    empty_after_exec = 0
    first_assistant_empty = 0
    n_apigen = 0
    n_nontool = 0
    for row in items:
        source = str(row.get("source") or "unknown")
        by_source[source] += 1
        if source == SOURCE_APIGEN_MT:
            n_apigen += 1
        messages = list(row.get("messages") or [])
        gold = messages[-1].get("content") if messages else ""
        label = label_assistant_content(gold)
        stop[label] += 1
        if label in NONTOOL_LABELS:
            n_nontool += 1
        idx = len(messages) - 1
        kind = classify_assistant_turn(messages, idx)
        if kind in {
            "empty_after_exec_terminal",
            "empty_after_exec_nonterminal",
        }:
            empty_after_exec += 1
        if is_first_assistant_empty_target(messages, idx):
            first_assistant_empty += 1
    n = len(items)
    return {
        "n": n,
        "by_source": _plain(by_source),
        "stop_label": _plain(stop),
        "stop_label_pct": {k: round(_pct(stop[k], n), 4) for k in STOP_LABELS},
        "empty_after_exec": empty_after_exec,
        "first_assistant_empty": first_assistant_empty,
        "pct_nontool": round(_pct(n_nontool, n), 4),
        "pct_apigen_mt": round(_pct(n_apigen, n), 4),
    }


def collect_stats(
    conversations: list[dict],
    *,
    unrolled: list[dict] | None = None,
    unroll_recipe: str = RECIPE_V7,
) -> dict[str, Any]:
    conv = scan_conversations(conversations)
    rows = unrolled
    if rows is None:
        rows, _stats = unroll_and_filter(
            conversations, check_bfcl_leak=False, **flags_for_recipe(unroll_recipe)
        )
    unr = scan_unrolled(rows)
    return {"conversations": conv, "unrolled": unr, "recipe": unroll_recipe}


def gate_reasons(stats: dict[str, Any]) -> list[str]:
    """Hard mix constraints from the v7 plan. Empty list = pass."""
    reasons: list[str] = []
    conv = stats.get("conversations") or {}
    unr = stats.get("unrolled") or {}
    pct_apigen = float(conv.get("pct_apigen_mt") or 0.0)
    if pct_apigen < V7_MIN_APIGEN_FRAC:
        reasons.append(
            f"pct_apigen_mt={pct_apigen:.3f} < {V7_MIN_APIGEN_FRAC} (conversation)"
        )
    pct_nontool = float(unr.get("pct_nontool") or 0.0)
    if not (V7_NONTOOL_MIN <= pct_nontool <= V7_NONTOOL_MAX):
        reasons.append(
            f"pct_nontool={pct_nontool:.3f} outside [{V7_NONTOOL_MIN}, {V7_NONTOOL_MAX}] "
            "(unrolled assistant targets)"
        )
    n_unr = int(unr.get("n") or 0)
    n_irr = int((unr.get("stop_label") or {}).get(STOP_IRRELEVANCE, 0))
    irr_share = _pct(n_irr, n_unr)
    if irr_share > V7_ABORT_IRRELEVANCE_FRAC:
        reasons.append(
            f"irrelevance_[] row share={irr_share:.3f} > {V7_ABORT_IRRELEVANCE_FRAC}"
        )
    empty_after = int(unr.get("empty_after_exec") or 0)
    if empty_after > 0:
        reasons.append(
            f"empty-after-exec targets after recipe={empty_after} (must be 0)"
        )
    return reasons


def format_report(stats: dict[str, Any]) -> str:
    conv = stats["conversations"]
    unr = stats["unrolled"]
    lines = [
        f"sft-v7 mix stats (recipe={stats.get('recipe')})",
        "conversations:",
        f"  n={conv['n']}  pct_apigen_mt={conv['pct_apigen_mt']:.4f}  "
        f"mean_assistant_turns={conv['mean_assistant_turns']:.3f}  "
        f"mean_tool_calls={conv['mean_tool_calls']:.3f}",
        f"  by_source={conv['by_source']}",
        f"  stop_label={conv['stop_label']}  pct={conv['stop_label_pct']}",
        "unrolled assistant targets:",
        f"  n={unr['n']}  pct_apigen_mt={unr['pct_apigen_mt']:.4f}  "
        f"pct_nontool={unr['pct_nontool']:.4f}",
        f"  empty_after_exec={unr['empty_after_exec']}  "
        f"first_assistant_empty={unr['first_assistant_empty']}",
        f"  by_source={unr['by_source']}",
        f"  stop_label={unr['stop_label']}  pct={unr['stop_label_pct']}",
    ]
    reasons = gate_reasons(stats)
    if reasons:
        lines.append("gates: FAIL")
        lines.extend(f"  - {r}" for r in reasons)
    else:
        lines.append("gates: PASS")
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
        description="Print v7 stop-taxonomy / source / empty-after-exec stats."
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=SPLITS_DIR_V7,
        help="Conversation JSONL dir (train/valid/corpus_test) or a JSONL file.",
    )
    parser.add_argument(
        "--mlx-dir",
        type=Path,
        default=None,
        help="Optional already-unrolled mlx JSONL dir (skips in-memory unroll).",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit 1 if hard mix constraints fail.",
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
    stats = collect_stats(conversations, unrolled=unrolled)
    print(format_report(stats))
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.gate:
        reasons = gate_reasons(stats)
        if reasons:
            print("mix_stats_v7: abort", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
