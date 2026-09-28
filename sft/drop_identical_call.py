"""Drop gold assistant turns that repeat the previous assistant tool call.

SFT v8 lever 1 (Run 21 diagnosis): v7's tool-call-heavy mix taught
observe→re-emit the same call until the step limit (``ls`` / VehicleControl
20× loops). At unroll time, skip assistant targets whose parsed IFFC call
set is identical to the previous assistant turn's call set.

This is **not** an empty-``[]`` filter and does **not** rewrite NL finals.
Non-tool / unparseable assistant text is left alone. Conversation-level
JSONL is unchanged; only unrolled gold rows are dropped.

Call **after** the v4 empty-after-exec filter and **before** the non-tool cap
so duplicate tool_call rows do not inflate ``pct_nontool`` after the cap.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from benchmark.iffc import parse_iffc_response


def canonical_tool_calls(content: Any) -> tuple[tuple[str, str], ...] | None:
    """Return a comparable tuple of ``(name, canonical_params_json)`` or ``None``.

    ``None`` means the text is not a nonempty IFFC tool-call list (``[]``,
    NL clarify/final, or unparseable). Those are not identical-*call* loops.
    """
    calls = parse_iffc_response(str(content or ""))
    if not calls:
        return None
    out: list[tuple[str, str]] = []
    for call in calls:
        name = str(call.get("function_name") or call.get("name") or "")
        params = call.get("parameters", call.get("arguments", {}))
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                params = {"_raw": params}
        if not isinstance(params, dict):
            params = {"_raw": params}
        out.append(
            (name, json.dumps(params, sort_keys=True, ensure_ascii=False, default=str))
        )
    return tuple(out)


def previous_assistant_content(messages: list[dict], index: int) -> str | None:
    """Content of the nearest earlier assistant turn, or ``None``."""
    if index < 0:
        return None
    for j in range(index - 1, -1, -1):
        if messages[j].get("role") == "assistant":
            return str(messages[j].get("content") or "")
    return None


def is_identical_to_previous_assistant_call(
    messages: list[dict], index: int | None = None
) -> bool:
    """True when ``messages[index]`` repeats the previous assistant tool call.

    ``index`` defaults to the last message (unrolled gold). Both turns must
    parse to a nonempty IFFC call list and compare equal after canonicalization.
    """
    if not messages:
        return False
    if index is None:
        index = len(messages) - 1
    if index < 0 or index >= len(messages):
        return False
    msg = messages[index]
    if msg.get("role") != "assistant":
        return False
    current = canonical_tool_calls(msg.get("content"))
    if current is None:
        return False
    prev = previous_assistant_content(messages, index)
    if prev is None:
        return False
    previous = canonical_tool_calls(prev)
    if previous is None:
        return False
    return current == previous


def max_identical_call_run(messages: list[dict]) -> int:
    """Longest run of consecutive assistant turns with the same tool-call set.

    User / system / observation turns in between do not break the run.
    Assistant turns that are not nonempty tool calls reset the run.
    A single distinct call has run length 1.
    """
    longest = 0
    current_run = 0
    prev_calls: tuple[tuple[str, str], ...] | None = None
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        calls = canonical_tool_calls(msg.get("content"))
        if calls is None:
            current_run = 0
            prev_calls = None
            continue
        if prev_calls is not None and calls == prev_calls:
            current_run += 1
        else:
            current_run = 1
            prev_calls = calls
        if current_run > longest:
            longest = current_run
    return longest


def drop_identical_call_unrolls(
    rows: Iterable[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """Drop unrolled rows whose gold repeats the previous assistant tool call.

    Original relative order of kept rows is preserved. Stats are for the
    prepare / mix_stats log.
    """
    items = list(rows)
    kept: list[dict] = []
    dropped_ids: list[str] = []
    dropped_by_source: dict[str, int] = {}
    for row in items:
        messages = list(row.get("messages") or [])
        if is_identical_to_previous_assistant_call(messages):
            dropped_ids.append(str(row.get("id", "")))
            source = str(row.get("source") or "unknown")
            dropped_by_source[source] = dropped_by_source.get(source, 0) + 1
            continue
        kept.append(row)
    stats: dict[str, Any] = {
        "n_before": len(items),
        "n_after": len(kept),
        "dropped": len(items) - len(kept),
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
        "dropped_ids_head": dropped_ids[:20],
        "note": (
            "Drop gold assistant turns whose parsed IFFC calls equal the "
            "previous assistant call (anti observe→re-emit loop). "
            "Does not rewrite NL finals or empty []."
        ),
    }
    return kept, stats


def format_identical_call_report(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "v8 identical-call drop: (not applied)"
    return "\n".join(
        [
            "v8 identical-call drop "
            f"(dropped {stats.get('dropped')} of {stats.get('n_before')} "
            f"unrolled rows → {stats.get('n_after')}):",
            f"  dropped_by_source: {stats.get('dropped_by_source')}",
            f"  note: {stats.get('note')}",
        ]
    )
