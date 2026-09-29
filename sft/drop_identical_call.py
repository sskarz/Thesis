"""Drop gold assistant turns that repeat a tool call (SFT v8.1).

SFT v8 lever 1 (Run 21 diagnosis): v7's tool-call-heavy mix taught
observe→re-emit until the step limit (``ls`` / VehicleControl 20× loops).
The first v8 pass only dropped **consecutive** identical assistant calls
(~32 unrolled rows) — too light for BFCL-style ``ls()`` /
``displayCarStatus`` A-B-A-B spam, where each adjacent pair differs.

v8.1 also drops gold assistant turns that re-emit a call **already
executed** (an earlier assistant call followed by a tool-result
observation) when the immediately previous user message is that
observation. Consecutive identical remains as a backstop.

This is **not** an empty-``[]`` filter and does **not** rewrite NL finals.
Non-tool / unparseable assistant text is left alone. Conversation-level
JSONL is unchanged; only unrolled gold rows are dropped.

Recipe flag stays ``v8`` (``splits-v8`` / ``mlx-v8``). This is a mix-recipe
bump, not a new experiment id.

Call **after** the v4 empty-after-exec filter and **before** the non-tool cap
so duplicate tool_call rows do not inflate ``pct_nontool`` after the cap.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from benchmark.iffc import parse_iffc_response

# Keep in sync with ``sft.filter_empty_after_exec`` (avoid importing that
# module here — it lazily imports this file from its CLI).
_EXEC_HEADER = "Function execution results"
_EXEC_FOOTER = "or [] if no further function calling is needed"

REASON_CONSECUTIVE = "consecutive_identical"
REASON_OBSERVE_SAME = "observe_same_call"
REASON_BOTH = "consecutive_and_observe_same"


def is_exec_observation(content: Any) -> bool:
    """True when a user message looks like an IFFC tool-result observation."""
    text = str(content or "")
    return _EXEC_HEADER in text or _EXEC_FOOTER in text


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


def executed_tool_calls_before(
    messages: list[dict], index: int
) -> set[tuple[str, str]]:
    """Individual ``(name, params_json)`` pairs already followed by an exec dump.

    Walks ``messages[:index]``. An assistant call set is marked executed only
    when a later user message (before the next assistant, or before ``index``)
    is an IFFC tool-result observation. New non-exec user turns clear a
    pending call without recording it.
    """
    seen: set[tuple[str, str]] = set()
    n = min(max(index, 0), len(messages))
    pending: tuple[tuple[str, str], ...] | None = None
    for i in range(n):
        msg = messages[i]
        role = msg.get("role")
        if role == "assistant":
            pending = canonical_tool_calls(msg.get("content"))
            continue
        if role == "user":
            if pending and is_exec_observation(msg.get("content")):
                seen.update(pending)
            pending = None
    return seen


def is_observe_same_call(
    messages: list[dict], index: int | None = None
) -> bool:
    """True when gold re-emits already-executed calls after a tool result.

    Requires:

    * ``messages[index]`` is an assistant nonempty IFFC call list
    * the immediately previous message is an IFFC exec observation
    * every call in the gold already appeared in an earlier executed call set

    This catches A-B-A-B ``ls()`` / ``displayCarStatus`` loops that consecutive
    identical comparison misses. A new user question (non-exec) does **not**
    match. A first-time different call after observe does **not** match.
    """
    if not messages:
        return False
    if index is None:
        index = len(messages) - 1
    if index <= 0 or index >= len(messages):
        return False
    msg = messages[index]
    if msg.get("role") != "assistant":
        return False
    current = canonical_tool_calls(msg.get("content"))
    if current is None:
        return False
    prev = messages[index - 1]
    if prev.get("role") != "user":
        return False
    if not is_exec_observation(prev.get("content")):
        return False
    executed = executed_tool_calls_before(messages, index)
    if not executed:
        return False
    return set(current) <= executed


def drop_reason_for_gold(
    messages: list[dict], index: int | None = None
) -> str | None:
    """Return a drop reason or ``None`` if the gold should be kept."""
    consecutive = is_identical_to_previous_assistant_call(messages, index)
    observe = is_observe_same_call(messages, index)
    if consecutive and observe:
        return REASON_BOTH
    if observe:
        return REASON_OBSERVE_SAME
    if consecutive:
        return REASON_CONSECUTIVE
    return None


def should_drop_identical_call_gold(
    messages: list[dict], index: int | None = None
) -> bool:
    """True when v8.1 should drop this assistant gold (consecutive or observe)."""
    return drop_reason_for_gold(messages, index) is not None


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


def count_observe_same_call_turns(messages: list[dict]) -> int:
    """How many assistant turns re-emit already-executed calls after observe."""
    n = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if is_observe_same_call(messages, i):
            n += 1
    return n


def max_observe_same_call_run(messages: list[dict]) -> int:
    """Longest run of consecutive observe→same-call assistant turns.

    Consecutive here means successive assistant turns (observations in
    between still count). A different first-time call after observe resets.
    """
    longest = 0
    current_run = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if is_observe_same_call(messages, i):
            current_run += 1
            if current_run > longest:
                longest = current_run
        else:
            current_run = 0
    return longest


def drop_identical_call_unrolls(
    rows: Iterable[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """Drop unrolled rows whose gold repeats a previous / already-executed call.

    Original relative order of kept rows is preserved. Stats are for the
    prepare / mix_stats log.
    """
    items = list(rows)
    kept: list[dict] = []
    dropped_ids: list[str] = []
    dropped_by_source: dict[str, int] = {}
    dropped_by_reason: dict[str, int] = {}
    for row in items:
        messages = list(row.get("messages") or [])
        reason = drop_reason_for_gold(messages)
        if reason is not None:
            dropped_ids.append(str(row.get("id", "")))
            source = str(row.get("source") or "unknown")
            dropped_by_source[source] = dropped_by_source.get(source, 0) + 1
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
            continue
        kept.append(row)
    n_observe = dropped_by_reason.get(REASON_OBSERVE_SAME, 0) + dropped_by_reason.get(
        REASON_BOTH, 0
    )
    n_consecutive = dropped_by_reason.get(
        REASON_CONSECUTIVE, 0
    ) + dropped_by_reason.get(REASON_BOTH, 0)
    stats: dict[str, Any] = {
        "n_before": len(items),
        "n_after": len(kept),
        "dropped": len(items) - len(kept),
        "dropped_by_source": dict(sorted(dropped_by_source.items())),
        "dropped_by_reason": dict(sorted(dropped_by_reason.items())),
        "dropped_consecutive": n_consecutive,
        "dropped_observe_same_call": n_observe,
        "dropped_ids_head": dropped_ids[:20],
        "note": (
            "v8.1: drop gold assistant turns that (a) equal the previous "
            "assistant call (consecutive identical) or (b) re-emit already-"
            "executed calls after a tool-result observation (observe→same-call, "
            "catches ls()/displayCarStatus A-B-A-B). Does not rewrite NL "
            "finals or empty []. Recipe flag remains v8."
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
            f"  dropped_by_reason: {stats.get('dropped_by_reason')}",
            f"  dropped_consecutive: {stats.get('dropped_consecutive')}  "
            f"dropped_observe_same_call: {stats.get('dropped_observe_same_call')}",
            f"  note: {stats.get('note')}",
        ]
    )
