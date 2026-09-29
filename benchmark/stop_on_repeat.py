"""Harness stop-on-repeat for ``--fc-mode iffc-stop-repeat``.

Also reused by ``--fc-mode iffc-ooda-card`` (same MIN=6 gate; the OODA card
is a prompt overlay, not a change to this heuristic).

After a tool-exec observation, force a turn stop (do not re-execute) when:

* ``consecutive_identical`` — proposed call set equals the previous assistant
  set. Fires immediately; this is the real decode-loop signal.
* ``observe_same_call`` — every proposed call was already executed **and**
  at least :data:`OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS` nonempty IFFC
  assistant tool-turns have already happened in the **current BFCL user-ask**
  (not the whole entry). Short VehicleControl paths (3–5 steps) re-call
  ``startEngine`` after ``lockDoors`` / ``pressBrakePedal``; that is state,
  not a loop, so the subset-of-executed rule is gated. Long A-B-A /
  hit-step loops still get cut once the turn exceeds the MIN.

The window resets on a non-exec user message (a fresh BFCL question), so a
later short VC turn is not poisoned by earlier asks. Exec observations do
not reset it. Baseline ``iffc`` is unchanged — this logic is gated on
``iffc-stop-repeat`` and ``iffc-ooda-card`` only. Shared canonicalization
mirrors ``sft/drop_identical_call.py`` but lives under ``benchmark/`` so
the harness does not import ``sft``.
"""

from __future__ import annotations

import json
from typing import Any

from .iffc import parse_iffc_response

_EXEC_HEADER = "Function execution results"
_EXEC_FOOTER = "or [] if no further function calling is needed"

REASON_CONSECUTIVE = "consecutive_identical"
REASON_OBSERVE_SAME = "observe_same_call"

# observe_same_call MIN: first eligible at the (MIN+1)-th assistant tool
# proposal of the current ask (so_far == MIN). MIN=6 keeps 3–5 step VC
# re-calls (even a 6-step path: so_far=5 < 6) while still cutting smoke
# `_82`-style A-B-A loops (~43 api) well before MULTI_TURN_MAX_STEPS=20.
# MIN=8 would also preserve VC; 6 is the tighter budget cut. Count is
# per current BFCL ask, not entry-wide (later short VC turns stay live).
OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS = 6


def is_exec_observation(content: Any) -> bool:
    """True when a user message looks like an IFFC tool-result observation."""
    text = str(content or "")
    return _EXEC_HEADER in text or _EXEC_FOOTER in text


def canonical_tool_calls(content: Any) -> tuple[tuple[str, str], ...] | None:
    """Comparable ``(name, canonical_params_json)`` tuples, or ``None``.

    ``None`` means not a nonempty IFFC tool-call list (``[]``, NL, or junk).
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
    """Content of the nearest earlier assistant turn before *index*, or ``None``."""
    if index < 0:
        return None
    for j in range(index - 1, -1, -1):
        if messages[j].get("role") == "assistant":
            return str(messages[j].get("content") or "")
    return None


def executed_tool_calls_before(
    messages: list[dict], index: int
) -> set[tuple[str, str]]:
    """Individual ``(name, params_json)`` pairs already followed by an exec dump."""
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


def assistant_tool_turns_so_far(messages: list[dict], index: int) -> int:
    """Nonempty IFFC assistant tool-turns in the current BFCL user-ask.

    The count is the number of already-emitted tool-call assistant messages
    after the last non-exec user message and before *index*. Exec
    observations do not start a new ask; a fresh BFCL question does.
    """
    n = min(max(index, 0), len(messages))
    start = 0
    for i in range(n):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        if is_exec_observation(msg.get("content")):
            continue
        start = i + 1
    count = 0
    for i in range(start, n):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        if canonical_tool_calls(msg.get("content")) is not None:
            count += 1
    return count


def force_stop_reason(
    chat_history: list[dict], proposed_content: str | None
) -> str | None:
    """Return a stop reason when *proposed_content* must not be re-executed.

    Only triggers when the last history message is an IFFC exec observation
    (post-tool). A fresh BFCL user question does not match, so baseline
    multi-turn progression after a new ask is preserved.

    Reasons:

    * ``consecutive_identical`` — proposed call set equals previous assistant
      (immediate; no MIN gate)
    * ``observe_same_call`` — every proposed call was already executed **and**
      ``assistant_tool_turns_so_far >= OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS``
    """
    current = canonical_tool_calls(proposed_content)
    if current is None or not chat_history:
        return None
    prev = chat_history[-1]
    if prev.get("role") != "user" or not is_exec_observation(prev.get("content")):
        return None

    prev_asst = previous_assistant_content(chat_history, len(chat_history))
    if prev_asst is not None:
        previous = canonical_tool_calls(prev_asst)
        if previous is not None and current == previous:
            return REASON_CONSECUTIVE

    executed = executed_tool_calls_before(chat_history, len(chat_history))
    if executed and set(current) <= executed:
        turns = assistant_tool_turns_so_far(chat_history, len(chat_history))
        if turns >= OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS:
            return REASON_OBSERVE_SAME
    return None
