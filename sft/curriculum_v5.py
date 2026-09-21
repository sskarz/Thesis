"""SFT v5 turn-0 completeness + env-state consistency (extends v4).

Run 15 (v4) largely fixed empty-``[]`` self-conditioning (2 empty-turn
primaries / 170 failures) but left **state_mismatch ~64%** and
**under_calling ~35% (mostly turn 0)**. v5 does **not** add epochs on the
v4 mix, does **not** treat BoN as the win, and does **not** copy BFCL
``multi_turn_base`` prompts into train.

Verified data gaps (not assumed root causes):

* **xLAM is single-turn** (``format_xlam_row``: query → one assistant JSON
  list, ``is_multi_turn=False``). A query that names two catalog tools with
  only one gold call teaches turn-0 under-call. There are no exec
  observations, so xLAM cannot teach env-state sequential calls.
* **ToolACE is multi-turn with IFFC exec observations**, but a first-turn
  gold that omits later first-episode calls whose arguments were already in
  the opening user message is an under-specified parallel set. Continue-
  after-exec calls that use a user-named entity the observation never
  confirmed teach state mismatch.

v5 (on top of v4 empty-target drops + recovery-after-``[]``):

1. **Turn-0 call-set completeness**
   * xLAM / first-call underspec: drop when the opening user text mentions
     ≥2 catalog tools and gold has only one call.
   * ToolACE first-episode: independently-knowable later calls (every
     nontrivial string param appears in the opening user text **and** the
     original continue is not state-inconsistent) are merged into the
     first nonempty gold (history-only rewrite; not a new ``[]``
     row). Later continue-after-exec rows that only repeat those absorbed
     calls are dropped so we do not teach calling them twice.
2. **Env-state consistency:** drop continue-after-exec gold whose
   nontrivial string params include a value the user named that is **not**
   in the exec observation or the previous assistant call (env never
   confirmed the entity). State-dependent continues (args grounded in
   exec / previous call) are kept — that is the sequential teacher.
3. Recovery-after-``[]`` (v3/v4) is unchanged: nonempty gold after a prior
   stop then a **new** user question is not continue-after-exec.

Synthetic structure is derived from existing gold, never from BFCL test
prompts. mlx-lm has no sample-weight field, so rows are dropped or
rewritten rather than downweighted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from benchmark.iffc import parse_iffc_response
from sft.format_iffc import dumps_iffc_calls, extract_json_array_of_objects

EMPTY_IFFC_TARGET = "[]"
# Markers from ``benchmark.iffc.format_iffc_execution_user_message``.
_EXEC_HEADER = "Function execution results"
_EXEC_FOOTER = "or [] if no further function calling is needed"


def is_empty_iffc_target(content: Any) -> bool:
    return str(content or "").strip() == EMPTY_IFFC_TARGET


def is_iffc_execution_observation(content: Any) -> bool:
    text = str(content or "")
    return _EXEC_HEADER in text or _EXEC_FOOTER in text


def has_nonempty_iffc_tool_call(content: Any) -> bool:
    text = str(content or "")
    if is_empty_iffc_target(text):
        return False
    return bool(parse_iffc_response(text))

# Tokens too generic to count as a tool mention in the user query.
_MENTION_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "api",
        "get",
        "set",
        "data",
        "info",
        "tool",
        "func",
        "call",
        "name",
        "type",
        "json",
        "http",
        "this",
        "that",
        "list",
        "item",
        "value",
    }
)
_GENERIC_PARAM_VALUES = frozenset(
    {
        "true",
        "false",
        "none",
        "null",
        "yes",
        "no",
        "on",
        "off",
        "ok",
        "n/a",
    }
)
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_TOKEN_RE = re.compile(r"[.\s_/\-]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def assistant_calls(content: Any) -> list[dict]:
    """Parse IFFC gold into ``{function_name, parameters}`` dicts."""
    text = str(content or "")
    if is_empty_iffc_target(text):
        return []
    raw = parse_iffc_response(text)
    out: list[dict] = []
    for call in raw:
        if not isinstance(call, dict):
            continue
        name = call.get("function_name") or call.get("name")
        if not name:
            continue
        params = call.get("parameters") or call.get("arguments") or {}
        if not isinstance(params, dict):
            params = {}
        out.append({"function_name": str(name), "parameters": params})
    return out


def call_key(call: dict) -> str:
    """Stable identity for deduping / absorption checks."""
    return json.dumps(
        {
            "function_name": call.get("function_name") or "",
            "parameters": call.get("parameters") or {},
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def available_tool_names(messages: list[dict]) -> list[str]:
    """Catalog names from the IFFC system prompt (JSON function list)."""
    for msg in messages:
        if msg.get("role") != "system":
            continue
        docs = extract_json_array_of_objects(str(msg.get("content") or ""))
        names = [str(d["name"]) for d in docs if isinstance(d, dict) and d.get("name")]
        if names:
            return names
    return []


def _name_tokens(name: str) -> list[str]:
    spaced = _CAMEL_RE.sub(r"\1 \2", name)
    parts = [p.lower() for p in _TOKEN_RE.split(spaced) if p]
    return parts


def tools_mentioned_in_text(text: str, names: Iterable[str]) -> list[str]:
    """Catalog tools whose distinctive name tokens appear in ``text``.

    Full-name substring (len ≥ 3) always counts. Otherwise a token of
    length ≥ 4 that is not a stopword must appear as a whole word.
    Short names (``ls``, ``cd``, ``mv``) count on a word-boundary match
    of the whole name.
    """
    query = str(text or "")
    if not query or not names:
        return []
    q_lower = query.lower()
    words = {w.lower() for w in _WORD_RE.findall(query)}
    hit: list[str] = []
    for name in names:
        raw = str(name or "").strip()
        if not raw:
            continue
        n_lower = raw.lower()
        if len(n_lower) >= 3 and n_lower in q_lower:
            hit.append(raw)
            continue
        if len(n_lower) <= 3 and n_lower in words:
            hit.append(raw)
            continue
        tokens = [
            t
            for t in _name_tokens(raw)
            if len(t) >= 4 and t not in _MENTION_STOP
        ]
        if tokens and any(t in words for t in tokens):
            hit.append(raw)
    return hit


def nontrivial_string_params(params: Any) -> list[str]:
    """String argument values that can ground (or contradict) env state."""
    found: list[str] = []
    if isinstance(params, str):
        s = params.strip()
        if len(s) >= 2 and s.lower() not in _GENERIC_PARAM_VALUES:
            found.append(s)
        return found
    if isinstance(params, dict):
        for value in params.values():
            found.extend(nontrivial_string_params(value))
        return found
    if isinstance(params, list):
        for value in params:
            found.extend(nontrivial_string_params(value))
        return found
    return found


def _haystack_contains(haystack: str, value: str) -> bool:
    if not value:
        return False
    return value.lower() in haystack.lower()


def first_non_exec_user_index(messages: list[dict]) -> int | None:
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        if not is_iffc_execution_observation(msg.get("content")):
            return i
    return None


def next_new_user_index(messages: list[dict], after: int) -> int:
    """Index of the next non-exec user after ``after``, or ``len(messages)``."""
    for i in range(after + 1, len(messages)):
        if messages[i].get("role") != "user":
            continue
        if not is_iffc_execution_observation(messages[i].get("content")):
            return i
    return len(messages)


def first_nonempty_assistant_index(messages: list[dict]) -> int | None:
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if has_nonempty_iffc_tool_call(msg.get("content")):
            return i
    return None


def is_first_nonempty_assistant(messages: list[dict], index: int) -> bool:
    found = first_nonempty_assistant_index(messages)
    return found is not None and found == index


def is_continue_after_exec(messages: list[dict], index: int) -> bool:
    """True when gold is a nonempty call immediately after an IFFC observation."""
    if index <= 0 or index >= len(messages):
        return False
    if messages[index].get("role") != "assistant":
        return False
    if not has_nonempty_iffc_tool_call(messages[index].get("content")):
        return False
    prev = messages[index - 1]
    if prev.get("role") != "user":
        return False
    return is_iffc_execution_observation(prev.get("content"))


def _previous_assistant_text(messages: list[dict], index: int) -> str:
    for j in range(index - 1, -1, -1):
        if messages[j].get("role") == "assistant":
            return str(messages[j].get("content") or "")
    return ""


def _user_text_before(messages: list[dict], index: int) -> str:
    parts: list[str] = []
    for j in range(index):
        if messages[j].get("role") != "user":
            continue
        if is_iffc_execution_observation(messages[j].get("content")):
            continue
        parts.append(str(messages[j].get("content") or ""))
    return "\n".join(parts)


def is_independently_knowable_call(call: dict, first_user_text: str) -> bool:
    """True when every nontrivial string param is already in the opening user."""
    strings = nontrivial_string_params(call.get("parameters"))
    if not strings:
        return True
    hay = first_user_text or ""
    return all(_haystack_contains(hay, s) for s in strings)


def independently_knowable_later_calls(
    messages: list[dict],
    first_asst_idx: int,
) -> list[dict]:
    """First-episode continue-after-exec calls whose args were in the opening user.

    These could have been issued in parallel at turn 0. State-dependent
    continues (args only in the exec dump) are excluded so sequential
    env-using calls stay sequential.
    """
    first_user = first_non_exec_user_index(messages)
    if first_user is None:
        return []
    episode_end = next_new_user_index(messages, first_user)
    if first_asst_idx < first_user or first_asst_idx >= episode_end:
        return []
    first_text = str(messages[first_user].get("content") or "")
    already = {call_key(c) for c in assistant_calls(messages[first_asst_idx].get("content"))}
    extras: list[dict] = []
    seen = set(already)
    for i in range(first_asst_idx + 1, episode_end):
        if messages[i].get("role") != "assistant":
            continue
        if not has_nonempty_iffc_tool_call(messages[i].get("content")):
            continue
        if not is_continue_after_exec(messages, i):
            continue
        if is_state_inconsistent_continue(messages, i):
            # User named the entity but the exec dump never confirmed it —
            # merging into turn 0 would teach the same state mismatch.
            continue
        for call in assistant_calls(messages[i].get("content")):
            key = call_key(call)
            if key in seen:
                continue
            if not is_independently_knowable_call(call, first_text):
                continue
            extras.append(call)
            seen.add(key)
    return extras


def merge_call_lists(base: list[dict], extras: list[dict]) -> list[dict]:
    """Append extras that are not already in ``base`` (order preserving)."""
    seen = {call_key(c) for c in base}
    out = list(base)
    for call in extras:
        key = call_key(call)
        if key in seen:
            continue
        out.append(call)
        seen.add(key)
    return out


def is_underspecified_turn0(messages: list[dict], index: int) -> bool:
    """Turn-0 gold has one call but the opening user named ≥2 catalog tools.

    Targets under-specified xLAM (and similar single-call first turns).
    Does not fire when gold already has 2+ calls, when fewer than two
    catalog tools are mentioned, or when this is not the first nonempty
    assistant responding to the opening user.
    """
    if not is_first_nonempty_assistant(messages, index):
        return False
    first_user = first_non_exec_user_index(messages)
    if first_user is None:
        return False
    # Only the response to the opening request — recovery after a
    # clarifying [] is a later user question (v4 recovery, leave it).
    prev_user = None
    for j in range(index - 1, -1, -1):
        if messages[j].get("role") == "user":
            prev_user = j
            break
    if prev_user != first_user:
        return False
    gold = assistant_calls(messages[index].get("content"))
    if len(gold) != 1:
        return False
    names = available_tool_names(messages)
    mentioned = tools_mentioned_in_text(
        str(messages[first_user].get("content") or ""), names
    )
    return len(mentioned) >= 2


def is_incomplete_turn0(messages: list[dict], index: int) -> bool:
    """First nonempty gold omits independently-knowable later first-episode calls."""
    if not is_first_nonempty_assistant(messages, index):
        return False
    return bool(independently_knowable_later_calls(messages, index))


def complete_turn0_gold(messages: list[dict], index: int) -> tuple[str, list[dict]]:
    """Merged first-turn gold (existing calls + independently-knowable extras)."""
    base = assistant_calls(messages[index].get("content"))
    extras = independently_knowable_later_calls(messages, index)
    merged = merge_call_lists(base, extras)
    return dumps_iffc_calls(merged), extras


def is_state_inconsistent_continue(messages: list[dict], index: int) -> bool:
    """Continue-after-exec uses a user-named entity the env never confirmed.

    A nontrivial string param is inconsistent when it appears in a prior
    non-exec user message but **not** in the exec observation or the
    previous assistant call. Schema-like enums the user never named are
    left alone (not a BFCL env-state teacher). Missing this drop is the
    ToolACE path that can teach ``cd(docs)`` after a listing that has no
    ``docs``.
    """
    if not is_continue_after_exec(messages, index):
        return False
    exec_text = str(messages[index - 1].get("content") or "")
    prev_asst = _previous_assistant_text(messages, index)
    env_hay = f"{exec_text}\n{prev_asst}"
    user_hay = _user_text_before(messages, index)
    if not user_hay:
        return False
    for call in assistant_calls(messages[index].get("content")):
        for value in nontrivial_string_params(call.get("parameters")):
            if _haystack_contains(env_hay, value):
                continue
            if _haystack_contains(user_hay, value):
                return True
    return False


def is_state_consistent_continue(messages: list[dict], index: int) -> bool:
    """Continue-after-exec with at least one string param grounded in env."""
    if not is_continue_after_exec(messages, index):
        return False
    if is_state_inconsistent_continue(messages, index):
        return False
    exec_text = str(messages[index - 1].get("content") or "")
    prev_asst = _previous_assistant_text(messages, index)
    env_hay = f"{exec_text}\n{prev_asst}"
    for call in assistant_calls(messages[index].get("content")):
        strings = nontrivial_string_params(call.get("parameters"))
        if any(_haystack_contains(env_hay, s) for s in strings):
            return True
        if not strings:
            # No entity args (ls / pwd style) after exec: sequential, not mismatch.
            return True
    return False


def is_absorbed_independent_followup(
    messages: list[dict],
    index: int,
    absorbed_keys: set[str],
) -> bool:
    """Later first-episode continue whose whole gold was merged into turn 0."""
    if not absorbed_keys:
        return False
    if not is_continue_after_exec(messages, index):
        return False
    first = first_nonempty_assistant_index(messages)
    if first is None or index <= first:
        return False
    first_user = first_non_exec_user_index(messages)
    if first_user is None:
        return False
    episode_end = next_new_user_index(messages, first_user)
    if index >= episode_end:
        return False
    gold = assistant_calls(messages[index].get("content"))
    if not gold:
        return False
    return all(call_key(c) in absorbed_keys for c in gold)


@dataclass
class Turn0Plan:
    """Per-conversation turn-0 completeness plan (parent messages)."""

    first_nonempty_idx: int | None = None
    extras: list[dict] = field(default_factory=list)
    merged_gold: str | None = None
    absorbed_keys: set[str] = field(default_factory=set)
    underspecified: bool = False
    incomplete: bool = False


def plan_turn0(messages: list[dict]) -> Turn0Plan:
    """Compute completeness extras once per parent conversation."""
    plan = Turn0Plan()
    idx = first_nonempty_assistant_index(messages)
    plan.first_nonempty_idx = idx
    if idx is None:
        return plan
    plan.underspecified = is_underspecified_turn0(messages, idx)
    extras = independently_knowable_later_calls(messages, idx)
    plan.extras = extras
    plan.incomplete = bool(extras)
    if extras:
        merged, extras = complete_turn0_gold(messages, idx)
        plan.merged_gold = merged
        plan.absorbed_keys = {call_key(c) for c in extras}
    return plan
