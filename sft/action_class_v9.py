"""Action-class labels for SFT v9 (lever2) unrolled assistant gold.

Classify the **last** message of each unrolled row (assistant gold) into the
ToolACE-style coverage classes used by ``quota_sample_v9``:

* ``mt_tool_chain_advance`` — nonempty IFFC call(s) that advance vs prior /
  already-executed calls
* ``tool_call_start`` — first assistant tool call in the prefix
* ``final_after_obs`` — NL final (not ``[]``) immediately after an IFFC exec
  dump (**loose**; stricter advancing-prior subtype is logged only)
* ``ask`` — clarify / miss-param NL
* ``refuse`` — refuse / cannot / out-of-scope NL
* ``irrelevance_[]`` — gold is ``[]``
* ``observe_same_call`` / ``consecutive_identical`` — soft-exclude from keep
  (logged; not primary teacher filters)
* ``other_tool`` / ``other_nl`` — residual rows (downsampled, not upsampled)

Never map NL finals → ``[]``. Reuses ``drop_identical_call`` predicates for
canonical calls / exec observations / observe→same detection.
"""

from __future__ import annotations

from typing import Any

from sft.drop_identical_call import (
    canonical_tool_calls,
    executed_tool_calls_before,
    is_exec_observation,
    is_identical_to_previous_assistant_call,
    is_observe_same_call,
    previous_assistant_content,
)
from sft.format_apigen_mt import (
    CLARIFY_KEYWORDS,
    label_assistant_content,
)

ACTION_MT_TOOL_CHAIN_ADVANCE = "mt_tool_chain_advance"
ACTION_TOOL_CALL_START = "tool_call_start"
ACTION_FINAL_AFTER_OBS = "final_after_obs"
ACTION_ASK = "ask"
ACTION_REFUSE = "refuse"
ACTION_IRRELEVANCE = "irrelevance_[]"
ACTION_OBSERVE_SAME = "observe_same_call"
ACTION_CONSECUTIVE = "consecutive_identical"
ACTION_OTHER_TOOL = "other_tool"
ACTION_OTHER_NL = "other_nl"

# Soft-exclude from quota keep set (sampler exclusion, not traj deletion).
EXCLUDE_FROM_KEEP = frozenset({ACTION_OBSERVE_SAME, ACTION_CONSECUTIVE})

# Primary quota classes (target bands enforced by quota_sample_v9).
QUOTA_CLASSES = (
    ACTION_MT_TOOL_CHAIN_ADVANCE,
    ACTION_TOOL_CALL_START,
    ACTION_FINAL_AFTER_OBS,
    ACTION_ASK,
    ACTION_REFUSE,
    ACTION_IRRELEVANCE,
)

REFUSE_CUES = (
    "cannot",
    "can't",
    "unable to",
    "out of scope",
    "not able to",
    "i refuse",
    "refuse to",
    "not allowed",
    "don't have access",
    "do not have access",
    "no access",
    "not supported",
    "unsupported",
    "beyond my",
    "outside my",
    "i'm not able",
    "i am not able",
    "sorry, i can't",
    "sorry, i cannot",
)


def _assistant_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def is_first_assistant_tool_call(messages: list[dict], index: int) -> bool:
    """True when ``messages[index]`` is the first nonempty IFFC tool call."""
    if index < 0 or index >= len(messages):
        return False
    if messages[index].get("role") != "assistant":
        return False
    if canonical_tool_calls(messages[index].get("content")) is None:
        return False
    for i in range(index):
        if messages[i].get("role") != "assistant":
            continue
        if canonical_tool_calls(messages[i].get("content")) is not None:
            return False
    return True


def prior_tools_look_advancing(messages: list[dict], index: int) -> bool:
    """Stricter subtype: earlier executed calls look non-repeating / advancing.

    Used only for logging / duplication preference — never hard-filters
    ``final_after_obs`` (CONFIRM: classify LOOSE).
    """
    if index <= 0:
        return False
    # Walk prior assistant tool turns; require at least one executed advance
    # (call set differs from previous assistant call) and no observe→same.
    seen_advance = False
    for i in range(index):
        if messages[i].get("role") != "assistant":
            continue
        if canonical_tool_calls(messages[i].get("content")) is None:
            continue
        if is_observe_same_call(messages, i):
            return False
        if is_identical_to_previous_assistant_call(messages, i):
            return False
        prev = previous_assistant_content(messages, i)
        if prev is not None and canonical_tool_calls(prev) is not None:
            cur = canonical_tool_calls(messages[i].get("content"))
            if cur is not None and cur != canonical_tool_calls(prev):
                seen_advance = True
        else:
            # first tool call counts as a start, not yet an advance chain
            pass
    # Also require that at least one prior call was executed (obs present).
    executed = executed_tool_calls_before(messages, index)
    return bool(executed) and (seen_advance or len(executed) >= 1)


def is_refuse_nl(content: Any) -> bool:
    text = str(content or "").strip()
    if not text or text == "[]":
        return False
    if canonical_tool_calls(text) is not None:
        return False
    lower = text.lower()
    return any(cue in lower for cue in REFUSE_CUES)


def is_ask_nl(content: Any) -> bool:
    text = str(content or "").strip()
    if not text or text == "[]":
        return False
    if canonical_tool_calls(text) is not None:
        return False
    if is_refuse_nl(text):
        return False
    lower = text.lower()
    return "?" in text or any(k in lower for k in CLARIFY_KEYWORDS)


def is_mt_tool_chain_advance(messages: list[dict], index: int) -> bool:
    """Nonempty IFFC call that advances vs previous / already-executed calls."""
    if index < 0 or index >= len(messages):
        return False
    if messages[index].get("role") != "assistant":
        return False
    current = canonical_tool_calls(messages[index].get("content"))
    if current is None:
        return False
    if is_first_assistant_tool_call(messages, index):
        return False
    if is_identical_to_previous_assistant_call(messages, index):
        return False
    if is_observe_same_call(messages, index):
        return False
    # Prefer: previous assistant call was executed (obs between) OR call set
    # differs from previous assistant call.
    prev = previous_assistant_content(messages, index)
    if prev is None:
        return False
    previous = canonical_tool_calls(prev)
    if previous is None:
        # previous assistant was NL; treating a later tool call as start-like
        return False
    if current == previous:
        return False
    # If immediately after exec, require not a subset of already-executed.
    if index > 0 and messages[index - 1].get("role") == "user":
        if is_exec_observation(messages[index - 1].get("content")):
            executed = executed_tool_calls_before(messages, index)
            if executed and set(current) <= executed:
                return False
            return True
    # Mid-chain without immediate obs still counts if call set differs.
    return current != previous


def classify_assistant_action(
    messages: list[dict], index: int | None = None
) -> str:
    """Return the primary action class for ``messages[index]`` (default: last)."""
    if not messages:
        return ACTION_OTHER_NL
    if index is None:
        index = len(messages) - 1
    if index < 0 or index >= len(messages):
        return ACTION_OTHER_NL
    msg = messages[index]
    if msg.get("role") != "assistant":
        return ACTION_OTHER_NL
    content = msg.get("content")
    text = str(content or "").strip()

    if text == "[]" or (not text and canonical_tool_calls(content) is None):
        return ACTION_IRRELEVANCE

    calls = canonical_tool_calls(content)
    if calls is not None:
        # Soft-exclude loop shapes first (sampler exclusion).
        consecutive = is_identical_to_previous_assistant_call(messages, index)
        observe = is_observe_same_call(messages, index)
        if observe:
            return ACTION_OBSERVE_SAME
        if consecutive:
            return ACTION_CONSECUTIVE
        if is_first_assistant_tool_call(messages, index):
            return ACTION_TOOL_CALL_START
        if is_mt_tool_chain_advance(messages, index):
            return ACTION_MT_TOOL_CHAIN_ADVANCE
        return ACTION_OTHER_TOOL

    # Natural language — never coerce to [].
    prev_is_exec = False
    if index > 0 and messages[index - 1].get("role") == "user":
        prev_is_exec = is_exec_observation(messages[index - 1].get("content"))

    if is_refuse_nl(content):
        return ACTION_REFUSE
    if is_ask_nl(content):
        return ACTION_ASK
    if prev_is_exec:
        # LOOSE final_after_obs (CONFIRM): any NL final after exec dump.
        return ACTION_FINAL_AFTER_OBS
    # Non-exec NL that isn't ask/refuse — leftover final-like text.
    stop = label_assistant_content(content)
    if stop == "clarify":
        return ACTION_ASK
    return ACTION_OTHER_NL


def classify_unrolled_row(row: dict) -> str:
    messages = list(row.get("messages") or [])
    return classify_assistant_action(messages)


def action_label_detail(messages: list[dict], index: int | None = None) -> dict:
    """Primary class plus logging subtypes (strict final_after_obs, etc.)."""
    if index is None:
        index = len(messages) - 1 if messages else -1
    primary = classify_assistant_action(messages, index)
    detail = {
        "action_class": primary,
        "final_after_obs_strict_advancing": False,
        "is_observe_same_call": False,
        "is_consecutive_identical": False,
    }
    if index < 0 or index >= len(messages):
        return detail
    detail["is_observe_same_call"] = is_observe_same_call(messages, index)
    detail["is_consecutive_identical"] = is_identical_to_previous_assistant_call(
        messages, index
    )
    if primary == ACTION_FINAL_AFTER_OBS:
        detail["final_after_obs_strict_advancing"] = prior_tools_look_advancing(
            messages, index
        )
    return detail


def attach_action_labels(record: dict) -> dict:
    """Return a shallow copy with ``action_labels`` on each assistant turn."""
    out = dict(record)
    messages = list(record.get("messages") or [])
    labels: list[str | None] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            labels.append(classify_assistant_action(messages, i))
        else:
            labels.append(None)
    out["action_labels"] = labels
    # Conversation-level stop_label stays compatible with v7 taxonomy when set.
    if messages:
        last_i = len(messages) - 1
        if messages[last_i].get("role") == "assistant":
            out["action_class"] = classify_assistant_action(messages, last_i)
    return out
