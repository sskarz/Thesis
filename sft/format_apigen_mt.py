"""Convert Salesforce/APIGen-MT-5k ShareGPT rows into IFFC chat JSONL.

ShareGPT ``from`` vocabulary (do **not** run these through ``format_toolace_row``):

* ``human`` → user
* ``function_call`` → assistant IFFC JSON via ``dumps_iffc_calls``
  (normalize ``name``+``arguments`` → ``function_name``+``parameters``)
* ``observation`` → user via ``format_iffc_execution_user_message``
* ``gpt`` → assistant **natural language** (clarify / final). Never auto-map
  NL done to ``[]`` (that was ToolACE's collapse teacher).

``tools`` (JSON string or list) → ``normalize_function_docs`` →
``build_iffc_system_prompt``. Optional truncated domain policy from ``system``
is appended **after** the Appendix B IFFC contract.

Reject unparseable ``function_call`` values, rows with no assistant turn, and
native ``tool_calls`` leaks.
"""

from __future__ import annotations

from typing import Any

from benchmark.iffc import build_iffc_system_prompt, parse_iffc_response

from sft.config import (
    APIGEN_MT_POLICY_MAX_CHARS,
    CHAT_FORMAT,
    SOURCE_APIGEN_MT,
)
from sft.format_iffc import (
    dumps_iffc_calls,
    n_tools_bin,
    normalize_function_docs,
    _count_gold_tools,
    _format_toolace_observation,
    _maybe_json,
)

STOP_TOOL_CALL = "tool_call"
STOP_CLARIFY = "clarify"
STOP_FINAL = "final_answer"
STOP_IRRELEVANCE = "irrelevance_[]"
STOP_LABELS = (STOP_TOOL_CALL, STOP_CLARIFY, STOP_FINAL, STOP_IRRELEVANCE)
NONTOOL_LABELS = frozenset({STOP_CLARIFY, STOP_FINAL, STOP_IRRELEVANCE})
CLARIFY_KEYWORDS = (
    "please provide",
    "could you",
    "which",
    "need more",
)

_FUNCTION_CALL_ROLES = frozenset({"function_call", "functioncall"})
_OBSERVATION_ROLES = frozenset({"observation", "tool", "function"})
_USER_ROLES = frozenset({"human", "user"})
_GPT_ROLES = frozenset({"gpt", "assistant"})
_KNOWN_ROLES = (
    _FUNCTION_CALL_ROLES | _OBSERVATION_ROLES | _USER_ROLES | _GPT_ROLES
)


def label_assistant_content(content: Any) -> str:
    """Assign a v7 stop-taxonomy label to one assistant target.

    Deterministic (no LLM): tool_call | irrelevance_[] | clarify | final_answer.
    """
    text = str(content or "")
    stripped = text.strip()
    if stripped == "[]":
        return STOP_IRRELEVANCE
    if parse_iffc_response(text):
        return STOP_TOOL_CALL
    if not stripped:
        return STOP_IRRELEVANCE
    lower = stripped.lower()
    if "?" in stripped or any(k in lower for k in CLARIFY_KEYWORDS):
        return STOP_CLARIFY
    return STOP_FINAL


def parse_apigen_tools(raw: Any) -> list[Any]:
    """Parse the Hub ``tools`` field into a list of function dicts."""
    parsed = _maybe_json(raw)
    if parsed is None or parsed == "":
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def parse_apigen_function_call_value(value: Any) -> list[dict] | None:
    """Normalize a ShareGPT ``function_call`` value to IFFC call dicts.

    Accepts a JSON object, a JSON string, or a list (parallel calls).
    Returns ``None`` when the value cannot be parsed as name+arguments.
    An empty list means a well-formed empty call set (not a reject).
    """
    parsed = _maybe_json(value)
    if parsed is None or parsed == "":
        return None
    if isinstance(parsed, dict):
        items: list[Any] = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return None

    calls: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        name = item.get("name") or item.get("function_name")
        if not name:
            return None
        params = item.get("arguments", item.get("parameters", {}))
        if isinstance(params, str):
            params = _maybe_json(params)
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return None
        calls.append({"function_name": str(name), "parameters": params})
    return calls


def _policy_suffix(system_text: Any, *, max_chars: int = APIGEN_MT_POLICY_MAX_CHARS) -> str:
    policy = str(system_text or "").strip()
    if not policy:
        return ""
    if len(policy) > max_chars:
        policy = policy[:max_chars].rstrip() + "\n…[truncated]"
    return "\n\nDomain policy:\n" + policy


def _assistant_labels(messages: list[dict]) -> list[str]:
    return [
        label_assistant_content(m.get("content"))
        for m in messages
        if m.get("role") == "assistant"
    ]


def format_apigen_mt_row(row: dict, row_index: int) -> dict | None:
    """Convert one Salesforce/APIGen-MT-5k ShareGPT row to an IFFC chat example.

    Returns ``None`` to skip (unparseable call, no assistant, empty dialog).
    Raises ``RuntimeError`` if native ``tool_calls`` leak into the record.
    """
    conversations = _maybe_json(row.get("conversations") or [])
    if not isinstance(conversations, list) or not conversations:
        return None

    openai_tools = normalize_function_docs(parse_apigen_tools(row.get("tools")))
    system_content = build_iffc_system_prompt(openai_tools) + _policy_suffix(
        row.get("system")
    )
    messages: list[dict] = [{"role": "system", "content": system_content}]

    prev_calls: list[dict] = []
    saw_assistant = False
    for turn in conversations:
        if not isinstance(turn, dict):
            return None
        if "tool_calls" in turn:
            return None
        role = (turn.get("from") or turn.get("role") or "").strip().lower()
        value = turn.get("value", turn.get("content", ""))
        if value is None:
            value = ""

        if role not in _KNOWN_ROLES:
            return None

        if role in _USER_ROLES:
            messages.append({"role": "user", "content": str(value)})
            prev_calls = []
        elif role in _FUNCTION_CALL_ROLES:
            calls = parse_apigen_function_call_value(value)
            if calls is None:
                return None
            messages.append({"role": "assistant", "content": dumps_iffc_calls(calls)})
            prev_calls = calls
            saw_assistant = True
        elif role in _OBSERVATION_ROLES:
            messages.append(
                {
                    "role": "user",
                    "content": _format_toolace_observation(str(value), prev_calls),
                }
            )
        elif role in _GPT_ROLES:
            # Natural language: clarify / final. Never coerce to [].
            text = str(value)
            messages.append({"role": "assistant", "content": text})
            prev_calls = []
            saw_assistant = True

    if not saw_assistant:
        return None
    if any("tool_calls" in m for m in messages):
        raise RuntimeError("native tool_calls leaked into APIGen-MT IFFC row")

    labels = _assistant_labels(messages)
    n_gold = _count_gold_tools(messages)
    raw_id = row.get("id", row.get("row_index", row_index))
    example_id = f"apigen_mt-{raw_id}"
    return {
        "id": example_id,
        "source": SOURCE_APIGEN_MT,
        "is_multi_turn": True,
        "n_tools_in_gold": n_gold,
        "n_tools_bin": n_tools_bin(n_gold),
        "chat_format": CHAT_FORMAT,
        "stop_label": labels[-1] if labels else None,
        "stop_labels": labels,
        "messages": messages,
    }
