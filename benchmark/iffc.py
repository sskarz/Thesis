"""Instruction-Followed Function Calling (IFFC) — arXiv:2608.22472 Appendix B.

IFFC does **not** use native ``tools`` / ``tool_choice``. The model is instructed
to emit a JSON list of ``{"function_name", "parameters"}`` dicts in assistant
text. Empty list ``[]`` means no (further) function calling this step.

NFC (native function calling) remains the default in ``config.FC_MODE``.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Verbatim Appendix B system prompt (placeholder {available_functions} is filled
# at runtime). Apostrophe in "doesn't" is U+2019 as in the paper.
IFFC_SYSTEM_PROMPT = (
    "You are an expert in function calling. You will be given a set of available "
    "functions and a user query. You should determine if the user query needs "
    "function calling, call the appropriate function with suitable arguments. "
    "Based on the question , you will need to make one or more function calls "
    "to achieve the purpose. Here are the available functions:\n"
    "{available_functions}\n"
    "For every user query you only answer in json format. You will return a list "
    "of dictionaries. If the user question doesn\u2019t require any function calling "
    "you will return an empty list ( e.g. [] ). If it does require function "
    "calling, for every function you add a dictionary which has two keys, "
    '"function_name" and "parameters". "parameters" is also a dict with function '
    "arguments and their values. Use the exact same format of function parameter "
    "values as the examples in the function definition."
)

IFFC_PARSE_RETRIES = 1

IFFC_RETRY_USER_MESSAGE = (
    "Your previous reply was not valid JSON in the required format. "
    'Reply with a JSON list of dictionaries, each with "function_name" and '
    '"parameters", or [] if no function calling is needed. Do not include any '
    "other text."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(
    r"<(think|reasoning|thought)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def format_available_functions(tools_or_bfcl_docs: list[dict] | None) -> str:
    """Serialize OpenAI tools or BFCL function docs for the IFFC placeholder."""
    funcs = _unwrap_function_schemas(tools_or_bfcl_docs or [])
    return json.dumps(funcs, indent=2, ensure_ascii=False)


def build_iffc_system_prompt(tools_or_bfcl_docs: list[dict] | None) -> str:
    """Fill ``IFFC_SYSTEM_PROMPT`` with serialized function schemas."""
    return IFFC_SYSTEM_PROMPT.replace(
        "{available_functions}",
        format_available_functions(tools_or_bfcl_docs),
    )


def ensure_iffc_system_message(
    chat_history: list[dict],
    tools_or_bfcl_docs: list[dict] | None,
) -> None:
    """Insert or refresh the IFFC system message when the active tool set changes."""
    content = build_iffc_system_prompt(tools_or_bfcl_docs)
    if chat_history and chat_history[0].get("role") == "system":
        if chat_history[0].get("content") != content:
            chat_history[0] = {"role": "system", "content": content}
        return
    chat_history.insert(0, {"role": "system", "content": content})


def assistant_text(message: dict) -> str:
    """Flatten assistant ``content`` plus optional ``reasoning_content``."""
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    elif content is None:
        content = ""
    else:
        content = str(content)

    reasoning = message.get("reasoning_content") or ""
    if reasoning and reasoning not in content:
        return f"{reasoning}\n{content}".strip()
    return content.strip()


def try_parse_iffc_response(content: str | None) -> list[dict] | None:
    """Extract a JSON call list from assistant text.

    Returns:
        * ``[]`` if the model emitted an empty list
        * a list of ``{"function_name", "parameters"}`` dicts
        * ``None`` if no JSON list/object could be parsed (caller may retry)
    """
    if content is None:
        return None
    text = _strip_wrappers(content)
    if not text:
        return None

    candidates: list[str] = []
    fences = _FENCE_RE.findall(text)
    if fences:
        candidates.extend(reversed(fences))
    candidates.append(text)

    for cand in candidates:
        parsed = _parse_json_candidate(cand)
        if parsed is not None:
            return parsed
    return None


def parse_iffc_response(content: str) -> list[dict]:
    """Extract IFFC calls from assistant text; unparseable input becomes ``[]``."""
    parsed = try_parse_iffc_response(content)
    return parsed if parsed is not None else []


def iffc_calls_to_func_strings(calls: list[dict]) -> list[str]:
    """Convert IFFC call dicts to BFCL ``name(arg=val, ...)`` strings."""
    result: list[str] = []
    for call in calls:
        name = call.get("function_name") or call.get("name") or "unknown"
        params = call.get("parameters", call.get("arguments", {}))
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                result.append(f"{name}(__arg_parse_error__={repr(params)})")
                continue
        if not isinstance(params, dict):
            result.append(f"{name}(__arg_parse_error__={repr(params)})")
            continue
        parts = [f"{k}={repr(v)}" for k, v in params.items()]
        result.append(f"{name}({', '.join(parts)})")
    return result


def format_iffc_execution_user_message(
    func_strings: list[str],
    exec_results: list[Any],
) -> str:
    """User-role observation after tool execution (IFFC stays instruction-following)."""
    lines = ["Function execution results:"]
    for call, exec_result in zip(func_strings, exec_results):
        lines.append(f"- {call} -> {exec_result}")
    if not func_strings:
        lines.append("- (no functions were executed)")
    lines.append("")
    lines.append(
        'Respond with a JSON list of function calls (each with "function_name" '
        "and \"parameters\"), or [] if no further function calling is needed."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _unwrap_function_schemas(tools_or_bfcl_docs: list[dict]) -> list[dict]:
    funcs: list[dict] = []
    for item in tools_or_bfcl_docs:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            funcs.append(item["function"])
        else:
            funcs.append(item)
    return funcs


def _strip_wrappers(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _parse_json_candidate(text: str) -> list[dict] | None:
    decoder = json.JSONDecoder()
    s = text.strip()
    i = 0
    last_valid: list[dict] | None = None
    while i < len(s):
        ch = s[i]
        if ch in "[{":
            try:
                obj, end = decoder.raw_decode(s, i)
            except json.JSONDecodeError:
                i += 1
                continue
            normalized = _normalize_calls(obj)
            if normalized is not None:
                last_valid = normalized
            i = end
            continue
        i += 1
    return last_valid


def _normalize_calls(value: Any) -> list[dict] | None:
    """Accept a list of call dicts, a single call dict, or ``[]``."""
    if isinstance(value, dict):
        if "function_name" in value or "name" in value:
            value = [value]
        elif isinstance(value.get("function_calls"), list):
            value = value["function_calls"]
        else:
            return None
    if not isinstance(value, list):
        return None

    calls: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        name = item.get("function_name") or item.get("name")
        if not name or not isinstance(name, str):
            return None
        params = item.get("parameters", item.get("arguments", {}))
        if params is None:
            params = {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(params, dict):
            return None
        calls.append({"function_name": name, "parameters": params})
    return calls
