"""Parser for Gemma native tool-call markup emitted in ``content``.

On llama.cpp with ``tool_choice=none`` (used for BFCL ``parallel`` /
``parallel_multiple`` to bypass the tools grammar), Gemma-4 emits tool
calls as raw text instead of OpenAI-style ``tool_calls``:

    <|tool_call>call:spotify.play{artist:<|"|>Taylor Swift<|"|>,duration:20}<tool_call|>

Notes on the format:

- Opener is ``<|tool_call>`` (slight variants exist).
- Body is ``call:FUNCTION_NAME{key:value,...}``. Names may contain dots.
- Strings are wrapped as ``<|"|>VALUE<|"|>`` rather than JSON quotes.
- Numbers, bools, and lists appear unquoted (``20``, ``true``, ``[15]``).
- The closer is often the malformed ``<tool_call|>`` (missing ``|`` after
  ``<``); tags are treated as optional once Gemma markers are present.
- Multiple calls are concatenated back-to-back.

This module converts that markup into the ``[{"func": {args}}, ...]``
format expected by the BFCL evaluator.
"""

from __future__ import annotations

import json
import re

STRING_WRAP = '<|"|>'

# Gemma-specific markers that distinguish this format from Qwen XML
# (``<tool_call>...</tool_call>``) and FunctionGemma (``<escape>`` /
# ``<start_function_call>``).
_GEMMA_MARKERS = ("<|tool_call", "<tool_call|>", STRING_WRAP)

_CALL_START = re.compile(r"call:([A-Za-z_][A-Za-z0-9_.]*)\s*\{")


def parse_gemma_tool_call_content(content: str) -> list[dict] | None:
    """Parse Gemma native tool-call markup into BFCL-compatible tool calls.

    Returns a list of ``{name: {args}}`` dicts, or ``None`` if *content*
    does not look like Gemma markup or no calls could be parsed.
    """
    if not isinstance(content, str) or not _looks_like_gemma(content):
        return None

    results: list[dict] = []
    for name, args_body in _extract_call_blocks(content):
        args = _parse_args(args_body)
        results.append({name: args})

    return results if results else None


def _looks_like_gemma(content: str) -> bool:
    return any(marker in content for marker in _GEMMA_MARKERS)


def _extract_call_blocks(content: str) -> list[tuple[str, str]]:
    """Return ``(function_name, args_inside_braces)`` pairs."""
    blocks: list[tuple[str, str]] = []
    i = 0
    while True:
        match = _CALL_START.search(content, i)
        if not match:
            break
        name = match.group(1)
        brace_open = match.end() - 1
        brace_close = _find_matching_brace(content, brace_open)
        if brace_close is None:
            i = match.end()
            continue
        args_body = content[brace_open + 1 : brace_close]
        blocks.append((name, args_body))
        i = brace_close + 1
    return blocks


def _find_matching_brace(s: str, open_idx: int) -> int | None:
    """Index of the ``}`` matching ``s[open_idx]``, skipping string wraps."""
    depth = 0
    i = open_idx
    n = len(s)
    wrap_len = len(STRING_WRAP)
    while i < n:
        if s.startswith(STRING_WRAP, i):
            i += wrap_len
            end = s.find(STRING_WRAP, i)
            if end == -1:
                return None
            i = end + wrap_len
            continue
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _parse_args(args_str: str) -> dict:
    """Split ``key:value`` pairs at commas, respecting wraps and nesting."""
    result: dict = {}
    for pair in _split_top_level(args_str, ","):
        pair = pair.strip()
        if not pair:
            continue
        colon_idx = _find_top_level_colon(pair)
        if colon_idx is None:
            continue
        key = pair[:colon_idx].strip()
        value_str = pair[colon_idx + 1 :].strip()
        if not key:
            continue
        result[key] = _parse_value(value_str)
    return result


def _find_top_level_colon(pair: str) -> int | None:
    """Index of the ``key:value`` colon, ignoring colons inside wraps/nesting."""
    depth = 0
    i = 0
    wrap_len = len(STRING_WRAP)
    while i < len(pair):
        if pair.startswith(STRING_WRAP, i):
            i += wrap_len
            end = pair.find(STRING_WRAP, i)
            if end == -1:
                return None
            i = end + wrap_len
            continue
        ch = pair[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
        i += 1
    return None


def _split_top_level(s: str, delimiter: str) -> list[str]:
    """Split *s* on *delimiter* at brace/bracket depth 0, outside wraps."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    wrap_len = len(STRING_WRAP)
    while i < len(s):
        if s.startswith(STRING_WRAP, i):
            current.append(STRING_WRAP)
            i += wrap_len
            end = s.find(STRING_WRAP, i)
            if end == -1:
                current.append(s[i:])
                break
            current.append(s[i:end])
            current.append(STRING_WRAP)
            i = end + wrap_len
            continue

        ch = s[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == delimiter and depth == 0:
            parts.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1

    if current:
        parts.append("".join(current))
    return parts


def _parse_value(value_str: str) -> str | int | float | bool | dict | list | None:
    """Interpret a single Gemma argument value."""
    value_str = value_str.strip()
    wrap_len = len(STRING_WRAP)

    if (
        value_str.startswith(STRING_WRAP)
        and value_str.endswith(STRING_WRAP)
        and len(value_str) >= 2 * wrap_len
    ):
        return value_str[wrap_len:-wrap_len]

    if value_str.startswith("{") and value_str.endswith("}"):
        return _parse_args(value_str[1:-1])

    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1]
        items = _split_top_level(inner, ",")
        return [_parse_value(item.strip()) for item in items if item.strip()]

    lowered = value_str.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None

    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass

    # JSON-quoted strings (and other JSON literals) if the model mixed formats.
    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return value_str
