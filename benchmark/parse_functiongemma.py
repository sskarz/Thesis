"""Parser for FunctionGemma's custom tool-call output format.

FunctionGemma uses control tokens instead of OpenAI-style tool_calls:

    <start_function_call>call:func_name{key:<escape>value<escape>,key2:<escape>value2<escape>}<end_function_call>

llama-server cannot parse this format natively, so the raw text ends up in
the ``content`` field.  This module converts that content into the standard
``[{"func_name": {"arg1": val1, ...}}]`` format used by the BFCL evaluator.
"""

import re


def parse_functiongemma_content(content: str) -> list[dict] | None:
    """Parse FunctionGemma's output format into BFCL-compatible tool calls.

    Returns a list of ``{name: {args}}`` dicts, or ``None`` if the content
    does not contain FunctionGemma control tokens.
    """
    if "<start_function_call>" not in content:
        return None

    blocks = re.findall(
        r"<start_function_call>(.*?)<end_function_call>", content, re.DOTALL
    )
    if not blocks:
        return None

    results: list[dict] = []
    for block in blocks:
        parsed = _parse_block(block.strip())
        if parsed is not None:
            results.append(parsed)

    return results if results else None


def _parse_block(block: str) -> dict | None:
    """Parse a single ``call:name{args}`` block."""
    if block.startswith("call:"):
        block = block[5:]

    brace_idx = block.find("{")
    if brace_idx == -1:
        # No arguments — bare function name
        return {block.strip(): {}}

    func_name = block[:brace_idx].strip()
    args_str = block[brace_idx:]

    # Strip outer braces
    if args_str.startswith("{") and args_str.endswith("}"):
        args_str = args_str[1:-1]
    else:
        return {func_name: {}}

    args = _parse_args(args_str)
    return {func_name: args}


def _parse_args(args_str: str) -> dict:
    """Split key:value pairs at commas respecting brace depth and escape regions."""
    pairs = _split_top_level(args_str, ",")
    result: dict = {}
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        colon_idx = pair.find(":")
        if colon_idx == -1:
            continue
        key = pair[:colon_idx].strip()
        value_str = pair[colon_idx + 1:].strip()
        result[key] = _parse_value(value_str)
    return result


def _split_top_level(s: str, delimiter: str) -> list[str]:
    """Split *s* on *delimiter* only at brace-depth 0 and outside escape regions."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_escape = False
    i = 0
    while i < len(s):
        # Check for <escape> tags
        if s[i:].startswith("<escape>"):
            in_escape = not in_escape
            current.append("<escape>")
            i += len("<escape>")
            continue

        ch = s[i]
        if not in_escape:
            if ch == "{":
                depth += 1
            elif ch == "}":
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


def _parse_value(value_str: str) -> str | int | float | bool | dict | list:
    """Interpret a FunctionGemma value token."""
    # Escaped string: <escape>...<escape>
    if value_str.startswith("<escape>") and value_str.endswith("<escape>"):
        inner = value_str[len("<escape>"):-len("<escape>")]
        return inner

    # Nested object: {...}
    if value_str.startswith("{") and value_str.endswith("}"):
        return _parse_args(value_str[1:-1])

    # Nested list: [...]
    if value_str.startswith("[") and value_str.endswith("]"):
        items = _split_top_level(value_str[1:-1], ",")
        return [_parse_value(item.strip()) for item in items if item.strip()]

    # Booleans
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # None / null
    if value_str.lower() in ("none", "null"):
        return None

    # Numeric
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass

    # Fallback: treat as string
    return value_str
