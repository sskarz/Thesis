"""Schema-aware type coercion for tool-call arguments.

Some models (notably Llama 3.2) serialize all tool-call arguments as JSON
strings — e.g. ``"5"`` instead of ``5``, ``"[1, 2]"`` instead of ``[1, 2]``.
The BFCL AST checker rejects these type mismatches even when the function name
and argument values are semantically correct.

This module post-processes parsed tool calls by coercing each argument value to
the type declared in the BFCL function schema.  It is safe to apply
unconditionally: values that already have the correct type pass through
unchanged, so models that produce correct types (e.g. Qwen) are unaffected.
"""

from __future__ import annotations

import json


def _try_parse_json(s: str):
    """Parse *s* as JSON, falling back to single-quote replacement."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return json.loads(s.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def coerce_value(value, schema: dict):
    """Coerce *value* to match the ``type`` declared in *schema*.

    Handles the common failure modes observed in Llama 3.2:
    - string → integer / number / float
    - string → array / tuple / list  (JSON or Python repr)
    - string → object / dict          (JSON or Python repr)
    - string → boolean
    - nested objects with string-typed leaf values
    """
    expected = schema.get("type", "string")

    # Already non-string — recurse into nested objects if needed.
    if not isinstance(value, str):
        if isinstance(value, dict) and expected in ("object", "dict"):
            nested_props = schema.get("properties", {})
            if nested_props:
                return {
                    k: coerce_value(v, nested_props.get(k, {}))
                    for k, v in value.items()
                }
        return value

    # --- value is a string; try to coerce based on expected type ---

    if expected == "integer":
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return value

    if expected in ("number", "float"):
        try:
            f = float(value)
            return int(f) if f == int(f) else f
        except ValueError:
            return value

    if expected in ("array", "tuple", "list"):
        parsed = _try_parse_json(value)
        if isinstance(parsed, list):
            return parsed
        return value

    if expected in ("object", "dict"):
        parsed = _try_parse_json(value)
        if isinstance(parsed, dict):
            nested_props = schema.get("properties", {})
            if nested_props:
                return {
                    k: coerce_value(v, nested_props.get(k, {}))
                    for k, v in parsed.items()
                }
            return parsed
        return value

    if expected == "boolean":
        low = value.lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False

    return value


def coerce_tool_call(tool_call: dict, func_defs: list[dict]) -> dict:
    """Coerce argument types in a single ``{func_name: {args}}`` dict.

    *func_defs* is the BFCL ``function`` list from the test entry — each item
    has ``name`` and ``parameters.properties`` with per-argument type info.
    """
    result = {}
    for func_name, args in tool_call.items():
        schema_props: dict = {}
        for fd in func_defs:
            if fd["name"] == func_name:
                schema_props = fd.get("parameters", {}).get("properties", {})
                break
        result[func_name] = {
            k: coerce_value(v, schema_props.get(k, {})) for k, v in args.items()
        }
    return result


def coerce_tool_calls(
    tool_calls: list[dict], func_defs: list[dict]
) -> list[dict]:
    """Coerce argument types for a list of tool calls."""
    return [coerce_tool_call(tc, func_defs) for tc in tool_calls]
