"""TagDispatch-style constrained decoding (CD) for multi-turn function calling.

This is a **deliberate CD ablation**, not a rebrand of NFC.

NFC (``--fc-mode nfc``) sends OpenAI ``tools`` / ``tool_choice=auto``. llama-server
may then compile a *lazy* tools GBNF from the chat template (known parallel
truncation bug on some templates). That native tools grammar is **not** the
CD experiment.

CD (``--fc-mode cd``) never sends ``tools`` / ``tool_choice``. It approximates
XGrammar-2 TagDispatch (Dong et al., arXiv:2601.04426) on stock llama-server:

1. **Dispatch pass** — GBNF ``grammar`` that only allows ``[]`` or a JSON
   array of ``{"function_name": <enum of live tools>}``.
2. **Dispatched pass** — llama-server ``json_schema`` (compiled to GBNF by
   ``json-schema-to-grammar``) for the *selected* tool's arguments.

This is a two-request approximation of TagDispatch, not a full XGrammar-2
engine: llama.cpp has no XGrammar; optional LLGuidance (``-DLLAMA_LLGUIDANCE=ON``)
is not required and is assumed OFF. Intra-request tag-triggered grammar
switching, Cross-Grammar Cache, and Earley mask caches are not available.

The persistent multi-turn transcript stays IFFC-shaped (JSON list in assistant
text; user-role execution feedback) so evaluation plumbing is shared. Only
the *completion* path is constrained.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Awaitable

from .iffc import (
    assistant_text,
    format_available_functions,
    try_parse_iffc_response,
)

# ---------------------------------------------------------------------------
# Prompts (instruction channel; constraints are the grammar / json_schema)
# ---------------------------------------------------------------------------

CD_DISPATCH_SYSTEM_PROMPT = (
    "You are an expert in function calling. You will be given a set of available "
    "functions and a user query. Decide whether any function must be called.\n"
    "Here are the available functions:\n"
    "{available_functions}\n"
    "Reply with JSON only. Return [] if no function calling is needed. "
    "If function calling is needed, return a JSON array of objects, each with "
    'exactly one key "function_name" whose value is the exact name of an '
    "available function. Do not include parameters in this step."
)

CD_ARGS_USER_MESSAGE = (
    "You selected these function names: {names}. "
    "Now reply with a JSON array of dictionaries, each with "
    '"function_name" (from that list) and "parameters" matching that '
    "function's schema. JSON only. No other text."
)

SyncPost = Callable[[dict], dict]
AsyncPost = Callable[[dict], Awaitable[dict]]


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

def unwrap_function(tool: dict) -> dict:
    """Return the inner function schema from an OpenAI tool or BFCL doc."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool


def tool_names(tools: list[dict] | None) -> list[str]:
    """Ordered unique names from OpenAI tools or BFCL function docs."""
    names: list[str] = []
    seen: set[str] = set()
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        name = unwrap_function(item).get("name")
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def tool_parameters_schema(tool: dict) -> dict:
    func = unwrap_function(tool)
    params = func.get("parameters")
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    return copy.deepcopy(params)


def tool_by_name(tools: list[dict] | None, name: str) -> dict | None:
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        if unwrap_function(item).get("name") == name:
            return item
    return None


# ---------------------------------------------------------------------------
# JSON Schema construction (TagDispatch tags + per-tool args)
# ---------------------------------------------------------------------------

def build_name_dispatch_schema(tools: list[dict] | None) -> dict:
    """JSON Schema for pass 1: ``[]`` or ``[{"function_name": <enum>}, ...]``.

    ``parameters`` is forbidden (``additionalProperties: false``).
    """
    names = tool_names(tools)
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "function_name": {"type": "string", "enum": names},
            },
            "required": ["function_name"],
            "additionalProperties": False,
        },
    }


def build_args_schema(
    tools: list[dict] | None,
    selected_names: list[str],
) -> dict:
    """JSON Schema for pass 2: one tagged object per selected name.

    Encodes TagDispatch's "tag t_i → sub-grammar G_i" as ``const`` name +
    that tool's parameter schema (``oneOf`` when several names were chosen).
    """
    if not selected_names:
        return {"type": "array", "maxItems": 0}

    variants: list[dict] = []
    for name in selected_names:
        tool = tool_by_name(tools, name)
        params = tool_parameters_schema(tool) if tool else {"type": "object", "properties": {}}
        variants.append(
            {
                "type": "object",
                "properties": {
                    "function_name": {"const": name, "type": "string"},
                    "parameters": params,
                },
                "required": ["function_name", "parameters"],
                "additionalProperties": False,
            }
        )

    # Deduplicate identical variants (same name selected twice → still oneOf
    # of unique tagged objects; array length still pinned to selection length).
    unique: list[dict] = []
    seen_names: set[str] = set()
    for variant in variants:
        n = variant["properties"]["function_name"]["const"]
        if n in seen_names:
            continue
        seen_names.add(n)
        unique.append(variant)

    items: dict
    if len(unique) == 1:
        items = unique[0]
    else:
        items = {"oneOf": unique}

    return {
        "type": "array",
        "minItems": len(selected_names),
        "maxItems": len(selected_names),
        "items": items,
    }


def build_tagged_union_schema(tools: list[dict] | None) -> dict:
    """Single-pass EBNF-style TagDispatch encoding (not used at runtime).

    ``oneOf`` over every live tool: const name + that tool's parameters, or
    empty array. Documented as the static-union alternative to two-pass CD.
    """
    names = tool_names(tools)
    if not names:
        return {"type": "array", "maxItems": 0}
    variants = []
    for name in names:
        tool = tool_by_name(tools, name)
        params = tool_parameters_schema(tool) if tool else {"type": "object", "properties": {}}
        variants.append(
            {
                "type": "object",
                "properties": {
                    "function_name": {"const": name, "type": "string"},
                    "parameters": params,
                },
                "required": ["function_name", "parameters"],
                "additionalProperties": False,
            }
        )
    return {
        "type": "array",
        "items": variants[0] if len(variants) == 1 else {"oneOf": variants},
    }


# ---------------------------------------------------------------------------
# Minimal JSON Schema checker (hermetic constraint-enforcement tests)
# ---------------------------------------------------------------------------

def instance_matches_schema(schema: dict, instance: Any) -> bool:
    """Subset JSON Schema validator for CD schemas (no $ref / format)."""
    if not isinstance(schema, dict):
        return True

    if "const" in schema and instance != schema["const"]:
        return False
    if "enum" in schema and instance not in schema["enum"]:
        return False
    if "oneOf" in schema:
        return any(instance_matches_schema(s, instance) for s in schema["oneOf"])
    if "anyOf" in schema:
        return any(instance_matches_schema(s, instance) for s in schema["anyOf"])

    t = schema.get("type")
    types: list[str] | None
    if t is None:
        types = None
    elif isinstance(t, list):
        types = [str(x) for x in t]
    else:
        types = [str(t)]

    if types is not None:
        if not any(_instance_has_type(instance, tt) for tt in types):
            return False
        # Recurse using a copy that pins a single matching type so nested
        # keywords (items / properties) still apply.
        if len(types) > 1:
            for tt in types:
                if _instance_has_type(instance, tt):
                    pinned = dict(schema)
                    pinned["type"] = tt
                    if instance_matches_schema(pinned, instance):
                        return True
            return False

    if _instance_has_type(instance, "array") and (
        "items" in schema or "minItems" in schema or "maxItems" in schema
    ):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            return False
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            return False
        items = schema.get("items")
        if items is False:
            return len(instance) == 0
        if isinstance(items, dict):
            return all(instance_matches_schema(items, x) for x in instance)
        return True

    if _instance_has_type(instance, "object") and (
        "properties" in schema
        or "required" in schema
        or "additionalProperties" in schema
    ):
        required = schema.get("required") or []
        if any(key not in instance for key in required):
            return False
        props = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                if not instance_matches_schema(props[key], value):
                    return False
            elif additional is False:
                return False
            elif isinstance(additional, dict) and not instance_matches_schema(
                additional, value
            ):
                return False
        return True

    return True


def _instance_has_type(instance: Any, t: str) -> bool:
    if t == "object":
        return isinstance(instance, dict)
    if t == "array":
        return isinstance(instance, list)
    if t == "string":
        return isinstance(instance, str)
    if t == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if t == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if t == "boolean":
        return isinstance(instance, bool)
    if t == "null":
        return instance is None
    return True


# ---------------------------------------------------------------------------
# GBNF for pass 1 (name enum). Sent as llama-server ``grammar``.
# ---------------------------------------------------------------------------

def _gbnf_emit_json_string(value: str) -> str:
    """GBNF quoted literal that generates the JSON encoding of *value*."""
    encoded = json.dumps(value, ensure_ascii=False)
    return '"' + encoded.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_name_dispatch_grammar(tools: list[dict] | None) -> str:
    """Tight GBNF: whitespace-tolerant ``[]`` or name-only call objects."""
    names = tool_names(tools)
    lines = [
        r"root ::= ws (empty | calls) ws",
        r"ws ::= [ \t\n]*",
        r'empty ::= "[" ws "]"',
    ]
    if not names:
        # Enum is empty: only [] is legal.
        return "\n".join(lines) + "\n"
    alts = " | ".join(_gbnf_emit_json_string(n) for n in names)
    lines.extend(
        [
            r'calls ::= "[" ws call (ws "," ws call)* ws "]"',
            r'call ::= "{" ws "\"function_name\"" ws ":" ws name ws "}"',
            f"name ::= {alts}",
        ]
    )
    return "\n".join(lines) + "\n"


def grammar_allowed_names(grammar: str) -> list[str]:
    """Extract the ``name ::=`` alternatives from a dispatch GBNF."""
    for line in grammar.splitlines():
        if line.startswith("name ::="):
            rhs = line.split("::=", 1)[1]
            raw_alts = [p.strip() for p in rhs.split("|")]
            names: list[str] = []
            for alt in raw_alts:
                # GBNF "\"cd\"" → JSON '"cd"' → cd
                if alt.startswith('"') and alt.endswith('"'):
                    inner = alt[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                    try:
                        names.append(json.loads(inner))
                    except json.JSONDecodeError:
                        continue
            return names
    return []


# ---------------------------------------------------------------------------
# llama-server payload (never tools / tool_choice)
# ---------------------------------------------------------------------------

def apply_dispatch_constraints(payload: dict, tools: list[dict] | None) -> dict:
    """Pass 1: native GBNF ``grammar``. Do not send json_schema (it would win)."""
    payload["grammar"] = build_name_dispatch_grammar(tools)
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    return payload


def apply_args_constraints(
    payload: dict,
    tools: list[dict] | None,
    selected_names: list[str],
) -> dict:
    """Pass 2: llama-server ``json_schema`` → GBNF via json-schema-to-grammar."""
    payload["json_schema"] = build_args_schema(tools, selected_names)
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload.pop("grammar", None)
    return payload


def assert_cd_payload(payload: dict, *, pass_kind: str) -> None:
    """Raise ``AssertionError`` if a payload looks like NFC tools-GBNF."""
    if "tools" in payload or "tool_choice" in payload:
        raise AssertionError(
            "CD must not send tools/tool_choice (that would activate NFC GBNF)"
        )
    if pass_kind == "dispatch":
        if not payload.get("grammar"):
            raise AssertionError("CD dispatch pass requires grammar")
        if "json_schema" in payload:
            raise AssertionError(
                "CD dispatch pass must not set json_schema (it overrides grammar)"
            )
    elif pass_kind == "args":
        if not payload.get("json_schema"):
            raise AssertionError("CD args pass requires json_schema")
    else:
        raise ValueError(f"unknown pass_kind {pass_kind!r}")


# ---------------------------------------------------------------------------
# System prompt on working copies (persistent history stays IFFC-shaped)
# ---------------------------------------------------------------------------

def build_dispatch_system_prompt(tools: list[dict] | None) -> str:
    return CD_DISPATCH_SYSTEM_PROMPT.replace(
        "{available_functions}",
        format_available_functions(tools),
    )


def _with_dispatch_system(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    msgs = copy.deepcopy(messages)
    content = build_dispatch_system_prompt(tools)
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {"role": "system", "content": content}
    else:
        msgs.insert(0, {"role": "system", "content": content})
    return msgs


def format_args_user_message(names: list[str]) -> str:
    return CD_ARGS_USER_MESSAGE.format(names=", ".join(names))


def _merge_timings(*responses: dict) -> dict:
    totals = {
        "prompt_n": 0,
        "prompt_ms": 0.0,
        "predicted_n": 0,
        "predicted_ms": 0.0,
    }
    for data in responses:
        src = data.get("timings") or {}
        for key in totals:
            totals[key] += src.get(key, 0) or 0
    return totals


def _assistant_response(content: str, timings: dict) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "timings": timings,
    }


def _content_from_response(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return assistant_text(choices[0].get("message") or {})


def _selected_names(parsed: list[dict] | None) -> list[str] | None:
    if parsed is None:
        return None
    return [c["function_name"] for c in parsed]


def _finalize_calls_json(parsed: list[dict]) -> str:
    return json.dumps(parsed, ensure_ascii=False)


def _build_dispatch_payload(
    messages: list[dict],
    tools: list[dict] | None,
    model: str,
) -> tuple[dict, list[dict]]:
    dispatch_msgs = _with_dispatch_system(messages, tools)
    payload: dict = {
        "model": model,
        "messages": dispatch_msgs,
        "temperature": 0.0,
    }
    apply_dispatch_constraints(payload, tools)
    return payload, dispatch_msgs


def _build_args_payload(
    dispatch_msgs: list[dict],
    pass1_content: str,
    tools: list[dict] | None,
    names: list[str],
    model: str,
) -> dict:
    payload: dict = {
        "model": model,
        "messages": dispatch_msgs
        + [
            {"role": "assistant", "content": pass1_content},
            {"role": "user", "content": format_args_user_message(names)},
        ],
        "temperature": 0.0,
    }
    apply_args_constraints(payload, tools, names)
    return payload


def _combine_two_pass(
    data1: dict,
    data2: dict | None,
    content: str,
    n_calls: int,
) -> tuple[dict, int]:
    timings = _merge_timings(data1) if data2 is None else _merge_timings(data1, data2)
    return _assistant_response(content, timings), n_calls


# ---------------------------------------------------------------------------
# Two-pass completion (sync + async)
# ---------------------------------------------------------------------------

def complete_step(
    messages: list[dict],
    tools: list[dict] | None,
    *,
    post: SyncPost,
    model: str,
) -> tuple[dict, int]:
    """Run TagDispatch two-pass CD. ``post(payload) -> chat-completions JSON``.

    Returns ``(synthetic_response, n_api_calls)``. The synthetic assistant
    ``content`` is IFFC JSON (``[]`` or a list of ``function_name``/``parameters``).
    Intermediate pass-1 text is **not** written to *messages*.
    """
    payload1, dispatch_msgs = _build_dispatch_payload(messages, tools, model)
    data1 = post(payload1)
    content1 = _content_from_response(data1)
    names = _selected_names(try_parse_iffc_response(content1))
    if names is None:
        return _combine_two_pass(data1, None, content1, 1)
    if not names:
        return _combine_two_pass(data1, None, "[]", 1)

    payload2 = _build_args_payload(dispatch_msgs, content1, tools, names, model)
    data2 = post(payload2)
    content2 = _content_from_response(data2)
    parsed2 = try_parse_iffc_response(content2)
    if parsed2 is None:
        return _combine_two_pass(data1, data2, content2, 2)
    return _combine_two_pass(data1, data2, _finalize_calls_json(parsed2), 2)


async def complete_step_async(
    messages: list[dict],
    tools: list[dict] | None,
    *,
    post: AsyncPost,
    model: str,
) -> tuple[dict, int]:
    """Async counterpart of :func:`complete_step`."""
    payload1, dispatch_msgs = _build_dispatch_payload(messages, tools, model)
    data1 = await post(payload1)
    content1 = _content_from_response(data1)
    names = _selected_names(try_parse_iffc_response(content1))
    if names is None:
        return _combine_two_pass(data1, None, content1, 1)
    if not names:
        return _combine_two_pass(data1, None, "[]", 1)

    payload2 = _build_args_payload(dispatch_msgs, content1, tools, names, model)
    data2 = await post(payload2)
    content2 = _content_from_response(data2)
    parsed2 = try_parse_iffc_response(content2)
    if parsed2 is None:
        return _combine_two_pass(data1, data2, content2, 2)
    return _combine_two_pass(data1, data2, _finalize_calls_json(parsed2), 2)
