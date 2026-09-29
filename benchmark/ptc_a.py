"""PTC-A: IFFC JSON-in-text plus a live tool-name DFA (arguments unconstrained).

``--fc-mode iffc-ptc-a`` is the isolated first ablation of PTC-Decoder
(Yu, Mu, & Wu, arXiv:2609.30836) on this harness: **TC-Decoder only**.

What it **is**
    One llama-server completion per decision step. Persistent transcript and
    parser stay IFFC (Appendix B JSON list of ``{"function_name","parameters"}``).
    The request attaches a GBNF ``grammar`` whose *name* alternatives are the
    *currently live* tool set (plus ``[]``). ``parameters`` is an unconstrained
    JSON object — any keys/values, including empty or extra fields.

What it **is not**
    * Not CD (``--fc-mode cd``): no pass-2 ``json_schema``, no TagDispatch
      two-request loop, no per-tool argument schemas.
    * Not CD pass-1 reused as-is: ``cd.build_name_dispatch_grammar`` forbids
      ``parameters`` and would silently drop args. PTC-A builds its own IFFC-
      shaped GBNF and only reuses name-only helpers (``tool_names``,
      ``grammar_allowed_names``, GBNF string literals).
    * Not PTC-B (Plan-to-Act stays off).
    * Not ``iffc-ooda-card`` (PR #61) and not ``iffc-stop-repeat``. First
      full-200 identity is BASE + PTC-A **solo**.
    * Not the default: ``config.FC_MODE`` stays ``nfc``.

llama-server: send ``grammar`` only. Never ``tools`` / ``tool_choice`` (NFC
tools-GBNF) and never ``json_schema`` (it overrides ``grammar``).
"""

from __future__ import annotations

import json
from typing import Any

from .cd import grammar_allowed_names, tool_names
from .cd import _gbnf_emit_json_string


# ---------------------------------------------------------------------------
# GBNF: live names constrained; parameters are free JSON objects
# ---------------------------------------------------------------------------

_UNCONSTRAINED_JSON_GBNF = r"""
json-value  ::= json-object | json-array | json-string | json-number | json-bool | json-null
json-object ::= "{" ws (json-member (ws "," ws json-member)*)? ws "}"
json-member ::= json-string ws ":" ws json-value
json-array  ::= "[" ws (json-value (ws "," ws json-value)*)? ws "]"
json-string ::= "\"" json-char* "\""
json-char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
json-number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
json-bool   ::= "true" | "false"
json-null   ::= "null"
""".lstrip()


def build_name_dfa_grammar(tools: list[dict] | None) -> str:
    """IFFC-shaped GBNF: ``[]`` or calls whose ``function_name`` is live.

    ``parameters`` is a generic JSON object (no per-tool required keys).
    """
    names = tool_names(tools)
    lines = [
        r"root ::= ws (empty | calls) ws",
        r"ws ::= [ \t\n]*",
        r'empty ::= "[" ws "]"',
    ]
    if not names:
        return "\n".join(lines) + "\n"
    alts = " | ".join(_gbnf_emit_json_string(n) for n in names)
    lines.extend(
        [
            r'calls ::= "[" ws call (ws "," ws call)* ws "]"',
            r'call ::= "{" ws "\"function_name\"" ws ":" ws name ws ","'
            r' ws "\"parameters\"" ws ":" ws json-object ws "}"',
            f"name ::= {alts}",
            "",
            _UNCONSTRAINED_JSON_GBNF.rstrip(),
        ]
    )
    return "\n".join(lines) + "\n"


def grammar_constrains_argument_schemas(grammar: str) -> bool:
    """True if *grammar* looks like CD pass-2 (per-tool arg schemas).

    PTC-A must stay name-only: unconstrained ``json-object``, no required
    property names from live tool schemas.
    """
    if "json_schema" in grammar:
        return True
    # CD pass-1 has no parameters at all; CD pass-2 is json_schema not GBNF.
    # A name-DFA that baked in required keys would skip the unconstrained
    # json-object production.
    if "parameters" in grammar and "json-object" not in grammar:
        return True
    return False


def instance_matches_name_dfa(
    tools: list[dict] | None,
    instance: Any,
) -> bool:
    """Hermetic accept/reject for the PTC-A name DFA (no llama-server).

    Accepts ``[]`` and IFFC objects whose ``function_name`` is in the live
    set. ``parameters`` may be any JSON value (typically an object); extra
    keys inside parameters are allowed. Out-of-set names are rejected.
    """
    names = set(tool_names(tools))
    if not isinstance(instance, list):
        return False
    if instance == []:
        return True
    for item in instance:
        if not isinstance(item, dict):
            return False
        if set(item.keys()) != {"function_name", "parameters"}:
            return False
        name = item.get("function_name")
        if not isinstance(name, str) or name not in names:
            return False
        # parameters unconstrained — any JSON-serializable value is fine
        try:
            json.dumps(item["parameters"])
        except (TypeError, ValueError):
            return False
    return True


def text_matches_name_dfa(tools: list[dict] | None, text: str) -> bool:
    """Parse assistant text as JSON and apply :func:`instance_matches_name_dfa`."""
    try:
        instance = json.loads(text)
    except json.JSONDecodeError:
        return False
    return instance_matches_name_dfa(tools, instance)


# ---------------------------------------------------------------------------
# llama-server payload (never tools / tool_choice / json_schema)
# ---------------------------------------------------------------------------

def apply_name_dfa_constraints(payload: dict, tools: list[dict] | None) -> dict:
    """Attach name-only GBNF. Strip NFC tools and CD json_schema if present."""
    payload["grammar"] = build_name_dfa_grammar(tools)
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload.pop("json_schema", None)
    return payload


def assert_ptc_a_payload(payload: dict, tools: list[dict] | None = None) -> None:
    """Raise ``AssertionError`` if the request is NFC tools-GBNF or CD pass-2."""
    if "tools" in payload or "tool_choice" in payload:
        raise AssertionError(
            "PTC-A must not send tools/tool_choice (that would activate NFC GBNF)"
        )
    if "json_schema" in payload:
        raise AssertionError(
            "PTC-A must not send json_schema (that would constrain arguments / "
            "override grammar and become CD pass-2)"
        )
    grammar = payload.get("grammar")
    if not grammar:
        raise AssertionError("PTC-A requires a name-DFA grammar")
    if grammar_constrains_argument_schemas(grammar):
        raise AssertionError("PTC-A grammar must not constrain argument schemas")
    allowed = grammar_allowed_names(grammar)
    live = tool_names(tools) if tools is not None else None
    if live is not None and allowed != live:
        raise AssertionError(
            f"PTC-A grammar names {allowed!r} != live tools {live!r}"
        )
    if live:
        if "parameters" not in grammar:
            raise AssertionError("PTC-A grammar must allow IFFC parameters")
        if "json-object" not in grammar:
            raise AssertionError(
                "PTC-A grammar must leave parameters as unconstrained JSON"
            )
