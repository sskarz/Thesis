"""Compact tool list for ``iffc-state-ooda-slim`` (R81).

Strict hoist: a BFCL class preamble ("This tool belongs to ...") is moved
under a ``Tool classes:`` header only when it is byte-identical for every
tool in that class's func-doc file. Schemas are serialized as JSON without
indentation; tool order is unchanged. Prompt text otherwise equals
``iffc-state-ooda``.
"""

from __future__ import annotations

import functools
import json
import re

from .iffc import _unwrap_function_schemas

_PRE_RE = re.compile(r"^(This tool belongs to .*?) Tool description: ", re.S)
_SEP = " Tool description: "


@functools.lru_cache(maxsize=1)
def class_preambles() -> tuple[dict[str, str], dict[str, str | None]]:
    """Return ``(name_to_class, class_to_hoistable_preamble_or_None)``."""
    from . import config
    from .generate_responses import load_jsonl
    from bfcl_eval.constants.executable_backend_config import (
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )

    name_to_class: dict[str, str] = {}
    hoist: dict[str, str | None] = {}
    for cls_name, doc_file in MULTI_TURN_FUNC_DOC_FILE_MAPPING.items():
        path = config.MULTI_TURN_FUNC_DOC_DIR / doc_file
        if not path.exists():
            continue
        pres = []
        for func in load_jsonl(path):
            name_to_class.setdefault(func["name"], cls_name)
            m = _PRE_RE.match(func.get("description") or "")
            pres.append(m.group(1) if m else None)
        ok = bool(pres) and None not in pres and len(set(pres)) == 1
        hoist[cls_name] = pres[0] if ok else None
    return name_to_class, hoist


def format_available_functions_compact(tools_or_bfcl_docs: list[dict] | None) -> str:
    funcs = _unwrap_function_schemas(tools_or_bfcl_docs or [])
    name_to_class, hoist = class_preambles()
    head: list[str] = []
    out: list[dict] = []
    for f in funcs:
        g = json.loads(json.dumps(f))
        pre = hoist.get(name_to_class.get(g.get("name", ""), ""), None)
        desc = g.get("description")
        if pre is not None and isinstance(desc, str) and desc.startswith(pre + _SEP):
            g["description"] = desc[len(pre + _SEP):]
            if pre not in head:
                head.append(pre)
        out.append(g)
    if [f.get("name") for f in out] != [f.get("name") for f in funcs]:
        raise AssertionError("compact tool list changed tool order")
    body = json.dumps(out, ensure_ascii=False)
    if not head:
        return body
    hl = "\n".join("- " + re.sub(r"^This tool belongs to (the )?", "", h) for h in head)
    return f"Tool classes:\n{hl}\nFunctions (JSON):\n" + body


def expand_compact(available_functions: str) -> list[dict]:
    """Inverse of the compact format (for schema-preservation tests)."""
    if available_functions.startswith("Tool classes:\n"):
        head_txt, body = available_functions.split("\nFunctions (JSON):\n", 1)
    else:
        head_txt, body = "", available_functions
    funcs = json.loads(body)
    name_to_class, hoist = class_preambles()
    for g in funcs:
        pre = hoist.get(name_to_class.get(g.get("name", ""), ""), None)
        if pre is not None and ("- " + re.sub(r"^This tool belongs to (the )?", "", pre)) in head_txt.split("\n"):
            g["description"] = pre + _SEP + g["description"]
    return funcs
