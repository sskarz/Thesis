"""Convert ToolACE + xLAM rows into IFFC chat JSONL for mlx-lm.

Target shape matches ``benchmark.iffc`` / ``--fc-mode iffc``:

* system: Appendix B prompt with available function schemas
* assistant: JSON list of ``{"function_name", "parameters"}`` (or ``[]``)
* tool results: user-role observation via ``format_iffc_execution_user_message``

Native ``tool_calls`` are never written into the adapter dataset.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from benchmark import config as bench_config
from benchmark.generate_responses import bfcl_to_openai_tools
from benchmark.iffc import (
    build_iffc_system_prompt,
    format_iffc_execution_user_message,
    iffc_calls_to_func_strings,
    parse_iffc_response,
)

from sft.config import (
    CHAT_FORMAT,
    FORMATTED_DIR,
    FORMATTED_TOOLACE,
    FORMATTED_XLAM,
    RAW_TOOLACE,
    RAW_XLAM,
    SOURCE_TOOLACE,
    SOURCE_XLAM,
)
from sft.io import read_jsonl, write_jsonl

_CALL_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


def dumps_iffc_calls(calls: list[dict]) -> str:
    """Serialize IFFC call dicts the way the harness parser expects."""
    normalized = [
        {
            "function_name": c["function_name"],
            "parameters": c.get("parameters") or {},
        }
        for c in calls
    ]
    return json.dumps(normalized, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------

def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


_OPENAI_OBJECT_TYPES = frozenset({"object", "dict"})


def _is_openai_json_schema(parameters: dict[str, Any]) -> bool:
    """True when ``parameters`` already looks like JSON Schema, not xLAM named specs.

    xLAM rows sometimes put a nested schema dict in ``parameters["type"]``.
    Membership against a set of strings would raise ``TypeError: unhashable
    type: 'dict'``, so only string ``type`` values are checked.
    """
    if "properties" in parameters:
        return True
    raw_type = parameters.get("type")
    return isinstance(raw_type, str) and raw_type.lower() in _OPENAI_OBJECT_TYPES


def _xlam_style_params_to_openai(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert ``{name: {type, description, required}}`` into JSON Schema."""
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}
    if _is_openai_json_schema(parameters):
        return parameters

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            continue
        prop: dict[str, Any] = {}
        raw_type = spec.get("type")
        if isinstance(raw_type, str):
            mapped = bench_config.GORILLA_TO_OPENAPI.get(raw_type, raw_type)
            if mapped is not None:
                prop["type"] = mapped
        if spec.get("description"):
            prop["description"] = spec["description"]
        if spec.get("default") not in (None, ""):
            prop["default"] = spec["default"]
        properties[name] = prop
        if spec.get("required") is True:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def normalize_function_docs(tools_or_docs: list[Any] | None) -> list[dict]:
    """Return OpenAI ``tools`` entries for ``build_iffc_system_prompt``."""
    docs: list[dict] = []
    for item in tools_or_docs or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if (
            isinstance(item_type, str)
            and item_type.lower() == "function"
            and isinstance(item.get("function"), dict)
        ):
            func = dict(item["function"])
        else:
            func = dict(item)
        name = func.get("name")
        if not name:
            continue
        params = func.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        docs.append(
            {
                "name": name,
                "description": func.get("description", "") or "",
                "parameters": _xlam_style_params_to_openai(params),
            }
        )
    return bfcl_to_openai_tools(docs)


def extract_json_array_of_objects(text: str) -> list[dict]:
    """Pull the first JSON list of objects (tool docs) out of a system string."""
    if not text:
        return []
    decoder = json.JSONDecoder()
    i = 0
    while True:
        i = text.find("[", i)
        if i < 0:
            return []
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if (
            isinstance(obj, list)
            and obj
            and all(isinstance(x, dict) for x in obj)
            and any("name" in x for x in obj)
        ):
            return obj
        i = max(end, i + 1)


# ---------------------------------------------------------------------------
# ToolACE call parser: ``[Name(arg=val, ...), Other()]``
# ---------------------------------------------------------------------------

def _matching_paren(s: str, open_idx: int) -> int:
    depth = 0
    in_str: str | None = None
    escape = False
    for i in range(open_idx, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in {'"', "'"}:
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced parentheses")


def _eval_ast(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        if isinstance(node, ast.Name):
            return node.id
        raise


def parse_kwarg_string(arg_src: str) -> dict[str, Any]:
    src = arg_src.strip()
    if not src:
        return {}
    tree = ast.parse(f"f({src})", mode="eval")
    call = tree.body
    if not isinstance(call, ast.Call):
        raise ValueError(f"not a call arg list: {src!r}")
    params: dict[str, Any] = {}
    for i, arg in enumerate(call.args):
        params[f"arg{i}"] = _eval_ast(arg)
    for kw in call.keywords:
        if kw.arg is None:
            continue
        params[kw.arg] = _eval_ast(kw.value)
    return params


def parse_toolace_call_list(text: str) -> list[dict] | None:
    """Parse a ToolACE ``[Func(k=v), ...]`` blob into IFFC call dicts.

    Returns:
        * a (possibly empty) list of calls when the text is a well-formed
          bracket list or plain language (plain language → ``[]``)
        * ``None`` when the text looks like a call list but cannot be parsed
    """
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []

    blob = raw
    if not (blob.startswith("[") and blob.endswith("]")):
        match = _CALL_LIST_RE.search(blob)
        if match is None:
            if looks_like_toolace_call_list(blob):
                return None  # truncated / unbalanced call list
            return []  # natural-language / clarifying turn
        blob = match.group(0)
        remainder = (raw[: match.start()] + raw[match.end() :]).strip()
        if remainder and "(" not in blob:
            return []

    inner = blob[1:-1].strip()
    if not inner:
        return []
    if "(" not in inner:
        return []

    calls: list[dict] = []
    i = 0
    try:
        while i < len(inner):
            while i < len(inner) and inner[i] in " \t\n\r,":
                i += 1
            if i >= len(inner):
                break
            paren = inner.find("(", i)
            if paren < 0:
                return None
            name = inner[i:paren].strip()
            if not name:
                return None
            close = _matching_paren(inner, paren)
            params = parse_kwarg_string(inner[paren + 1 : close])
            calls.append({"function_name": name, "parameters": params})
            i = close + 1
    except (ValueError, SyntaxError, TypeError):
        return None
    return calls


def looks_like_toolace_call_list(text: str) -> bool:
    raw = (text or "").strip()
    return bool(raw.startswith("[") and "(" in raw)


# ---------------------------------------------------------------------------
# ToolACE / xLAM → IFFC records
# ---------------------------------------------------------------------------

def n_tools_bin(n_tools_in_gold: int) -> str:
    return "ge2" if n_tools_in_gold >= 2 else "1"


def _count_gold_tools(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        total += len(parse_iffc_response(str(msg.get("content") or "")))
    return total


def format_toolace_row(row: dict, row_index: int) -> dict | None:
    """Convert one Team-ACE/ToolACE row to an IFFC chat example."""
    system_text = row.get("system") or ""
    conversations = row.get("conversations") or []
    conversations = _maybe_json(conversations)
    if not isinstance(conversations, list) or not conversations:
        return None

    tools = extract_json_array_of_objects(system_text)
    openai_tools = normalize_function_docs(tools)
    messages: list[dict] = [
        {"role": "system", "content": build_iffc_system_prompt(openai_tools)}
    ]

    prev_calls: list[dict] = []
    saw_assistant = False
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("from") or turn.get("role") or "").lower()
        value = turn.get("value", turn.get("content", ""))
        if value is None:
            value = ""
        value = str(value)

        if role in {"human", "user"}:
            messages.append({"role": "user", "content": value})
            prev_calls = []
        elif role in {"gpt", "assistant"}:
            calls = parse_toolace_call_list(value)
            if calls is None:
                if looks_like_toolace_call_list(value):
                    return None  # malformed gold call list
                calls = []
            messages.append({"role": "assistant", "content": dumps_iffc_calls(calls)})
            prev_calls = calls
            saw_assistant = True
        elif role in {"tool", "function", "observation"}:
            messages.append(
                {
                    "role": "user",
                    "content": _format_toolace_observation(value, prev_calls),
                }
            )
        else:
            # Unknown role — keep as user context rather than dropping the dialog.
            messages.append({"role": "user", "content": value})

    if not saw_assistant:
        return None
    if any("tool_calls" in m for m in messages):
        raise RuntimeError("native tool_calls leaked into ToolACE IFFC row")

    n_gold = _count_gold_tools(messages)
    example_id = f"toolace-{row.get('row_index', row_index)}"
    return {
        "id": example_id,
        "source": SOURCE_TOOLACE,
        "is_multi_turn": True,
        "n_tools_in_gold": n_gold,
        "n_tools_bin": n_tools_bin(n_gold),
        "chat_format": CHAT_FORMAT,
        "messages": messages,
    }


def _format_toolace_observation(value: str, prev_calls: list[dict]) -> str:
    parsed = _maybe_json(value)
    func_strings: list[str] = []
    exec_results: list[Any] = []

    if isinstance(parsed, list) and parsed:
        prev_by_name: dict[str, list[str]] = {}
        for s in iffc_calls_to_func_strings(prev_calls):
            name = s.split("(", 1)[0]
            prev_by_name.setdefault(name, []).append(s)

        for item in parsed:
            if isinstance(item, dict):
                name = str(item.get("name") or "unknown")
                result = item.get("results", item.get("result", item))
                matched = prev_by_name.get(name) or []
                func_strings.append(matched.pop(0) if matched else f"{name}()")
                exec_results.append(result)
            else:
                func_strings.append("unknown()")
                exec_results.append(item)
    else:
        if prev_calls:
            func_strings = iffc_calls_to_func_strings(prev_calls)
            exec_results = [parsed] * len(func_strings)
        else:
            func_strings = ["(unparsed_tool_result)"]
            exec_results = [parsed]

    return format_iffc_execution_user_message(func_strings, exec_results)


def format_xlam_row(row: dict, row_index: int) -> dict | None:
    """Convert one Salesforce/xlam-function-calling-60k row to IFFC chat."""
    query = row.get("query")
    if not query:
        return None
    tools = _maybe_json(row.get("tools") or [])
    answers = _maybe_json(row.get("answers") or [])
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list):
        tools = []
    if isinstance(answers, dict):
        answers = [answers]
    if not isinstance(answers, list):
        return None

    openai_tools = normalize_function_docs(tools)
    calls: list[dict] = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        name = ans.get("name") or ans.get("function_name")
        if not name:
            continue
        params = ans.get("arguments", ans.get("parameters", {}))
        if isinstance(params, str):
            params = _maybe_json(params)
        if not isinstance(params, dict):
            params = {}
        calls.append({"function_name": str(name), "parameters": params})

    messages = [
        {"role": "system", "content": build_iffc_system_prompt(openai_tools)},
        {"role": "user", "content": str(query)},
        {"role": "assistant", "content": dumps_iffc_calls(calls)},
    ]
    if any("tool_calls" in m for m in messages):
        raise RuntimeError("native tool_calls leaked into xLAM IFFC row")

    raw_id = row.get("id", row.get("row_index", row_index))
    example_id = f"xlam-{raw_id}"
    n_gold = len(calls)
    return {
        "id": example_id,
        "source": SOURCE_XLAM,
        "is_multi_turn": False,
        "n_tools_in_gold": n_gold,
        "n_tools_bin": n_tools_bin(n_gold),
        "chat_format": CHAT_FORMAT,
        "messages": messages,
    }


def format_rows(
    rows: list[dict],
    source: str,
) -> tuple[list[dict], int]:
    """Format a list of raw rows. Returns ``(records, n_skipped)``."""
    out: list[dict] = []
    skipped = 0
    formatter = format_toolace_row if source == SOURCE_TOOLACE else format_xlam_row
    for i, row in enumerate(rows):
        rec = formatter(row, i)
        if rec is None:
            skipped += 1
            continue
        out.append(rec)
    return out, skipped


def unroll_conversation(record: dict) -> list[dict]:
    """Emit one mlx-lm chat example per assistant turn.

    ``mlx_lm.lora --mask-prompt`` only trains on the **final** message. Unrolling
    so every example ends on an assistant turn is how multi-turn IFFC targets
    actually receive loss. Split membership stays at conversation id.

    This helper is unfiltered. ``sft.train_mlx.prepare_mlx_data`` runs
    ``sft.filter_empty_after_exec`` (SFT v3 by default): drops **all**
    empty-after-exec ``[]`` targets (terminal included) and may insert a
    history-only ``[]`` on the next user→call unroll so recovery is trained
    without making ``[]`` itself a frequent target.
    """
    messages = record.get("messages") or []
    rows: list[dict] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix = messages[: i + 1]
        if not prefix or prefix[-1].get("role") != "assistant":
            continue
        rows.append(
            {
                "id": f"{record['id']}#asst-{i}",
                "parent_id": record["id"],
                "source": record.get("source"),
                "is_multi_turn": record.get("is_multi_turn"),
                "n_tools_in_gold": record.get("n_tools_in_gold"),
                "n_tools_bin": record.get("n_tools_bin"),
                "chat_format": CHAT_FORMAT,
                "messages": prefix,
            }
        )
    return rows


def unroll_records(records: list[dict]) -> list[dict]:
    """Unroll every assistant turn with **no** empty-after-exec filter.

    Training data should go through ``sft.filter_empty_after_exec.unroll_and_filter``
    (used by ``sft.train_mlx.prepare_mlx_data``; SFT v3 default).
    """
    out: list[dict] = []
    for rec in records:
        out.extend(unroll_conversation(rec))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Format ToolACE + xLAM raw JSONL into IFFC chat JSONL."
    )
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args(argv)

    raw_toolace = RAW_TOOLACE
    raw_xlam = RAW_XLAM
    out_toolace = FORMATTED_TOOLACE
    out_xlam = FORMATTED_XLAM
    if args.raw_dir:
        raw_root = Path(args.raw_dir)
        raw_toolace = raw_root / "toolace.jsonl"
        raw_xlam = raw_root / "xlam.jsonl"
    if args.out_dir:
        out_root = Path(args.out_dir)
        out_toolace = out_root / "toolace.jsonl"
        out_xlam = out_root / "xlam.jsonl"
    else:
        FORMATTED_DIR.mkdir(parents=True, exist_ok=True)

    if not raw_toolace.exists() and not raw_xlam.exists():
        parser.error(
            f"No raw files at {raw_toolace} or {raw_xlam}. "
            "Run: uv run python -m sft.download_datasets"
        )

    if raw_toolace.exists():
        recs, skipped = format_rows(read_jsonl(raw_toolace), SOURCE_TOOLACE)
        write_jsonl(out_toolace, recs)
        print(f"ToolACE: wrote {len(recs)} → {out_toolace} (skipped {skipped})")
    else:
        print(f"skip ToolACE (missing {raw_toolace})")

    if raw_xlam.exists():
        recs, skipped = format_rows(read_jsonl(raw_xlam), SOURCE_XLAM)
        write_jsonl(out_xlam, recs)
        print(f"xLAM: wrote {len(recs)} → {out_xlam} (skipped {skipped})")
    else:
        print(f"skip xLAM (missing {raw_xlam})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
