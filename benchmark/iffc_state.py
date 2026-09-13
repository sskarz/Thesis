"""IFFC + harness-written STATE overlay (METHODS §5 / PLAN #6 working memory).

``--fc-mode iffc-state`` keeps the IFFC request path (no native ``tools`` /
``tool_choice``; JSON-in-text parse; same BFCL checker) and injects a compact
``[STATE]...[/STATE]`` snapshot built from **actual tool execution results**.

This is **not** GST (no 7-step curriculum, not on the NFC channel), **not**
LLM summarization, **not** CD/GBNF, and **not** shell→GFS.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
from typing import Any

from .iffc import (
    IFFC_RETRY_USER_MESSAGE,
    build_iffc_system_prompt,
    format_iffc_execution_user_message,
)

STATE_OPEN = "[STATE]"
STATE_CLOSE = "[/STATE]"
KEEP_RECENT_TURNS = 2

# Caps keep the block small enough for SLM context (especially IFFC's ~6k
# system prompt). Older raw dumps are dropped; this snapshot is the memory.
_MAX_FILES = 24
_MAX_FIELDS = 24
_MAX_LAST = 6
_MAX_LISTING = 12
_MAX_VALUE_CHARS = 80
_MAX_BLOCK_CHARS = 1200

_EXEC_DUMP_PREFIX = "Function execution results:"

# GFS / common path-like result keys from BFCL executable backends.
_CWD_KEYS = ("current_working_directory", "cwd", "working_directory", "pwd")
_LISTING_KEYS = ("current_directory_content", "matches")
_SKIP_FIELD_KEYS = frozenset(
    {
        *_CWD_KEYS,
        *_LISTING_KEYS,
        "file_content",
        "terminal_output",
        "matching_lines",
        "error",
        "log",
    }
)
_FILE_ARG_KEYS = (
    "file_name",
    "filename",
    "source",
    "destination",
    "dir_name",
    "path",
    "file",
)
_ERROR_RE = re.compile(r"^\s*(error|exception)\b", re.IGNORECASE)

STATE_ADDENDUM = (
    f"A {STATE_OPEN}...{STATE_CLOSE} block in user messages is a verified "
    "snapshot of environment state taken from actual tool execution results "
    "(not model text). Trust STATE over older raw dumps that may have been "
    "compressed. Continue to answer only with a JSON list of dictionaries "
    'with "function_name" and "parameters", or [] if no further function '
    "calling is needed."
)


def build_iffc_state_system_prompt(tools_or_bfcl_docs: list[dict] | None) -> str:
    """Appendix B IFFC prompt plus a short STATE-trust addendum (no GST)."""
    return build_iffc_system_prompt(tools_or_bfcl_docs) + "\n\n" + STATE_ADDENDUM


def ensure_iffc_state_system_message(
    chat_history: list[dict],
    tools_or_bfcl_docs: list[dict] | None,
) -> None:
    """Insert or refresh the IFFC+STATE system message when tools change."""
    content = build_iffc_state_system_prompt(tools_or_bfcl_docs)
    if chat_history and chat_history[0].get("role") == "system":
        if chat_history[0].get("content") != content:
            chat_history[0] = {"role": "system", "content": content}
        return
    chat_history.insert(0, {"role": "system", "content": content})


def parse_func_string(func_string: str) -> tuple[str, dict[str, Any]]:
    """Parse a BFCL ``name(arg=val, ...)`` string into ``(name, kwargs)``."""
    text = (func_string or "").strip()
    if not text:
        return "unknown", {}
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        name, _, _ = text.partition("(")
        return name or text, {}
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            name = text.split("(", 1)[0]
        args: dict[str, Any] = {}
        for kw in node.keywords:
            if not kw.arg:
                continue
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                args[kw.arg] = ast.unparse(kw.value)
        return name, args
    name, _, _ = text.partition("(")
    return name or text, {}


def coerce_exec_result(result: Any) -> Any:
    """Turn an execution observation into dict/list/scalar when possible."""
    if result is None or isinstance(result, (dict, list, int, float, bool)):
        return result
    text = str(result).strip()
    if not text or text == "None":
        return None
    for loader in (ast.literal_eval, json.loads):
        try:
            return loader(text)
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return text


def is_error_result(result: Any) -> bool:
    parsed = coerce_exec_result(result)
    if isinstance(parsed, dict) and parsed.get("error") not in (None, "", False):
        return True
    if isinstance(parsed, str) and _ERROR_RE.search(parsed):
        return True
    return False


class StateTracker:
    """Accumulates verified env facts from executed calls + observations."""

    def __init__(self) -> None:
        self.cwd: str | None = None
        self.files: dict[str, str] = {}
        self.listing: list[str] = []
        self.fields: dict[str, str] = {}
        self.last: list[str] = []
        self.last_error: str | None = None

    def update(self, func_strings: list[str], exec_results: list[Any]) -> None:
        for call, raw in zip(func_strings, exec_results):
            name, args = parse_func_string(call)
            parsed = coerce_exec_result(raw)
            error = is_error_result(raw)
            self._record_last(name, args, parsed, error)
            if error:
                self.last_error = _compact_value(parsed)
                continue
            self.last_error = None
            self._apply_cwd(name, args, parsed)
            self._apply_files(name, args, parsed)
            self._apply_fields(parsed)

    def render(self) -> str:
        lines = [STATE_OPEN]
        if self.cwd:
            lines.append(f"cwd: {self.cwd}")
        present = [n for n, st in self.files.items() if st != "removed"]
        if present:
            lines.append("files: " + ", ".join(present[:_MAX_FILES]))
        if self.listing:
            lines.append("listing: " + ", ".join(self.listing[:_MAX_LISTING]))
        if self.fields:
            pairs = [f"{k}={v}" for k, v in list(self.fields.items())[:_MAX_FIELDS]]
            lines.append("fields: " + "; ".join(pairs))
        if self.last:
            lines.append("last: " + "; ".join(self.last[-_MAX_LAST:]))
        if self.last_error:
            lines.append(f"error: {self.last_error}")
        if len(lines) == 1:
            lines.append("(empty)")
        lines.append(STATE_CLOSE)
        block = "\n".join(lines)
        if len(block) > _MAX_BLOCK_CHARS:
            block = block[: _MAX_BLOCK_CHARS - 20].rstrip() + "\n" + STATE_CLOSE
        return block

    def _record_last(
        self,
        name: str,
        args: dict[str, Any],
        parsed: Any,
        error: bool,
    ) -> None:
        outcome = "error" if error else _last_outcome(parsed)
        arg_hint = _brief_args(args)
        piece = f"{name}({arg_hint})->{outcome}" if arg_hint else f"{name}()->{outcome}"
        self.last.append(_compact_value(piece, limit=100))
        if len(self.last) > _MAX_LAST * 2:
            self.last = self.last[-_MAX_LAST:]

    def _apply_cwd(self, name: str, args: dict[str, Any], parsed: Any) -> None:
        cwd_from_result = _extract_cwd(parsed)
        if cwd_from_result:
            self.cwd = _join_cwd(self.cwd, cwd_from_result)
            return
        if name != "cd":
            if name == "pwd" and isinstance(parsed, str) and parsed.strip():
                self.cwd = _join_cwd(self.cwd, parsed.strip())
            return
        folder = args.get("folder") or args.get("path")
        if not isinstance(folder, str) or not folder:
            return
        folder = folder.rstrip("/") or "/"
        if folder == ".":
            return
        if folder == "..":
            if self.cwd:
                parent = posixpath.dirname(self.cwd.rstrip("/")) or "/"
                self.cwd = parent
            return
        self.cwd = _join_cwd(self.cwd, folder)

    def _apply_files(self, name: str, args: dict[str, Any], parsed: Any) -> None:
        listing = _extract_listing(parsed)
        if listing:
            self.listing = listing[:_MAX_LISTING]
            for item in listing:
                self.files[item] = "listed"
            self._trim_files()

        if name in {"mkdir", "touch"}:
            target = args.get("dir_name") or args.get("file_name")
            if isinstance(target, str) and target:
                self.files[target] = "created"
        elif name == "echo":
            target = args.get("file_name")
            if isinstance(target, str) and target:
                self.files[target] = "written"
        elif name in {"cat", "grep", "wc"}:
            target = args.get("file_name") or args.get("filename")
            if isinstance(target, str) and target:
                self.files.setdefault(target, "touched")
        elif name == "rm":
            target = args.get("file_name")
            if isinstance(target, str) and target:
                self.files[target] = "removed"
        elif name == "rmdir":
            target = args.get("dir_name")
            if isinstance(target, str) and target:
                self.files[target] = "removed"
        elif name in {"mv", "cp"}:
            src = args.get("source")
            dest = args.get("destination")
            if isinstance(src, str) and src and name == "mv":
                self.files[src] = "removed"
            if isinstance(dest, str) and dest:
                self.files[dest] = "created"
        else:
            for key in _FILE_ARG_KEYS:
                val = args.get(key)
                if (
                    isinstance(val, str)
                    and val
                    and "/" not in val
                    and val not in {".", ".."}
                ):
                    self.files.setdefault(val, "touched")
        self._trim_files()

    def _apply_fields(self, parsed: Any) -> None:
        if not isinstance(parsed, dict):
            return
        for key, value in parsed.items():
            if not isinstance(key, str) or key.startswith("_"):
                continue
            if key in _SKIP_FIELD_KEYS:
                continue
            compact = _compact_value(value)
            if self.fields.get(key) != compact:
                self.fields[key] = compact
        while len(self.fields) > _MAX_FIELDS:
            self.fields.pop(next(iter(self.fields)))

    def _trim_files(self) -> None:
        while len(self.files) > _MAX_FILES:
            self.files.pop(next(iter(self.files)))


def attach_state_block(execution_message: str, tracker: StateTracker) -> str:
    """Insert the STATE snapshot into an IFFC execution user message."""
    block = tracker.render()
    marker = 'Respond with a JSON list of function calls'
    idx = execution_message.find(marker)
    if idx >= 0:
        return (
            execution_message[:idx].rstrip()
            + "\n\n"
            + block
            + "\n\n"
            + execution_message[idx:]
        )
    return execution_message.rstrip() + "\n\n" + block


def record_and_format(
    tracker: StateTracker,
    func_strings: list[str],
    exec_results: list[Any],
) -> str:
    """Update STATE from execution, then format the IFFC user observation + STATE."""
    tracker.update(func_strings, exec_results)
    base = format_iffc_execution_user_message(func_strings, exec_results)
    return attach_state_block(base, tracker)


def is_execution_dump(message: dict) -> bool:
    """True for IFFC tool-result user messages (raw dumps, optionally with STATE)."""
    if message.get("role") != "user":
        return False
    content = message.get("content") or ""
    if not isinstance(content, str):
        return False
    if content.startswith(_EXEC_DUMP_PREFIX) or _EXEC_DUMP_PREFIX in content:
        return True
    return STATE_OPEN in content and STATE_CLOSE in content


def is_user_turn_start(message: dict) -> bool:
    """True for a BFCL user question (not execution dumps or IFFC retries)."""
    if message.get("role") != "user":
        return False
    content = message.get("content") or ""
    if not isinstance(content, str):
        return False
    if content == IFFC_RETRY_USER_MESSAGE:
        return False
    if is_execution_dump(message):
        return False
    return True


def compress_older_dumps(
    chat_history: list[dict],
    keep_recent: int = KEEP_RECENT_TURNS,
) -> None:
    """Drop raw execution dumps from turns older than the last *keep_recent*.

    Last ~2 user/assistant+tool turns stay verbatim. Older dumps are replaced
    by the accumulated STATE snapshot (already on the latest observation).
    """
    starts = [i for i, msg in enumerate(chat_history) if is_user_turn_start(msg)]
    if len(starts) <= keep_recent:
        return
    cutoff = starts[-keep_recent]
    chat_history[:] = [
        msg
        for i, msg in enumerate(chat_history)
        if not (i < cutoff and is_execution_dump(msg))
    ]


def iffc_state_payload_omits_tools(payload: dict) -> bool:
    """True when the chat payload stays on the IFFC (no native tools) channel."""
    return "tools" not in payload and "tool_choice" not in payload


def iffc_state_payload_has_state(payload: dict) -> bool:
    """True when some message carries a harness-written STATE block."""
    for message in payload.get("messages") or []:
        text = message.get("content") or ""
        if isinstance(text, str) and STATE_OPEN in text and STATE_CLOSE in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _compact_value(value: Any, limit: int = _MAX_VALUE_CHARS) -> str:
    if value is None:
        return "ok"
    if isinstance(value, dict):
        if "error" in value:
            value = value["error"]
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            value = ",".join(f"{k}={_compact_value(v, 24)}" for k, v in list(value.items())[:4])
    elif isinstance(value, list):
        value = ",".join(_compact_value(v, 24) for v in value[:8])
    text = str(value).replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _brief_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts = []
    for key, val in list(args.items())[:3]:
        parts.append(f"{key}={_compact_value(val, 32)}")
    return ", ".join(parts)


def _last_outcome(parsed: Any) -> str:
    if parsed is None:
        return "ok"
    if isinstance(parsed, dict):
        if "error" in parsed:
            return f"error:{_compact_value(parsed.get('error'), 40)}"
        if "current_working_directory" in parsed:
            return f"cwd={parsed['current_working_directory']}"
        if "current_directory_content" in parsed:
            items = parsed["current_directory_content"]
            if isinstance(items, list):
                return "[" + ",".join(str(x) for x in items[:6]) + "]"
        keys = [k for k in parsed if not str(k).startswith("_")][:4]
        if keys:
            return ",".join(f"{k}={_compact_value(parsed[k], 20)}" for k in keys)
        return "ok"
    return _compact_value(parsed, 40)


def _extract_cwd(parsed: Any) -> str | None:
    if isinstance(parsed, dict):
        for key in _CWD_KEYS:
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    if isinstance(parsed, str):
        match = re.search(
            r"(?:current(?: working)? directory|cwd)\s*[:=]\s*(\S+)",
            parsed,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(".,;\"'")
    return None


def _extract_listing(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    for key in _LISTING_KEYS:
        val = parsed.get(key)
        if isinstance(val, list):
            return [str(x) for x in val if x is not None]
    return []


def _join_cwd(previous: str | None, incoming: str) -> str:
    incoming = incoming.strip()
    if not incoming:
        return previous or "/"
    if incoming.startswith("/"):
        return posixpath.normpath(incoming)
    # GFS ``cd`` often returns only the leaf name, not a full path.
    if previous:
        if incoming == posixpath.basename(previous.rstrip("/")):
            return previous
        return posixpath.normpath(posixpath.join(previous, incoming))
    return posixpath.normpath("/" + incoming.lstrip("/"))
