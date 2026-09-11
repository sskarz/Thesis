"""Allowlisted parser: command text → ToolCall or ParseError.

Deny-by-default. The model may wrap a command in markdown or a thought line;
we still only execute a single allowlisted argv.
"""

from __future__ import annotations

import re
import shlex

from .dialect import (
    ALLOWED_COMMANDS,
    ALLOWED_FLAGS,
    DONE_ALIASES,
    VALUED_OPTIONS,
)
from .types import ParseError, Redirect, ToolCall

_FENCE_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Anything that would reintroduce a real shell. Redirect tokens are stripped
# before this check.
_UNSAFE_RE = re.compile(
    r"(\$\(|`|<\(|>\(|\$\{|\|\||&&|;|\||(?<!>)&(?!>))"
)

_DENIED_VERBS = frozenset(
    {
        "python",
        "python3",
        "bash",
        "sh",
        "zsh",
        "curl",
        "wget",
        "chmod",
        "chown",
        "sudo",
        "env",
        "xargs",
        "awk",
        "sed",
        "perl",
        "node",
        "tar",
        "exec",
    }
)


def parse_command(text: str) -> ToolCall | ParseError:
    """Parse model output into one structured call."""
    raw = text if text is not None else ""
    line = _extract_command_line(raw)
    if line is None:
        return ParseError("no command found in model output", raw=raw, kind="parse")

    if _UNSAFE_RE.search(line):
        return ParseError(
            "unsafe shell syntax (pipes, &&, ;, or substitution are not allowed)",
            raw=raw,
            kind="unsafe",
        )

    try:
        tokens = shlex.split(line, posix=True)
    except ValueError as exc:
        return ParseError(f"could not tokenize: {exc}", raw=raw, kind="parse")

    if not tokens:
        return ParseError("empty command", raw=raw, kind="parse")

    tokens, redirect, redirect_err = _split_redirect(tokens)
    if redirect_err:
        return ParseError(redirect_err, raw=raw, kind="parse")
    if not tokens:
        return ParseError("redirect without a command", raw=raw, kind="parse")

    verb = tokens[0]
    if verb in DONE_ALIASES:
        if len(tokens) > 1 or redirect is not None:
            return ParseError("done takes no arguments", raw=raw, kind="parse")
        return ToolCall(name="done", raw=line)

    if verb in _DENIED_VERBS:
        return ParseError(
            f"command {verb!r} is not in the allowlist",
            raw=raw,
            kind="deny",
        )
    if verb not in ALLOWED_COMMANDS:
        return ParseError(
            f"unknown command {verb!r}; not in the allowlist",
            raw=raw,
            kind="deny",
        )

    flags: set[str] = set()
    options: list[tuple[str, str]] = []
    positional: list[str] = []
    allowed_flags = ALLOWED_FLAGS.get(verb, frozenset())
    valued = VALUED_OPTIONS.get(verb, frozenset())
    rest = tokens[1:]
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--":
            positional.extend(rest[i + 1 :])
            break
        if tok.startswith("-") and len(tok) > 1 and not tok.startswith("---"):
            name = tok.lstrip("-")
            # Combined short flags: -la → l, a  (not for valued options)
            if tok.startswith("--"):
                opt = name
                if opt in valued:
                    if i + 1 >= len(rest):
                        return ParseError(f"{verb} -{opt} requires a value", raw=raw)
                    options.append((opt, rest[i + 1]))
                    i += 2
                    continue
                return ParseError(f"unknown flag {tok} for {verb}", raw=raw, kind="deny")
            if len(name) == 1 and name in valued:
                if i + 1 >= len(rest):
                    return ParseError(f"{verb} -{name} requires a value", raw=raw)
                options.append((name, rest[i + 1]))
                i += 2
                continue
            if all(ch in allowed_flags for ch in name):
                flags.update(name)
                i += 1
                continue
            # find -name / -type use a single dash but are valued options
            if name in valued:
                if i + 1 >= len(rest):
                    return ParseError(f"{verb} -{name} requires a value", raw=raw)
                options.append((name, rest[i + 1]))
                i += 2
                continue
            return ParseError(f"unknown flag {tok} for {verb}", raw=raw, kind="deny")
        positional.append(tok)
        i += 1

    arg_err = _validate_argv(verb, flags, options, positional)
    if arg_err:
        return ParseError(arg_err, raw=raw, kind="parse")

    return ToolCall(
        name=verb,
        flags=frozenset(flags),
        options=tuple(options),
        positional=tuple(positional),
        redirect=redirect,
        raw=line,
    )


def to_argv(call: ToolCall) -> list[str]:
    """Reconstruct a host argv list (no shell interpolation).

    Harbor / Terminus adapters should exec this list directly.
    Redirect is *not* included; the adapter applies it after capturing stdout.
    """
    argv = [call.name]
    for flag in sorted(call.flags):
        argv.append(f"-{flag}")
    # Positionals before valued options so `find /dst -name x` round-trips
    # in the usual Unix order (Harbor exec argv, no shell).
    argv.extend(call.positional)
    for key, value in call.options:
        argv.extend([f"-{key}", value])
    return argv


def _extract_command_line(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    lines: list[str] = []
    for raw_line in text.splitlines():
        ln = raw_line.strip()
        if not ln:
            continue
        if ln.startswith("$ "):
            ln = ln[2:].strip()
        elif ln.startswith("> ") and not ln.startswith(">>"):
            # Prompt-style prefix, not a redirect (redirects live inside the command).
            ln = ln[2:].strip()
        lines.append(ln)
    if not lines:
        return None
    for ln in reversed(lines):
        verb = ln.split(None, 1)[0] if ln.split() else ""
        verb = verb.lstrip("$")
        if verb in ALLOWED_COMMANDS or verb in DONE_ALIASES:
            return ln
    return lines[-1]


def _split_redirect(
    tokens: list[str],
) -> tuple[list[str], Redirect | None, str | None]:
    """Peel a trailing  > path  or  >> path  (or concatenated >path)."""
    if not tokens:
        return tokens, None, None

    # Concatenated form: echo hi >/notes/a.txt
    last = tokens[-1]
    if last.startswith(">>") and last != ">>":
        return tokens[:-1], Redirect(">>", last[2:]), None
    if last.startswith(">") and last != ">" and not last.startswith(">>"):
        return tokens[:-1], Redirect(">", last[1:]), None

    if last in {">", ">>"}:
        return tokens[:-1], None, "redirect is missing a path"

    if len(tokens) >= 2 and tokens[-2] in {">", ">>"}:
        op: str = tokens[-2]
        path = tokens[-1]
        if path in {">", ">>"}:
            return tokens, None, "invalid redirect path"
        return tokens[:-2], Redirect(">" if op == ">" else ">>", path), None

    # Reject redirects buried in the middle; the dialect is suffix-only.
    for tok in tokens[:-1]:
        if tok in {">", ">>"} or (tok.startswith(">") and tok != ">>"):
            return tokens, None, "redirect must be the last tokens of the command"
    return tokens, None, None


def _validate_argv(
    verb: str,
    flags: set[str],
    options: list[tuple[str, str]],
    positional: list[str],
) -> str | None:
    if verb == "pwd" and positional:
        return "pwd takes no arguments"
    if verb == "cd" and len(positional) > 1:
        return "cd takes at most one path"
    if verb == "cat" and not positional:
        return "cat requires a path (no stdin in this dialect)"
    if verb == "mkdir" and not positional:
        return "mkdir requires a path"
    if verb in {"mv", "cp"} and len(positional) != 2:
        return f"{verb} requires source and destination"
    if verb == "rm" and not positional:
        return "rm requires a path"
    if verb == "rm" and any(p in {"/", "."} for p in positional) and "r" in flags:
        return "refusing to remove /"
    if verb == "grep":
        if len(positional) < 2:
            return "grep requires a pattern and a path"
        if len(positional) > 2:
            return "grep takes one pattern and one path"
    if verb == "wc" and not positional:
        return "wc requires a path (no stdin in this dialect)"
    if verb == "find":
        for key, value in options:
            if key == "type" and value not in {"f", "d"}:
                return "find -type must be f or d"
    return None
