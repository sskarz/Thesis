"""Map allowlisted shell ToolCalls → BFCL GorillaFileSystem func-call strings.

Primary next-eval seam for this experiment (not a Terminal-Bench path).
Does **not** import ``benchmark/`` or ``bfcl_eval``; strings match what
``generate_multi_turn._tool_calls_to_func_strings`` already feeds to
``execute_multi_turn_func_call``.

GorillaFileSystem constraints (from bfcl_eval gorilla_file_system.py):

- ``cd(folder)`` — one folder level; no nested paths
- ``ls(a=False)`` — current directory only
- ``cat`` / ``grep`` / ``wc`` / ``rm`` / ``mkdir`` / ``touch`` — local name,
  not a path
- ``mv`` / ``cp`` — source and destination are local names
- ``echo(content, file_name=None)`` — writing requires the file to already
  exist (``touch`` first); no POSIX redirect
- ``find(path='.', name=None)`` — ``name`` is a substring, not a glob
- ``mkdir`` — current directory only; no ``-p``

Unmappable dialect features raise ``MappingError`` so the loop can repair
instead of pretending POSIX.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from .types import ToolCall


class MappingError(ValueError):
    """Dialect command cannot be expressed as a single GFS tool call."""


def _local_name(path: str, *, allow_dot: bool = False) -> str:
    if path in {".", ""} and allow_dot:
        return "."
    if path in {".."}:
        return ".."
    if path in {"/", "."}:
        raise MappingError(
            f"GFS has no path {path!r} as a file name; use a cwd-local name"
        )
    p = PurePosixPath(path)
    if path.startswith("/") or len(p.parts) > 1:
        raise MappingError(
            f"GFS is cwd-local; cannot map path {path!r} (cd there first, then use {p.name!r})"
        )
    return path.rstrip("/") or path


def to_bfcl_func_string(call: ToolCall) -> str:
    """Return e.g. ``cd(folder='notes')`` for ``cd notes``."""
    if call.name == "done":
        raise MappingError("done is a harness stop token, not a GFS tool")

    def fmt(func: str, **kwargs: object) -> str:
        if not kwargs:
            return f"{func}()"
        parts = []
        for key, value in kwargs.items():
            parts.append(f"{key}={value!r}")
        return f"{func}({', '.join(parts)})"

    if call.redirect is not None and call.name != "echo":
        raise MappingError(
            "GFS has no generic stdout redirect; only echo(content, file_name=...) writes"
        )

    if call.name == "pwd":
        return fmt("pwd")

    if call.name == "ls":
        kwargs: dict[str, object] = {}
        if "a" in call.flags:
            kwargs["a"] = True
        # -l / -1 are display flags; GFS ls() has no equivalent.
        if call.positional:
            raise MappingError("GFS ls() lists the current directory only (no path arg)")
        return fmt("ls", **kwargs)

    if call.name == "cd":
        folder = call.positional[0] if call.positional else "/"
        if folder not in {".", "..", "/"} and "/" in folder.rstrip("/"):
            raise MappingError(
                "GFS cd(folder=...) allows only one folder level at a time"
            )
        return fmt("cd", folder=folder)

    if call.name == "mkdir":
        if "p" in call.flags:
            raise MappingError("GFS mkdir has no -p; create one cwd-local dir_name")
        if len(call.positional) != 1:
            raise MappingError("GFS mkdir takes one dir_name")
        return fmt("mkdir", dir_name=_local_name(call.positional[0]))

    if call.name == "cat":
        if len(call.positional) != 1:
            raise MappingError("GFS cat takes one file_name")
        return fmt("cat", file_name=_local_name(call.positional[0]))

    if call.name == "rm":
        if len(call.positional) != 1:
            raise MappingError("GFS rm takes one file_name")
        return fmt("rm", file_name=_local_name(call.positional[0]))

    if call.name in {"mv", "cp"}:
        src, dst = call.positional[0], call.positional[1]
        return fmt(
            call.name,
            source=_local_name(src),
            destination=_local_name(dst),
        )

    if call.name == "grep":
        if "r" in call.flags:
            raise MappingError("GFS grep is a single cwd-local file (no -r)")
        pattern, path = call.positional[0], call.positional[1]
        return fmt("grep", file_name=_local_name(path), pattern=pattern)

    if call.name == "touch":
        if len(call.positional) != 1:
            raise MappingError("GFS touch takes one file_name")
        return fmt("touch", file_name=_local_name(call.positional[0]))

    if call.name == "sort":
        if len(call.positional) != 1:
            raise MappingError("GFS sort takes one file_name")
        return fmt("sort", file_name=_local_name(call.positional[0]))

    if call.name == "rmdir":
        if len(call.positional) != 1:
            raise MappingError("GFS rmdir takes one dir_name")
        return fmt("rmdir", dir_name=_local_name(call.positional[0]))

    if call.name == "diff":
        if len(call.positional) != 2:
            raise MappingError("GFS diff takes two file names")
        return fmt(
            "diff",
            file_name1=_local_name(call.positional[0]),
            file_name2=_local_name(call.positional[1]),
        )

    if call.name == "tail":
        if len(call.positional) != 1:
            raise MappingError("GFS tail takes one file_name")
        kwargs: dict[str, object] = {
            "file_name": _local_name(call.positional[0]),
        }
        for key, value in call.options:
            if key == "n":
                kwargs["lines"] = int(value) if str(value).isdigit() else value
        return fmt("tail", **kwargs)

    if call.name == "du":
        if call.positional:
            raise MappingError("GFS du() has no path argument")
        kwargs = {}
        if "h" in call.flags:
            kwargs["human_readable"] = True
        return fmt("du", **kwargs)

    if call.name == "wc":
        path = call.positional[0]
        mode = "l"
        if "w" in call.flags:
            mode = "w"
        elif "c" in call.flags:
            mode = "c"
        return fmt("wc", file_name=_local_name(path), mode=mode)

    if call.name == "find":
        path = call.positional[0] if call.positional else "."
        name = None
        for key, value in call.options:
            if key == "name":
                name = value
            elif key == "type":
                raise MappingError("GFS find has no -type")
        kwargs = {"path": path}
        if name is not None:
            kwargs["name"] = name
        return fmt("find", **kwargs)

    if call.name == "echo":
        content = " ".join(call.positional)
        if call.redirect is not None:
            if call.redirect.op != ">":
                raise MappingError("GFS echo has no append; only overwrite via file_name")
            return fmt(
                "echo",
                content=content,
                file_name=_local_name(call.redirect.path),
            )
        return fmt("echo", content=content)

    raise MappingError(f"no GorillaFileSystem mapping for {call.name!r}")
