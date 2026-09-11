"""Backend protocol, memory/temp-dir executors, Harbor seam.

Command semantics are implemented once against FileStore. HarborAdapter is
intentionally unimplemented: it documents how ToolCall maps onto a real
sandbox exec without pulling Harbor into this repo.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from .parser import to_argv
from .types import Observation, ToolCall
from .vfs import FileStore, MemoryStore, TempDirStore, match_name


class Backend(Protocol):
    def execute(self, call: ToolCall) -> Observation: ...
    def exists(self, path: str) -> bool: ...
    def is_dir(self, path: str) -> bool: ...
    def is_file(self, path: str) -> bool: ...
    def read(self, path: str) -> str | None: ...
    def cwd(self) -> str: ...
    def snapshot_files(self) -> dict[str, str]: ...


class StoreBackend:
    """Execute allowlisted ToolCalls against a FileStore."""

    def __init__(self, store: FileStore) -> None:
        self.store = store

    def exists(self, path: str) -> bool:
        return self.store.exists(path)

    def is_dir(self, path: str) -> bool:
        return self.store.is_dir(path)

    def is_file(self, path: str) -> bool:
        return self.store.is_file(path)

    def read(self, path: str) -> str | None:
        try:
            return self.store.read(path)
        except (FileNotFoundError, IsADirectoryError):
            return None

    def cwd(self) -> str:
        return self.store.cwd()

    def snapshot_files(self) -> dict[str, str]:
        return self.store.snapshot_files()

    def execute(self, call: ToolCall) -> Observation:
        if call.name == "done":
            return Observation(ok=True, stdout="")
        handler = getattr(self, f"_op_{call.name}", None)
        if handler is None:
            return Observation(ok=False, stderr=f"{call.name}: not implemented", exit_code=127)
        try:
            obs = handler(call)
        except FileNotFoundError as exc:
            return Observation(ok=False, stderr=f"{call.name}: {exc}: No such file or directory", exit_code=1)
        except IsADirectoryError as exc:
            return Observation(ok=False, stderr=f"{call.name}: {exc}: Is a directory", exit_code=1)
        except NotADirectoryError as exc:
            return Observation(ok=False, stderr=f"{call.name}: {exc}: Not a directory", exit_code=1)
        except FileExistsError as exc:
            return Observation(ok=False, stderr=f"{call.name}: {exc}: File exists", exit_code=1)
        except PermissionError as exc:
            return Observation(ok=False, stderr=f"{call.name}: {exc}", exit_code=1)
        except ValueError as exc:
            return Observation(ok=False, stderr=str(exc), exit_code=1)

        if call.redirect is not None and obs.ok:
            try:
                self.store.write(
                    call.redirect.path,
                    obs.stdout,
                    append=call.redirect.op == ">>",
                )
            except FileNotFoundError as exc:
                return Observation(
                    ok=False,
                    stderr=f"redirect: {exc}: No such file or directory",
                    exit_code=1,
                )
            except IsADirectoryError as exc:
                return Observation(ok=False, stderr=f"redirect: {exc}: Is a directory", exit_code=1)
            return Observation(ok=True, stdout="", exit_code=0)
        return obs

    def _op_pwd(self, call: ToolCall) -> Observation:
        return Observation(ok=True, stdout=self.store.cwd() + "\n")

    def _op_cd(self, call: ToolCall) -> Observation:
        target = call.positional[0] if call.positional else "/"
        self.store.chdir(target)
        return Observation(ok=True, stdout="")

    def _op_ls(self, call: ToolCall) -> Observation:
        path = call.positional[0] if call.positional else self.store.cwd()
        if self.store.is_file(path):
            name = path.rsplit("/", 1)[-1]
            return Observation(ok=True, stdout=name + "\n")
        entries = self.store.listdir(path)
        if "a" not in call.flags:
            entries = [(n, d) for n, d in entries if not n.startswith(".")]
        lines = []
        for name, is_dir in entries:
            if "l" in call.flags:
                kind = "d" if is_dir else "-"
                lines.append(f"{kind} {name}")
            else:
                lines.append(name)
        stdout = "\n".join(lines)
        if stdout:
            stdout += "\n"
        return Observation(ok=True, stdout=stdout)

    def _op_cat(self, call: ToolCall) -> Observation:
        chunks = [self.store.read(p) for p in call.positional]
        return Observation(ok=True, stdout="".join(chunks))

    def _op_echo(self, call: ToolCall) -> Observation:
        text = " ".join(call.positional)
        if "n" not in call.flags:
            text += "\n"
        return Observation(ok=True, stdout=text)

    def _op_mkdir(self, call: ToolCall) -> Observation:
        for path in call.positional:
            self.store.mkdir(path, parents="p" in call.flags)
        return Observation(ok=True, stdout="")

    def _op_rm(self, call: ToolCall) -> Observation:
        for path in call.positional:
            try:
                self.store.remove(path, recursive="r" in call.flags)
            except FileNotFoundError:
                if "f" not in call.flags:
                    raise
        return Observation(ok=True, stdout="")

    def _op_mv(self, call: ToolCall) -> Observation:
        self.store.move(call.positional[0], call.positional[1])
        return Observation(ok=True, stdout="")

    def _op_cp(self, call: ToolCall) -> Observation:
        self.store.copy(
            call.positional[0],
            call.positional[1],
            recursive="r" in call.flags,
        )
        return Observation(ok=True, stdout="")

    def _op_grep(self, call: ToolCall) -> Observation:
        pattern, path = call.positional[0], call.positional[1]
        flags = re.IGNORECASE if "i" in call.flags else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"grep: invalid pattern: {exc}") from exc

        if self.store.is_dir(path) and "r" not in call.flags:
            raise IsADirectoryError(path)

        files = list(self.store.walk_files(path)) if "r" in call.flags or self.store.is_dir(path) else [self.store.resolve(path)]
        matches: list[str] = []
        total = 0
        for fpath in files:
            try:
                content = self.store.read(fpath)
            except (IsADirectoryError, FileNotFoundError):
                continue
            file_hits = 0
            for i, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    file_hits += 1
                    if "c" not in call.flags:
                        prefix = ""
                        if "n" in call.flags:
                            prefix = f"{i}:"
                        if "r" in call.flags:
                            matches.append(f"{fpath}:{prefix}{line}")
                        else:
                            matches.append(f"{prefix}{line}")
            total += file_hits
        if "c" in call.flags:
            return Observation(ok=True, stdout=f"{total}\n")
        stdout = "\n".join(matches)
        if stdout:
            stdout += "\n"
        return Observation(ok=True, stdout=stdout)

    def _op_wc(self, call: ToolCall) -> Observation:
        path = call.positional[0]
        content = self.store.read(path)
        lines = content.count("\n")
        if content and not content.endswith("\n"):
            lines += 1
        words = len(content.split())
        chars = len(content.encode("utf-8"))
        show_l, show_w, show_c = "l" in call.flags, "w" in call.flags, "c" in call.flags
        if not (show_l or show_w or show_c):
            show_l = show_w = show_c = True
        parts = []
        if show_l:
            parts.append(str(lines))
        if show_w:
            parts.append(str(words))
        if show_c:
            parts.append(str(chars))
        return Observation(ok=True, stdout=" ".join(parts) + f" {path}\n")

    def _op_find(self, call: ToolCall) -> Observation:
        start = call.positional[0] if call.positional else self.store.cwd()
        name_pat = None
        type_filter = None
        for key, value in call.options:
            if key == "name":
                name_pat = value
            elif key == "type":
                type_filter = value
        hits: list[str] = []
        if type_filter in (None, "d"):
            # include start dir and descendants
            if self.store.is_dir(start):
                stack = [self.store.resolve(start)]
                seen = set()
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    if name_pat is None or match_name(cur, name_pat):
                        hits.append(cur)
                    try:
                        for child, is_dir in self.store.listdir(cur):
                            child_path = cur.rstrip("/") + "/" + child if cur != "/" else "/" + child
                            if is_dir:
                                stack.append(child_path)
                    except FileNotFoundError:
                        continue
            elif type_filter == "d":
                pass
        if type_filter in (None, "f"):
            for fpath in self.store.walk_files(start):
                if name_pat is not None and not match_name(fpath, name_pat):
                    continue
                if fpath not in hits:
                    hits.append(fpath)
        if type_filter == "f":
            hits = [h for h in hits if self.store.is_file(h)]
        elif type_filter == "d":
            hits = [h for h in hits if self.store.is_dir(h)]
        hits = sorted(set(hits))
        stdout = "\n".join(hits)
        if stdout:
            stdout += "\n"
        return Observation(ok=True, stdout=stdout)


def memory_backend(
    files: dict[str, str] | None = None,
    dirs: list[str] | None = None,
) -> StoreBackend:
    return StoreBackend(MemoryStore(files=files, dirs=dirs))


def tempdir_backend(
    files: dict[str, str] | None = None,
    dirs: list[str] | None = None,
    root: Path | None = None,
) -> StoreBackend:
    return StoreBackend(TempDirStore(root=root, files=files, dirs=dirs))


class HarborAdapter:
    """Seam for Terminal-Bench / Harbor Terminus-style evaluation.

    Not implemented in this spike (no Harbor / Docker dependency).

    Intended mapping:

        call = parse_command(model_text)
        argv = to_argv(call)          # never shell=True
        result = await env.exec(argv) # Harbor BaseEnvironment
        # if call.redirect: write result.stdout to that path via a second exec
        return Observation(...)

    See docs/vfs-shell-action-space.md §9.
    """

    def execute(self, call: ToolCall) -> Observation:
        raise NotImplementedError(
            "HarborAdapter is a documented seam, not a runtime. "
            f"Would exec argv={to_argv(call)!r} inside a Harbor sandbox. "
            "See docs/vfs-shell-action-space.md."
        )

    @staticmethod
    def argv_for(call: ToolCall) -> list[str]:
        return to_argv(call)
