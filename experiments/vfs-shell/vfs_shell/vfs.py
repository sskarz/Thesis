"""In-memory and temp-dir virtual filesystems.

Both stores speak POSIX paths rooted at `/` and refuse to escape the sandbox
root. Command semantics live in `backend.py`; this module is the file tree.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Iterator, Protocol


class FileStore(Protocol):
    def cwd(self) -> str: ...
    def resolve(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...
    def is_dir(self, path: str) -> bool: ...
    def is_file(self, path: str) -> bool: ...
    def listdir(self, path: str) -> list[tuple[str, bool]]: ...
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str, append: bool = False) -> None: ...
    def mkdir(self, path: str, parents: bool = False) -> None: ...
    def remove(self, path: str, recursive: bool = False) -> None: ...
    def move(self, src: str, dst: str) -> None: ...
    def copy(self, src: str, dst: str, recursive: bool = False) -> None: ...
    def chdir(self, path: str) -> None: ...
    def walk_files(self, path: str) -> Iterator[str]: ...
    def snapshot_files(self) -> dict[str, str]: ...
    def snapshot_dirs(self) -> set[str]: ...


def normalize_path(path: str, cwd: str = "/") -> str:
    """Resolve `.` / `..` and keep the result inside `/`."""
    if not path:
        path = cwd
    path = path.replace("\\", "/")
    abs_path = path if path.startswith("/") else str(PurePosixPath(cwd) / path)
    parts: list[str] = []
    for part in PurePosixPath(abs_path).parts:
        if part in ("/", ""):
            continue
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts) if parts else "/"


def parent_of(path: str) -> str:
    if path == "/":
        return "/"
    return str(PurePosixPath(path).parent)


class MemoryStore:
    """Dict-backed VFS. Default backend for hermetic tests."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        dirs: list[str] | None = None,
    ) -> None:
        self._cwd = "/"
        self._dirs: set[str] = {"/"}
        self._files: dict[str, str] = {}
        for d in dirs or []:
            self.mkdir(d, parents=True)
        for path, content in (files or {}).items():
            parent = parent_of(normalize_path(path, "/"))
            self.mkdir(parent, parents=True)
            self._files[normalize_path(path, "/")] = content

    def cwd(self) -> str:
        return self._cwd

    def resolve(self, path: str) -> str:
        return normalize_path(path, self._cwd)

    def exists(self, path: str) -> bool:
        p = self.resolve(path)
        return p in self._dirs or p in self._files

    def is_dir(self, path: str) -> bool:
        return self.resolve(path) in self._dirs

    def is_file(self, path: str) -> bool:
        return self.resolve(path) in self._files

    def listdir(self, path: str) -> list[tuple[str, bool]]:
        p = self.resolve(path)
        if p not in self._dirs:
            raise FileNotFoundError(p)
        entries: list[tuple[str, bool]] = []
        prefix = p if p == "/" else p + "/"
        seen: set[str] = set()
        for d in self._dirs:
            if d == p:
                continue
            if d.startswith(prefix):
                rest = d[len(prefix) :]
                name = rest.split("/")[0]
                if name and name not in seen:
                    seen.add(name)
                    entries.append((name, True))
        for f in self._files:
            if not f.startswith(prefix):
                continue
            rest = f[len(prefix) :]
            if not rest or "/" in rest:
                continue
            if rest not in seen:
                seen.add(rest)
                entries.append((rest, False))
        return sorted(entries, key=lambda x: x[0])

    def read(self, path: str) -> str:
        p = self.resolve(path)
        if p not in self._files:
            if p in self._dirs:
                raise IsADirectoryError(p)
            raise FileNotFoundError(p)
        return self._files[p]

    def write(self, path: str, content: str, append: bool = False) -> None:
        p = self.resolve(path)
        if p in self._dirs:
            raise IsADirectoryError(p)
        parent = parent_of(p)
        if parent not in self._dirs:
            raise FileNotFoundError(parent)
        if append and p in self._files:
            self._files[p] += content
        else:
            self._files[p] = content

    def mkdir(self, path: str, parents: bool = False) -> None:
        p = self.resolve(path)
        if p in self._files:
            raise FileExistsError(p)
        if p in self._dirs:
            if parents:
                return
            raise FileExistsError(p)
        parent = parent_of(p)
        if parent not in self._dirs:
            if not parents:
                raise FileNotFoundError(parent)
            self.mkdir(parent, parents=True)
        self._dirs.add(p)

    def remove(self, path: str, recursive: bool = False) -> None:
        p = self.resolve(path)
        if p == "/":
            raise PermissionError("refusing to remove /")
        if p in self._files:
            del self._files[p]
            return
        if p not in self._dirs:
            raise FileNotFoundError(p)
        if not recursive:
            raise IsADirectoryError(p)
        prefix = p + "/"
        for f in list(self._files):
            if f.startswith(prefix):
                del self._files[f]
        for d in list(self._dirs):
            if d.startswith(prefix) or d == p:
                self._dirs.discard(d)

    def move(self, src: str, dst: str) -> None:
        s = self.resolve(src)
        d = self.resolve(dst)
        if not self.exists(s):
            raise FileNotFoundError(s)
        if self.is_dir(d):
            d = (d.rstrip("/") + "/" + PurePosixPath(s).name) if d != "/" else "/" + PurePosixPath(s).name
            d = normalize_path(d, "/")
        if self.is_file(s):
            parent = parent_of(d)
            if parent not in self._dirs:
                raise FileNotFoundError(parent)
            if d in self._dirs:
                raise IsADirectoryError(d)
            self._files[d] = self._files.pop(s)
            return
        # directory move
        if d.startswith(s.rstrip("/") + "/") or d == s:
            raise PermissionError("cannot move a directory into itself")
        parent = parent_of(d)
        if parent not in self._dirs:
            raise FileNotFoundError(parent)
        mapping_files = {}
        for f, content in list(self._files.items()):
            if f == s or f.startswith(s + "/"):
                mapping_files[d + f[len(s) :]] = self._files.pop(f)
        self._files.update(mapping_files)
        dirs_to_add = []
        for directory in list(self._dirs):
            if directory == s or directory.startswith(s + "/"):
                dirs_to_add.append(d + directory[len(s) :])
                self._dirs.discard(directory)
        self._dirs.update(dirs_to_add)

    def copy(self, src: str, dst: str, recursive: bool = False) -> None:
        s = self.resolve(src)
        d = self.resolve(dst)
        if not self.exists(s):
            raise FileNotFoundError(s)
        if self.is_dir(d):
            d = (d.rstrip("/") + "/" + PurePosixPath(s).name) if d != "/" else "/" + PurePosixPath(s).name
            d = normalize_path(d, "/")
        if self.is_file(s):
            parent = parent_of(d)
            if parent not in self._dirs:
                raise FileNotFoundError(parent)
            self._files[d] = self._files[s]
            return
        if not recursive:
            raise IsADirectoryError(s)
        parent = parent_of(d)
        if parent not in self._dirs:
            raise FileNotFoundError(parent)
        for directory in list(self._dirs):
            if directory == s or directory.startswith(s + "/"):
                self._dirs.add(d + directory[len(s) :])
        for f, content in list(self._files.items()):
            if f == s or f.startswith(s + "/"):
                self._files[d + f[len(s) :]] = content

    def chdir(self, path: str) -> None:
        p = self.resolve(path)
        if p not in self._dirs:
            if p in self._files:
                raise NotADirectoryError(p)
            raise FileNotFoundError(p)
        self._cwd = p

    def walk_files(self, path: str) -> Iterator[str]:
        p = self.resolve(path)
        if self.is_file(p):
            yield p
            return
        if p not in self._dirs:
            raise FileNotFoundError(p)
        prefix = p if p == "/" else p + "/"
        for f in sorted(self._files):
            if p == "/" or f.startswith(prefix) or f == p:
                yield f

    def snapshot_files(self) -> dict[str, str]:
        return dict(self._files)

    def snapshot_dirs(self) -> set[str]:
        return set(self._dirs)


class TempDirStore:
    """Same dialect on a real temp directory (no subprocess, no /bin/sh)."""

    def __init__(
        self,
        root: Path | None = None,
        files: dict[str, str] | None = None,
        dirs: list[str] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else Path(
            tempfile.mkdtemp(prefix="vfs-shell-")
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._cwd = "/"
        for d in dirs or []:
            self.mkdir(d, parents=True)
        for path, content in (files or {}).items():
            parent = parent_of(normalize_path(path, "/"))
            self.mkdir(parent, parents=True)
            self.write(path, content)

    def cwd(self) -> str:
        return self._cwd

    def resolve(self, path: str) -> str:
        return normalize_path(path, self._cwd)

    def _real(self, vfs_path: str) -> Path:
        vfs_path = self.resolve(vfs_path)
        rel = vfs_path.lstrip("/")
        real = (self.root / rel).resolve() if rel else self.root.resolve()
        root = self.root.resolve()
        if real != root and root not in real.parents:
            raise PermissionError("path escapes sandbox")
        return real

    def exists(self, path: str) -> bool:
        return self._real(path).exists()

    def is_dir(self, path: str) -> bool:
        return self._real(path).is_dir()

    def is_file(self, path: str) -> bool:
        return self._real(path).is_file()

    def listdir(self, path: str) -> list[tuple[str, bool]]:
        real = self._real(path)
        if not real.is_dir():
            if not real.exists():
                raise FileNotFoundError(self.resolve(path))
            raise NotADirectoryError(self.resolve(path))
        entries = []
        for child in sorted(real.iterdir(), key=lambda p: p.name):
            entries.append((child.name, child.is_dir()))
        return entries

    def read(self, path: str) -> str:
        real = self._real(path)
        if real.is_dir():
            raise IsADirectoryError(self.resolve(path))
        if not real.is_file():
            raise FileNotFoundError(self.resolve(path))
        return real.read_text(encoding="utf-8")

    def write(self, path: str, content: str, append: bool = False) -> None:
        real = self._real(path)
        if real.is_dir():
            raise IsADirectoryError(self.resolve(path))
        if not real.parent.exists():
            raise FileNotFoundError(parent_of(self.resolve(path)))
        mode = "a" if append else "w"
        with real.open(mode, encoding="utf-8") as fh:
            fh.write(content)

    def mkdir(self, path: str, parents: bool = False) -> None:
        p = self.resolve(path)
        if p == "/":
            return
        real = self._real(p)
        if real.is_file():
            raise FileExistsError(p)
        if real.is_dir() and not parents:
            raise FileExistsError(p)
        real.mkdir(parents=parents, exist_ok=parents)

    def remove(self, path: str, recursive: bool = False) -> None:
        p = self.resolve(path)
        if p == "/":
            raise PermissionError("refusing to remove /")
        real = self._real(p)
        if not real.exists():
            raise FileNotFoundError(p)
        if real.is_dir() and not recursive:
            raise IsADirectoryError(p)
        if real.is_dir():
            shutil.rmtree(real)
        else:
            real.unlink()

    def move(self, src: str, dst: str) -> None:
        s = self.resolve(src)
        d = self.resolve(dst)
        src_real = self._real(s)
        if not src_real.exists():
            raise FileNotFoundError(s)
        dst_real = self._real(d)
        if dst_real.is_dir():
            dst_real = dst_real / src_real.name
        dst_real.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_real), str(dst_real))

    def copy(self, src: str, dst: str, recursive: bool = False) -> None:
        s = self.resolve(src)
        d = self.resolve(dst)
        src_real = self._real(s)
        if not src_real.exists():
            raise FileNotFoundError(s)
        dst_real = self._real(d)
        if dst_real.is_dir():
            dst_real = dst_real / src_real.name
        if src_real.is_dir():
            if not recursive:
                raise IsADirectoryError(s)
            shutil.copytree(src_real, dst_real)
            return
        dst_real.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_real, dst_real)

    def chdir(self, path: str) -> None:
        p = self.resolve(path)
        real = self._real(p)
        if real.is_file():
            raise NotADirectoryError(p)
        if not real.is_dir():
            raise FileNotFoundError(p)
        self._cwd = p

    def walk_files(self, path: str) -> Iterator[str]:
        p = self.resolve(path)
        real = self._real(p)
        if real.is_file():
            yield p
            return
        if not real.is_dir():
            raise FileNotFoundError(p)
        for dirpath, _dirnames, filenames in os.walk(real):
            for name in filenames:
                full = Path(dirpath) / name
                rel = full.relative_to(self.root.resolve()).as_posix()
                yield "/" + rel if rel != "." else "/"

    def snapshot_files(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for vfs_path in self.walk_files("/"):
            try:
                out[vfs_path] = self.read(vfs_path)
            except (IsADirectoryError, FileNotFoundError, UnicodeDecodeError):
                continue
        return out

    def snapshot_dirs(self) -> set[str]:
        dirs = {"/"}
        root = self.root.resolve()
        for dirpath, dirnames, _files in os.walk(root):
            for name in dirnames:
                full = Path(dirpath) / name
                rel = full.relative_to(root).as_posix()
                dirs.add("/" + rel)
        return dirs


def match_name(path: str, glob_pat: str) -> bool:
    return fnmatch(PurePosixPath(path).name, glob_pat)
