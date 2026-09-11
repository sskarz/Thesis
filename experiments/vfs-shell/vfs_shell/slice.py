"""Filter BFCL multi-turn entries to GorillaFileSystem slices."""

from __future__ import annotations

GFS_CLASS = "GorillaFileSystem"
SLICE_FS_ONLY = "fs-only"
SLICE_ANY_FS = "any-fs"
SLICES = (SLICE_FS_ONLY, SLICE_ANY_FS)


def involved_classes(entry: dict) -> list[str]:
    return list(entry.get("involved_classes") or [])


def is_fs_only(entry: dict) -> bool:
    return involved_classes(entry) == [GFS_CLASS]


def is_any_fs(entry: dict) -> bool:
    return GFS_CLASS in involved_classes(entry)


def filter_slice(entries: list[dict], slice_name: str) -> list[dict]:
    if slice_name == SLICE_FS_ONLY:
        return [e for e in entries if is_fs_only(e)]
    if slice_name == SLICE_ANY_FS:
        return [e for e in entries if is_any_fs(e)]
    raise ValueError(f"Unknown slice {slice_name!r}. Choose from: {', '.join(SLICES)}")
