"""Toy tasks that require consecutive commands. Outcome-scored, not AST-scored."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .backend import Backend
from .types import CheckResult

CheckFn = Callable[[Backend], list[CheckResult]]


@dataclass
class ToyTask:
    id: str
    instruction: str
    oracle: tuple[str, ...]
    initial_files: dict[str, str] = field(default_factory=dict)
    initial_dirs: tuple[str, ...] = ()
    max_turns: int = 16
    _evaluate: CheckFn | None = field(default=None, repr=False)

    def evaluate(self, backend: Backend) -> list[CheckResult]:
        if self._evaluate is None:
            return [CheckResult("missing_checks", False, "task has no evaluator")]
        return self._evaluate(backend)


def _ok(name: str, passed: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail)


def archive_note() -> ToyTask:
    """Create a dir, write a file, grep it, move it into an archive folder."""

    def checks(backend: Backend) -> list[CheckResult]:
        content = backend.read("/notes/archive/todo.txt") or ""
        return [
            _ok("archive_dir", backend.is_dir("/notes/archive")),
            _ok(
                "archived_file",
                backend.is_file("/notes/archive/todo.txt"),
                "expected /notes/archive/todo.txt",
            ),
            _ok("contains_milk", "milk" in content, repr(content)),
            _ok(
                "original_moved",
                not backend.exists("/notes/todo.txt"),
                "source path should be gone after mv",
            ),
        ]

    return ToyTask(
        id="archive_note",
        instruction=(
            "Create /notes. Write a file /notes/todo.txt containing the line "
            "'buy milk'. Confirm with grep that the file mentions milk. Then "
            "move it to /notes/archive/todo.txt (create the archive directory)."
        ),
        oracle=(
            "mkdir -p /notes/archive",
            "echo buy milk > /notes/todo.txt",
            "grep milk /notes/todo.txt",
            "mv /notes/todo.txt /notes/archive/todo.txt",
            "done",
        ),
        _evaluate=checks,
    )


def error_harvest() -> ToyTask:
    """Grep seeded logs, redirect matches, then relocate the report."""

    log = "INFO start\nERROR boom\nINFO ok\nERROR fail\nINFO done\n"

    def checks(backend: Backend) -> list[CheckResult]:
        report = backend.read("/logs/archive/errors.txt") or ""
        return [
            _ok("archive_dir", backend.is_dir("/logs/archive")),
            _ok("report_exists", backend.is_file("/logs/archive/errors.txt")),
            _ok("has_boom", "boom" in report, repr(report)),
            _ok("has_fail", "fail" in report, repr(report)),
            _ok("no_info", "INFO" not in report, repr(report)),
            _ok(
                "staging_gone",
                not backend.exists("/logs/errors.txt"),
                "expected mv to relocate the report",
            ),
        ]

    return ToyTask(
        id="error_harvest",
        instruction=(
            "Search /logs/app.log for lines containing ERROR. Write those "
            "lines to /logs/errors.txt, then move that file to "
            "/logs/archive/errors.txt (create directories as needed)."
        ),
        oracle=(
            "mkdir -p /logs/archive",
            "grep ERROR /logs/app.log > /logs/errors.txt",
            "mv /logs/errors.txt /logs/archive/errors.txt",
            "done",
        ),
        initial_files={"/logs/app.log": log},
        _evaluate=checks,
    )


def relocate_and_count() -> ToyTask:
    """Nested mkdir, copy, delete original, find, wc — consecutive FS ops."""

    def checks(backend: Backend) -> list[CheckResult]:
        content = backend.read("/dst/readme.txt") or ""
        return [
            _ok("copied", backend.is_file("/dst/readme.txt")),
            _ok("content", "hello edge" in content, repr(content)),
            _ok(
                "source_removed",
                not backend.exists("/src/bin/readme.txt"),
                "cp then rm should leave only the destination",
            ),
            _ok("src_bin_remains", backend.is_dir("/src/bin")),
        ]

    return ToyTask(
        id="relocate_and_count",
        instruction=(
            "Create /src/bin. Write /src/bin/readme.txt with the text "
            "'hello edge'. Copy it to /dst/readme.txt (create /dst), remove "
            "the original file, then use find to confirm the copy remains."
        ),
        oracle=(
            "mkdir -p /src/bin",
            "echo hello edge > /src/bin/readme.txt",
            "mkdir /dst",
            "cp /src/bin/readme.txt /dst/readme.txt",
            "rm /src/bin/readme.txt",
            "find /dst -name readme.txt",
            "wc -w /dst/readme.txt",
            "done",
        ),
        _evaluate=checks,
    )


TASKS: dict[str, ToyTask] = {
    t.id: t
    for t in (archive_note(), error_harvest(), relocate_and_count())
}


def get_task(task_id: str) -> ToyTask:
    try:
        factory = {
            "archive_note": archive_note,
            "error_harvest": error_harvest,
            "relocate_and_count": relocate_and_count,
        }[task_id]
    except KeyError as exc:
        known = ", ".join(TASKS)
        raise KeyError(f"unknown task {task_id!r}; known: {known}") from exc
    return factory()
