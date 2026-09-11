"""CLI: run toy tasks with the scripted model (or llama-server later)."""

from __future__ import annotations

import argparse
import sys

from .backend import memory_backend, tempdir_backend
from .loop import run_episode
from .models import LlamaServerModel, ScriptedModel
from .tasks import TASKS, get_task
from .types import EpisodeResult


def _make_backend(kind: str, task):
    files = dict(task.initial_files)
    dirs = list(task.initial_dirs)
    if kind == "memory":
        return memory_backend(files=files, dirs=dirs)
    if kind == "tempdir":
        return tempdir_backend(files=files, dirs=dirs)
    raise ValueError(f"unknown backend {kind!r}")


def _print_result(result: EpisodeResult) -> None:
    status = "PASS" if result.success else "FAIL"
    print(f"[{status}] {result.task_id}  turns={len(result.turns)}  "
          f"parsed_errors={result.parse_errors}  stop={result.stopped}")
    for check in result.checks:
        mark = "ok" if check.passed else "FAIL"
        extra = f"  ({check.detail})" if check.detail and not check.passed else ""
        print(f"  [{mark}] {check.name}{extra}")
    if not result.success:
        print("  trace:")
        for i, turn in enumerate(result.turns, start=1):
            verb = turn.call.name if turn.call else f"PARSE:{turn.parse_error.kind if turn.parse_error else '?'}"
            print(f"    {i}. {verb}: {turn.raw!r} -> {turn.observation.render()!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VFS/shell-as-action toy harness (not BFCL)."
    )
    parser.add_argument("--task", help="task id; default is --all")
    parser.add_argument("--all", action="store_true", help="run every toy task")
    parser.add_argument("--list", action="store_true", help="list task ids")
    parser.add_argument(
        "--backend",
        choices=("memory", "tempdir"),
        default="memory",
        help="file store (default: in-memory VFS)",
    )
    parser.add_argument(
        "--model",
        choices=("scripted", "llama-server"),
        default="scripted",
        help="scripted oracle (tests) or local llama-server stub",
    )
    parser.add_argument("--server-url", default=None, help="llama-server base URL")
    args = parser.parse_args(argv)

    if args.list:
        for task_id, task in TASKS.items():
            print(f"{task_id}: {task.instruction.splitlines()[0][:80]}")
        return 0

    ids = list(TASKS) if (args.all or not args.task) else [args.task]
    if args.task and args.task not in TASKS and not args.all:
        print(f"unknown task {args.task!r}; known: {', '.join(TASKS)}", file=sys.stderr)
        return 2

    failed = 0
    for task_id in ids:
        task = get_task(task_id)
        backend = _make_backend(args.backend, task)
        if args.model == "scripted":
            model = ScriptedModel(list(task.oracle))
        else:
            model = LlamaServerModel(base_url=args.server_url)
        result = run_episode(task, model, backend)
        _print_result(result)
        if not result.success:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
