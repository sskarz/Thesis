"""Toy tasks must pass under a scripted oracle and fail if a step is skipped."""

from vfs_shell.backend import HarborAdapter, memory_backend, tempdir_backend
from vfs_shell.loop import run_episode
from vfs_shell.models import LlamaServerModel, ScriptedModel
from vfs_shell.parser import parse_command
from vfs_shell.tasks import TASKS, get_task
from vfs_shell.types import ToolCall


def test_all_scripted_toy_tasks_pass_memory():
    passed = []
    for task_id in TASKS:
        task = get_task(task_id)
        backend = memory_backend(files=dict(task.initial_files), dirs=list(task.initial_dirs))
        result = run_episode(task, ScriptedModel(list(task.oracle)), backend)
        assert result.success, (
            f"{task_id} failed checks={[c.name for c in result.checks if not c.passed]} "
            f"trace={[t.raw for t in result.turns]}"
        )
        assert result.executed_calls >= 2
        passed.append(task_id)
    assert len(passed) >= 2


def test_archive_note_also_passes_on_tempdir():
    task = get_task("archive_note")
    backend = tempdir_backend(files=dict(task.initial_files), dirs=list(task.initial_dirs))
    result = run_episode(task, ScriptedModel(list(task.oracle)), backend)
    assert result.success, [c for c in result.checks if not c.passed]


def test_skipped_move_fails_outcome_checks():
    task = get_task("archive_note")
    script = [
        "mkdir -p /notes/archive",
        "echo buy milk > /notes/todo.txt",
        "grep milk /notes/todo.txt",
        "done",
    ]
    backend = memory_backend()
    result = run_episode(task, ScriptedModel(script), backend)
    assert not result.success
    failed = {c.name for c in result.checks if not c.passed}
    assert "archived_file" in failed
    assert "original_moved" in failed


def test_parse_error_repair_then_progress():
    task = get_task("archive_note")
    script = [
        "cat /notes/todo.txt | grep milk",
        *task.oracle,
    ]
    backend = memory_backend()
    result = run_episode(task, ScriptedModel(script), backend)
    assert result.parse_errors >= 1
    assert result.success


def test_harbor_adapter_is_documented_seam():
    adapter = HarborAdapter()
    call = parse_command("ls /")
    assert isinstance(call, ToolCall)
    assert adapter.argv_for(call) == ["ls", "/"]
    try:
        adapter.execute(call)
    except NotImplementedError as exc:
        assert "Harbor" in str(exc) or "stretch" in str(exc).lower()
    else:
        raise AssertionError("HarborAdapter.execute should stay unimplemented")


def test_llama_server_url_defaults_to_v1():
    model = LlamaServerModel(base_url="http://127.0.0.1:8080")
    assert model.base_url.endswith("/v1")
    already = LlamaServerModel(base_url="http://localhost:8080/v1")
    assert already.base_url == "http://localhost:8080/v1"
