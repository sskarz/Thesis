from vfs_shell.bfcl_fs import MappingError, from_bfcl_func_string, to_bfcl_func_string
from vfs_shell.parser import parse_command


# Canonical GFS dialect cases: shell → BFCL → shell must match.
ROUND_TRIP_CASES = [
    ("pwd", "pwd()"),
    ("ls -a", "ls(a=True)"),
    ("cd notes", "cd(folder='notes')"),
    ("mkdir archive", "mkdir(dir_name='archive')"),
    ("cat todo.txt", "cat(file_name='todo.txt')"),
    ("mv todo.txt archive", "mv(source='todo.txt', destination='archive')"),
    ("cp a.txt b.txt", "cp(source='a.txt', destination='b.txt')"),
    ("rm spam.txt", "rm(file_name='spam.txt')"),
    ("grep ERROR app.log", "grep(file_name='app.log', pattern='ERROR')"),
    ("wc -w readme.txt", "wc(file_name='readme.txt', mode='w')"),
    ("find . -name test", "find(path='.', name='test')"),
    ("echo buy milk", "echo(content='buy milk')"),
    ("echo buy milk > todo.txt", "echo(content='buy milk', file_name='todo.txt')"),
]


def _map(text: str) -> str:
    call = parse_command(text)
    assert not isinstance(call, type(None))
    from vfs_shell.types import ToolCall

    assert isinstance(call, ToolCall)
    return to_bfcl_func_string(call)


def test_maps_core_gfs_calls():
    for shell, expected in ROUND_TRIP_CASES:
        assert _map(shell) == expected


def test_round_trip_shell_bfcl_shell():
    for shell, expected_bfcl in ROUND_TRIP_CASES:
        bfcl = _map(shell)
        assert bfcl == expected_bfcl
        assert from_bfcl_func_string(bfcl) == shell


def test_reverse_maps_remaining_gfs_dialect():
    extras = [
        "ls",
        "touch notes.txt",
        "sort log.txt",
        "tail -n 20 log.txt",
        "diff a.txt b.txt",
        "rmdir archive",
        "du",
        "du -h",
        "wc -l readme.txt",
        "find .",
        "cd ..",
    ]
    for shell in extras:
        assert from_bfcl_func_string(_map(shell)) == shell


def test_refuses_gfs_incompatible_paths_and_flags():
    for text, needle in [
        ("mkdir -p notes", "-p"),
        ("cd notes/archive", "one folder"),
        ("cat /notes/todo.txt", "cwd-local"),
        ("ls /notes", "current directory"),
        ("grep ERROR app.log > errors.txt", "redirect"),
    ]:
        call = parse_command(text)
        from vfs_shell.types import ToolCall

        assert isinstance(call, ToolCall)
        try:
            to_bfcl_func_string(call)
        except MappingError as exc:
            assert needle.lower() in str(exc).lower()
        else:
            raise AssertionError(f"expected MappingError for {text!r}")

