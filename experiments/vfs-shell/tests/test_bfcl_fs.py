from vfs_shell.bfcl_fs import MappingError, to_bfcl_func_string
from vfs_shell.parser import parse_command


def _map(text: str) -> str:
    call = parse_command(text)
    assert not isinstance(call, type(None))
    from vfs_shell.types import ToolCall

    assert isinstance(call, ToolCall)
    return to_bfcl_func_string(call)


def test_maps_core_gfs_calls():
    assert _map("pwd") == "pwd()"
    assert _map("ls -a") == "ls(a=True)"
    assert _map("cd notes") == "cd(folder='notes')"
    assert _map("mkdir archive") == "mkdir(dir_name='archive')"
    assert _map("cat todo.txt") == "cat(file_name='todo.txt')"
    assert _map("mv todo.txt archive") == "mv(source='todo.txt', destination='archive')"
    assert _map("cp a.txt b.txt") == "cp(source='a.txt', destination='b.txt')"
    assert _map("rm spam.txt") == "rm(file_name='spam.txt')"
    assert _map("grep ERROR app.log") == "grep(file_name='app.log', pattern='ERROR')"
    assert _map("wc -w readme.txt") == "wc(file_name='readme.txt', mode='w')"
    assert _map("find . -name test") == "find(path='.', name='test')"
    assert _map("echo buy milk") == "echo(content='buy milk')"
    assert _map("echo buy milk > todo.txt") == (
        "echo(content='buy milk', file_name='todo.txt')"
    )


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
