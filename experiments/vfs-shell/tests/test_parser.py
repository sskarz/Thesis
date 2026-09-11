from vfs_shell.parser import parse_command, to_argv
from vfs_shell.types import ParseError, ToolCall


def test_parses_grep_with_redirect():
    call = parse_command("grep ERROR /logs/app.log > /logs/errors.txt")
    assert isinstance(call, ToolCall)
    assert call.name == "grep"
    assert call.positional == ("ERROR", "/logs/app.log")
    assert call.redirect is not None
    assert call.redirect.op == ">"
    assert call.redirect.path == "/logs/errors.txt"


def test_parses_combined_short_flags_and_thought_wrapper():
    text = "I should list files.\n```bash\nls -la /notes\n```\n"
    call = parse_command(text)
    assert isinstance(call, ToolCall)
    assert call.name == "ls"
    assert call.flags == frozenset({"l", "a"})
    assert call.positional == ("/notes",)


def test_done_alias():
    call = parse_command("finish")
    assert isinstance(call, ToolCall)
    assert call.name == "done"


def test_denies_pipe():
    err = parse_command("cat /a.txt | grep milk")
    assert isinstance(err, ParseError)
    assert err.kind == "unsafe"


def test_denies_python():
    err = parse_command("python -c 'print(1)'")
    assert isinstance(err, ParseError)
    assert err.kind == "deny"


def test_denies_unknown_flag():
    err = parse_command("ls --color /")
    assert isinstance(err, ParseError)


def test_to_argv_roundtrip_without_redirect():
    call = parse_command("find /dst -name readme.txt -type f")
    assert isinstance(call, ToolCall)
    assert to_argv(call) == ["find", "/dst", "-name", "readme.txt", "-type", "f"]


def test_refuses_rm_root():
    err = parse_command("rm -r /")
    assert isinstance(err, ParseError)
