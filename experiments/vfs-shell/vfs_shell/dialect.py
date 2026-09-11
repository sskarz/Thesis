"""Constrained command dialect: vocab, prompt, and deny rules.

Kept small on purpose (SWE-agent ACI lesson: raw bash is not the best LM
interface). Pipes, conjunctions, and substitution are out of vocab so an SLM
must use extra turns for composition — that is the consecutive-calling test.
"""

from __future__ import annotations

ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "ls",
        "cat",
        "grep",
        "cd",
        "pwd",
        "mkdir",
        "mv",
        "cp",
        "rm",
        "echo",
        "wc",
        "find",
        "done",
    }
)

# Short flags accepted per verb. Unknown flags are a parse error, not a guess.
ALLOWED_FLAGS: dict[str, frozenset[str]] = {
    "ls": frozenset({"l", "a", "1"}),
    "cat": frozenset(),
    "grep": frozenset({"n", "i", "r", "c"}),
    "cd": frozenset(),
    "pwd": frozenset(),
    "mkdir": frozenset({"p"}),
    "mv": frozenset(),
    "cp": frozenset({"r"}),
    "rm": frozenset({"r", "f"}),
    "echo": frozenset({"n"}),
    "wc": frozenset({"l", "w", "c"}),
    "find": frozenset(),
    "done": frozenset(),
}

# find-style options that take a value: -name PATTERN, -type f|d
VALUED_OPTIONS: dict[str, frozenset[str]] = {
    "find": frozenset({"name", "type"}),
}

DONE_ALIASES: frozenset[str] = frozenset({"done", "finish", "submit"})

COMMAND_HELP = (
    "ls [-l] [path]     list directory\n"
    "cat <path>         print file\n"
    "grep [-n][-i][-r][-c] <pattern> <path>\n"
    "cd [path]          change directory\n"
    "pwd                print working directory\n"
    "mkdir [-p] <path>  create directory\n"
    "mv <src> <dst>     move / rename\n"
    "cp [-r] <src> <dst>  copy\n"
    "rm [-r][-f] <path> remove\n"
    "echo [-n] <text> [> path | >> path]\n"
    "wc [-l][-w][-c] <path>\n"
    "find [path] [-name glob] [-type f|d]\n"
    "done               end the episode (optional; state is still scored)\n"
    "Redirect: any command may end with  > path  or  >> path  (stdout only).\n"
    "Forbidden: pipes |, &&, ||, ;, $(), backticks, python/curl/sh."
)

SYSTEM_PROMPT = """You are a file-system agent. You solve tasks by emitting Unix-like commands.

Rules:
- Emit exactly ONE command per turn. No surrounding JSON.
- Use only: ls cat grep cd pwd mkdir mv cp rm echo wc find done
- No pipes, no &&, no ;, no command substitution. Use extra turns instead.
- Paths are POSIX and rooted at /. You cannot leave /.
- After a command, you will see its stdout/stderr. Use that to decide the next command.
- When the task is complete, emit: done

""" + COMMAND_HELP


def repair_message(error: str) -> str:
    allowed = " ".join(sorted(ALLOWED_COMMANDS))
    return (
        f"PARSE ERROR: {error}\n"
        f"Allowed commands: {allowed}\n"
        "Emit exactly one command, for example: mkdir -p /notes\n"
        "Do not use pipes, &&, or JSON."
    )
