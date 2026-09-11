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
        "touch",
        "sort",
        "diff",
        "tail",
        "du",
        "rmdir",
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
    "touch": frozenset(),
    "sort": frozenset(),
    "diff": frozenset(),
    "tail": frozenset(),
    "du": frozenset({"h"}),
    "rmdir": frozenset(),
    "done": frozenset(),
}

# Options that take a value: find -name PATTERN, tail -n N
VALUED_OPTIONS: dict[str, frozenset[str]] = {
    "find": frozenset({"name", "type"}),
    "tail": frozenset({"n"}),
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

# Prompt for BFCL GorillaFileSystem (cwd-local names, no mkdir -p).
GFS_SYSTEM_PROMPT = """You operate a simple in-memory file system by emitting Unix-like commands.

Rules:
- Emit exactly ONE command per step. No JSON, no tool_calls, no markdown fences.
- No pipes, no &&, no ;, no $() or backticks.
- Names are local to the current directory unless noted. cd moves one folder level at a time (., .., /, or a child name).
- mkdir has no -p. echo > file only works if the file already exists (touch first).
- After each command you will see its result. Then emit the next command, or `done` when this user request is finished.

Commands:
  pwd
  ls [-a]
  cd <folder>
  mkdir <dir_name>
  rmdir <dir_name>
  touch <file_name>
  echo <text>
  echo <text> > <file_name>
  cat <file_name>
  grep <pattern> <file_name>
  find [path] [-name substring]
  wc [-l|-w|-c] <file_name>
  sort <file_name>
  tail -n <N> <file_name>
  diff <file1> <file2>
  mv <src> <dst>
  cp <src> <dst>
  rm <file_name>
  du [-h]
  done
"""


def repair_message(error: str) -> str:
    allowed = " ".join(sorted(ALLOWED_COMMANDS))
    return (
        f"PARSE ERROR: {error}\n"
        f"Allowed commands: {allowed}\n"
        "Emit exactly one command, for example: mkdir -p /notes\n"
        "Do not use pipes, &&, or JSON."
    )


def gfs_repair_message(error: str) -> str:
    return (
        f"PARSE/MAP ERROR: {error}\n"
        "Emit exactly one GorillaFileSystem command, for example: ls -a\n"
        "cwd-local names only. No mkdir -p, no pipes, no JSON. "
        "When this user request is done, emit: done"
    )
