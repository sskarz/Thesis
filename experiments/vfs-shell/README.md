# VFS / shell-as-action experiment

Miniature harness for **consecutive command calling**: the model emits
allowlisted Unix-like text (`ls`, `cat`, `grep`, …); a parser maps that onto
a virtual filesystem; success is **final state**, not AST match.

This is a parallel spike. It does **not** touch the BFCL pipeline in
`benchmark/`, and it is **not** the IFFC/NFC structured-output track.

Design: [`docs/vfs-shell-action-space.md`](../../docs/vfs-shell-action-space.md)

## What “success” means

A toy task **passes** when programmatic checks on the VFS all hold after the
episode (files in the right place, required substrings present, sources
moved). The command trace is recorded for debugging; it is not the score.

The bundled proof uses a **scripted model** (oracle command list). That shows
the loop composes consecutive ops. It is **not** an SLM accuracy number.

| Task | Consecutive ops (oracle) | Outcome checks |
|------|--------------------------|----------------|
| `archive_note` | mkdir → echo-redirect → grep → mv | `/notes/archive/todo.txt` contains `milk`; original gone |
| `error_harvest` | mkdir → grep-redirect → mv | `/logs/archive/errors.txt` has ERROR lines only |
| `relocate_and_count` | mkdir → echo → mkdir → cp → rm → find → wc | copy at `/dst/readme.txt`; source file removed |

A scripted model that **skips** a step must fail those checks.

## How to run

From the repo root (hermetic; no llama-server):

```bash
# All three toy tasks, in-memory VFS, scripted oracle
uv run python experiments/vfs-shell/run.py --all

# One task on a real temp directory (still no /bin/sh)
uv run python experiments/vfs-shell/run.py --task archive_note --backend tempdir

uv run python experiments/vfs-shell/run.py --list
```

Tests live next to the experiment so they do not import `benchmark/` / BFCL:

```bash
uv run pytest experiments/vfs-shell/tests -q
```

Expected: every toy task PASS under `--model scripted`.

### Later: local llama-server (stub only)

The client talks to the same OpenAI-compatible endpoint as `benchmark/`, but
sends **no `tools` array** — the dialect lives in the system prompt.

```bash
# after llama-server is listening on :8080
uv run python experiments/vfs-shell/run.py --all --model llama-server
# or: LLAMA_SERVER_URL=http://127.0.0.1:8080 uv run python experiments/vfs-shell/run.py --all --model llama-server
```

Live SLM runs are **not** part of this spike’s success criteria.

## Layout

```
experiments/vfs-shell/
  README.md              this file
  run.py                 CLI entry (adds this dir to sys.path)
  vfs_shell/
    dialect.py           allowlist + system prompt
    parser.py            text → ToolCall (deny pipes / $())
    vfs.py               MemoryStore + TempDirStore
    backend.py           executor + HarborAdapter seam
    loop.py              multi-turn observe → next command
    models.py            ScriptedModel + LlamaServerModel
    tasks.py             three outcome-scored toy tasks
```

## Next steps toward Terminal-Bench (do not boil the ocean)

1. Keep the dialect frozen. `ToolCall` / `to_argv()` is the agent action.
2. Fill in `HarborAdapter.execute`: `to_argv(call)` → Harbor
   `BaseEnvironment.exec` with **no** `shell=True`. Redirects are a second
   write, not a bash operator inside the container.
3. Wrap `run_episode` as a Harbor `BaseAgent` (Terminus-style single action
   per turn, with this parser in front so unconstrained bash never runs).
4. Run **3–5 easy filesystem tasks** from Harbor examples / Terminal-Bench,
   not the full hard split. Use each task’s own tests as the score.
5. Only then point `LlamaServerModel` at Qwen2.5-3B / Qwen3-4B.

Do not add `harbor` or `terminal-bench` to `pyproject.toml` until that eval
is actually scheduled. Do not route this through `benchmark/run_all.py`.
