# VFS / shell-as-action experiment

Miniature harness for **consecutive command calling**: the model emits
allowlisted Unix-like text (`ls`, `cat`, `grep`, …); a parser maps that onto
a virtual filesystem (today) or BFCL GorillaFileSystem tools (next). Toy-task
success is **final state**, not AST match.

This is a parallel spike. It does **not** change `benchmark/` defaults, and it
is **not** the IFFC/NFC structured-output track. The thesis **primary metric
stays BFCL v4 `multi_turn_base`**; the dialect is a different *interface* to
FS-shaped tools.

Design (includes benchmark ranking): [`docs/vfs-shell-action-space.md`](../../docs/vfs-shell-action-space.md)

## What “success” means (this spike)

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
    bfcl_fs.py           ToolCall → GorillaFileSystem func-string (next eval)
    vfs.py               MemoryStore + TempDirStore
    backend.py           executor + HarborAdapter stretch seam
    loop.py              multi-turn observe → next command
    models.py            ScriptedModel + LlamaServerModel
    tasks.py             three outcome-scored toy tasks
```

## Next steps (ranked; do not boil the ocean)

1. **Keep this toy harness** as the unit-test proof of parse → execute → observe.
2. **Primary: BFCL-FS-shell.** Filter `multi_turn_base` to GorillaFileSystem
   entries. Emit dialect text → `to_bfcl_func_string` → existing
   `execute_multi_turn_func_call` / `multi_turn_checker`. Compare to the
   JSON-tool baseline on the **same** entries. Mapper sketch:
   `vfs_shell/bfcl_fs.py` (cwd-local GFS constraints; `mkdir -p` refuses).
3. **Optional: InterCode-Bash** as an external interactive-bash / FS-delta check.
4. **Optional: AgentBench OS** if a second Docker OS setting is useful.
5. **Stretch only: Terminal-Bench / Harbor.** SE/CLI-heavy; farther from
   “edge SLM tool calling for agentic apps.” Do not add `harbor` to
   `pyproject.toml` unless that appendix is scheduled.
6. **Later (no training in this PR): SFT** an SLM to emit the allowlisted
   dialect, then map commands → GFS tools. Hypothesis: bash/code pretraining
   reduces JSON-schema format burden on consecutive FS tool use. See design
   doc §10.

Do not route this through `benchmark/run_all.py` until an opt-in flag
(default off) exists.
