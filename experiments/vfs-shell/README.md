# VFS / shell-as-action experiment

Miniature harness for **consecutive command calling**: the model emits
allowlisted Unix-like text (`ls`, `cat`, `grep`, …); a parser maps that onto
a toy VFS (scripted tasks) or BFCL GorillaFileSystem tools
(`run_bfcl_fs.py`). Toy-task success is **final state**; BFCL-FS uses the
existing `multi_turn_checker`.

This is a parallel spike. It does **not** change `benchmark/` defaults, and it
is **not** the IFFC/NFC structured-output track. The thesis **primary metric
stays BFCL v4 `multi_turn_base`**; the dialect is a different *interface* to
FS-shaped tools.

Live `run_bfcl_fs.py` uses BFCL **GorillaFileSystem**
(`execute_multi_turn_func_call` / `multi_turn_checker`). The toy in-memory
`MemoryStore` VFS (and temp-dir store) is only for hermetic `run.py` toys and
unit tests — it is not the live eval environment.

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

Toy-task live SLM runs are optional. The BFCL-FS harness below is the eval
entrypoint for Gemma-4-E2B-it.

## Layout

```
experiments/vfs-shell/
  README.md              this file
  run.py                 CLI entry (adds this dir to sys.path)
  vfs_shell/
    dialect.py           allowlist + toy / GFS system prompts
    parser.py            text → ToolCall (deny pipes / $())
    bfcl_fs.py           ToolCall ↔ GorillaFileSystem func-string
    bfcl_fs_eval.py      multi-turn shell loop + multi_turn_checker
    slice.py             fs-only / any-fs filters
    vfs.py               MemoryStore + TempDirStore
    backend.py           toy VFS executor + Harbor stretch seam
    loop.py              toy multi-turn observe → next command
    models.py            ScriptedModel + LlamaServerModel
    tasks.py             three outcome-scored toy tasks
  run_bfcl_fs.py         FS-slice CLI (does not touch NFC/IFFC)
  fixtures/              recorded oracle + slice fixtures
```

## BFCL-FS eval harness (shell → GorillaFileSystem)

Format ablation on the **FS slice** of `multi_turn_base`. Does **not** change
NFC or IFFC (`benchmark.run_all --fc-mode nfc|iffc`). The model emits Unix-like
text; observations and JSONL `shell_emissions` keep that shell form. The
`result` field is still BFCL func-strings for `multi_turn_checker`.

| Slice | Filter | Size on v4 `multi_turn_base` |
|-------|--------|------------------------------|
| `fs-only` | `involved_classes == ["GorillaFileSystem"]` | 13 |
| `any-fs` | `"GorillaFileSystem" in involved_classes` | 50 (includes mixed Twitter/Math/…) |

`any-fs` mixed backends still only expose FS commands; non-FS user turns will
not be solvable in this ablation. Prefer `fs-only` for the fair NFC/IFFC
comparison on the same IDs.

### Dry-run / hermetic (no llama-server)

```bash
# List IDs (uses installed bfcl-eval data)
uv run python experiments/vfs-shell/run_bfcl_fs.py --slice fs-only --dry-run
uv run python experiments/vfs-shell/run_bfcl_fs.py --slice any-fs --dry-run

# Recorded oracle on the first fs-only entry + multi_turn_checker
uv run python experiments/vfs-shell/run_bfcl_fs.py \
  --slice fs-only --limit 1 \
  --recorded experiments/vfs-shell/fixtures/recorded_multi_turn_base_1.jsonl

# Full hermetic suite (mapper + toys + FS eval)
uv run pytest experiments/vfs-shell/tests -q
```

### Live Mac eval (Gemma-4-E2B-it on :8080)

Same server Sanskar already uses for NFC Run 4 / IFFC Run 5. Do **not** pass
`tools` — this harness never sends them.

If the server is not up:

```bash
cd llama.cpp
./build/bin/llama-server --jinja -fa on \
  -hf bartowski/google_gemma-4-E2B-it-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 65536 --temp 0.0 \
  -np 4 -cb -ub 1024
```

Then, from the repo root (leave that server running):

```bash
# FS-only (~13) — primary fair slice
uv run python experiments/vfs-shell/run_bfcl_fs.py \
  --model gemma4-e2b \
  --slice fs-only \
  --server-url http://localhost:8080 \
  --out experiments/vfs-shell/results/shell_fs_only.jsonl

# any-FS (~50)
uv run python experiments/vfs-shell/run_bfcl_fs.py \
  --model gemma4-e2b \
  --slice any-fs \
  --server-url http://localhost:8080 \
  --out experiments/vfs-shell/results/shell_any_fs.jsonl
```

### Compare to NFC / IFFC controls (do not re-run unless you want a refresh)

Controls stay on `benchmark.run_all` and the default `--fc-mode nfc`:

```bash
# NFC control (Run 4: 54/200 = 27% on full multi_turn_base)
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode nfc

# IFFC control (Run 5: 61/200 = 30.5% on full multi_turn_base)
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc
```

Shell-FS prints `correct/total` **on the slice only**. Do not compare 13- or
50-entry accuracy to the full 200. For an apples-to-apples number, score NFC
and IFFC traces on the IDs from `--dry-run`.

## Next steps (out of this PR)

InterCode / AgentBench / Terminal-Bench are not this harness. SFT on VFS
commands is later work (design doc §10).
