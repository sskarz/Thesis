# Virtual filesystem / shell-as-action space

A parallel experiment for consecutive tool calling. It is **not** the BFCL /
IFFC / NFC track: nothing here is wired into `benchmark/` defaults.

**Scaffold:** [`experiments/vfs-shell/`](../experiments/vfs-shell/)  
**Run / scoring:** [`experiments/vfs-shell/README.md`](../experiments/vfs-shell/README.md)

## 1. Claim this spike tests

Small models (<5B, quantized, Apple Silicon / edge) are often asked to emit
JSON tool calls (`{"name": ..., "arguments": ...}`) whose syntax they were not
primarily trained on. A competing interface is **familiar terminal text**:
`ls`, `cat`, `grep`, `cd`, … — a small, parseable command dialect that a
harness maps onto backend operations.

The question for the thesis is **multi-turn correctness** (did the *sequence*
of actions leave the environment in the right state?), not “can we build a
software-engineering agent.”

## 2. How Hivemind-style command intercept works

[Activeloop Hivemind](https://github.com/activeloopai/hivemind) is a shared
memory layer for coding agents. The part that matters here is **not** the
memory product; it is the **intercept compiler**:

```
model emits Bash (the native tool of Claude Code / Codex / Cursor)
        │
        ▼
pre-tool hook: does this command touch ~/.deeplake/memory/?
        │
        ├─ no  → pass through to the real shell
        └─ yes → allowlisted parser / compiler
                    │
                    ├─ known builtin (ls, cat, grep, find, mv, …)
                    │     → compile to backend ops (SQL-backed VFS)
                    │     → return stdout as if the files were on disk
                    └─ unrecognized / unsafe (python, curl, $(), find -exec)
                          → deny (do not execute on the host)
```

Documented properties of that compiler (from the public repo, not from our
measurements):

- Path `~/.deeplake/memory/` is a **virtual filesystem** backed by tables,
  not a host directory the model can escape into.
- On the order of **~70 allowlisted builtins** run in the VFS; anything else
  is denied.
- Pipelines and redirections that *touch* the memory path are intercepted
  (`cat a | grep x`, `echo hi > ~/.deeplake/memory/note.md`).
- Interpreters, fetchers, and `find -exec` / `--to-command` are blocked
  because they re-enter the host.

The thesis-relevant pattern is therefore:

**natural command text → constrained parser → structured backend call**

rather than **JSON schema → grammar-constrained `tool_calls`**.

This scaffold copies that *shape* (allowlist + VFS + deny-by-default) at toy
scale. It does not reimplement Hivemind, Deeplake, or agent-memory.

## 3. Why Terminal-Bench fits this idea better than BFCL

| | **BFCL v4 multi-turn** (existing pipeline) | **Terminal-Bench** (this experiment’s target) |
|---|---|---|
| Action the model emits | OpenAI-style JSON / XML `tool_calls` | Terminal commands in a container |
| Checker | AST match to a reference call (single-turn); execution of *named* Python APIs (multi-turn) | **Outcome tests** on the final environment |
| Environment | Simulated function backends (`GorillaFileSystem`, etc.) | Docker / Harbor sandbox; real CLI |
| What “wrong format, right intent” scores | Usually **fail** (parse / AST) | Can still **pass** if the files/state are correct |
| Natural interface for `ls`/`grep`/`mv` | Indirect (JSON args to `cd(folder=…)`) | Direct |

BFCL is the right benchmark for the **IFFC / NFC / structured-output**
question (does the model emit a valid function call under a schema?). That
work stays on the other branch and in `benchmark/`.

It is the *wrong* primary benchmark for a **shell dialect as the action
space**, because:

1. BFCL multi-turn still expects *function-call objects*, not `mkdir -p /notes`.
2. AST checking scores *form*, not “did consecutive actions compose.”
3. Teaching an SLM two action languages (JSON tools *and* bash) to evaluate
   a bash-shaped idea confounds the measurement.

**Terminal-Bench** ([tbench.ai](https://www.tbench.ai), Harbor,
[arXiv:2601.11868](https://arxiv.org/abs/2601.11868)) is built for this
shape: English instruction → agent loop in a container → **tests inspect the
final container state, not the command trace**. Terminus-2 is Harbor’s
reference agent (interactive tmux / shell, not a JSON tool catalog). A
*thin subset* of easy file-system tasks is enough to measure consecutive
calling; a full leaderboard run is explicitly out of scope for this spike.

**Related (not the scaffold’s runtime):**

- **InterCode-Bash** (Yang et al., 2023) — interactive bash, execution
  feedback, Docker; reward from stdout *and* filesystem delta.
- **AgentBench OS** (Liu et al., 2023) — Ubuntu Docker, success rate on
  deterministic OS questions / operation sequences.

## 4. Mapping: constrained dialect → tool / exec API

### Dialect (this spike)

One command per turn, POSIX-like argv, **no** pipes, `&&`, `;`, or
`$()` / backticks. Allowlist:

`ls`, `cat`, `grep`, `cd`, `pwd`, `mkdir`, `mv`, `cp`, `rm`, `echo`, `wc`,
`find`, plus `done` (optional stop token).

Optional shell-like **stdout redirect** (`>` / `>>`) is parsed as part of
the dialect (Hivemind intercepts redirects on VFS paths too) and mapped to a
`write` backend op. That keeps “grep matches into a file” expressible
without pipes.

### Parser → structured call

```
"grep -n ERROR /logs/app.log > /logs/errors.txt"
        │  shlex + allowlist + deny unsafe tokens
        ▼
ToolCall(
  name="grep",
  flags={"n"},
  positional=("ERROR", "/logs/app.log"),
  redirect=Redirect(op=">", path="/logs/errors.txt"),
)
```

Unknown verbs, pipes, and command substitution return a **parse error
observation** (nothing executes). The loop can feed that back — a cheap
repair signal for SLMs.

### Backend

`ToolCall` is the stable interface. Two stores implement it today:

- **Memory VFS** — in-process tree; default for tests.
- **Temp-dir VFS** — same ops on a real `tempfile` tree (no `/bin/sh`).

A **Harbor / Terminus adapter** is a documented seam: `ToolCall` → argv
list → `environment.exec(argv)` (no `shell=True`). The toy tasks never
require that path.

### Loop

```
instruction → model emits command text
           → parse (or repair observation)
           → execute → stdout/stderr
           → append to transcript
           → repeat until `done` or max turns
           → score = programmatic checks on final state
```

This is consecutive tool calling: each command may depend on the previous
observation (`grep` then `mv` the file that matched).

## 5. How to score multi-turn correctness

**Do not** AST-match the command string against a gold script.

Score **state / outcome**, in the Terminal-Bench sense:

- files exist / do not exist at expected paths
- file contents contain required substrings
- directories were created; cwd is whatever the task asked for

The harness still *records* the parsed `ToolCall` trace so we can later
report auxiliary metrics (parse-success rate, turns-to-success, repair
count) without making them the pass/fail criterion.

A scripted “model” that emits the oracle command list must pass; a scripted
model that skips a step must fail. That is the miniature proof the loop
measures composition, not parser self-consistency.

## 6. Fit for SLMs <5B on Apple Silicon / edge

Why this interface is *plausible* for the thesis hardware (M4 Pro Metal,
later Jetson Orin Nano Super):

- **Small vocab.** A dozen verbs plus short flags is a tighter language
  model than arbitrary JSON schemas with nested objects. SLMs already saw
  Unix commands in pretraining; they saw fewer well-formed `tool_calls`.
- **Parseability.** `shlex` + allowlist fails loud. JSON that is 90%
  correct often fails the BFCL AST checker with no recoverable structure.
- **Repair loops.** A one-line `PARSE ERROR: pipes are not allowed` is a
  cheaper retry than regenerating a full JSON blob under a grammar.
- **No extra grammar compiler.** The existing llama.cpp GBNF path is
  valuable for IFFC; this track can stay at `temperature=0` text completion
  against an OpenAI-compatible `llama-server` (stub in the scaffold).
- **Context.** Command + short stdout is compact versus dumping tool JSON
  schemas every turn — relevant at 4k–8k context on edge.

This is a **hypothesis**, not a result. The spike only proves the harness.
Live <5B numbers belong in a later Harbor / llama-server run.

## 7. Explicit non-goals

- **Not a SWE-agent.** No patch editor, linter, test runner, or repo-scale
  navigation. SWE-agent’s result that a custom ACI beat a raw Linux shell
  (Yang et al., 2024) is a *caution*: unconstrained bash is not automatically
  the best interface. This dialect stays small on purpose.
- **Not “unconstrained shell as silver bullet.”** Pipes, `python -c`,
  `curl`, and `find -exec` are denied. If the model needs composition, it
  uses extra turns (consecutive calling), not a hidden bash.
- **Not rewriting IFFC / NFC.** JSON/XML tool-call parsers, GBNF, and BFCL
  AST evaluation stay in `benchmark/`. This experiment must not change
  those defaults.
- **Not a Terminal-Bench leaderboard submission.** No Harbor dependency, no
  Docker task pack, no reported TB2 score.
- **Not Hivemind memory / skills.** No SQL VFS, no session capture.

## 8. Prior art (brief, no new numbers)

- **CodeAct** (Wang et al., 2024, [arXiv:2402.01030](https://arxiv.org/abs/2402.01030)) —
  executable Python as a unified action space, with interpreter feedback.
  Same family of idea (code-shaped actions); different language (Python vs
  a tiny Unix dialect).
- **SWE-agent ACI** (Yang et al., 2024, [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)) —
  a *small* LM-oriented command set (search, view, edit) outperformed the
  default Linux shell on SWE-bench in their study. Lesson used here:
  constrain and structure the interface; do not hand the model `/bin/bash`.
- **Hivemind** ([activeloopai/hivemind](https://github.com/activeloopai/hivemind)) —
  allowlisted bash → compiler → virtual FS. Mechanism we mimic at toy scale.
- **Terminal-Bench / Harbor / Terminus** ([tbench.ai](https://www.tbench.ai),
  [harbor-framework](https://github.com/laude-institute/harbor)) —
  outcome-verified CLI tasks; the intended next eval, not this PR’s runtime.
- **InterCode-Bash** (Yang et al., 2023, [arXiv:2306.14898](https://arxiv.org/abs/2306.14898));
  **AgentBench OS** (Liu et al., 2023, [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)) —
  earlier interactive OS/bash success-rate settings.

## 9. Path to a Terminus-style eval (without boiling the ocean)

1. Keep this dialect + parser frozen; treat `ToolCall` as the agent’s only
   action type.
2. Implement `HarborAdapter.execute` as argv → `BaseEnvironment.exec`
   (already sketched in `vfs_shell/backend.py`).
3. Pick **3–5 easy file-system tasks** from Terminal-Bench / Harbor examples
   (create/move/grep/count), not the full 2.0 hard split.
4. Wrap the existing `run_episode` loop as a Harbor `BaseAgent` that emits
   one allowlisted command per turn (Terminus-style *single tool*, but with
   our parser in front so unconstrained bash never hits the container).
5. Score with the task’s own tests (final state). Report parse-success and
   turns as diagnostics only.
6. Point `LlamaServerModel` at local `llama-server` (`--jinja`, `--temp 0.0`)
   for Qwen2.5-3B / Qwen3-4B / FunctionGemma — same models as the BFCL
   pipeline, different action space.

Stop there until those few tasks produce a real SLM number. Do not import
Harbor into `pyproject.toml` until that run is actually scheduled.
