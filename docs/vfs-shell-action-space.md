# Virtual filesystem / shell-as-action space

A parallel experiment for consecutive tool calling. It does **not** change
`benchmark/` defaults and is **not** the IFFC/NFC structured-output track.
The thesis **primary metric remains BFCL v4 `multi_turn_base`**; this spike
asks whether a **terminal/VFS command dialect** is a better *interface* to
those same FS tools.

**Scaffold:** [`experiments/vfs-shell/`](../experiments/vfs-shell/)  
**Run / scoring:** [`experiments/vfs-shell/README.md`](../experiments/vfs-shell/README.md)

## 1. Claim this spike tests

Small models (<5B, quantized, Apple Silicon / edge) are often asked to emit
JSON tool calls (`{"name": ..., "arguments": ...}`) whose syntax they were not
primarily trained on. A competing interface is **familiar terminal text**:
`ls`, `cat`, `grep`, `cd`, … — a small, parseable command dialect that a
harness maps onto backend operations / named tools.

The question for the thesis is **multi-turn correctness** (did the *sequence*
of actions leave the environment in the right state and invoke the right
tools?), not “can we build a software-engineering or sysadmin agent.”

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

**natural command text → constrained parser → structured backend call / tool**

rather than **JSON schema → grammar-constrained `tool_calls`**.

This scaffold copies that *shape* (allowlist + VFS + deny-by-default) at toy
scale. It does not reimplement Hivemind, Deeplake, or agent-memory.

Applied to BFCL: the model would emit `cd notes` instead of
`{"name": "cd", "arguments": "{\"folder\": \"notes\"}"}`; the mapper produces
the GorillaFileSystem call `cd(folder='notes')` that
`execute_multi_turn_func_call` already knows how to run.

## 3. Which benchmarks fit this thesis (ranked)

Thesis target: **optimize consecutive / multi-turn tool calling for SLMs
<5B in agentic apps.** Current primary reported metric:
**BFCL v4 `multi_turn_base`** (execution checker on stateful backends,
including GorillaFileSystem). The action-space experiment should stay
close to that claim — FS-shaped tools, not SWE/sysadmin.

| Rank | Setting | Why it fits | Why it does not |
|------|---------|-------------|-----------------|
| **1 — primary next** | **BFCL multi-turn GorillaFileSystem** (subset of `multi_turn_base` / miss_* / long_context that `involved_classes` includes `GorillaFileSystem`) | Tools already *are* `ls`, `cd`, `cat`, `mv`, `cp`, `mkdir`, `pwd`, `grep`, `find`, `echo`, `wc`, `rm`. Multi-turn scoring is **execution + final FS state** (plus required ground-truth steps), not single-turn AST. A shell dialect → BFCL func-string mapper is the closest test of “same thesis metric, different interface.” Existing `benchmark/generate_multi_turn.py` / `evaluate_multi_turn.py` stay the eval backend. | GFS is cwd-local and one-folder-at-a-time (`cd` rejects nested paths; `cat`/`mv` dest take a name, not a path). The dialect is a bit richer (`mkdir -p`, redirects). Mapper must be explicit about those gaps — not pretend POSIX. Other `multi_turn_base` backends (Twitter, Travel, Vehicle, …) are **not** a natural shell dialect. |
| **2** | **InterCode-Bash** (Yang et al., 2023, [arXiv:2306.14898](https://arxiv.org/abs/2306.14898)) | Native interactive multi-turn bash; reward from stdout **and** filesystem delta; Docker; closer to “command as action” than JSON tools. | Broader bash than our allowlist; not the thesis’s existing metric; extra harness. Use if we want an external FS-command check after BFCL-FS-shell. |
| **3** | **AgentBench OS** (Liu et al., 2023, [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)) | Ubuntu Docker, multi-turn bash, success rate on deterministic OS questions / operation sequences. | Same “real OS” distance as InterCode; farther from BFCL tool-calling claim; easy to slide into sysadmin. Optional corroboration only. |
| **4 — stretch / external** | **Terminal-Bench** ([tbench.ai](https://www.tbench.ai), Harbor, [arXiv:2601.11868](https://arxiv.org/abs/2601.11868)) | Outcome tests on final container state; Terminus-style one-action-per-turn. | SE/CLI-heavy (compile, debug, servers). Farther from “edge SLM tool calling for agentic apps.” **Not** the default or best bench for this thesis. |

**Recommendation:** implement **BFCL-FS-shell** next (mapper + FS-subset of
`multi_turn_base`). Treat **InterCode-Bash** as the first *external* FS
check if the mapper result is interesting. Do **not** plan a Terminal-Bench
leaderboard run unless the thesis later needs a stretch “real CLI” appendix.

IFFC/NFC (JSON/XML `tool_calls`, GBNF) remains the right track for
*format* on the same BFCL metric — that work stays on the other branch.
This experiment only changes the **surface language** the SLM emits for
FS-like tools.

## 4. Mapping: constrained dialect → tool / exec API

### Dialect (this spike)

One command per turn, POSIX-like argv, **no** pipes, `&&`, `;`, or
`$()` / backticks. Allowlist:

`ls`, `cat`, `grep`, `cd`, `pwd`, `mkdir`, `mv`, `cp`, `rm`, `echo`, `wc`,
`find`, plus `done` (optional stop token).

Optional shell-like **stdout redirect** (`>` / `>>`) is parsed as part of
the dialect (Hivemind intercepts redirects on VFS paths too) and mapped to a
`write` backend op — or, on the BFCL path, to `echo(content=…, file_name=…)`
when the redirect target is a **local name**.

### Parser → structured call

```
"grep ERROR app.log"
        │  shlex + allowlist + deny unsafe tokens
        ▼
ToolCall(name="grep", positional=("ERROR", "app.log"))
        │  BFCL-FS mapper (next experiment, sketched in vfs_shell/bfcl_fs.py)
        ▼
"grep(file_name='app.log', pattern='ERROR')"
        │  existing execute_multi_turn_func_call
        ▼
GorillaFileSystem.grep(...)  → observation → next turn
```

Unknown verbs, pipes, and command substitution return a **parse error
observation** (nothing executes). The loop can feed that back — a cheap
repair signal for SLMs.

### Backend (this spike vs next)

`ToolCall` is the stable interface.

**Today (toy harness, keep as-is):**

- **Memory VFS** — in-process tree; default for tests.
- **Temp-dir VFS** — same ops on a real `tempfile` tree (no `/bin/sh`).

**Next (not wired into `benchmark/run_all.py` in this PR):**

- **`to_bfcl_func_string` / `from_bfcl_func_string`** — dialect ↔ GorillaFileSystem
  call string (`cd notes` ↔ `cd(folder='notes')`). Documented overlap and refusals
  live in `experiments/vfs-shell/vfs_shell/bfcl_fs.py`. Live
  `run_bfcl_fs.py` executes those strings on **BFCL GorillaFileSystem**, not the
  toy `MemoryStore` VFS (toys stay hermetic in `run.py` only).
- **InterCode / AgentBench adapters** — later; same `ToolCall`, different
  exec sandbox.
- **HarborAdapter** — stretch only (Terminal-Bench). `ToolCall` → argv →
  `environment.exec` with no `shell=True`. Not a thesis milestone.

### Loop

```
instruction → model emits command text
           → parse (or repair observation)
           → execute (toy VFS *or* mapped BFCL tool)
           → append to transcript
           → repeat until `done` or max turns
           → score = FS state / BFCL multi_turn_checker
```

This is consecutive tool calling: each command may depend on the previous
observation (`grep` then `mv` the file that matched).

## 5. How to score multi-turn correctness

**Toy harness (this PR):** do **not** AST-match the command string against a
gold script. Score **state / outcome** on the VFS:

- files exist / do not exist at expected paths
- file contents contain required substrings
- directories were created

The harness still *records* the parsed `ToolCall` trace for diagnostics
(parse-success rate, turns-to-success, repair count).

A scripted “model” that emits the oracle command list must pass; a scripted
model that skips a step must fail.

**BFCL-FS-shell (recommended next):** reuse `multi_turn_checker` on the
mapped func-call traces. BFCL multi-turn already combines **final instance
state** with required ground-truth tool steps (exploratory `pwd`/`ls` need
not match). That keeps the thesis metric comparable to the JSON-tool
baseline on the same entries — only the model’s *emission format* changes.

Do not invent a new leaderboard number in this spike.

## 6. Fit for SLMs <5B on Apple Silicon / edge

Why this interface is *plausible* for the thesis hardware (M4 Pro Metal,
later Jetson Orin Nano Super):

- **Small vocab.** A dozen verbs plus short flags is a tighter language
  model than arbitrary JSON schemas with nested objects. SLMs already saw
  Unix commands in pretraining; they saw fewer well-formed `tool_calls`.
- **Parseability.** `shlex` + allowlist fails loud. JSON that is 90%
  correct often fails a schema/AST parse with no recoverable structure.
- **Repair loops.** A one-line `PARSE ERROR: pipes are not allowed` is a
  cheaper retry than regenerating a full JSON blob under a grammar.
- **No extra grammar compiler required** for the command track (IFFC can
  keep GBNF). This track can stay at `temperature=0` text completion
  against an OpenAI-compatible `llama-server` (stub in the scaffold).
- **Context.** Command + short stdout is compact versus dumping tool JSON
  schemas every turn — relevant at 4k–8k context on edge.

This is a **hypothesis**, not a result. The spike only proves the harness
and the mapper sketch. Live <5B numbers belong on BFCL-FS-shell (and
optionally InterCode), not on Terminal-Bench.

## 7. Explicit non-goals

- **Not a SWE-agent.** No patch editor, linter, test runner, or repo-scale
  navigation. SWE-agent’s result that a custom ACI beat a raw Linux shell
  (Yang et al., 2024) is a *caution*: unconstrained bash is not automatically
  the best interface. This dialect stays small on purpose.
- **Not “unconstrained shell as silver bullet.”** Pipes, `python -c`,
  `curl`, and `find -exec` are denied. If the model needs composition, it
  uses extra turns (consecutive calling), not a hidden bash.
- **Not rewriting IFFC / NFC.** JSON/XML tool-call parsers, GBNF, and the
  default BFCL generation path stay in `benchmark/`.
- **Not a Terminal-Bench (or Harbor) milestone.** No Harbor dependency, no
  Docker task pack, no reported TB score. Stretch only.
- **Not sysadmin / full Ubuntu agentics.** AgentBench OS is rank 3, not a
  product goal.
- **Not Hivemind memory / skills.** No SQL VFS, no session capture.
- **Not training in this PR.** Fine-tune / SFT on VFS commands is
  **§10 later work**.

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
- **BFCL v3/v4 multi-turn** ([Gorilla blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)) —
  GorillaFileSystem plus other stateful APIs; execution checker; the
  thesis’s existing metric.
- **InterCode-Bash** (Yang et al., 2023, [arXiv:2306.14898](https://arxiv.org/abs/2306.14898));
  **AgentBench OS** (Liu et al., 2023, [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)) —
  interactive OS/bash success-rate settings (ranks 2–3 above).
- **Terminal-Bench / Harbor / Terminus** — outcome-verified CLI tasks;
  stretch / external only (rank 4).

## 9. Recommended next steps (do not boil the ocean)

1. **Keep the toy harness.** Frozen dialect + parser + VFS loop; scripted
   oracles stay the unit-test proof.
2. **BFCL-FS-shell (primary; landed).** CLI:
   `experiments/vfs-shell/run_bfcl_fs.py --slice fs-only|any-fs`.
   Filter `multi_turn_base` to `GorillaFileSystem`. Model emits dialect
   text (no `tools` array). `parse_command` → `to_bfcl_func_string` →
   existing `execute_multi_turn_func_call` / `multi_turn_checker`.
   Observations reverse-map to shell (`ls -a -> …`). Result JSONL keeps
   BFCL `result` plus raw `shell_emissions`.
   Compare against NFC/IFFC **on the same `--dry-run` IDs**. Mac commands
   are in `experiments/vfs-shell/README.md`.
3. Respect GFS constraints in the mapper: cwd-local names, `cd` one folder
   level, `echo` to an existing file (`touch` first), no `mkdir -p`.
   Unmappable dialect features → parse/mapping error observation, not a
   silent POSIX pretend.
4. **InterCode-Bash (optional second).** Same dialect, execution reward /
   FS delta, if a check outside BFCL is needed.
5. **AgentBench OS / Terminal-Bench.** Only if the committee wants an
   external CLI appendix. Do not add `harbor` or `terminal-bench` to
   `pyproject.toml` until that is actually scheduled.
6. Point `LlamaServerModel` at local `llama-server` (`--jinja`, `--temp 0.0`)
   for Qwen2.5-3B / Qwen3-4B / FunctionGemma — same models as the BFCL
   pipeline, FS-shell action space.

Do not route this through `benchmark/run_all.py` until the mapper is an
opt-in flag (e.g. `--action-space shell`) that defaults **off**.

## 10. Later experiment (do not implement now): SFT on VFS commands

**Hypothesis.** Pretraining already contains large amounts of bash and
code. Fine-tuning an SLM to emit the *allowlisted* shell dialect against a
VFS (Hivemind-style), then mapping commands → GorillaFileSystem / toy-VFS
ops at eval time, should reduce **format burden** relative to JSON tool
schemas for multi-turn FS tool use — especially consecutive `ls`/`cd`/`mv`
chains.

**Not in this PR:** no datasets, no QLoRA, no training loop.

**Sketch for a later methods chapter:**

1. Collect trajectories: toy-harness oracles, BFCL-FS episodes translated
   from gold `cd(folder=…)` traces into dialect text, optional InterCode
   successes filtered to the allowlist.
2. SFT (QLoRA on Qwen2.5-3B / Qwen3-4B, same hardware story as METHODS §18)
   with target = command line(s), not JSON `tool_calls`.
3. Eval with the mapper on the **same** BFCL-FS subset so the metric stays
   `multi_turn_base` (FS slice). Ablate: JSON-tool SFT vs shell-dialect SFT
   vs prompt-only dialect.
4. Report accuracy, parse-error rate, turns-to-success, tokens — not a
   Terminal-Bench score.

Until that run exists, treat §6 as a hypothesis and §9 step 2 as the
inference-only comparison.
