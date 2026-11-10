# Distillation data build (Stage 1b + 2) and teacher reference run

2026-11-06, Mac, branch `thesis/distill-stage0` (local only, nothing pushed).
Teacher `deepseek/deepseek-v4.1-flash-1017` via OpenRouter, reasoning off, pinned to
provider DeepInfra (DeepSeek's own endpoint is blocked by the OpenRouter privacy
setting). Loop mode for the teacher: `iffc-ooda-card`. All numbers are from logs under
`sft/data/distill/` (gitignored); nothing is in `benchmark.db`.

## Task authoring (`s2a/`)
300 slots, anchor rotation over the 8 backends (file system twice per cycle) ->
**92 valid tasks**, 1,306 teacher calls, $0.3368. Reject reasons (counts of slots
that ended rejected): gold call errored 111, unparseable JSON 48, ungrounded gold
26, duplicate 19, bad initial state 4. Every valid task: gold runs clean in the
real BFCL backend, passes the stock checker against itself, has no overlap with the
BFCL exclusion list, every specific gold value is stated or looked up, and
free-text arguments are dictated verbatim by the user.

Per-backend valid tasks: Trading 40, File system 27, Ticket 24, Twitter 21, Vehicle
18, Math 18, Message 10, Travel 7 (tasks with two backends count for both).

## Solving + verification (`s2a/solve/`)
92 tasks x 4 attempts (attempt 0 greedy like the student, 1-3 at temp 0.7), stock BFCL
checker. **188/368 attempts verified; 55/92 tasks have at least one verified
attempt; greedy attempt alone verified 48/92.** 259 s wall at 8 workers, $0.1217,
0 API errors. Pilot-era comparison: the final task rules took the greedy rate from
6/21 (first pilot) and 0/13 to 48/92.

## Dataset `mlx-t1/` (not yet used for anything)
Verified episodes only, at most 2 per task: 103 episodes, 55 tasks -> **706 rows
(638 train / 68 valid, split by task)**. Stop rows were 38.9% before the cap
(867 rows) and are capped to **24.9%** (161 stop rows dropped at random, seeded).
Look-up calls 376/797 = 47.2% (name-prefix heuristic, not the autopsy regex).
For reference the earlier data: null test stop 23.1% / look-up 37.9%; self-distill
stop 33.2% / look-up 19.7% (THESIS.md Run 74; different heuristic for look-up).
Rows per backend: Trading 363, File system 175, Twitter 164, Message 146, Ticket 123,
Math 121, Travel 87, Vehicle 81. Max rows from one task 25.
Token length with Gemma's chat template: median 5,510, max 7,936, 0 rows over 8,192.

**Leak statement (handoff section 4):** the scan over all 706 rows found 0 rows
containing a BFCL multi-turn task id, 0 rows sharing an 8-gram with any BFCL user
turn, 0 rows with an exact BFCL user turn. Exclusion list: 800 ids, 1,153 unique
user-turn texts, sha256 recorded in `mlx-t1/build_log.json`. Tool names are not
treated as leaks because training must use the BFCL backends.

## Teacher reference run on the 200 tasks (evaluation only, never training data)
Same harness mode, greedy, 1 attempt, stock checker: **155/200 verified**, 168 s wall,
1,891 calls, $0.1005, 0 API errors, 2 stop-on-repeat flags. Files in
`teacher_ceiling_200_EVAL_ONLY/`. Not a registered benchmark run (not in
`benchmark.db`), served by DeepInfra fp8 rather than DeepSeek's endpoint, one
attempt: a reference ceiling, not a paper row without Sans's decision. Gemma for
comparison (from the paper tables): 61/200 JSON text, 77/200 full stack.

## Things to know / open
1. **History format mismatch (hypothesis).** The earlier rows' histories contain
   Gemma's own reasoning text (the harness persists `reasoning_content`); teacher
   histories are clean JSON, and teacher targets are plain pretty-printed JSON with
   no code fence. Whether this matters is untested.
2. **Coverage skew.** Travel (7 tasks) and Vehicle are thin; Trading is 363 rows.
3. **Task style.** Dictated-text and minimal-gold rules make tasks more explicit than
   BFCL's; the verified set is what the teacher solves under the strict checker.
4. **Next (needs Sans's OK + time estimate):** null-scale training (156 rows,
   39 updates, rank 8, LR 5e-6, same config as Run 73) through convert / LoRA /
   fuse / GGUF Q4_K_M / serve, then the gate.

---

# Update after the Codex review (same day)

Review by Codex (`gpt-6.1-sol`, medium effort, read-only): **NO-GO for the first
dataset (`mlx-t1`)**. It re-verified zero contamination (0 id, exact-turn, 8-gram or
>=0.85-similar hits; max similarity 0.7957) but found real quality problems. Addressed in `mlx-t2`, then a second Codex pass
(`docs/reviews/2026-11-06-distillation-data-codex-round2.md`) said NO-GO again and was addressed in
`mlx-t3` (below). `mlx-t1` is superseded by `mlx-t2` and must not be used.

| Codex finding | What I did |
|---|---|
| Critical: gold can run cleanly yet do the wrong thing (task `s2a_0103`: `cp` into a non-folder silently creates a FILE named `vendors`) | Added an independent semantic check (`cp`/`mv` destination without an extension must be a child folder; the backend reports `to '<dest>/<name>'`). Now in `validate()` and in a per-task audit. **9 of 92 tasks quarantined.** |
| Accepted episodes still contain steps whose own calls errored (23 targets) | Rows whose calls errored (seen in the next row's execution results) are dropped; recovery rows stay. **16 dropped.** |
| Grounding / contamination claims overreach | Wording fixed ("checker-accepted"). Literal-overlap report persisted in `build_log.json` (shared gold values are mostly backend built-ins). Grounding stays a heuristic, not a proof. |
| 156-row run not reproducible; 219 duplicate sequences | Exact-duplicate rows removed (**309**), seeded stratified manifest `train_156.jsonl` (whole episodes round-robin over the rarest exercised backend). |
| Training config not materialised | `sft/lora_config_distill_t1.yaml`: identical to the Run 73 config except the adapter path. 156 iterations = 39 optimizer updates (mlx-lm counts micro-batches). |
| Token audit not persisted | Stored in `build_log.json`: median 5,640, max 7,936, 0 rows over 8,192 (HF chat template). |
| Rerun isolation / dedup race | Per-process nonce on backend instance names; duplicate check repeated under the lock at acceptance. |

## Dataset `mlx-t2/` (current)
Verified episodes of the 83 non-quarantined tasks, max 2 per task: 97 episodes, 51
tasks -> **398 rows (365 train, 33 valid)**, stop share 24.9%, look-up share 47.0%
(name-prefix heuristic). Leak scan: 0 / 0 / 0 again. Frozen null-scale subset
(`mlx-t2-null156/`): **156 unique rows, 26 tasks, stop 21.1%, look-up 46.2%**,
rows by exercised backend: Trading 81, Message 47, Twitter 32, Math 32, Travel 28,
File system 27, Ticket 27, Vehicle 20. Hashes in `build_log.json`.


# Round 2 (Codex NO-GO on `mlx-t2`) -> `mlx-t3`
Verified OK by Codex: contamination on the 156-row set (0 id / exact / 8-gram / >=0.85 hits, max
similarity 0.7957), frozen files match recorded hashes, config equals Run 73 except adapter path,
156 micro-batches / accumulation 4 = 39 optimizer updates, row format and masking.
Defects it found and what changed:
* copy/move check skipped dotted destinations (`'..'` created a file named `..`): now any path or
  parent-directory destination is rejected (`semantic_issue`).
* five tasks quarantined by hand from its findings (`manual_quarantine.json`): 0005 (precision
  means significant digits), 0067 and 0098 (`..` moves), 0105 (`received_count` for "sent"), 0165
  (approximated unobserved division, over budget).
* teacher targets that guessed values (filenames never listed, unobserved numbers) are now dropped
  row by row (`ungrounded_target`: a value must appear in user-visible text, a file name with an
  extension must appear verbatim, free text must be dictated). 10 rows dropped.
* per-invocation nonce for backend instance names (author and solver).
Not claimed: that every remaining step is optimal. Grounding is a heuristic, the checker is the
filter, and Codex has not yet reviewed `mlx-t3`.

## Dataset `mlx-t3/` (current)
51 -> 46 tasks (14 auto-quarantined + 5 manual overlap), 88 episodes, **342 rows (311 train / 31 valid)**,
stop 24.9%, look-up 47.3%, leak scan 0/0/0, tokens median 5,459 max 7,982, none over 8,192.
Frozen null-scale subset `mlx-t3-null156/`: **156 rows, 25 tasks, stop 22.4%, look-up 47.7%**; rows by
exercised backend Trading 92, Travel 27, Message 40, Math 37, Ticket 26, Twitter 23, File system 21,
Vehicle 16. Launch directly with `mlx_lm.lora --config sft/lora_config_distill_t1.yaml` (the default
`sft/train_mlx.py` entry point rebuilds data and applies older overrides).
