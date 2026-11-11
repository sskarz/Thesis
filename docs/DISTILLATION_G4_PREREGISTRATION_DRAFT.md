# G4: BFCL 200 evaluation of the trained arm, pre-registration DRAFT (written 2026-11-11; NOT approved; nothing is run)

**Status.** Draft for the advisor's review (note 2026-11-11 08:27 UTC) and one Codex pass. **No run on the 200 BFCL tasks starts without Sans's own go in the chat** that names the run, its time estimate and its cost. This file only fixes what must be settled before such a run. Items marked **[OPEN]** need a decision by Sans; items marked **[PROPOSAL]** are my suggestion and are not numbers from the record. Numbers not marked come from the paper (`paper/main.tex` on main), `METHODS.md`, `benchmark.db` (read-only) or `docs/TRAINING_RESULTS_LOG.md`.

## 1. Question
Does the trained arm, served with the compact-list stack, solve more of the 200 BFCL v4 `multi_turn_base` tasks than the untuned model with the same stack, in a paired comparison on the same tasks, server build and day? The screening result (arm A: 118 of 155 on F + K2 against 59, 61 and 54 for the untuned model) is **not** a BFCL result: F and K2 come from the same generator and teacher as the training tasks.

## 2. The reference stack: what must be pinned before any run
* **Runs 84 and 85** (`benchmark.db`, `runs` table; paper Table of the shorter list): Run 84 = `fc_mode=iffc-state-ooda-slim`, 88 of 200 (44.0%); Run 85 = `fc_mode=iffc-state-ooda`, 72 of 200 (36.0%), both `multi_turn_base`, base Q4_K_M, "R70 flags", order on-off, 2026-10-29 21:22 to 2026-10-30 02:50 UTC (about 2 h 29 min and 2 h 59 min). The `runs` table stores no command line (`server_params` is empty), so the **exact command line cannot be read back from the database**. **[OPEN, blocks launch]** reconstruct it from `METHODS.md` (the Run 78 to 85 sections), the `R70 flags` definition, and the worktree state; list server flags (`-c`, `-np`, `-ub`, `--jinja`, `-fa`), `run_all` flags and the thinking setting; get Sans's or the advisor's confirmation. Mode code: `iffc-state-ooda-slim` exists on `thesis/compact-list-r81` (`8f62b8a`, the commit Runs 84/85 are attributed to) and as the port `dfdbc2b` (`thesis/slim-port`, used for G2 and G3). Offline render of the 200 opening prompts is identical between the two (200 of 200, text and message hash).
* **Adapter server renders the same prompts.** Requirement before the run: a server-side capture (first request of at least 20 BFCL tasks) of the adapter server and of the untuned server must match the offline render of the same mode, task by task (hash of the rendered text), with the right think-marker count for the arm (0 for thinking off, 1 for on). G3's parity check did this for training rows only, not for BFCL prompts.
* **Server build and flags:** llama.cpp `b10894-d344123fe`, bar Q4_K_M GGUF with prefix `923c4c86177d2ee1`, LoRA GGUF applied at runtime at scale 1.0 for the arm, flags as in the G3 chain (`--jinja -fa on -c 65536 -np 4 -cb -ub 1024`, `--temp 0.0`) **[OPEN: confirm these equal the flags of Runs 84/85]**.

## 2b. Binding the evaluated adapter to the screened one
Before launch, freeze and record: the checkpoint directory and its `adapters.safetensors` SHA-256; the converted LoRA GGUF path and SHA-256; the screening reports (`G3_*_RUN1.json`, `G3_*_REPORT.json`) that scored exactly that GGUF (their `adapter_sha` files in `runlogs/`). At server start the chain checks `/lora-adapters`: exactly one adapter, path and SHA-256 equal to the frozen GGUF, scale 1.0; the pass-1 checkpoint and any other conversion are refused by name. Comparator servers start with no LoRA (`/lora-adapters` empty). The identity JSON is saved with each run.

## 2c. Deadlines and cleanup (enforced by the run script, as in the G3 chain)
* **Per-run cap [PROPOSAL]:** 4.5 h (1.5 times the longest recorded run, 2 h 59 min); the evaluator is killed at the cap and the run is void, not partial.
* **Whole-chain cap [PROPOSAL]:** number of runs times the per-run cap plus 1 h, from a persisted start time.
* **Cleanup on every exit** (success, failure, cancel, timeout, signal): a trap stops only the processes the chain started (evaluator process group and its own `llama-server`), then verifies that port 8080 has no listener and no `llama-server` of the chain remains; the check result is written to the log. A failed cleanup check is reported as a failure.

## 3. Comparators and the primary comparison
Arm A is trained and served with thinking OFF. The untuned model scored better with thinking ON on the screening banks (59 and 61 against 54 of 155), and the paper's reported stack uses the default template (thinking on). So an untuned comparison with thinking off alone would flatter the arm.
* **Comparator 1 (primary): untuned model, same stack, thinking ON,** a fresh run on the same server build and day (not the 88 of 200 from Run 84, which was a different day; the paper's own runs of one configuration differed by up to 6 tasks).
* **Comparator 2 (secondary): untuned model, thinking OFF,** fresh run, because it is the arm's own serving mode.
* **Arm:** the arm chosen by the registered rule in the G3c pre-registration (higher run-1 FK composite, ties to arm A) **if the combined retrain is done, the combined model; otherwise arm A.** **[OPEN: Sans decides which model goes to G4.]** Arm B is thinking ON and would be compared with comparator 1 only.
* **Decision rule (which one decides):** the claim "a trained model beats the untuned model" holds only if the arm beats comparator 1 under the criterion in section 5. A win over comparator 2 only is reported as such and is not the claim.

## 4. Design and time
* **Runs (primary plan) [PROPOSAL]:** untuned thinking ON, arm, then arm again and untuned thinking ON again in the opposite order (the paper's own two-pair, opposite-order design for the shorter-list test), so 4 runs of 200 tasks, plus 1 run of untuned thinking OFF = 5 runs. The first pair alone is 2 runs.
* **Time per run:** 2.5 to 3.0 h on the record for the untuned model (Runs 84 and 85). For the adapter with thinking off I have no measurement on the BFCL 200 (on bank F the pass-1 probe ran 81 tasks in 1,508 s with a second server on the same GPU; not comparable). Estimate **[ANALYSIS, rough]**: 2 to 3 h per run, so **one pair 5 to 6 h, five runs 11 to 14 h.** Servers restart between runs; nothing else on the GPU.
* **Cost:** $0 (local). Nothing is spent on the API.
* **One run or a repeat:** the paper's own repeats of one configuration moved 15 to 18 tasks between runs and 6 tasks in total score (77 against 71 and 72). A single pair cannot tell a 3-point difference from noise. The primary plan therefore uses two pairs; a one-pair plan is allowed only if Sans chooses it, and then the claim is limited to "first pair".

## 5. Test and margin (from the paper's own paired method)
* **Test:** exact two-sided McNemar on per-task pass/fail over the same 200 task ids (paper, "Statistics"), alpha 0.05, uncorrected, as in the paper's shorter-list test; with two pairs, the paper's task-level sign test (each task counted once, ties dropped) as the combined test.
* **Margin [PROPOSAL, not from the record]:** a win needs both p below 0.05 and a gain of at least 8 tasks (4 points). Reason: the paper's untuned stack moved by up to 6 tasks between identical runs; 8 is above that. **[OPEN: Sans fixes the margin before the run.]**
* **Wording that stays:** the paper's statistics language is not strengthened. If p is below 0.05 for one pair only, it is reported as "suggestive"; the multiplicity rule is: one primary comparison, the others reported as exploratory.
* **Not claimed:** anything about other models, other splits or edge hardware (scope is fixed).

## 6. No BFCL text in the training data (checks to run and record before launch)
* The generator saw only synthetic prompts built from backend co-occurrence weights and themes, never BFCL task text; the author prompts and weights files (`sft/g2b_weights.json`, the author's `THEMES` lists) are committed and can be inspected. BFCL tool descriptions are part of the harness prompt by design (they are the task's tools).
* **Gates already passed on the frozen rows** (`check_provenance.py`): excluded task ids, 8-gram overlap with BFCL and held-out banks, exact-turn match, and the same checks on teacher sentences (arm B) (G3 pre-registration section 3; results log sections 6 and 8b). Arm A's frozen rows: 0 hits after removing one task.
* The advisor's extra check on arm A (2026-11-11): training task ids share nothing with F or K2, and 0 of 441 long bank strings appear verbatim in the training user text. This is a **partial** audit: it does not cover paraphrases or shared state templates, and the Codex review of the arm A result states that leakage was not audited end to end. **[PROPOSAL]** add before launch: (a) 8-gram and 5-gram overlap of the training user text with all 200 BFCL user turns, reported as counts; (b) overlap of the initial-state JSON key sets between training tasks and BFCL tasks, reported as counts (shared state templates are expected because the same simulated backends are used, and should be stated, not hidden).
* If any check shows a BFCL id or an exact BFCL turn in training rows, the run does not start.

## 7. Launch conditions (all must hold)
1. Sans's go in the chat naming the run(s), the time estimate and the cost; the plan is fixed in a dated amendment.
2. Section 2 items closed: exact command line reconstructed and confirmed; flags equal; prompt-render capture equal.
3. Which model goes to G4 is decided and recorded (arm A, arm B or the combined retrain). **The model must have passed both registered FK gates: run 1 at least 70 of 155 and the restarted repeat at least 70 of 155** (arm A: 118 and 118). A model that missed either gate does not go to the BFCL 200 unless Sans, in the chat and after seeing the screening result, overrides this in writing; the override is quoted in a dated amendment and every report of that run says so.
4. Section 6 counts recorded and clean.
5. Section 2b hashes frozen and section 2c caps set in the run script; server build identical to `b10894-d344123fe`; GPU free; swap under 8 GB; no training job running.
6. Codex review of this document and of the run script, major problems only, no major problem.
7. The 200 test ids are used only for scoring; no row of any later training is built from them or from the outcome.

## 8. Limits stated with any G4 result
Single model family and seed; greedy decoding but repeats of one configuration still differ (paper); the screening banks F and K2 share their author with the training data so their scores are not evidence about BFCL; the untuned comparators are fresh runs on the day, not the 2026-10-29 runs; training used synthetic tasks from one teacher; thinking-off versus thinking-on is a serving difference between the arm and comparator 1, stated next to every number.
