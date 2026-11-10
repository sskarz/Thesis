# Distillation: state of play and handoff for the next agent (written 2026-11-09, after Round 5)

Purpose: everything a fresh agent needs to pick up the fine-tuning thread, or to decide to leave it alone, without re-deriving what was tried. This supersedes the *state* in `docs/DISTILLATION_HANDOFF.md` (written 2026-11-06 before any round ran); that file's hard rules and contamination section are still valid and are repeated in section 4. Nothing here was run for this document: every number is copied from the run log or a findings doc, and each section names its source. If a number you need is not in these docs, do not invent it; ask Sans.

## 1. One-paragraph status

Thesis: "State, Not Syntax: Harness Optimization for Small-Model Multi-Turn Tool Calling" (M.S., Cal State LA). Only Gemma-4-E2B-it is evaluated, on BFCL v4 `multi_turn_base`, on one Mac. Sans asked for fine-tuning with a stronger teacher (DeepSeek V4.1 Flash, text-level distillation) to beat or complement the harness improvements. Across Rounds 2-5, **no fine-tuned adapter beat the untuned base under the pre-registered screens**, and a harness retry nudge (Round 4 H) equalled base. The last open experiment (Round 5 Stage B: on-policy same-state DPO vs chosen-only SFT) was stopped by its own data-supply floor (120 train pairs < 150), so DPO has **never been trained** in this pipeline. Codex's (and my) recommendation as of 2026-11-09 was: stop fine-tuning for the thesis, write the negative results up (Appendix B, per the paper rules), and keep the harness gains as the final result; Sans had not yet decided at the time of writing.

## 2. Reading order

1. This file.
2. `docs/DISTILLATION_RUNLOG.md` (chronological, every number with its source file).
3. Findings: `DISTILLATION_ROUND2_FINDINGS.md`, `DISTILLATION_ROUND3_FINDINGS.md`, `DISTILLATION_ROUND4_FINDINGS.md`, `DISTILLATION_ROUND5_FINDINGS.md`.
4. Pre-registered plans (the rules that decided each outcome): `DISTILLATION_ROUND3_PLAN.md`, `DISTILLATION_ROUND4_PLAN.md` (+ `_H_SPEC.md`), `DISTILLATION_ROUND5_PLAN.md` (v2 + v2.1/2.2/2.3).
5. Method brainstorms: `DISTILLATION_NEXT_METHODS.md`, `docs/reviews/codex_r26_next_methods_brainstorm.md`, `docs/reviews/codex_r34_next_step.md`.
6. Codex reviews of each step: `docs/reviews/codex_r*.md` and the older `2026-11-0*` files. Memory pointer: `~/.claude/projects/-Users-sanskarthapa-Documents-GitHub-thesis-vault/memory/thesis-distillation-pipeline.md`.

## 3. Cumulative results (all against the untuned Q4_K_M base served from the same bar GGUF)

| Round | What | Result (source: runlog / findings) |
|---|---|---|
| pre-2 | earlier Thesis-repo fine-tunes, 200 tasks | best SFT 15.0%, DPO "Run 51" 26.5% (53/200), base JSON-text 30.5% |
| 2 | t1/t2/t3/c1 adapters, 23-task dev bank and mid20 controls | all lost; the 17/20 gate is not reliably reproducible (bar rerun 15/20, base through our pipeline 16/20, runtime-LoRA c1 14, fused c1 13, t1 10). Training-target format mismatch was found (JSON after `<|turn>model` vs served `<|channel>thought...`). |
| 3 | native-format, outcome-verified, hint-free rows; A (304 rows/43 tasks) and AB (320 rows/95 tasks); dev_all 103 | base 44/44; AB 41, A 36; bar 47. Neither advances. |
| 3b | `<|think|>` marker fix (mlx-lm omitted it; server renders it) | AB_think 40, A_think 37. Fix did not change the outcome. Probe: adapter raised in-sample target agreement 9->17/35 but held-out 25->24. |
| 3c | LoRA on q,o,gate,up,down (+k,v where present), layers 19-34 | 36/103. Worse. (All 35 layers OOMs on Metal.) |
| 4 | Stage A on fresh bank F (81): base 35/35, H (harness retry nudge) 35, AB 30. | H does not advance. The base replies `[]` after a tool error 31/40 (B1); H turns most into calls (30/49) but the pass count does not move (+7/-7). Stage B: strict hint-free recovery rows 45 (< 100) so training arms C/C_match/C+H were never trained. |
| 5 A1 | BF16 vs Q4 probe | no detected precision rescue (DiD 0 < 4). |
| 5 A2 | S0 ordinary SFT vs Sw answer-weighted SFT, clean 315+35 rows; FK (155 tasks), base composites 68/67, need >= 77 and > 68 | S0 59 (29+30), Sw 64 (33+31). Neither advances. |
| 5 B | on-policy same-state pairs, 472 states x 6 samples | 120 train pairs/60 tasks < 150 floor: stopped. DPO/control never trained. |

Run-to-run variation to keep in mind: identical greedy server runs on F exchanged 5 gains and 5 losses (equal totals); on dev_all two base runs differed on 6 tasks (3 each way). Treat differences of a handful of tasks as noise.

## 4. Hard rules (carry over unchanged)

* Never invent or recompute numbers; numbers come from `benchmark.db` and logs. Wording never stronger than counts support. Gate/screen results are tripwires and resource-allocation screens, not accuracies or significance tests.
* The 200 BFCL tasks, mid20, fresh-20, the confirm bank, dev banks, F, K2 are never training data. Provenance checks (`sft/check_provenance.py`: IDs, gold signatures, 8-gram text, turn-similarity, BFCL text) must pass before any training; extend them to both branches of preference pairs and to rendered `reasoning_content`.
* F and K2 per-task text: aggregate-only (do not read per-task content). No tuning on dev_all/F per-task. Never screen or select on validation loss.
* Protected-bank ladder (confirm bank, mid20, fresh-20, full 200) runs only after a screen is passed AND explicit Sans approval with an estimate.
* Scope fixed: Gemma-4-E2B-it only; do not re-add other evaluated models. Do not edit `paper/main.tex` or the CSULA thesis from this thread.
* Git: branch `thesis/distill-stage0` (never `main`), local commits only; no push, PR, merge or upload without Sans's explicit OK each time. No AI attribution lines in commits or PR text.
* API keys only from env (`OPENROUTER_API_KEY`); never print, log or commit one. Data and logs stay out of git (`sft/data/distill/` is gitignored).
* Budget: Round 4 allowance $15 (about $3.2 spent in Round 4); Round 5 cap $2.5 ($1.4453 spent, K2 only). Cumulative about $4.6 (Round 4 $3.2 + Round 5 $1.4453) of the $15 allowance; reconcile from call logs before any new paid chain, with `--max-cost` on each call. A new round needs a new budget decision from Sans.
* Codex review at each step: `codex exec -m gpt-6.1-sol -c model_reasoning_effort=medium -s read-only -c shell_environment_policy.inherit=none -o <out> - < <prompt>`; never change `~/.codex/config.toml`. Ping Sans (PushNotification) for decisions, stopping rules, budget issues, results.
* Pre-register before running: rules, screening bank, thresholds, caps, stop conditions; log deviations honestly (Round 5 recorded two: the skipped S0 replacement run and a K2 chain that was not fail-fast).

## 5. Infrastructure map

Worktree: `~/Documents/GitHub/Thesis-wt-distill` (branch `thesis/distill-stage0`). Python: `~/Documents/GitHub/Thesis/.venv/bin/python`. llama.cpp: `~/Documents/GitHub/Thesis/llama.cpp/build/bin/llama-server` (build `b10894-d344123fe`). MLX base: `~/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx` (mlx-lm 0.32.0, bf16). Bar GGUF: bartowski `google_gemma-4-E2B-it-Q4_K_M.gguf` (sha256 prefix `923c4c86177d2ee1`). Serving: `bash sft/serve_gguf.sh <gguf> <logfile> [lora.gguf]` (Run 69 argv, port 8080, `-np 4`, `-c 65536`, `--temp 0.0`).

Serving fact (important): training is MLX **bf16** LoRA; serving is an **unmerged fp16 runtime LoRA** (`sft/mlx_adapter_to_gguf_lora.py`, scale 1.0) on the **Q4_K_M** GGUF. No QAT was ever used. The trained base and the served base differ in precision (A1 measured the difference: 24/70 greedy actions differ between the Q4 base and a locally converted BF16 base, but the held-out adapter-minus-base effect is 0 at both).

Pipeline, in order of use:

* Authoring / verification: chain scripts under `sft/data/distill/{f,g,k}/` (e.g. `f/author_f.sh`, `f/chainF1.sh`, `k/chainK1.sh`), teacher solve k=4, strict dedupe vs every earlier bank and the 200 benchmark tasks; banks F (81), G (79 curriculum), K2 (74).
* Student rollouts: `benchmark.distill_student_eval --tasks <bank.jsonl> --log-dir <dir> --label <name> --samples 0 --workers 4 --fc-mode iffc-ooda-card` (sample 0 = greedy; `--samples k --temp 0.7` for pools). Shared helpers: `sft/data/distill/r5/evallib.sh` (`serve NAME [lora]`, `run NAME FCMODE BANK N OUTDIR`, validity checks: expected rows, unique tasks, zero `api_error` episodes, 2.5h cap).
* Labelling: `benchmark/distill_onpolicy.py label` (keep = `act_ok` or `stop_correct`), `hybrid`, and native sampling `benchmark/distill_rationalize.py`.
* Data builders: `sft/build_recovery_rows.py` (Round 4 R/C candidates), `sft/r5_state_audit.py` (Round 5 state selection and guards), `sft/r5_pairs.py` (subcommands `states`, `sample`, `pairs`, `expand`), `sft/add_think_marker.py`, `sft/check_provenance.py`.
* Training: `sft/train_mlx.py` (stock), `sft/train_mlx_weighted.py --mode {plain|weighted} --model <MLX base> --train --data <dir> --config <yaml>` (note: stock mlx-lm supervises one extra `<pad>` target per row), `sft/r5_dpo.py` (own DPO / chosen-only SFT trainer; usage in its docstring; synthetic-pair preflight output at `sft/data/distill/r5/runlogs/dpo_preflight_dpo.out`; measured step ~25 s for a 5.8k-token pair, peak 24.7 GB). Configs: `sft/lora_config_r3_*.yaml`, `sft/lora_config_r5_{S0,Sw}.yaml`.
* Conversion: `sft/mlx_adapter_to_gguf_lora.py <adapter_dir> <out.gguf>`.
* Diagnostics: `sft/diag_mlx.py`, `sft/stageA_diag.py`, `sft/token_segments_audit.py`, `sft/r5_precision_probe.py`, `sft/probe_action_change.py`.
* Chain scripts (templates for unattended runs): `sft/data/distill/r5/chain*.sh`.

Where the data is: all of `sft/data/distill/` is gitignored and exists only in the worktree on Sans's Mac. If the worktree is deleted, the banks, rollouts, labels, pairs, samples and adapters are gone; only code, docs and reviews survive in git. Back up `sft/data/distill/{f,g,k,r3,r5}` if there is any chance of needing them.

## 6. Pitfalls already paid for (do not repeat)

1. **Think-marker mismatch.** mlx-lm renders training prompts without `<|think|>`; llama-server renders it by default. Prepend `<|think|>\n` to the system content for training (`sft/add_think_marker.py`), and when probing through the server **strip** the content marker so exactly one marker is rendered (A1 attempt 1 rendered it twice). Assert the count of `<|think|>` in the rendered prompt via `/apply-template`.
2. **q-only in effect.** Layers 19-34 share KV, so a LoRA over them has no working v_proj/k_proj; "all modules" on 16 layers (Round 3c) was worse and all 35 layers OOMs on Metal at ~9.5k-token sequences. Trainable parameters: 524,288 (rank 8, scale 2.0).
3. **Validation loss is mostly format.** Answer NLL collapses (held-out 0.847 -> 0.0035) while reasoning NLL moves ~10%; validation loss did not predict screen outcomes. Never select on it.
4. **Training-vs-served format.** Native thought-channel responses must be the training target (`<|channel>thought...<channel|>` + JSON answer + `<turn|>`+ newline); the terminal "termination" segment is `<turn|>` + a normal newline, and the stock mask also supervises one extra pad target.
5. **Greedy server runs are not identical.** Expect ~5 flips each way on F between identical runs. Restart the server between base repeats; log both.
6. **Strict JSON-list filter.** Fenced (` ```json `) outputs must be rejected; the permissive parser silently accepted them (25 rows in Round 3, 15 in Round 4). Use `json.loads` + list check.
7. **Row uniqueness keys.** Pre-registered uniqueness is on (task_id, step), not on prompt text (Round 4: 60 prompts, only 53 (task, step) pairs).
8. **Detector bugs.** Execution results sit in the *last user message after the OODA card*; an early post-error detector inspected the wrong message and falsely "disproved" the premise (Codex r24). Validate detectors against an independent marker.
9. **Hybrid-ID collisions, prompt-cache thrash.** Episode ids must include the source student episode; schedule jobs sample-major so concurrent workers see different tasks (same-task requests thrash llama-server's per-slot prompt cache, ~4x slower).
10. **Harness `context_overflow` flags** count as API-error episodes in the validity check: a flagged run must be replaced completely (failed attempt preserved), not patched.
11. **Dedupe order matters.** Eligibility before dedupe, sort by (task_id, episode_id, step, req_index), then drop identical prompts with conflicting targets (Round 5 found 3 qualifying states lost otherwise, plus 16-18 conflicting-target prompts).
12. **Labeller semantics.** `keep` = `act_ok`/`stop_correct`; "act_ok" means the teacher action executes cleanly, not that the continuation succeeds. Teacher-action preference/imitation is therefore not outcome-verified learning.
13. **Teacher solves share episode ids with student verdicts.** Executed-history reconstruction must read only student verdict dirs (`r3/student_pool_old`, `r3/student_pool2`, `g/student_pool`), not `teacher_solve`.
14. **Shell hazards on this machine.** Foreground `sleep` followed by commands can be blocked by the harness; use backgrounded until-loop waiters. Beware `sed` multi-line replaces and empty `$PY`; Metal memory can be tight (24.7 GB peak for DPO with an eval server running).

## 7. What is untried or unresolved (candidate directions, none approved)

All of the following are post-outcome relative to Round 5 and need a new written pre-registration and Sans's approval before running. They are listed with the honest reasons they might or might not be worth it.

* **Same-state DPO / chosen-only SFT on the existing 120 pairs.** The only unanswered objective question. Cost: provenance audit of both branches first, a real-pair preflight (`sft/r5_dpo.py --preflight`, never run on real pairs), ~6h DPO cap + ~3.5h control + FK evals. Any result is exploratory and cannot retroactively pass the Stage B floor. A pass on FK (>= 77 and > 68, then restarted repeat mean >= 76.8) would justify protected-bank confirmation, not a thesis claim; attribution to the preference signal needs the control.
* **More pair supply.** The floor failed on data, not on optimization. Levers: more samples per state (sampling ran 12 generations/min; 3.5h for 2,832), more or different eligible states (fresh student rollouts, new curricula), a looser matcher admitting unresolved alternatives (post-outcome). The floor itself is a protocol threshold, not an efficacy threshold.
* **Teacher-interleaved rollouts (Codex r26 "E7").** Let the teacher act at student failure points and continue from the teacher's state; produces the state distribution the student actually reaches while giving corrective actions. Not implemented.
* **GRPO-style / checker-based outcome reward.** The only route to *outcome-verified* learning (the checker already scores episodes). Substantial new implementation and compute on one Mac; Codex rated its immediate value low relative to cost.
* **Rejection-sampling SFT on exact-matching on-policy responses.** Close to Round 3's arm A (student's own verified replies, which scored 36-37); Codex judged low marginal value.
* **Different capacity / targets.** LoRA is effectively q-only on 16 layers; widening failed in Round 3c. Real alternatives would be full fine-tuning (memory-bound on this Mac), a different layer range, or a merged/QAT-style path (never used; Sans floated "full precision" and A1 found no rescue signal).
* **Harness side.** The positive thesis results are harness-based. The Round 4 finding that the base answers `[]` after tool errors (31/40 post-error replies on F) and that H converts most to calls without changing passes suggests failures are about *what* to call next (arguments, lookups, state) rather than whether to retry. Any new harness hypothesis should be pre-registered like the fine-tuning ones.

Meaningful positive bar (pre-registered in Round 5, reuse unless Sans changes it): run 1 composite >= 77 of 155 on FK and > 68, then a restarted repeat with mean >= 76.8 and each run > 68. Five tasks of movement alone is not evidence.

## 8. What to write in the thesis regardless of next steps (per Codex r34)

State the hypotheses, the registered rules, the negative SFT screens (Rounds 3-5), the behaviour-versus-success distinction (H changed behaviour, not passes), and the data-supply stop for DPO. Keep training history in Appendix B under the current paper rules and summarize implications in the discussion. Avoid: "fine-tuning cannot help", "DPO failed" (untrained), "recovery training failed" (untrained), "harness optimization universally wins". F is an adaptively informed screening bank; existing harness claims keep their selection and multiple-comparison caveats.

## 9. Open items at hand-off

* Sans's decision: stop and write up (default) vs a labelled exploratory DPO run vs more sampling under a new protocol.
* Round 5 Stage B artifacts not finished: branch-level provenance audit; real-pair preflight; the exception-propagation fix for `turn_complete` / `same_effect` (Codex r33); missing registered diagnostics (separate prompt/reasoning/answer token lengths, pre/post-filter breakdown).
* Round 5 manifest (`sft/data/distill/r5/manifest.json`) does not yet include the A2 configs, clean-data hashes, adapter/GGUF hashes, chain scripts or an F-hash runtime assertion (Codex r32 audit gaps; not evidence of a wrong adapter or bank). `evallib.sh` should persist `/lora-adapters` JSON and check exact path and scale.
* Everything is committed locally on `thesis/distill-stage0`; nothing was pushed. No process is running. Round 5 API spend $1.4453 of $2.5; Round 4 about $3.2 of $15.
