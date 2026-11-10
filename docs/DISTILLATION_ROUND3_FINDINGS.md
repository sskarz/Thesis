# Distillation round 3: findings (2026-11-07)

> **ADDENDUM (2026-11-07, after Codex r14): the result below is confounded by a training/serving prompt mismatch and should not be read as a test of the recipe.**
> Training prompts were rendered by mlx-lm with the Gemma-4 template default (`enable_thinking` false): no `<|think|>` marker. llama-server renders the same messages with thinking on: `<|turn>system\n<|think|>\n...`. Checked with the server's /apply-template against the MLX tokenizer: all 659 round-3 rows (AB train 320, A train 304, valid 35) differ from the served prompt; prepending `<|think|>\n` to the system message makes every one identical (0 mismatches). So both adapters were trained on prompts that never occur at serving time, including arm A, whose targets are the student's own replies generated under the served (thinking-on) prompt. The dev-bank scores above are real measurements of those adapters, but they do not isolate the data. A re-train on the corrected prompts (sft/add_think_marker.py, mlx-A-think / mlx-AB-think) is the next step; nothing else changes. Round 2's t1/t2/t3 should be checked for the same mismatch (not yet done).
> Also from the same review: "Training itself behaved" means only that teacher-forced validation loss decreased; answer/end-turn tokens are ~17% (A) and ~16% (AB) of supervised tokens, so loss falling mostly reflects the reasoning text. Action quality and MLX-vs-llama.cpp parity were never tested (probe not run).


Setup, fixed before any result: docs/DISTILLATION_ROUND3_PLAN.md (with the Codex r11 revisions). Scope: Gemma-4-E2B-it, BFCL v4 multi_turn_base-style tasks, one Mac. Teacher DeepSeek V4.1 Flash, text-level only. Never trained on: the 200 BFCL tasks, mid20, fresh-20, confirm bank, dev bank or any dev2 task.

## Result
Neither fine-tuned candidate reached the screening bar on the 103-task held-out dev bank, so the ladder stopped there. mid20, fresh-20, the confirm bank and the full 200 were not run (they were gated on the dev screen).

| run (bar GGUF bartowski Q4_K_M, sha256 prefix 923c4c86177d2ee1, R69 argv, greedy) | dev_all passes /103 | source |
|---|---|---|
| base B1 | 44 | sft/data/distill/r3/eval_B1/verdicts.jsonl |
| base B2 (server restarted) | 44 | eval_B2/verdicts.jsonl |
| AB, runtime LoRA, lr 2e-5 | 41 | eval_AB1/verdicts.jsonl |
| A, runtime LoRA, lr 2e-5 | 36 | eval_A1/verdicts.jsonl |

Screening bar (pre-registered): >= max(B1, B2) + 3 = 47. All runs: 0 API-error episodes.
Paired counts (tasks, same greedy protocol):
* AB vs B1: gained 6, lost 9. Against both base runs: 5 tasks AB passed that both base runs failed, 8 tasks AB failed that both passed.
* A vs B1: gained 3, lost 11. Against both base runs: 3 gained, 8 lost.
* A-only passes vs AB: 5; AB-only: 10.
* Base vs base: 6 tasks flipped (3 each way), 41 passed in both.

## Reading
* Both fine-tunes land below the base on this bank (AB 3 passes lower, A 8 lower). The two base runs each passed 44 tasks with six task-level flips; two base runs and one run per candidate do not establish the distribution of re-run differences. This is screening evidence from one greedy run per candidate, not an accuracy estimate and not a significance claim.
* Neither candidate cleared the base screen. AB passed 5 more tasks than A (A: student's own verified replies, 43 training tasks; AB: 95 tasks). AB - A is a package comparison (more tasks, teacher-informed rows, different token exposure) and cannot isolate a teacher contribution.
* Both the training and dev sets contain teacher-authored, teacher-verified synthetic tasks. Their relative difficulty was not established, and these runs do not identify the cause of the lower dev pass counts. Training rows also went through student-success/reachability selection.
* Teacher-forced validation loss decreased (see addendum for what this does and does not show): validation loss (35 task-disjoint training-valid rows) fell from 0.270 to 0.193 (AB, 2e-5), 0.218 (AB, 5e-6), 0.194 (A, 2e-5). Lower validation loss did not translate into more passes. Validation loss was used only for the pre-declared dose rule.
* Measurement note: the two base runs differed by 0 passes with 6 task-level flips. That does not establish how reliably a 3-pass improvement could be detected.

## Procedure and declared deviations
Native-format rows (reasoning + answer in the model's own template format), outcome-verified against the stock checker, hint-free for arm B (student sampled without any hint, exact function-string match; 161 of 300 candidate rows kept, biased toward actions the student can already reach), runtime LoRA on the exact bar GGUF (no fuse/requantize), same optimizer updates for A and AB (320 microbatches = 80 updates), dose fixed by the validation-loss rule before any candidate dev evaluation (base dev runs B1/B2 were already recorded). Deviation: hint-free sampling ran 5 sampled attempts per row rather than the declared 8. Not done / not recorded: the planned supervised-token totals, adapter norm and the 100-prompt action-change probe (sft/probe_action_change.py was written but not run), so full compliance with plan item 6 cannot be verified from the records; the dose was chosen by the validation-loss rule only.
Spend: about $2.18 of the $4 cap (per-call usage logs, not an account balance).

## Not claimed
No claim about BFCL multi_turn_base accuracy for either adapter (never run on mid20/fresh/200). No claim that distillation cannot work; the claim is limited to this recipe (LoRA rank 8 on q_proj of layers 19-34, 80 updates, ~300 rows, synthetic tasks) on this bank.

## Caveats worth a reviewer's eye
* Adapter lands on q_proj only (shared-KV layers have no v_proj), which limits capacity; a rank/target change would be a new experiment, not a fix of this one.
* Single greedy run per candidate; the plan's repeat rule applies only to positive screens.

---
## Round 3b (2026-11-07/08): bug-fixed retrain + dose probes

Change: identical rows, order, hyperparameters and seed; only the `<|think|>\n` system prefix added so the training prompt equals the served prompt (sft/add_think_marker.py; 0/659 mismatches vs the server's /apply-template). Same serving (bar GGUF 923c4c86177d2ee1, R69 argv, runtime LoRA scale 1.0), same dev_all (103), greedy, 0 API-error episodes in every run.

| run | dev_all passes /103 | paired vs B1 (gained / lost) | vs both base runs (gained / lost) |
|---|---|---|---|
| base B1 / B2 | 44 / 44 | | |
| AB (no marker) | 41 | +6 / -9 | 5 / 8 |
| A (no marker) | 36 | +3 / -11 | 3 / 8 |
| AB think-marker | 40 | +6 / -10 | 5 / 8 |
| A think-marker | 37 | +6 / -13 | 5 / 10 |

Validation loss (35 task-disjoint training-valid rows): AB think 0.267 -> 0.194, A think 0.267 -> 0.195. All four candidates are below both base runs; none approaches the pre-registered bar of 47. The screening decision did not change with the prompt fix (AB 41 -> 40, A 36 -> 37, all below 47). Task-level results did move (corrected AB vs original AB: 7 gained, 8 lost; corrected A vs original A: 7 gained, 6 lost), and with one run per candidate the run-to-run variability of a candidate is not measured, so these runs show that the fix did not rescue the screen, not that the mismatch contributes nothing. The ladder (confirm, mid20, fresh-20, full 200) was not run.

Failure types (grouping the checker's error strings; sft/data/distill/r3/eval_*/verdicts.jsonl): base B1/B2 ~30 "execution results so far do not match" (missing results), 10-11 "model response list is empty", ~19 state-instance mismatches; the adapters show the same mix (execution-results mismatch 34-40, empty-turn 9-12, state mismatch 16-20) with mean API calls per episode 7.0-7.2 vs 7.1 for base. The categories are broadly similar across base and adapters; this does not establish that failure behaviour is unchanged.

Dose / generalisation probe (think-marker AB adapter, runtime LoRA scale applied on top of the trained alpha/r = 2; greedy action under the served prompt; sft/probe_action_change.py; files in sft/data/distill/r3/probe/):
* Held-out training-valid prompts (35 prompts, tasks never in training): exact target-action agreement scale 0 (= base) 25/35, scale 1: 24, scale 2: 21, scale 4: 19, scale 8: 7 (16/35 parsed). Greedy action differs from base on 10/35 prompts at scale 1.
* In-sample (first 35 TRAINING rows, all arm-B rows): scale 0: 9/35, scale 1: 17/35, scale 4: 10/35.
Reading (diagnostic only; 35 prompts from 10 tasks, 19 arm-B and 16 arm-A rows, versus a training subset of 35 arm-B rows from 26 tasks; unmatched, correlated subsets; exact action-string agreement is not task success; probe caps generation at 1500 tokens and saved no finish reasons, so the scale-8 parse collapse is uninterpreted): the adapter increased agreement on this training subset (9 -> 17) without improving agreement on this small held-out diagnostic (25 -> 24; it changed 10 of 35 actions, gaining 3 and losing 4). Larger runtime scales reduced agreement, which says nothing about more updates, another learning rate or different supervision.

Capacity note (config and adapter inspection): `num_layers: 16` selects layers 19-34; those layers share earlier KV, so `v_proj` adds nothing there, and the saved adapters contain q_proj only (MLP and output projections are untouched). Trainable parameters 0.524M of 4.63B = 0.011% (this alone does not diagnose insufficient capacity). AB trains on 95 tasks, A on 43. Candidate explanation, untested: the update is narrow. Biderman et al. report that attention-only LoRA underperformed MLP/all-module adaptation in their code experiments and that rank also mattered; that is a pointer, not evidence about this model.
Spend: about $2.18 of the $4 cap (per-call usage logs, not an account balance).

## Not claimed
No claim about BFCL multi_turn_base accuracy for either adapter (never run on mid20/fresh/200). No claim that distillation cannot work; the claim is limited to this recipe (LoRA rank 8 on q_proj of layers 19-34, 80 updates, ~300 rows, synthetic tasks) on this bank.

## Caveats worth a reviewer's eye
* Adapter lands on q_proj only (shared-KV layers have no v_proj), which limits capacity; a rank/target change would be a new experiment, not a fix of this one.
* Single greedy run per candidate; the plan's repeat rule applies only to positive screens.

---
## Round 3b (2026-11-07/08): bug-fixed retrain + dose probes

Change: identical rows, order, hyperparameters and seed; only the `<|think|>\n` system prefix added so the training prompt equals the served prompt (sft/add_think_marker.py; 0/659 mismatches vs the server's /apply-template). Same serving (bar GGUF 923c4c86177d2ee1, R69 argv, runtime LoRA scale 1.0), same dev_all (103), greedy, 0 API-error episodes in every run.

| run | dev_all passes /103 | paired vs B1 (gained / lost) | vs both base runs (gained / lost) |
|---|---|---|---|
| base B1 / B2 | 44 / 44 | | |
| AB (no marker) | 41 | +6 / -9 | 5 / 8 |
| A (no marker) | 36 | +3 / -11 | 3 / 8 |
| AB think-marker | 40 | +6 / -10 | 5 / 8 |
| A think-marker | 37 | +6 / -13 | 5 / 10 |

Validation loss (35 task-disjoint training-valid rows): AB think 0.267 -> 0.194, A think 0.267 -> 0.195. All four candidates are below both base runs; none approaches the pre-registered bar of 47. The screening decision did not change with the prompt fix (AB 41 -> 40, A 36 -> 37, all below 47). Task-level results did move (corrected AB vs original AB: 7 gained, 8 lost; corrected A vs original A: 7 gained, 6 lost), and with one run per candidate the run-to-run variability of a candidate is not measured, so these runs show that the fix did not rescue the screen, not that the mismatch contributes nothing. The ladder (confirm, mid20, fresh-20, full 200) was not run.

Failure types (grouping the checker's error strings; sft/data/distill/r3/eval_*/verdicts.jsonl): base B1/B2 ~30 "execution results so far do not match" (missing results), 10-11 "model response list is empty", ~19 state-instance mismatches; the adapters show the same mix (execution-results mismatch 34-40, empty-turn 9-12, state mismatch 16-20) with mean API calls per episode 7.0-7.2 vs 7.1 for base. The categories are broadly similar across base and adapters; this does not establish that failure behaviour is unchanged.

Dose / generalisation probe (think-marker AB adapter, runtime LoRA scale applied on top of the trained alpha/r = 2; greedy action under the served prompt; sft/probe_action_change.py; files in sft/data/distill/r3/probe/):
* Held-out training-valid prompts (35 prompts, tasks never in training): exact target-action agreement scale 0 (= base) 25/35, scale 1: 24, scale 2: 21, scale 4: 19, scale 8: 7 (16/35 parsed). Greedy action differs from base on 10/35 prompts at scale 1.
* In-sample (first 35 TRAINING rows, all arm-B rows): scale 0: 9/35, scale 1: 17/35, scale 4: 10/35.
Reading (diagnostic only; 35 prompts from 10 tasks, 19 arm-B and 16 arm-A rows, versus a training subset of 35 arm-B rows from 26 tasks; unmatched, correlated subsets; exact action-string agreement is not task success; probe caps generation at 1500 tokens and saved no finish reasons, so the scale-8 parse collapse is uninterpreted): the adapter increased agreement on this training subset (9 -> 17) without improving agreement on this small held-out diagnostic (25 -> 24; it changed 10 of 35 actions, gaining 3 and losing 4). Larger runtime scales reduced agreement, which says nothing about more updates, another learning rate or different supervision.

Capacity note (from config and trainable-parameter count, 0.524M of 4.63B = 0.011%): LoRA sits on q_proj only, in 16 layers (19-34) because the last layers share KV; MLP and attention outputs are untouched. Combined with ~95 training tasks and 80 updates, the update may be too narrow and too small to change multi-step behaviour. This is a candidate explanation, untested; the LoRA literature (Biderman et al.) reports target modules matter more than rank.
Spend: no additional API cost in round 3b.

### Pre-registration status (round 3b)
Round 3b is a post-dev bug-correction attempt added after the original screen; the original and corrected runs are all reported (two attempts per arm). The scale/in-sample probes are post-result exploratory diagnostics, not the pre-dev probe required by plan item 6 (probe, adapter norm and supervised-token totals were omitted before the first dev run). No checkpoint, dose or winner was selected on dev. Dev has now informed diagnosis and the proposals below, so the confirm bank, mid20, fresh-20 and the full 200 stay untouched, and the next comparison must be pre-declared.

### Codex r15 on what remains untested
Verified: think-marker datasets equal the originals except for the system prefix (same rows, order, targets); configs differ only in adapter_path; GGUF and MLX templates are byte-identical; across all 739 logged eval_think_AB calls the full histories rendered with thinking on equal default rendering plus the prefix and token counts match logged prompt_tokens; trainer tokenization has one BOS; every eval call finished with `stop` and max prompt length 12,229 < 16,384 slot context (dev truncation unlikely).
Still untested: effect of the harness flattening earlier-turn reasoning into assistant content (training copies it, so it is not a train/serve mismatch, but its behavioural effect is unmeasured); BF16-MLX vs Q4-llama.cpp behavioural transfer (frozen-prompt log-prob comparison with/without adapter); separate action vs reasoning loss.

### Proposed next experiments (Codex r15 order; none started)
1. Cheap diagnostics: frozen-prompt MLX-vs-server parity with the adapter; small training-only memorisation test; separate action/reasoning loss.
2. LoRA on all linear modules of all layers (enumerate actual modules; verify GGUF conversion support), data and update count fixed.
3. More diverse verified multi-turn data (recovery, dependency chains, stop/continue; APIGen-MT-style methodology).
4. Checker-derived preference learning (fresh same-state chosen/rejected pairs or iterated rejection SFT, then GRPO via mlx-lm-lora; executable multi-turn rollout integration still needs validation).
5. Action-weighted supervision / longer training selected on training-only diagnostics, one axis at a time.

### Round 3b diagnostics (MLX bf16, think_AB adapter vs base; sft/diag_mlx.py; sft/data/distill/r3/probe/mlx_*.json)
Same 35 held-out valid rows and 35 in-sample training rows (arm B) as the server probes. Teacher-forced per-token NLL, split at `<channel|>`:
| | reasoning NLL base -> adapter | answer (JSON) NLL base -> adapter |
|---|---|---|
| held-out 35 | 0.188 -> 0.166 | 0.847 -> 0.0035 |
| in-sample 35 | 0.241 -> 0.220 | 0.269 -> 0.0010 |
MLX greedy action = target: in-sample 16/35 (server+LoRA 17/35), held-out 26/35 (server+LoRA 24/35, server base 25/35).
Reading (diagnostic, 70 rows, one adapter; the means above are unweighted means of per-row token means): (1) aggregate exact-target agreement is similar between MLX and server+LoRA on these probes (16 vs 17 in-sample, 26 vs 24 held-out) but row-level agreement differs substantially (MLX and server adapter actions match on only 19/35 training and 26/35 held-out rows), so backend parity is not established; base-MLX generation was not run. (2) Token-weighted, answer tokens account for 51% (held-out) and 43% (in-sample) of the NLL reduction and reasoning tokens for the rest; the answer is conditioned on gold reasoning, so this does not identify formatting vs decision learning, and the diagnostic does not reproduce the trainer's validation loss.

### Round 3c: LoRA on expanded attention/MLP targets (last 16 layers), AB think-marker data
One change vs the think_AB run: target modules widened from q_proj to attention q/o and MLP gate/up/down on the same layers 19-34 (160 adapter tensors, no k/v because KV is shared; 6,356,992 trainable params = 0.137%, vs 524,288 = 0.011%; per_layer_input_gate and per_layer_projection untouched, so not literally all linear modules). An all-35-layers variant ran out of Metal memory at the first long sequences and produced no adapter (sft/lora_config_r3_allmods_AB.yaml is the 16-layer config actually used). Same rows, order, lr 2e-5, 80 updates, seed; same serving and dev_all screen. Validation loss 0.267 -> 0.174 (q-only: 0.194).
dev_all: **36/103** (base 44, 44; bar 47); vs B1 +6/-14, vs B2 +3/-11; vs both base runs 3 gained / 11 lost; 0 API-error episodes (eval_allmods16_AB/verdicts.jsonl). Lower validation loss and 12x the trainable parameters did not produce more passes; the result is below base and below the q-only adapter (40), validation loss did not track screening passes here (no split-loss diagnostic was run for this adapter). One run per candidate; not a significance claim. Exploratory (dev already informed this choice); no ladder run.

### Codex r16 recommendation (ranked)
1. Stop this series and write it up as a recipe-specific screening failure with training/held-out divergence (no claim that distillation fails in general). 2. If continuing: diverse verified multi-turn data (recovery, dependencies, stop/continue), with a predeclared fresh comparison. 3. Then checker-derived preferences (same-state chosen/rejected pairs or rejection SFT before RL). 4. Lower lr / fewer updates as a bounded control (probes do not show excess training is the cause). 5. Bigger N of the same data: low priority. Keep confirm/mid20/fresh/200 untouched until a candidate clears a fresh screen.
