# Distillation: what we may be missing, and the next methods to test

Status: planning only. Nothing here is approved. Any training, rollout or paid-API step needs Sans's OK with a time estimate first. Written 2026-11-08, after Round 3c (five candidates, 36-41 / 103 on dev_all vs base 44 / 44, bar 47).

All counts below come from `sft/data/distill/r3/eval_*/verdicts.jsonl`, the training-row metadata, or the model config. Anything not measured is labelled as a hypothesis.

## 1. Where we are

| Candidate (dev_all, 103 tasks) | Passes |
|---|---:|
| Base B1 / B2 (same GGUF, two runs) | 44 / 44 |
| AB (no think marker) | 41 |
| AB (think marker) | 40 |
| A (no marker) | 36 |
| A (think marker) | 37 |
| Expanded attention/MLP targets, AB think | 36 |

Screening bar was 47. Two identical base runs have the same aggregate score but differ on 6 of 103 tasks (3 each way). That shows task-level flips between identical runs; it does not give a score-noise bound. The supported conclusion is only that no candidate met the screen in its observed run.

Project notes also record that earlier fine-tunes (SFT-IFFC v4-v7 and distillation rounds 1-2) did not beat the untuned base. I have not re-audited those runs here. If that holds, "more rows" alone is not a sufficient story.

## 2. Measured facts that shape the next step (headroom analysis, 2026-11-08)

Pooling the 7 dev runs (2 base, 5 fine-tuned) over the 103 tasks:

- 25 tasks pass in all 7 runs; 43 tasks pass in none; 35 are mixed.
- 47 tasks pass in at least one base run; 60 pass in at least one of the 7 runs.
- 13 tasks fail in both base runs but pass in some fine-tuned run; 16 tasks pass in both base runs but fail in some fine-tuned run. These flips are a mix of adapter effect and run noise; this analysis cannot separate them.
- Never-pass tasks are concentrated in classes: of tasks touching each class, never-pass/total is GorillaFileSystem 12/17, TicketAPI 14/21, MathAPI 11/22, TradingBot 19/41, TravelAPI 4/13, MessageAPI 5/14, TwitterAPI 7/24, VehicleControlAPI 8/32.
- Training data did cover the hard classes: tasks per class in the 95 training tasks are GorillaFileSystem 24, TicketAPI 22, MathAPI 18, TradingBot 51, TwitterAPI 20, VehicleControlAPI 16, MessageAPI 8, TravelAPI 4. So the miss is not simply "class absent from training".
- Most common failure messages on base run B1 (59 non-pass episodes; top six message groups shown, 53 episodes): execution-results mismatch 30, state-instance mismatch 8 (TicketAPI) + 4 (TradingBot), empty model response 11 (turns 0, 1, 3 combined).

Where the never-pass tasks fail (base run B1 only; 43 tasks passing in no run):

- Checker message: 22 execution-results mismatch, 6 empty model response, 15 end-state mismatch (TicketAPI 6, TradingBot 4, VehicleControlAPI 2, MessageAPI 2, TwitterAPI 1).
- For the 28 messages that name a turn: turn 0 = 14, turn 1 = 9, turn 2 = 4, turn 4 = 1. So 14 of 43 never-pass tasks already fail on the first turn, before any multi-turn state has built up. This is B1 only and does not say why those turns fail.
- Never-pass tasks by number of turns: 2 turns 10, 3 turns 14, 4 turns 16, 5 turns 3 (all dev: 36 / 29 / 30 / 8).

Train/serve facts:

- The MLX training base (`sft/models/gemma-4-e2b-it-mlx`) is bf16, no quantization in `config.json`. The served model is the bartowski Q4_K_M GGUF. Adapters were therefore trained against unquantized weights and applied on top of a 4-bit-quantized model. Whether this gap matters is untested.
- For the same adapter, MLX and the llama.cpp server pick the same action on 26 of 35 held-out and 19 of 35 training probes (Round 3b). Base-model parity result is in section 4, check P (30 of 35 identical actions).

## 3. What we may be missing (ranked by how much evidence we have)

1. **Train/serve numerics (bf16 train, Q4_K_M serve).** Evidence: configuration fact above plus row-level MLX/server disagreement. Not shown to cause the miss. Cheapest to test.
2. **Failures are not used.** Every arm was positive-only SFT; A was a single round of rejection sampling. The mixed and never-pass tasks are where a verifier signal on failures could add information that success-only SFT does not use (this does not show success-only SFT cannot work).
3. **Narrow task coverage.** Adapters raised in-sample agreement (9 to 17 of 35) without moving held-out agreement (25 to 24). Only 95 training tasks / 320 rows. Hypothesis: the adapter fits task-specific content. Not tested directly.
4. **Few stochastic samples per task.** The seven dev evaluations were greedy. Student rollouts for training-pool tasks already exist at T=0.7, k=2 over 76 tasks each (`student_pool_old`: 49 of 152 episodes pass, 26 tasks with any pass, 3 mixed; `student_pool2`: 44 of 152, 23 with any pass, 2 mixed). k=2 is too few to estimate per-task pass rates, so we do not know how many tasks have 0 < p < 1. Analyze these before approving a new pilot.
5. **Possible capability ceiling (untested).** 43 of 103 tasks pass in none of seven related greedy runs. That is compatible with a capability limit but does not establish one; reading failing episodes (check R) can suggest mechanisms, not prove limits.
6. **Dev bank reuse.** dev_all has now screened 7 runs. A fresh bank is needed before any new candidate is judged, and a bank reused across successive adaptive choices becomes a tuning bank; use a new one per decision stage or fix the choices in advance.

What I am not claiming: that distillation cannot work at this scale. The literature gains I found (Bespoke GRPO on Qwen2.5-7B with a binary checker reward and 4xH200; APIGen-MT-style data at thousands of trajectories) are at larger scale and, for GRPO, had no SFT baseline. At 1.7B the controlled comparisons I found had GRPO and SFT within about a point of each other. I found no harness-vs-training head-to-head at roughly 2B on one machine.

## 4. Free checks first (no training, no API)

- **H. Headroom (done).** Section 2. Result: there are mixed tasks to learn from, but 43 of 103 never pass.
- **P. Base-model parity (done, 2026-11-08).** Greedy generation on the 35 held-out probe prompts (`mlx-AB-think/valid.jsonl`, from 10 tasks), bf16 MLX base vs the Q4_K_M server base (adapter scale 0): exact-target 27 (MLX) vs 25 (server); identical parsed action on 30 of 35 rows; both correct on 24, server-only 1, MLX-only 3. All 35 parsed in MLX. With the adapter on the same prompts the two backends gave the same action on only 26 of 35 (Round 3b). So on this probe the base models mostly agree across bf16/MLX and Q4_K_M/llama.cpp, and the adapter widens the disagreement (5 rows differ for base, 9 with adapter). The comparison mixes quantization with backend, template and kernel differences, so it cannot isolate quantization. Small sample from 10 tasks; the widening could be chance. This does not rule out a train/serve numerics effect and is weak evidence for it; it also does not show it is the cause of the dev miss. Artifacts: `sft/data/distill/r3/probe/mlx_base_valid_gen.json`, `runlogs/15_mlx_base_valid_gen.log`.
- **R. Failure read (done, 2026-11-08; results in section 4a).**
- **Q. Quant-gap feasibility.** Check whether the Q4_K_M GGUF can be dequantized to bf16 safetensors that MLX loads (or whether MLX's own 4-bit is close enough to Q4_K_M to be a useful proxy). Outcome is a yes/no on M1, not a training run.

## 4a. Results of the free checks (2026-11-08)

All numbers come from existing artifacts; no model, server or API call was made for this section.

**R. Failure read, base run B1 (15 never-pass dev tasks, chosen by a fixed rule: sorted by task id, first 4 GorillaFileSystem / 4 TicketAPI / 4 TradingBot / 3 MathAPI by priority order of class).** Labels are my reading of model calls vs gold calls in `verdicts.jsonl` and `dev_all.jsonl`; they are non-exclusive and subjective (Codex r18 recounted independently and differed on two rows, which are corrected above), and cover 15 of 43 tasks in one run.

| Label | Episodes (of 15) | Example |
|---|---:|---|
| Wrong file name (extension dropped or description used as name) | 5 | `cat(file_name='colors')` for `colors.txt`; `wc(file_name='stops')` |
| Guessed an id or skipped the lookup call (counting calls that bypass a needed lookup) | 5 | `resolve_ticket(ticket_id=4)` with no `get_user_tickets()`; several guessed `cancel_order` ids |
| Empty model response for a turn | 3 | turn with no calls |
| Partial action list in a turn (some requested calls missing; excludes empty-only turns) | 6 | only `ticket_get_login_status()` of four calls; `set_navigation` missing |
| Unit strings in wrong form | 3 | `unit_out='pounds'` vs gold `'lb'` |
| Wrong directory navigation | 2 | `cd('clients')` where gold is `cd('..')` |
| Extra or wrong-purpose call or argument | 2 | `create_ticket(..., description=...)`; `du(file_name='chapter1')` |

Reading: in this sample the labelled failures are mostly argument-convention, lookup, and partial-action errors. The call comparison supports a convention/lookup hypothesis; it cannot show that these failures need no substantial reasoning. This is one run and 15 tasks, so it suggests a mechanism and does not prove anything about the other 28.

**Training coverage of those behaviors (train split only: the 320 rows the adapters were trained on; validation rows excluded).** The final-step targets contain 382 function calls over 62 distinct functions. `file_name`, `file_name1` and `file_name2` arguments: 26 in total (23 with `.txt`, 3 without; 20 are plain `file_name`). Unit-string arguments: none for `lb`/`kg`-style units (`imperial_si_conversion` appears once). Of 80 distinct functions in the dev gold calls, 24 have zero occurrences in training targets (for example `ticket_login`, `percentage`, `get_ticket`). Zero-coverage does not clearly separate the groups: 17 of 43 never-pass tasks and 12 of 25 always-pass tasks use at least one such function. This is fragile: counting the 35 validation rows as well gives 21 of 80 uncovered functions and 10 of 43 vs 11 of 25. Thin coverage fits the data but is not shown to be the cause.

**Existing T=0.7, k=2 student rollouts** (`student_pool_old` and `student_pool2`, 152 distinct training-pool tasks, 304 episodes, no overlap with dev ids): passes out of 2 per task: 0/2 = 103 tasks, 1/2 = 5, 2/2 = 44. The teacher verified 76 of the 152 tasks; the student passed at least one sample on 23 of those and 0/2 on 53. 39 of those 53 appear in the AB data (34 in the train split, 5 only in validation). Consequences:
- Outcomes are nearly all-or-nothing per task at k=2 (5 mixed of 152). Methods that need within-task contrast (same-task pass/fail pairs, GRPO groups) would have little to work with from this pool. k=2 is small, so this is an indication, not a measurement of per-task pass rates.
- The informative gap is teacher-passes / student-fails (53 tasks). 34 of them are in the train split and presumably fed arm B; I did not trace each row's arm. Arm B plus arm A together did not move held-out results.

**Provenance check built and run (hardened after Codex r18 found a false pass).** `sft/check_provenance.py` (uses `benchmark/distill_exclusions.py`) on 355 AB-think rows from 105 training tasks vs the 103-task dev bank and all BFCL multi-turn splits (1,153 turn texts):
- 0 rows with excluded BFCL ids, 8-gram overlap or exact-turn matches; `check_task` (includes the 0.85 similarity criterion) rejects 0 of 105 training tasks; 0 shared task ids; 0 near-duplicate task pairs by source question text.
- **The content-level check does not pass:** 5 rows (2 training tasks: `distill_s2a_0016`, `distill_s2a_0285`) share a word 8-gram with the user turns of 3 dev tasks. The shared strings are fixture phrases that two teacher-authored tasks both used: "book swap this Saturday at the community hall" (a tweet in the training task's world state; dev `distill_r3_0274` asks for it as a post), the credential string "refresh token rt-9981 grant type read_write" (training `s2a_0285` vs dev `r3_1079`), and "nearest airport to Rivermist" (training `s2a_0285`, in assistant text, vs dev `r3_0854`). These are different tasks, not a copied task, but they are shared text and the strict check flags them. Effect on results looks small: across the seven runs `r3_0274` fails 7 of 7, `r3_1079` passes 7 of 7, `r3_0854` passes 6 of 7 (fails only think_A), so these three tasks contribute at most one observed pass-count difference (think_A on `r3_0854`); their causal effect is not measured. The training data is not rebuilt or relaxed for this; for any future bank the overlap rule decides whether such tasks are dropped before training.
- The script also rejects empty held-out/source text, empty row messages, duplicate ids, and applies the BFCL similarity criterion to the actual row text (plain user turns and the `goal:` line of OODA cards), not only to source questions; these paths were found open by Codex r18/r19 and closed. Known limits: it does not inspect assistant or tool-result text against BFCL similarity, and it does not check hidden fields outside `messages`.
- Negative tests: injecting three training tasks into the held-out file failed (id overlap, near-duplicates); pasting a dev user turn into a training row while keeping its metadata also fails, and so does a lightly edited BFCL turn (similarity above 0.85). Both previously passed, which is the bug found in review. A pass means only that the implemented checks found nothing, not that leakage is absent.
- Report: `sft/data/distill/r3/provenance_AB_think.json` (input SHA-256s, code hashes, hash of the BFCL turn texts).

## 5. Methods to test, in order

Each method gets a written comparison before it runs: candidate vs the same two base runs, the same screening rule, and a fresh bank of tasks never used for training, tuning or earlier screening. With 6 of 103 flips between identical base runs, a pre-registered bar must exceed noise; the current +3 over the better base run is a tripwire, not a significance test.

### M1. Quantization-matched LoRA (QLoRA-style train/serve match)
- Idea: train the adapter on weights numerically closer to what is served.
- How: if Q feasibility passes, train the existing Round 3b recipe (think-marked AB, q/v, layers 19-34, same seed) on the dequantized Q4_K_M weights. Only change is the base weights.
- Why early: one-variable change against an existing recipe; uses existing data; no API spend. It is an optional diagnostic control, not a presumed fix; its cost advantage over other methods is not established.
- Cost (approval needed; any generation-based feasibility test also needs approval and an estimate): conversion plus one training run (about the same as the earlier AB runs) plus one 103-task dev eval, roughly 1 hour per eval at 4-way concurrency by the episode-time arithmetic (3.2 episode-hours / 4 for B1). Exact training wall time to be confirmed from `runlogs/train_*.log` before asking for approval.
- Pass condition: because dev_all has been used for seven screens, evaluate on a new bank with new approved base runs and a bank-specific rule written beforehand; historical B1/B2 numbers cannot baseline a different bank. Also report MLX/server action agreement on held-out (26 of 35 for the current adapter).
- Risk: P found base MLX-bf16 and server-Q4 agree on 30 of 35 probe actions, so a large numerics gap on the base is not visible; the evidence for M1 is only the wider adapter disagreement (26 of 35). Keep M1 as a one-variable control, not a presumed fix.

### M2. Diverse, larger verified data (APIGen-MT / ToolACE style)
- Idea: address narrow task coverage. Hundreds of new distinct tasks, not more rows from the same 95.
- How: teacher-authored tasks over the same 8 API classes with a held-out-by-construction split (by task and by call pattern), checker-verified, then native-format, hint-free rows as in Round 3. Keep arms A/B separate so AB-A remains readable.
- Need: new API budget (round 3 spent about $2.18 of a $4 cap, per the run log, not re-verified here), a fresh dev bank, an exclusion check against the 200 tasks, mid20, fresh-20, confirm and dev banks.
- Cost: needs an estimate after pilot token counts; 114 audited tasks are outside dev and train by id (108 also excluding validation); both counts need an executable exclusion check before use.
- Pass condition: pre-registered before data is built: bank size, rule, and a scale-sweep control (same recipe at 1/3 and full data) so we can see whether data size moves anything.
- Risk: earlier fine-tunes did not beat base (see section 1); this may repeat. The scale-sweep control would show that.

### M2b. Convention curriculum (targeted variant of M2)
- Idea: section 4a shows labelled failures that are mostly argument conventions (file names with extension, unit abbreviations), lookup-before-act (history/ticket ids), finishing every requested action, and `cd('..')` navigation. Build verified training tasks that exercise each behavior many times across varied wording, instead of broad coverage.
- Control: the same behaviors handled in the harness (tool-description or argument-normalization changes) with no training, evaluated on the same fresh bank. The thesis premise is that harness changes carry this model; a fine-tune has to beat the harnessed base, not only the raw base, so the comparison must be run with the same harness on both arms.
- Needs: new tasks and rollouts (approval and estimate), a fresh bank, the provenance check on the new data.
- Risk: 34 of the 53 gap tasks were already in the train split without improving held-out results, so the curriculum must differ in density and variety, not only in kind.

### M3. Failure-aware preference pairs (DPO or rejection-SFT with negatives)
- Idea: use failure signal on training-pool tasks, which positive-only SFT ignores. Dev tasks (including the 43 never-pass and 35 mixed ones counted above) supply neither pairs nor recovery trajectories; M3/M4 draw only from tasks that pass an explicit exclusion check against the 200, mid20, fresh-20, confirm and dev banks.
- How: sample k rollouts per task at T > 0 from the base (or current adapter) on training-pool tasks; for tasks with mixed outcomes, find the first divergent assistant turn between a passing and failing rollout from the same prefix and build (chosen, rejected) pairs; train DPO with a small beta and a SFT term, via mlx-lm-lora (DPO is supported; check Gemma-4 support first).
- Cost: dominated by rollouts. B1 averaged 113 s per episode and 7 API-style model calls. Example arithmetic: 150 tasks x k=4 = 600 episodes x 113 s / 4 parallel is about 4.7 hours. This is an estimate to be re-measured at T > 0.
- Pre-step: the existing T=0.7, k=2 pools are analyzed in section 4a (5 mixed tasks of 152), so same-task pass/fail pairs from them would be scarce. Cross-source pairs (teacher-chosen vs student-rejected at the first divergent step, on the 53 gap tasks) are the alternative, but arm B already used teacher-informed data from this set. Only if that is not enough, a pass@k pilot on about 40 training-pool tasks at k=4 (needs approval and a time estimate) to count tasks with 0 < p < 1. If few, M3 (same-task pairs) and GRPO stop there; M4 can still use teacher-gap tasks and newly found successes, and a k=4 pilot only measures observed mixed outcomes, not true per-task pass rates. Cross-source DPO pairs still need the same-prefix validator.
- Simpler alternative first: rejection-SFT or expert iteration (M4) before DPO. Rejection-SFT does not train on negatives; only DPO uses them. Compare the simpler method before adding DPO complexity.
- Pair validator: build a strict same-prefix checker (identical history up to the divergent turn, identical tool state) before trusting any pair.
- Risk: pair construction from divergent prefixes is not validated for this harness; the multi-turn state makes "same prefix" strict.

### M4. Expert iteration (multi-round on-policy RFT)
- Idea: Round 3's arm A was one round of the student's own verified replies. Do 2 rounds: train on passes, resample from the adapted model, keep new passes on previously failing tasks, retrain from base.
- How: reuse the M3 rollouts for round 1 to avoid extra cost; compare 1-round vs 2-round and a no-training resample control.
- Cost: one more rollout sweep plus training per round.
- Risk: positive-only again; on-policy data reduces forgetting in the literature I found, but we saw no gain from arm A.

### M5. GRPO with checker reward (mlx-lm-lora)
- Idea: the Bespoke-style recipe, closest to what the strongest published result did.
- Gate: only if M3 or M4 shows a signal on a fresh bank. Needs a multi-turn rollout loop wired to the BFCL checker; mlx-lm-lora's multi-turn rollout integration is unvalidated for this setup.
- Cost: highest; time estimate only after a rollout-throughput measurement on this Mac.

### M6. Bounded controls (low priority, pre-declare if run)
- Lower learning rate or fewer updates; adapter scale below 1.0 at serving time. The earlier probes show the adapter flipping actions both ways (scale 1: +3/-4 of 35), and nothing yet shows excessive training is the cause.
- Run only if pre-declared with the rule fixed before looking at dev (the Round 3 rule forbade dev-triggered fallbacks).

## 6. Recommended order

0. Built: executable exclusion/provenance check (`sft/check_provenance.py`). Still to build before any run: model/adapter/template manifests, paired task-level comparison with task-clustered uncertainty, and predefined stopping and budget rules.
1. R, P, H, provenance check and the existing-pool analysis are done (section 4a). Q (dequantization feasibility) remains and may need a generation test, which needs approval.
2. M1 as a one-variable control if Q shows it is feasible (P gave only weak evidence for a gap).
3. Analyze the existing T=0.7, k=2 pools; only then, if needed and approved, a pass@k pilot (about 40 tasks, k=4).
4. M2 only with a fresh bank and a scale-sweep control; M3 and M4 share rollouts.
5. M5 only if 3-4 show signal.
6. Write up Rounds 1-3c either way: recipe-specific negative result, prompt-template mismatch, validation loss not tracking passes, training/held-out divergence. Do not claim distillation fails in general.

## 7. Guardrails (unchanged)

Gemma-4-E2B-it only; BFCL v4 multi_turn_base; one Mac. The 200 tasks, mid20, fresh-20, confirm bank and dev banks are never training data. No benchmark runs, training or paid API without Sans's approval and a time estimate. Branch `thesis/distill-*`, local commits only. Gate numbers are tripwires, not accuracy claims. Data and logs stay out of git.
