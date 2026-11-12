# Distillation round 5: answer-weighted SFT and on-policy preference pairs (findings, 2026-11-09)

Setup, fixed before any result: `docs/DISTILLATION_ROUND5_PLAN.md` (v2 plus amendments v2.1, v2.2, v2.3). Scope: Gemma-4-E2B-it only, BFCL v4 `multi_turn_base`-style tasks, one Mac, teacher DeepSeek V4.1 Flash (text-level, via OpenRouter/DeepInfra). Serving: bar GGUF (bartowski Q4_K_M, sha256 prefix `923c4c86177d2ee1`), llama.cpp build `b10894-d344123fe`, Run 69 argv, greedy, harness mode `iffc-ooda-card`. All numbers below come from `verdicts.jsonl` / logs under `sft/data/distill/r5/` (gitignored) and `docs/DISTILLATION_RUNLOG.md`. Codex reviews: r26 (brainstorm), r27/r28 (plan), r29 (A0), r30 (A1 + trainers), r31 (pair builder), r32 (A2), r33 (Stage B stop), r34 (next step). Reviews are in `docs/reviews/`.

Authorization: Sans, 2026-11-08: "Do A then B, codex review at every step of the way. Ping me when u need me I'm going to bed."

## Bottom line

Everything Round 5 tested is negative or inconclusive-by-rule:

* **A1 (precision probe):** no detected precision rescue.
* **A2 (answer-weighted SFT vs an ordinary-SFT control):** both arms scored below both base runs on the screening bank. Neither advances.
* **B (same-state on-policy preference pairs, DPO vs chosen-only SFT):** stopped by the pre-registered pair floor (120 training pairs < 150). DPO and its control were never trained, so nothing can be said about DPO's efficacy from this round.

Cumulative record across all rounds: no tested fine-tune or harness variant has beaten the untuned base on a held-out bank under the pre-registered rules.

## Screening bank and rule

FK = F (81 teacher-verified tasks, built in Round 4) + K2 (74 tasks, built in Round 5, frozen sha256 `e2ac40e4ece912d07a3646a3b6a787c4dd880bd8a2b7dc1f2a4e966bfc98c8ef`) = 155 tasks. F is an adaptively informed screening bank (its aggregates helped motivate Round 5); neither F nor K2 is training data.

Base runs (bar GGUF, greedy, server restarted between runs, 0 API-error episodes): F 35/81 twice; K2 33/74 and 32/74. Composite base runs: B1 = 68, B2 = 67 of N = 155; mu_B = 67.5; B* = 68.

Rule: a candidate run 1 advances to a repeat iff composite >= ceil(mu_B + 0.06 N) = ceil(76.8) = **77** AND > B* = 68; after one restarted repeat it passes the screen iff the mean is >= 76.8 and each run is > 68. This is a **resource-allocation screen**, not a significance test. The two F base runs exchanged 5 gains and 5 losses with equal totals; that is a fact about those two runs, not an interval estimate.

## A1: BF16 vs Q4 precision probe (diagnostic, $0)

Frozen 70 prompts (first 35 valid + first 35 train rows of `mlx-AB-think`; 36 tasks), greedy, exactly one `<|think|>` marker rendered (attempt 1 rendered it twice and was archived), 0 HTTP errors, 0 truncations, rendered prompt and token ids identical across servers (70/70).

Exact target-action agreement (held-out valid /35, train /35): Q4 base 25 / 9; Q4 base repeat 25 / 8; Q4 + think_AB adapter 25 / 16; BF16 base 24 / 19; BF16 + think_AB adapter 24 / 17. Adapter-minus-base on held-out is 0 at both precisions, so the difference-in-differences is 0 (< the pre-registered trigger of 4): **precision-rescue trigger not met**.

Descriptive extra: the Q4 base and the locally converted BF16 base differ on 24 of 70 greedy actions, versus 3 of 70 between two Q4 runs. The conversion provenance differs (local HF snapshot -> BF16 vs bartowski's Q4 path), so this does not isolate quantization. A task-level BF16 comparison was not run (BF16 is not the thesis baseline; it would need Sans's approval).

## A2: answer-weighted SFT (Sw) vs ordinary SFT (S0), clean identical rows

Rows: `mlx-AB-think` with the two flagged tasks (`distill_s2a_0016`, `distill_s2a_0285`) removed from both arms: 315 train + 35 valid. Both arms: Gemma-4-E2B-it MLX bf16 base, LoRA rank 8, scale 2.0, q_proj on layers 19-34 (these layers share KV, so v_proj has no effect; 524,288 trainable parameters), lr 2e-5, grad-accum 4, 320 microbatches = 80 updates, seed 61, max_seq 9500, prompt masking, `<|think|>\n` in the system content. Served as an unmerged fp16 runtime LoRA GGUF (scale 1.0) on the bar GGUF. Only the loss differs:

* S0: stock mlx-lm loss.
* Sw: `0.9 * mean CE(answer segment) + 0.1 * mean CE(reasoning)`, separator assigned to the answer segment, absent segment renormalized (`sft/train_mlx_weighted.py`, tests in `sft/test_train_weighted.py`). The non-reasoning tokens (separator, answer JSON, termination) are about 15.8% of supervised tokens in the unweighted data (`sft/token_segments_audit.py`).

| arm | F (of 81) | K2 (of 74) | composite | advances |
|---|---|---|---|---|
| base B1 / B2 | 35 / 35 | 33 / 32 | 68 / 67 | n/a |
| S0 ordinary SFT | 29 | 30 | 59 | no |
| Sw answer-weighted | 33 | 31 | 64 | no |

Threshold 77 and > 68. Neither arm reaches either. Both are below both base runs. Sw minus S0 is +5 composite passes; that is descriptive only and no efficacy claim.

Training validation loss (reported for completeness, never used to select anything): S0 0.267 -> 0.195; Sw 1.625 -> 0.558. They are on different scales (Sw is the weighted objective) and are not comparable.

Integrity notes: S0's first K2 run contained one harness-flagged episode (`context_overflow:turn_1`, scored not-passed), which tripped the zero-API-error validity check. The registered rule requires a complete replacement; I first skipped it (a protocol deviation Codex r32 caught, recorded in the runlog), then ran the replacement: 30/74, 0 API-error episodes. The first attempt is preserved at `sft/data/distill/r5/evals/K2_S0_attempt1`. Codex r32 verified: identical training configuration except the objective; the Sw loss was actually applied (monkeypatch reaches both training and validation); correct adapters and GGUFs served (scale 1.0); bank hashes; pass-count recounts.

## B: on-policy same-state preference pairs (stopped by the pair floor)

**States.** Teacher-labelled kept states (labels from Round 3 old/new + curriculum G) that are recovery states (the student's request follows a tool error) or canonical-correction states (the teacher action differs canonically from the student action), after guard audits against dev_all, confirm, F, K2 and BFCL text: 472 states / 180 tasks (153 recovery, 319 correction-only). Five source tasks dropped (the two declared above plus `distill_g_0184`, `distill_r3_0604`, `distill_r3_1176`); 18 states with conflicting targets for an identical prompt removed.

**Sampling.** Adapter-free bar GGUF (verified `/lora-adapters == []`), each state sampled greedy + 5 at T=0.7, no explicit `max_tokens` (the historical native sampler sent none), exactly one `<|think|>` marker: 472 x 6 = 2,832 generations. 0 error records, 0 missing keys, all HTTP 200, all `finish_reason=stop`. Finished 3h28m after start (cap 4h). Cost $0 (local).

**Matcher (fixed in v2.3 before sampling).** Strict JSON-list parse; canonical calls (argument-key order canonicalized, call order kept). Precedence: excluded (truncated / API error / unparsable / no reasoning) -> exact -> execution_error (replay returns an error via `iffc_state.is_error_result`) -> premature_stop (empty response while the teacher acts and the turn is not complete) -> return_equivalent (excluded) -> unresolved (excluded). **Verified inferior = execution_error or premature_stop.** Chosen = exact-match sample with nonempty reasoning <= 3,500 chars; rejected = verified-inferior sample with nonempty reasoning. Pairs = `zip(chosen, rejected)[:2]` by sample index per state; both branches <= 9,500 tokens; validation = tasks ordered by sha256(task_id), first ceil(10%) of tasks with pairs.

**Response classes (2,832):** exact 408 (5 of them with reasoning > 3,500 chars), execution_error 350, premature_stop 956, return_equivalent 169, unresolved 676, unparsable 273. States with any exact response: 173; with a chosen-eligible exact response after the reasoning cap: 171; with a verified-inferior response: 299; with both: **99**.

**Pair funnel.** 143 candidate pairs -> 11 dropped for a branch > 9,500 tokens -> 132 retained across 67 tasks -> **120 train pairs from 60 tasks + 12 validation pairs from 7 tasks.** Rejected class among train pairs: 50 execution_error, 70 premature_stop. Recovery / correction-only: 46 / 74. By source (train pairs): g 38, r3 42, s2a 40 (`docs/reviews/r5_pair_diagnostics.json`). The chosen response is the greedy sample in 14/120 pairs (11.7%). Original student-teacher agreement on the pair states: 22/120 train pairs (18.3%, pair-weighted), or 13/82 unique train states (15.9%). Mean chosen-minus-rejected total branch length +18.6 tokens (so length is not an obvious confound).

**Registered floor:** >= 150 train pairs from >= 50 tasks, applied after all filters and the validation split. Result: 120 pairs / 60 tasks -> `r5_pairs.py pairs` exited 2 -> **Stage B stopped by rule.** Pooling the 12 validation pairs (132 total) or keeping the 11 overlength candidates (143 total) would not reach 150 either (Codex r33). No filter, pairing, split or floor was changed after seeing outcomes.

**Not done:** the final train-and-validation branch provenance audit (`sft/check_provenance.py` expanded over both preference branches and rendered reasoning); the pair set is therefore pre-audit and not provenance-certified. The trainer's preflight on the **real** pair set (longest real pair, `sft_chosen` preflight) was also not run. The saved diagnostics do not contain every registered pre/post-filter diagnostic (e.g. separate prompt/reasoning/answer token lengths).

**Known code concern (Codex r33):** `benchmark/distill_onpolicy.py:turn_complete` and `same_effect` swallow checker/replay exceptions (returning False) instead of propagating them, so an infrastructure failure could be misclassified as `premature_stop` or `unresolved`. The independent recount observed zero such exceptions, so no count changes, but a future run should add an exception-propagating mode and log `replay_exception` separately, as v2.3 intends.

## What this does and does not show

* Answer-weighted loss, at this scale (315 training rows from 95 tasks, 80 optimizer updates, q-only adapter on 16 layers), did not lift the model to the base level on FK: 64 vs 67-68, and plain SFT 59. Sw scored 5 more composite passes than the plain-loss control S0 (descriptive only), but both are below base.
* On-policy pair supply from this pipeline is the binding constraint, not the optimizer: only 99 of 472 states yield both a teacher-matching and a verified-inferior student response, and the student agreed with the teacher on 18.3% of the pair states originally.
* **It does not show** that DPO fails (never trained), that recovery training fails (never trained), or that fine-tuning cannot help this model. It shows that these specific recipes (Rounds 3-5) did not beat the untuned base under these screens, and that this particular pair-supply pipeline did not reach its registered floor.
* Passes on FK are noisy: two identical greedy base runs on F exchanged 5 gains and 5 losses with equal totals. Small differences (e.g. Sw vs S0, +5) are not evidence.

## Cost

Round 5 API spend: K2 authoring $1.2966 + K2 solve $0.1487 = **$1.4453 of the $2.5 Round 5 cap**. Round 4 was about $3.2 of its $15 allowance. Stages A1, A2 and B were local ($0). Wall-clock: S0 training ~1.5h, Sw training ~1.5h, each FK evaluation ~55-60 min per arm, pair sampling 3h28m.

## Artifact locations (all gitignored; live only in the worktree on Sans's Mac)

`sft/data/distill/r5/`: `manifest.json`, `states.jsonl`, `samples.jsonl` (+ manifest), `pairs/{train,val}.jsonl`, `pairs/diagnostics.json`, `evals/{F,K2}_{B1,B2,S0,Sw}`, `evals/K2_S0_attempt1`, `mlx-AB-think-clean/`, `probe_*.json`, `runlogs/`, chain scripts `chain*.sh`, `evallib.sh`. Adapters: `sft/adapters-r5-{S0,Sw}`; GGUF LoRAs: `sft/lora-gguf/r5_{S0,Sw}.gguf`. Banks: `sft/data/distill/f/tasks_F.jsonl`, `sft/data/distill/k/tasks_K2.jsonl`.
