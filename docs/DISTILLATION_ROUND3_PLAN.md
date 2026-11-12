# Distillation round 3: plan (written 2026-11-07 before any round-3 run)

Approved by Sans in chat ("Yeah go for it"), after the report that round 2 ended in a negative result. Scope is unchanged:
Gemma-4-E2B-it only, BFCL v4 multi_turn_base, one Mac, text-level teacher (DeepSeek V4.1 Flash via OpenRouter/DeepInfra).
Never trained on: the 200 BFCL tasks, mid20, the fresh-20 bank, the confirm bank (22), dev bank (23) or any new dev2 task.

## Question
Do native-format, outcome-verified, hint-free rows, applied as a runtime LoRA on the exact bar GGUF (bartowski Q4_K_M),
beat that same GGUF on a held-out bank large enough to see a 4-5 task effect? Round 2 could not tell, because (a) the
training targets were not in the format the model serves, (b) conversion/requantization was a confound, (c) the 23-task
dev bank and the 20-task gate cannot resolve effects of 2-3 passes.

## What changes versus round 2
1. **Targets are native by construction**: each row's last message is the model's own reasoning (`reasoning_content`) plus
   its answer, rendered into the `<|channel>thought ... <channel|>` block by the chat template. No hinted
   rationalization (Codex r9 found paraphrased leaks). A teacher action only becomes a row if the STUDENT, sampled
   without any hint, produces an effect-equal action; otherwise the row is dropped (and the drop rate is reported:
   this biases toward actions the student can already reach).
2. **Two arms, one control-by-design**:
   * arm A (self-verified, RFT): rows from the student's own episodes that pass the stock checker. Contains no teacher
     information except the task and its gold check.
   * arm AB: A plus arm B = teacher-informed rows: failed student episodes, teacher labels at the student's states
     (gold-checked stop/act), hybrid rollouts kept only if the whole episode passes the stock checker, then hint-free
     native sampling at each teacher step. AB - A isolates what the teacher adds.
3. **No fuse/requantize**: adapters run at runtime on the exact bartowski GGUF (`sft/mlx_adapter_to_gguf_lora.py`,
   `serve_gguf.sh` third argument). Training base stays the bf16 MLX base.
4. **Measurement**: dev_all = 23 old dev + dev2 (new, target >= 80) held-out teacher-verified tasks. The bar GGUF is run on
   dev_all TWICE (server restarted between) to measure its own run-to-run spread before any candidate is judged.
5. **Task supply**: author ~300 new tasks (seed 21, prefix `distill_r3`), `--existing` over every bank, semantic audit,
   exclusion check, teacher-verified (k=4); cross-similarity check dev2 vs all training tasks (>= 0.85 -> drop the
   training task). Split stratified by backend: dev2 and train pool 2. Old train_pool minus quarantined (0004, 0216,
   manual quarantine) joins the pool.

## Pipeline and estimates (4 parallel slots, ~26 s per amortized student episode from basepipe_pool)
| step | what | GPU/API | est. |
|---|---|---|---|
| 1 | author 300 + validate + teacher-verify k=4 | API <= $2.5 | 40 min |
| 1' | in parallel: serve bar GGUF, student sampling k=2 temp 0.7 on old pool (78) | GPU | 70 min |
| 2 | bar GGUF on dev_all x2 (B1, B2) | GPU | 2 x ~50 min |
| 3 | student sampling k=2 temp 0.7 on pool 2 | GPU | ~60-90 min |
| 4 | teacher labels + hybrid rescue (failed episodes) | API <= $0.5 | 20 min |
| 5 | native rows (arm A, arm B hint-free), leak scans, Codex row review | GPU light | 40 min |
| 6 | train tA, tAB (same hparams as t1/c1: rank 8, scale 2, 16 layers, lr 5e-6, 1 epoch, grad-accum 4) | GPU | 2 x ~40 min |
| 7 | convert adapters to GGUF LoRA, eval tA and tAB on dev_all (runtime LoRA on bar GGUF) | GPU | 2 x ~50 min |
Total wall about 8-9 h; API hard cap $4 (logged 0.70 spent in round 2, ~1.55 total so far).

## Decision rules (fixed now, before results)
* Dev advance rule: candidate advances iff dev_all passes >= max(B1, B2) + 3 AND the candidate's passes are not below
  min(B1, B2). If the two base runs differ by more than 4, no candidate advances and the finding is "the measurement
  noise exceeds the effect we can detect"; report that, stop.
* Dose: one pre-declared variant allowed only if both tA and tAB are indistinguishable from base on dev_all (within the
  base spread): same data, lr 2e-5, nothing else. No other tuning on dev.
* Advancing candidate: confirm bank (22) once (cand >= mean base on a base run of the confirm bank); then mid20 once
  (tripwire: >= 15, the bar-rerun score, and report regressions vs Run 69 and vs the rerun), then fresh-20 once
  (>= 3, base). Only a candidate that clears all of those gets the full 200 (about 3 h), compared with the base's full
  200 from benchmark.db (and one base rerun if the DB has only one).
* All of this is a tripwire ladder, not accuracy estimation; wording about statistics never gets stronger than what
  the counts support. A negative result is reported as a negative result.
* No candidate selection on mid20/fresh/confirm; none gets re-run until it passes.

## Leak and hygiene gates (unchanged, applied to every row)
ID / exact-turn / 8-gram / similarity >= 0.85 vs BFCL 200, mid20, fresh bank, dev, dev2, confirm; literal overlap
report-only; stop share capped at 25%; ungrounded and failed-action targets dropped; token length <= 9000 (memory
checked on the longest row); rows deduplicated; train/valid task-disjoint.

## Not in scope
Pushing, PRs, merging; RAG; other models; editing paper/main.tex or the CSULA thesis; any key in logs or git.

---
## Revision after Codex plan review (r11, 2026-11-07); this section supersedes the text above where they differ
Codex: design not yet able to resolve a 4-5 task effect; fix provenance/hint enforcement first. Changes adopted:
1. **Screening pilot, inconclusive allowed.** Advance rule `>= max(B1,B2)+3` is a screening threshold only (the "not below
   min" clause is dropped as redundant). Paired task-level gains/losses and base-vs-base flips are recorded; a
   candidate that screens positive is re-run once (and base is already run twice), interleaved. A base-vs-base gap
   above 4 triggers a measurement investigation, it does not by itself prove the effect is undetectable. Wording stays
   at "screening evidence", never an accuracy or significance claim.
2. **Hint-free is enforced in code**: `distill_rationalize --hinted 0 --samples 8`; the run asserts no `hinted` row;
   acceptance is exact function-string equality of the action (stricter than effect equality; no `same_effect` is used
   to substitute rows). Dropped rows are written to a sidecar for reachability audit (by correction type: early-stop fix,
   recovery, other act).
3. **Hybrid IDs** now include the source student episode (`<student_episode>__h<switch>__s<k>`), fixing the replay-script
   collision when two failed samples of one task share a switch position.
4. **Student sampling uses sample ids >= 1 at temperature 0.7** (sample 0 is greedy in the helper); the gate-like
   greedy runs use sample 0.
5. **Arm A rows are the student's own original successful replies, never resampled.** Arm AB = A + B (teacher-informed,
   hint-free native). Both arms get the SAME number of optimizer updates (the smaller set is repeated to match), so AB - A
   reflects the teacher-informed rows rather than step count; it is still a package comparison, not a clean teacher
   ablation (tasks themselves are teacher-authored and teacher-verified in both arms). A:B ratio is reported.
6. **Dose is measured, not assumed.** Before dev evaluation: log optimizer updates, supervised tokens, validation loss,
   adapter norm; probe action change on 100 held-out training-valid prompts (greedy action under runtime LoRA vs base).
   Two doses (lr 5e-6 and 2e-5) are trained on AB only; the dose is chosen by this predeclared rule: take 2e-5 unless its
   validation loss is higher than at 5e-6, in which case 5e-6. The dose is fixed before any dev result; dev results never
   change it (the earlier dev-triggered lr fallback is removed).
7. **Family-level split.** dev2 / pool 2 split by gold-function-sequence signature (whole groups on one side), then
   cross-split audit; counts of authored / rejected / verified / retained are reported. Claims about dev results are
   limited to teacher-verifiable synthetic tasks; the external claim comes from BFCL only.
8. **One frozen selection rule**: primary candidate = AB at the chosen dose; A is the diagnostic arm. If both screen
   positive the winner is the one with the higher mean over its repeats; ties go to A. Confirm bank, mid20 and fresh-20
   are single untouched checks for the winner, reported as regression diagnostics against a contemporaneous base run
   on the confirm bank and Run 90 on mid20; the full 200 follows only if dev screening was positive on repeat and mid20
   is >= 15.
9. **Manifest**: model/adapter/GGUF sha256, converter revision, server argv, seeds, sample ids, temperature, concurrency
   are recorded in the run log at each step; API errors and incomplete episodes are excluded and counted, never scored as
   fails silently.
