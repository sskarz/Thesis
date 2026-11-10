# Distillation round 2: failure modes, controls, and why the 17/20 gate cannot be cleared reliably

2026-11-07, run while Sans slept (instruction: find the failure modes, fix them, keep going until the
score beats the goal, use Codex for direction). Full action log: `docs/DISTILLATION_RUNLOG.md`.
Every number below is from `benchmark.db` (runs 86-90, read-only) or the saved eval reports under
`sft/data/distill/student_eval/` and `sft/data/distill/gate/`.

## Headline
1. **The reference gate is not reliably reproducible.** The nominal base configuration (same bartowski GGUF, no
   adapter, same harness settings, same 20 tasks) scored **17/20 historically (Run 69) and 15/20 on rerun
   (Run 90)**, losing two of Run 69's passes. Complete environmental equivalence was not established (macOS
   27.0.0 -> 27.3.0, harness revisions that preserve this mode's path, no stored GGUF hash or cache state for
   Run 69), so this is operational reproducibility evidence, not an isolated noise measurement. The untouched-base
   rerun failed the predefined gate (15/20, two regressions); the historical reference passed.
2. **t1 (teacher-only rows) scored 10/20, below the three observed untuned scores of 17, 15 and 16** (the 16 used
   our own conversion pipeline). This is a regression tripwire, not an estimated base range and not an established
   effect of the teacher data. The self-output control c1 (trained on the base's own replies; not a no-op, it
   updates weights) scored 13 (fused) and 14 (runtime LoRA); these controls do not isolate training, conversion and
   execution variability.
3. **No model I produced beat the goal.** I did not keep re-running candidates until one reached 17: with run-to-run
   variation of this size that would be selection on noise, and I will not report it as an improvement.

## mid20 controls (same harness: iffc-ooda-card, -np 4, temp 0, llama.cpp b10894-d344123fe)
| run | model | mid20 | passes lost vs Run 69 | new passes |
|---|---|---|---|---|
| 69 | bar: bartowski GGUF | 17/20 | 0 | - |
| 90 | **bar rerun** (bartowski, no adapter) | 15/20 | 2 (45, 146) | - |
| 88 | untuned base through OUR convert+quantize path | 16/20 | 3 (14, 91, 146) | 115, 195 |
| 89 | bar GGUF + **runtime** no-op LoRA c1 (own greedy replies) | 14/20 | 5 (45, 7, 14, 60, 146) | 152, 195 |
| 87 | c1 fused + re-quantized | 13/20 | 5 (12, 14, 150, 60, 112) | 115 |
| 86 | distill-t1 (154 teacher rows, fused) | 10/20 | 8 (23, 7, 14, 91, 150, 146, 112, 186) | 115 |

Reading (hedged; n = 20, one run per model, tripwire not accuracy): requantization is not necessary for an observed
regression (c1 runtime 14 vs c1 fused 13), and its contribution remains unresolved; the pipeline-matched base
scored 16. Differences of one to three passes here cannot be separated from execution variability.

## dev bank (23 fresh teacher-verified authored tasks, never trained on; untuned base via our pipeline = 7/23)
| model | dev |
|---|---|
| untuned base (our pipeline) | 7/23 |
| t1 teacher-only rows | 6/23 |
| t2 on-policy teacher labels | 4/23 |
| t3 hybrid (student prefix + teacher finish, outcome-verified) | 5/23 |
| c1 control (base's own replies) | 4/23 |
Each fine-tune lost base passes and gained none, including the control trained on the base's own output.
The dev bank cannot separate data effects from a shared perturbation effect (n = 23, one run each).
Holdouts are ID-disjoint (23 / 22 / 78 tasks) and no candidate trained on a dev or confirm id or exact turn, but
confirm tasks 0259 / 0287 resemble pool tasks 0004 / 0216 (similarity 0.86 / 0.89) and t1 trained on 0004:
confirm-bank similarity contamination remains for t1. t2 onward excluded 0004 / 0216.

## Failure modes found
* Student stops or gives up early (empty turns) and skips steps the stock checker needs: the checker needs the
  model's execution results to contain the gold's results, so a skipped lookup (e.g. retweet without the
  gold-required get_tweet; closing a ticket by a guessed id instead of listing tickets) fails the task
  (dev 0338 = missing results; dev 0156 = state mismatch from a guessed id). Codex narrowed this: extra calls are not always free (they can change state).
* Training-target format mismatch (established): the served model answers `<|channel>thought ... <channel|>`
  then JSON, our rows put JSON directly after `<|turn>model`. Starting validation loss was ~1.0-1.2 per token
  with JSON-only targets and 0.281 with native reasoning + answer targets. The prompt side matches serving
  exactly (token counts equal, `<|think|>` present). Causation of the pass losses is NOT established:
  reasoning was still present in 96.6-98.8% of requests for every model.
* Not supported by the data: "training taught the model to skip thinking" (tokens/request 296-328 for base and
  all runs; collapsed v11 had 31). Re-quantization is not necessary for an observed regression; its contribution
  remains unresolved.

## What was built (all offline-tested, committed locally, reviewed by Codex in 9 rounds, docs/reviews/)
Teacher adapter; task author + validator (real BFCL backends, semantic audit, leak checks); outcome-verified
solver; student-state relabeling with gold-checked stops/acts; hybrid rollouts (student prefix, teacher finish,
verified end to end); rationalization into native reasoning + answer format; local-student recorder;
MLX-adapter -> GGUF-LoRA converter so an adapter runs at runtime on the exact bar GGUF;
dev / confirm / pool banks (45 verified tasks held out); gate scripts. Logged API cost across all calls: about $1.55 in total (about $0.70 in round 2), from per-call usage logs, not a verified
account balance.

## Recommendation / decisions for Sans
* Stop treating mid20 >= 17 as a pass criterion. Either (a) redefine the gate as "no worse than the bar's
  own rerun distribution" (needs repeats), or (b) judge candidates only on the full 200 with a paired test
  against the base (about 3 h per run; the base itself needs 2-3 reruns for noise).
* Teacher-only rows: report t1 as a regression tripwire (10/20 vs untuned 15-17), not as an estimated effect.
* If a next attempt is wanted: use native-format targets (verified format fix), runtime LoRA on the exact bar
  GGUF, hybrid outcome-verified rows with hinted rationalizations reviewed or dropped, and a much larger
  held-out dev bank (>= 100 tasks) so effects of 3-4 passes become visible.
* Not done: fresh-20 bank for any candidate; t4 (native-format teacher rows) was aborted after Codex flagged
  leaked hints; confirm bank (22 tasks) is untouched.
