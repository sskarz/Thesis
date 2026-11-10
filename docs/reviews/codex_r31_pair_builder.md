**REVISE before sampling.** State construction and the final audit check out. The matcher, sampling recovery, and pair-level provenance need fixes. No models were run and no files were written; F/K2 inspection remained aggregate-only.

**VERIFIED — selection, counts, and histories**

- [State selection](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/r5_state_audit.py:41) implements eligibility first, registered sorting, prompt+target deduplication, then removal of every conflicting-target prompt. I reproduced **149 duplicates and 18 conflicting states removed**.
- Recomputed guard predicates match the [final audit JSON](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/data/distill/r5/state_audit_final.json) and drop list: **482 states / 183 tasks → 472 / 180**, comprising **153 recovery / 319 correction-only**. The four contaminated states span three dropped source tasks. The [final runlog entry](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:120) is accurate.
- All **462 episode IDs are unique** across the three student pools. For **every saved state**, attached history, task, messages, turn, student action, and `step == executed_before` match recomputed `label_dir.align()`. Tool-observation counts also equal `step`; all three original label reports have zero unconsumed-alignment episodes. **I found no wrong-history state.**
- Deterministic replay of **30 states**, covering source/recovery/label strata and deep histories, reproduced **27 `act_ok` and all three `stop_correct`** labels without mismatches.

**VERIFIED — correct parts of sampling and assembly**

- Defaults are greedy plus five T=0.7 samples; messages are passed unchanged; adapter absence is asserted.
- For all **472 prompts**, training’s system-content `<|think|>` marker renders identically to the local template with `enable_thinking=True`, with exactly one marker. Actual serving configuration still needs a preflight assertion.
- Canonicalization sorts argument keys while preserving call order.
- Nonempty reasoning is required for both branches. `classify()` intentionally counts long-reasoning matches as `exact`, but the **chosen list filters `len(reasoning_content) <= 3500` before zipping**. Rejected reasoning has no character cap. `finish_reason != "stop"` excludes either branch.
- Sample-index ordering and `zip(chosen, rejected)[:2]` match registration: no cross product or reuse of a single chosen candidate. Both complete branch lengths use the same tokenizer/template calls as [the trainer](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/r5_dpo.py:32), with the amended **9500-token** cap and no truncation.
- Validation uses SHA-256 task ordering and `ceil(10%)` of tasks with retained pairs. Train/validation membership is task-disjoint. The chosen-only trainer consumes the same pair records and shuffled indices, preserving actual chosen multiplicity.

**BUG — matcher differs from v2.1 item 2.**  
At [classification](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/r5_pairs.py:115):

1. `_is_error` is used instead of registered `iffc_state.is_error_result`. These differ: `"Exception: bad"` and `"Error: bad"` are missed; `{"error": null}` and `{"error": ""}` are incorrectly treated as errors.
2. The `label == []` branch skips execution checking and introduces **`act_when_done` as verified inferior**. That class is absent from the amended registered precedence. This affects three eligible stop-label states.
3. Arbitrary replay exceptions are treated as verified execution errors, potentially converting checker/infrastructure failures into rejection evidence.

**Exact edits:** import/use `is_error_result`; remove the `if not label` shortcut and remove `act_when_done` from rejected eligibility. Apply **exact → execution error → premature stop → return-equivalent → unresolved** uniformly, including empty teacher labels. Log unexpected replay exceptions separately and fail classification rather than treating them as verified inferior. Retaining `act_when_done` would require an explicit pre-outcome amendment.

**BUG — 1024 does not reproduce the historical native sampler’s limit.**  
The claim at [sampling line 96](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/r5_pairs.py:96) is unsupported. Although `TeacherConfig.max_tokens` defaults to 1024, [the local backend returns before applying it](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/benchmark/teacher_api.py:198). `distill_rationalize` therefore sends no explicit limit. Historical control logs contain **five completions above 1024 tokens**, reaching **1640**.

**Exact edits:** correct the script description and settle the limit before sampling: either reproduce the historical request/server limit, or explicitly register **1024 as a new limit**. Do not describe it as inherited from `TeacherConfig`.

**BUG — failed samples cannot resume, and incomplete sample sets can silently build pairs.**  
[Resume handling](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/r5_pairs.py:81) marks every saved key complete, including failures. The advertised rerun consequently skips failed jobs and fails the same final assertion again. Pair loading accepts missing, duplicated, unknown, or out-of-range sample records.

**Exact edits:**

- Preserve attempt records; consider a key complete only after a successful, validated HTTP response. Retry failed/missing keys and select one authoritative successful record per key.
- Call `raise_for_status()` and validate response fields.
- Before pairing, require exactly indices **0–5 for each of 472 states**, unique authoritative keys, no unresolved API errors, and matching state/config hashes.
- Log API errors separately from truncation.

**CONCERN — server provenance and the four-hour limit are not enforced here.**  
An empty adapter list also passes on the wrong base model. The sampler does not assert bar GGUF/build/template or enforce its registered deadline.

**Exact edits:** pin `/props`, model/build and tokenizer/template provenance; assert one rendered think marker and prompt-token identity; bind resumed samples to that manifest. Enforce a **14,400-second job deadline** in the sampler or its launcher.

**CONCERN — `parse_strict` is stronger than the historical “strict JSON-list filter.”**  
[The historical filter](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/benchmark/distill_native.py:146) requires successful whole-string `json.loads` yielding a list. Current `parse_strict` additionally rejects aliases, extra call keys, and some parameter representations accepted by historical normalization. It correctly rejects fences and surrounding prose. Across all **1103 historical native responses**, I found **no disagreement** between bare-list acceptance and the new schema check.

**Exact edit:** explicitly freeze this additional schema restriction before outcomes, or reproduce historical bare-list gating followed by its existing call normalization. Keep permissive text extraction diagnostic-only.

**BUG — final preference-branch leakage protection is missing.**  
Source isolation is verified: selected task IDs are disjoint from dev_all, confirm, F and K2; all-split BFCL checks cover protected benchmark subsets. `g/tasks_F.jsonl` is curriculum G, distinct from screening `f/tasks_F.jsonl`. K2’s hash matches its frozen value.

However, [the provenance checker](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/check_provenance.py:67) does not expand `chosen`/`rejected`, and [its content scan](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/check_provenance.py:149) omits reasoning. Newly generated reasoning or rejected answers could therefore introduce held-out text without detection. The source audit cannot certify future generations.

**Exact edits:** expand **both branches of train and validation pairs**, inspect rendered content including reasoning against held-out and BFCL text/IDs, and produce aggregate-only reports with explicit K2 input. Drop every affected source task and recompute pairing/splitting/counts under the registered rule.

**BUG — diagnostics are incomplete; two smaller reporting errors also need edits.**  
[Current diagnostics](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/r5_pairs.py:170) do not report the registered strata and lengths before/after every filter. Missing items include original student agreement, source, prompt/answer lengths, branch-length difference, exclusion reasons, and stagewise uniqueness.

**Exact edits:**

- Add stagewise diagnostics, including `exact_but_reasoning_over_3500`, API failures, truncation, overlength and provenance exclusions.
- Replace `"val_tasks": sorted(valt) and len(valt)` with **`len(valt)`**. Currently an empty split reports `[]`, otherwise an integer.
- Count unique chosen **response content**, not merely `(state_id, sample_idx)`. Identical responses at different indices currently inflate uniqueness and hide multiplicity. Retain occurrence counts separately.
- Report duplicate-content pairs separately; resolve the “distinct pairs” interpretation without changing registered zip ordering.
- After writing diagnostics, exit nonzero when the post-split floor fails. The trainer already asserts the floor, but pair construction currently only reports a boolean.

**CONCERN — floor feasibility is plausible, with substantial risk.**

The old yields measure finding a chosen response, not finding both an accepted chosen and a verified-inferior rejection:

| Extrapolation to 472 states | States with a chosen response |
|---|---:|
| Raw R 60/166, C 142/166 | ≈328 |
| Strict R 45/166, C 135/166 | ≈301 |
| Strict R; correction-control sensitivity using 24/42 | ≈224 |

The third scenario matters because all **319 correction-only states** differ from the student, whereas the old controls were predominantly student-equivalent. Its 24/42 rate is still pre-strict, so it is not a clean conservative bound.

Assuming one retained pair per usable state and roughly 10% pair loss to validation:

| Scenario | 50% also produce a usable rejection | 75% do |
|---|---:|---:|
| 301 chosen states | ≈135 train pairs | ≈203 |
| 224 chosen states | ≈101 | ≈151 |

Second pairs can improve these figures; truncation, unresolved alternatives, length and leakage filters reduce them. The **50-task floor requires at least 56 pair-bearing tasks before splitting**.

Thus **150 pairs / 50 tasks is reachable, but not reliably predicted by existing evidence**. The registered stop rule is meaningful here; no sampling outcome should trigger a change to filtering, pairing, or the floor.