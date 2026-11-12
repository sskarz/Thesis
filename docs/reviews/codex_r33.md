**VERIFIED — Stage B must stop.** Independent classification and pair reconstruction reproduce the saved artifacts. One state-count label and one denominator in the runlog need correction. Full protocol certification remains incomplete because final branch provenance was skipped.

Read-only review completed; no model runs or files written. F/K2 text remains aggregate-only.

**VERIFIED — sampling and reconstruction**

- 472 unique states, 180 tasks; exactly 6 indices per state: **2,832 unique records**, no duplicates, errors, missing keys, or unexpected indices.
- All records are HTTP 200 with `finish_reason=stop`; temperatures match greedy plus five samples at 0.7.
- Sampling manifest matches the states SHA256, records `lora=[]`, build `b10894-d344123fe`, and the bar GGUF path. That path resolves to blob `923c4c86177d2ee173a7f5b4fa3d0ac65f5962ab15e6d6a5bc250aec4fd7bf7e`, matching the recorded hash.
- Independent classification using v2.3 precedence gives:

| Class | Responses |
|---|---:|
| exact | 408 |
| execution_error | 350 |
| premature_stop | 956 |
| return_equivalent | 169 |
| unresolved | 676 |
| unparsable | 273 |

No replay/checker exceptions occurred. No responses were excluded for truncation or missing reasoning.

- **173 states have any exact response; 171 have a chosen-eligible exact response** after the reasoning cap. Verified-inferior states: **299**; states with both eligible classes: **99**. Five exact responses exceed 3,500 reasoning characters.
- Registered zip selection produces **143 candidate pairs**, then **11 overlength drops**, leaving **132 pairs / 67 tasks**.
- Reconstruction exactly matches saved pair order, prompts, responses, token lengths and rejection classes.
- Split is exactly SHA256 task ordering, first `ceil(0.10 × 67) = 7` tasks for validation: **train 120/60 tasks; validation 12/7 tasks**, with no task overlap.

**VERIFIED — floor and outcome-dependent selection**

The [registered rule](~/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND5_PLAN.md:57) requires **≥150 train pairs from ≥50 tasks after filtering and splitting**. The task requirement passes; the pair requirement fails. Logs record pair-builder exit **2** and no subsequent B training.

The current plan and builder match frozen commit `a8668f1`, preceding sampling; the saved outputs match that implementation. I found no evidence of outcome-based filter, pairing, split or floor changes.

**132 total pairs / 67 tasks does not change the decision.** Pooling validation is outside the rule and still yields fewer than 150 pairs. Even retaining all 11 overlength candidates would yield only 143 total pairs. No registered-consistent reading of these artifacts meets the floor.

**BUG — exact runlog edits**

At [runlog line 148](~/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:148), replace the state-count sentence with:

> States with any exact response: 173; with a chosen-eligible exact response after the reasoning cap: 171; with a verified-inferior response: 299; with both chosen-eligible exact and verified-inferior responses: 99.

At [line 149](~/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:149), replace:

> student originally agreed with teacher on 18.3% of states; mean branch length difference 18.6 tokens.

with:

> Original student–teacher agreement is 22/120 train pairs (18.3%, pair-weighted), or 13/82 unique train states (15.9%); mean chosen-minus-rejected total branch length is +18.6 tokens.

Also clarify the pair funnel:

> Pairs: 143 candidates before length filtering; 11 dropped because either branch exceeded 9500 tokens; 132 retained before splitting across 67 tasks; 120 train pairs from 60 tasks and 12 validation pairs from 7 tasks.

Other reported training-pair summaries reproduce: rejected classes **50/70**, recovery/correction-only **46/74**, greedy chosen **14/120 = 11.7%**. These counts support descriptive teacher-action preferences, not better recovery or continuation success.

**BUG — exception handling does not fully implement v2.3**

[Classifier exception handling](~/Documents/GitHub/Thesis-wt-distill/sft/r5_pairs.py:164) calls helpers that swallow exceptions:

- [`turn_complete`](~/Documents/GitHub/Thesis-wt-distill/benchmark/distill_onpolicy.py:93) returns `False` on checker failure, potentially converting infrastructure failure into `premature_stop`.
- [`same_effect`](~/Documents/GitHub/Thesis-wt-distill/benchmark/distill_onpolicy.py:128) returns `False` on replay failure, hiding it as unresolved.

Exact fix: add an optional exception-propagating mode to both helpers, use it from `r5_pairs.classify`, and catch those exceptions there as `replay_exception`. Preserve existing defaults for other callers. **No current recount changes:** the independent pass observed zero such exceptions.

**CONCERN — audit completeness, not the stop decision**

The [v2.3 final-row audit](~/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND5_PLAN.md:56) requires both branches of train and validation pairs to be checked. “Moot after the stop” overstates compliance.

Replace [line 151](~/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:151) with:

> Diagnostics copy: docs/reviews/r5_pair_diagnostics.json. The required final train-and-validation branch provenance audit was not performed; these counts are pre-audit and the pair set is not provenance-certified. Stage B remains stopped: any task exclusions and recomputed split can retain at most 132 total pairs, below the 150-train-pair floor.

The diagnostics also lack the complete registered before/after-every-filter breakdown, including separate prompt/reasoning/answer token lengths. Add:

> The saved diagnostics do not yet contain every registered pre/post-filter diagnostic.

**Options only**

Within the registered plan:

- Retain the B stop and report A results with their existing validity caveats.
- Complete the missing provenance and descriptive diagnostics; apply registered task drops and recompute if needed.
- Complete any outstanding A integrity replacement under its existing rule.
- Bring any proposal beyond the plan to Sans as a separate decision.

Post-outcome deviations would include lowering the floor, pooling validation for training, changing the split, adding samples or source states, expanding pair combinations, relaxing parsing/reasoning/length rules, admitting unresolved/equivalent responses, hinted rescue, or training DPO/chosen-only SFT despite the failed floor. Any separately approved experiment must be identified as a new post-outcome protocol; it cannot retroactively make this Stage B pass.