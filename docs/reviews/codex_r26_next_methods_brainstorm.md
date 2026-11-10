# Fine-tuning brainstorm: diagnosis and next experiments

**My leading diagnosis is a data-selection and learning-objective problem.** The pipeline concentrates on behavior the student already reproduces, then optimizes teacher-forced continuations whose loss is a weak proxy for autonomous task completion. Quantization mismatch remains worth testing, but the repo does not establish it as the main cause. Increasing adapter capacity alone has already disappointed.

**Recommended direction:** run precision and loss diagnostics first; then build genuinely same-state, outcome-verified preference pairs and compare DPO against SFT on the identical chosen responses. Teacher-interleaved rollouts are the strongest subsequent extension. Full GRPO is premature.

This review was read-only. No training, model generation, benchmark evaluation, or paid API calls were made. Bank F per-task text was not read.

## 1. What the evidence actually establishes

| Observation | Supported conclusion |
|---|---|
| Five adapters score **36–41/103**, against base **44/44**, with a screening bar of **47**. | None passed the screen. This does not identify the causal failure or establish that every adapter reduces expected accuracy. |
| Correcting the thinking marker gives AB **40**, A **37**. | The original mismatch was real; correcting it did not rescue the screen. |
| Expanding q-only LoRA to q/o+MLP gives **36/103**, despite lower validation loss. | Narrow q-only capacity cannot carry the whole explanation. |
| On fresh F, base scores **35/35**, H **35**, AB **30**, out of 81. | The AB deficit recurs on another synthetic bank. H changes behavior without improving aggregate completion. |
| Base runs exchange **3 tasks each way** on dev103 and **5 each way** on F81. | Greedy serving is not task-deterministic. These observations do **not** establish a ±5 confidence interval or a reliable detection threshold. |
| Recovery collection yields **45 strict JSON rows, representing 42 unique task-step pairs**. | The unique-row threshold of 100 was not reached. The findings document conflates strict row count with uniqueness. |

Sources: [Round 3 findings](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND3_FINDINGS.md:46), [expanded-target result](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND3_FINDINGS.md:125), [Round 4 results and corrected counts](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:87).

## 2. Ranked explanations for the losses

The ranking below concerns explanatory support. Evaluation noise is a separate limitation on what we can infer.

### 1. Selection favors already-reachable behavior and discards difficult corrections

**Strongest evidence.**

Arm B retains a teacher-informed action only when the unhinted student reproduces that action. Round 3 kept **161/300** candidates. Round 4 exposes the bottleneck more clearly: only **42 unique strict recovery states** survive.

This makes the training distribution selective in two ways:

- Whole-episode success favors trajectories that already work.
- Native rationalization favors actions the current student can already reach within a small sampling budget.

That is useful for preserving native format, but it removes precisely the teacher corrections that could expand capability. It also rejects function-string differences that might be semantically equivalent. The nonempty-reasoning and length requirements add further selection.

The result is closer to **reinforcing selected existing behavior** than transferring the teacher’s full corrective policy. This does not make the approach useless, but it limits the information available to training. [Native filtering implementation](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/benchmark/distill_rationalize.py:43)

### 2. Behavioral coverage is thin, even where API-class coverage exists

**Strong evidence for thin coverage; causal contribution remains untested.**

AB has **95 training tasks and 320 rows**. Training touches the difficult API classes, so “the class was missing” is insufficient. However, the recorded coverage audit finds:

- **24 of 80** functions in dev reference calls absent from training targets.
- Almost no unit-convention examples.
- Very few after-error training rows.
- Failures involving skipped lookups, guessed identifiers, incomplete action lists, file names, and navigation.

The adapter improves exact agreement on the selected training probe, **9→17/35**, while held-out agreement moves **25→24/35**. That fits task-specific learning with limited behavioral transfer.

The right expansion is more distinct decision patterns and states, rather than additional rows from the same few trajectories. [Coverage and failure audit](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_NEXT_METHODS.md:62)

### 3. Teacher forcing teaches continuations more readily than autonomous decisions

**Strong diagnostic concern.**

The answer loss is measured after supplying the gold reasoning. On held-out probes, answer NLL falls **0.847→0.0035**, yet autonomous action agreement does not improve.

A plausible mechanism is:

1. Given the recorded reasoning, emitting its implied JSON becomes easy.
2. During autonomous generation, the model produces different reasoning.
3. The improved conditional answer distribution is reached less often than the loss suggests.

Reasoning tokens dominate supervised token count, but “the model only learned reasoning/style” is too strong: answer tokens account for **51% of held-out NLL reduction** in the recorded diagnostic. The more defensible concern is **conditioning and exposure**, not merely token imbalance.

Action-weighted loss deserves a control experiment, but the already tiny conditional answer NLL makes it an uncertain cure. [Split-loss diagnostics](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND3_FINDINGS.md:116)

### 4. Failure supervision lacks sufficiently strong counterfactual labels

**A clear missing ingredient.**

Current SFT mostly rewards successful outputs. It does not explicitly compare a failed student action with a better alternative from the identical state.

Importantly, an existing `act_ok` teacher label means the action executes cleanly and passes grounding checks. It does **not** prove successful continuation. Whole-episode hybrid verification is stronger, but teacher takeover success also does not prove that one teacher correction followed by the student succeeds.

DPO is promising only after this distinction is handled. Otherwise, it can confidently prefer executable but unhelpful actions. [Label checks](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/benchmark/distill_onpolicy.py:97), [hybrid verification](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/benchmark/distill_hybrid.py:56)

### 5. Train/serve numerics could amplify adapter changes

**Plausible; presently weak causal evidence.**

The base backends agree on **30/35** held-out actions; with the adapter they agree on **26/35**. That suggests checking transfer, but it combines quantization, backend, kernels, and potentially model-conversion differences. The prompts come from only ten tasks.

Moreover, MLX+adapter itself achieves only **16/35** training and **26/35** held-out target actions. There is no demonstrated large hidden fine-tuning win in bf16 waiting to be recovered at serving.

Keep numerics as a cheap diagnostic, rather than the leading rescue theory. [Parity findings](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_NEXT_METHODS.md:58)

### 6. Thinking and history representation may interact with learning

**Original bug confirmed; residual mechanism untested.**

The missing thinking marker affected all original training prompts. Later checks establish corrected template parity across logged histories, one BOS, and no evident dev context overflow.

What remains unresolved is behavioral: earlier reasoning is flattened into assistant content. Training copies that representation, so this is not an established train/serve mismatch. It could nevertheless encourage reliance on verbose historical reasoning or stale conclusions.

A new reasoning-history representation would need matched training and inference. Changing inference alone would answer a different question. [Template audit](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND3_FINDINGS.md:103)

### 7. Capacity, dose, and forgetting remain possible contributors

**Weaker than the data/objective explanations.**

The wider adapter has roughly twelve times as many trainable parameters and performs worse. This weakens “q-only cannot learn it” as the primary explanation. It does not rule out better layer placement, rank, or regularization.

Likewise, larger runtime scales hurt probe agreement, but runtime scaling is not equivalent to more training. The dose comparison selected learning rate using validation loss, whose relationship to completion is demonstrably poor.

A bounded lower-dose control is reasonable. Another broad capacity sweep is low priority. All-layer adaptation already encountered Metal OOM. [Capacity result](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND3_FINDINGS.md:125)

### What Round 4 changes

H turns many post-error empty replies into new calls, yet achieves **35/81**, with **7 gains and 7 losses** versus base.

This argues against optimizing “retry more” as the objective. Useful training must distinguish:

- Correcting an argument.
- Performing a missing lookup.
- Completing omitted actions.
- Recovering from a reversible mistake.
- Recognizing that another call would compound the damage.

A retry-frequency metric cannot substitute for task success.

## 3. What “full precision” would change

You currently use **bf16 LoRA training on an unquantized base**, followed by an **unmerged fp16 runtime adapter on Q4_K_M serving weights**. You are not doing QAT, and this is not QLoRA training.

| Proposed change | Meaning and consequence |
|---|---|
| Serve an F16/BF16 base with the existing adapter | Removes low-bit serving weights. No retraining is initially necessary. Best first diagnostic. |
| Train and serve in the same higher-precision backend | Improves numerical alignment, but changes the serving system and may require more integration. |
| Train all parameters | Full-parameter fine-tuning; independent of precision. Much greater memory/optimizer cost and currently unattractive. |
| Use float32 throughout | A separate, expensive numerical choice with little present justification. |
| Train against dequantized Q4_K_M weights | A numerical matching control. Dequantization preserves quantization error; it does not restore original weights or guarantee kernel parity. |

QLoRA specifically trains adapters through a frozen quantized base. Its results do not establish exact matching between MLX quantization and llama.cpp Q4_K_M. Keeping the runtime adapter unmerged avoids introducing another merge/requantization transformation, but does not remove the existing bf16/Q4 gap. [QLoRA paper](https://arxiv.org/abs/2305.14314)

**Thesis cost:** an F16 adapter beating a Q4 base is not a clean fine-tuning comparison. Compare base versus adapter at each precision. A higher-precision result can be reported as a diagnostic or separate condition; the existing Q4 headline needs its own matched comparison.

## 4. Evaluation and cost rules for the candidate experiments

These are **proposals for a new registration**, not changes to completed rounds.

**Shared screen S:** freeze one candidate and its controls before opening outcomes on a fresh bank of roughly 100–150 tasks. Run base and candidate twice in counterbalanced order. A practical continuation rule could require a mean gain equivalent to **six tasks per 100**, with positive paired net gains in both repetitions. This remains a resource-allocation screen, not proof of significance.

Final evidence should come from an independent protected comparison, with uncertainty clustered by task across repetitions. Do not treat repeated evaluations of the same tasks as independent additional tasks. Small effects can remain inconclusive even after repeating.

**Cost anchors:** recorded Q4 evaluations take approximately **27–57 minutes per arm** on 81–103 tasks. The 79-task, two-sample curriculum rollout took about **59 minutes**. Existing small SFT runs suggest **80–95 minutes**. New methods can be slower.

The estimates below are planning ranges, not measured forecasts. Add approximately **2–4 hours for four Q4 evaluation runs** and, if needed, roughly **$2–3 for a fresh verified bank**. Development time is separate. All model runs and API use still require Sans’s approval.

## 5. Eleven concrete candidate experiments

### E1. Same-backend precision comparison

**Hypothesis:** Q4 serving disproportionately erases or redirects the adapter’s benefit.

- **Data/code:** existing frozen training/validation prompts and adapter. Serve Q4 and F16 from the same pinned source/build; compare base and adapter in each. Conversion tools exist; add manifests and paired action reports.
- **Cost:** approximately 1–3 local hours for a small probe, potentially substantially longer for full F16 episodes; **$0 API**. Several hours of setup.
- **Pass rule:** advance the precision hypothesis only if the adapter-minus-base effect improves consistently at F16 on independent task-level outcomes. Higher base/adapter action agreement alone is insufficient.
- **Risk/detectability:** a small probe diagnoses gross transfer problems, not task-accuracy differences. A full comparison still faces screening noise. F16 latency, context, and concurrency must be measured.

**Priority:** first diagnostic; no new training.

### E2. Answer-weighted SFT on identical rows

**Hypothesis:** current loss allocation spends update capacity on reasoning continuations at the expense of executable decisions.

- **Data/code:** unchanged corrected AB rows, targets, updates, and adapter modules. Add token masks; compare ordinary SFT with a predeclared answer-weighted objective, such as 90% normalized answer loss and 10% reasoning loss. Include termination tokens. Existing prompt masking is insufficient.
- **Cost:** about 1.5–3 local hours per training arm; **$0 API**; roughly a day to implement and verify masks.
- **Pass rule:** S against base, plus improvement over the matched ordinary-SFT control. Report autonomous generation and separate losses.
- **Risk/detectability:** low conditional answer NLL suggests this may not help. Small completion gains will remain unresolved; large changes in empty/under-call behavior are easier to diagnose.

An answer-only endpoint is useful later, but changing both loss weighting and reasoning format simultaneously would obscure the result.

### E3. Behavior-balanced data expansion

**Hypothesis:** coverage of decision patterns matters more than repeating existing successes.

- **Data/code:** teacher-verified tasks spanning lookup-before-act, identifier grounding, navigation, valid-value discovery, complete action lists, and correct stopping. Reuse authoring, verification, exclusion, and native-row tooling. Hold architecture, loss, and updates fixed.
- **Cost:** approximately **$4–7 API** for a bounded expansion; **6–12 local hours** for rollout/filtering/training, depending on retention. Several hours of orchestration.
- **Pass rule:** S; compare against an equal-row/token control made from existing data. Report distinct task/family coverage.
- **Risk/detectability:** the native gate can discard the new difficult content again. Broad same-teacher synthetic gains may not transfer to BFCL. Only a substantial completion gain is detectable on a small bank.

Do not name this a pure scale experiment if both amount and behavioral composition change.

### E4. Two-round expert iteration / rejection fine-tuning

**Hypothesis:** successful trajectories from the updated policy expose learnable states missed by one-round arm A.

- **Data/code:** real student rollouts, whole-episode BFCL verification, native success rows—all exist. Add iteration orchestration and task-balanced sampling. Train round one, refresh rollouts, then train on newly successful trajectories.
- **Cost:** **6–12 local hours** for a small two-round pilot; **$0 API** using existing training tasks, plus optional task authoring. About a day of engineering.
- **Pass rule:** round two must beat both round one and a matched no-update resampling control, then satisfy S. Track newly solved training tasks separately.
- **Risk/detectability:** arm A already failed; this must add new successful states rather than repeat the same easy trajectories. If the pilot yields few new successes, stop before another evaluation campaign.

### E5. Same-state, answer-focused DPO with a chosen-only SFT control

**Hypothesis:** explicitly suppressing observed wrong actions adds information beyond imitating the correction.

**Exact pair source:**

1. Take a failed student action from a training-pool state.
2. Replay the identical history and environment state.
3. Obtain a teacher correction or alternative student action.
4. Verify continuation after each branch using a fixed continuation policy and budget.
5. Keep strict, unambiguous preferences; retain branch provenance and state hashes.

Teacher-success/student-failure tasks are the practical source. Existing two-sample pools contain only **five mixed tasks**, so pure student pass/fail pairing is presently sparse.

- **Data/code:** reuse label/replay/hybrid components. Build counterfactual branches, same-prefix validation, pair deduplication, reference-log-prob caching, and a preference trainer. No validated in-repo DPO trainer exists.
- **Cost:** **6–12 local hours** including branch collection and training; approximately **$0.2–1.5 API** for targeted labels/continuations; **1–3 days engineering**.
- **Pass rule:** S, and DPO must outperform SFT on the **identical chosen examples**, with matched task weighting and declared update budgets.
- **Risk/detectability:** branch outcomes can depend more on the continuation policy than the first action. Few trustworthy pairs make both generalization and detection difficult.

**Answer-only caveat:** scoring only answer tokens while chosen and rejected have different unscored reasoning yields a conditional-action preference objective. It is not ordinary full-response DPO. Report it explicitly as a modified objective.

For the cleanest action comparison, branch after a shared, unhinted reasoning prefix where feasible; reject inconsistent branches. Otherwise retain separate reasoning, mask its loss, and acknowledge the conditioning difference. Do not attach a correct action to contradictory reasoning.

Teacher logits are unnecessary for DPO; policy/reference log probabilities come from local models. [DPO paper](https://arxiv.org/abs/2305.18290)

### E6. Trajectory preferences, with bounded objective alternatives

**Hypothesis:** whole-trajectory preferences align better with completion than locally ranked actions.

- **Pairs:** same task and initial state; checker-passing trajectory chosen, failing trajectory rejected. Divergent later histories are legitimate here. Score generated assistant tokens only, excluding tool observations, user turns, and padding.
- **Data/code:** existing recorded trajectories; add trajectory packing, generation masks, length accounting, and reference scores.
- **Cost:** **4–10 local hours** using existing data plus a small sampling pilot; **$0 API** unless new teacher trajectories are needed; **1–3 days engineering**.
- **Pass rule:** S and superiority to a chosen-only trajectory-SFT control.
- **Risk/detectability:** sparse pairs and long-horizon credit assignment. Penalizing an entire failed trajectory can suppress many good actions.

Predeclare **one** alternative after identifying a specific problem:

| Alternative | Concrete use | Main concern |
|---|---|---|
| **IPO** | Same validated pairs; replace DPO with a finite-margin objective. | Does not repair bad pair labels. |
| **SimPO** | Same pairs; reference-free, length-normalized sequence scoring. | Length normalization can interact with useful multi-call outputs. |
| **KTO** | Unpaired checker-labelled pass/fail trajectories, balanced by task/family. | A failed episode does not make every action undesirable. |

KTO is the natural fallback when trustworthy pairs are scarce; it is not a reason to label every step of a failed rollout negative. [IPO](https://arxiv.org/abs/2310.12036), [SimPO](https://arxiv.org/abs/2405.14734), [KTO](https://arxiv.org/abs/2402.01306)

### E7. Teacher-interleaved rollout distillation

**Hypothesis:** bounded teacher interventions let the student visit useful downstream states that permanent teacher takeover or unassisted sampling misses.

- **Data/code:** extend the existing hybrid client to return control to the student after one intervention. Use a fixed intervention schedule or trigger derived from training-side errors. Record who generated every action; verify complete assisted trajectories.
- **Training target:** executed teacher corrections and verified student continuations. Evaluate entirely without the teacher.
- **Cost:** **6–14 local hours**, approximately **$0.5–2 API** for a bounded pilot; **1–2 days engineering**.
- **Pass rule:** at matched rollout/token budgets, improve over permanent takeover distillation and unassisted RFT, then satisfy S.
- **Risk/detectability:** assisted success can overstate autonomous capability. Native filtering may erase the valuable teacher interventions again; any relaxation requires a new explicit protocol.

With text-only DeepSeek access this is **intervention-based sequence distillation**, rather than a faithful implementation of logits-based Guided-OPD or PivotOPD.

### E8. GRPO-lite with real multi-turn checker reward

**Hypothesis:** policy optimization can exploit genuine within-task outcome variation better than SFT/preferences.

- **Data/code:** sample groups of four rollouts on training tasks; use stock checker success as the initial reward. Build old-policy log probabilities, clipped updates, reference regularization, rollout masks, and zero-variance-group handling.
- **Cost:** roughly **12–36 local hours** for a small multi-round experiment; **$0 API** on existing tasks; several days of engineering.
- **Pass rule:** first require an informative pilot—for example, mixed rewards in at least ten of forty task groups. Then beat matched RFT at the same rollout budget and satisfy S.
- **Risk/detectability:** all-fail groups provide no relative signal. Serving Q4 while optimizing bf16 also complicates claims of on-policy training. Use a consistent rollout/training policy or declare the approximation.

Do not begin by rewarding retry count or number of successful tool executions: both can rise without task success.

### E9. Observable verifier plus best-of-N selection

**Hypothesis:** candidate diversity already contains useful alternatives that a verifier can select more effectively than majority voting.

- **Data/code:** generate a small candidate set; rank using schema validity, observed identifiers, duplicate calls, execution errors, and observable state consistency. Fit any learned selector only on training-pool checker labels. Existing sampling and execution records help; selector and integration need building.
- **Cost:** approximately **3–8 local hours** for collection/pilot, with inference potentially **2–5×** more expensive; **$0 API**; **1–2 days engineering**.
- **Pass rule:** satisfy S against greedy and beat equal-budget voting. Report completion versus tokens/latency.
- **Risk/detectability:** a verifier can reject obvious errors without identifying the correct plan. Small accuracy gains may be economically unattractive and statistically unresolved.

**Boundary:** using the gold BFCL checker to select evaluation candidates is an oracle experiment. Use it freely for training labels; report training-only oracle pass@N as headroom. An evaluation-time selector must not access reference calls or hidden target state.

### E10. Minimal output constraints

**Hypothesis:** a narrow final-answer grammar removes residual formatting failures without disrupting decisions.

- **Data/code:** constrain only the final JSON envelope after reasoning; preserve `[]`, multiple calls, and normal reasoning. Avoid another two-pass tool-name/argument design.
- **Cost:** approximately **1–3 local hours** for a pilot; **$0 API**; half a day to a day of backend work.
- **Pass rule:** S, plus fewer actual parse failures without increased under-calling.
- **Risk/detectability:** low priority. Historical constrained decoding scored **21/200**, and the tool-name automaton **27/200**. If current parsing failures are rare, available headroom is too small.

Those historical systems differ from this narrow proposal, but they are strong reasons to demand a measured syntax bottleneck first. [Historical results](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/PAPER_WRITING_PROCESS.md:27)

### E11. Observable-state supervision

**Hypothesis:** explicit supervision of remembered facts and pending obligations improves decisions beyond action imitation.

- **Data/code:** derive state targets from information actually exposed in tool results: known IDs, current directory, completed requests, unresolved slots. Add a compact state-summary training task or auxiliary loss while retaining the existing action format.
- **Cost:** **3–8 local hours** for a small build/train pilot; **$0 API** with deterministic targets; **1–2 days engineering**.
- **Pass rule:** improve held-out observable-state accuracy and satisfy S against an equal-token action-only control.
- **Risk/detectability:** backend object fields can leak facts the model never observed. Accurate summaries may still fail to improve action selection. State accuracy provides a more sensitive mechanism measure, but completion remains primary.

This complements the thesis directly, provided the fine-tuned model receives the same harness support as the base.

## 6. Recommended top three and execution order

### Ranked by potential practical value

1. **E5: Same-state DPO, with identical-chosen SFT as the essential control.**  
   It directly tests whether negative evidence helps beyond the successful-action imitation already tried. Pair quality is the central investment.

2. **E7: Teacher-interleaved rollouts.**  
   This addresses access to useful states and corrective behaviors. It is especially relevant if preference construction finds that one correction rarely enables a successful student continuation.

3. **E2: Answer-weighted SFT.**  
   Cheap and scientifically clean. Its expected upside is uncertain because gold-reasoning-conditioned answer loss is already tiny.

**Execution order differs from that ranking:** precision diagnostics → E2 → a shared branch-data pilot → chosen-only SFT → DPO → teacher interleaving if needed.

RFT belongs in that sequence as a substantive baseline. GRPO should wait until the rollout pilot demonstrates enough reward variation and simpler learning methods show some signal.

### Cheap diagnostics first

**Existing-artifact analysis, no new model calls:**

- Reconcile strict rows versus unique states.
- Tabulate usable preference pairs by source, task, and family.
- Separate `act_ok`, whole-teacher-suffix success, and correction-plus-student success.
- Audit answer/reasoning/termination token masks. `diag_mlx.py` has a small count inconsistency: its answer-token count includes the separator while its answer NLL excludes it.
- Freeze model, tokenizer, template, adapter, conversion, and harness hashes. Some scripts hardcode the other checkout or select a cached snapshot implicitly.
- Reconstruct the declared Round 4 error-free-in-both-bases subset.

**Zero-API-cost diagnostics requiring approved local generation:**

- E1’s same-backend Q4/F16 base-and-adapter probe.
- Evaluate answers under recorded versus freshly generated reasoning on training-side prompts.
- A bounded **40-task, k=4** training-pool rollout pilot, recording finish reasons and true wall time.
- Counterfactual teacher-correction branches followed by the student.

The existing records support about **$11.8 remaining by logged usage**, rather than an account-balance check. Reserve fresh evaluation and contingency money before expanding training data.

## 7. Protocol choices that can disadvantage fine-tuning

### The screen has limited sensitivity and asymmetric selection

Comparing one candidate run against the **maximum of two base runs** favors the base. Repeating only positive candidates further makes observed deficits and gains asymmetrically characterized.

That is defensible for conservative resource allocation, but it cannot support a strong claim that all candidates truly hurt. Future finalist comparisons should use symmetric repetitions and paired task outcomes.

### The harness comparison is not yet fully answered

The distillation evaluator defaults to `iffc-ooda-card`. The paper’s strongest configuration also includes state-summary behavior and the shorter tool list. These are distinct conditions.

A candidate can fail with the current card harness yet complement the stronger harness differently. Conversely, beating a weaker configuration does not establish improvement over the best harnessed base.

For one finalist, use a small predeclared factorial comparison:

| | Current card harness | Strongest frozen harness |
|---|---|---|
| Base | Contemporary reference | Contemporary reference |
| Fine-tune | Matched condition | Matched condition |

Training prompts must match the intended harness. Historical scores from different banks cannot fill these cells.

### Fine-tuning has stricter data restrictions than harness engineering

The harness has benefited from benchmark-informed iteration; fine-tuning is appropriately excluded from benchmark tasks. This makes the thesis’s engineering comparison meaningful, but weakens universal conclusions about “harness versus learning.”

Keep the disclosed tuning leakage. Do not repair fairness by training on evaluation data.

### Native filtering may be overly restrictive for capability transfer

Requiring unhinted reproduction protects format and provenance, but selects against difficult teacher corrections. A separately registered action-supervision experiment could test that restriction. It must not be presented as continuation of the stopped Round 4 arm.

Similarly, **100 unique recovery rows is a feasibility guardrail**, not an established minimum needed for learning. Forty-two states do not establish that recovery training cannot work.

### Screening banks are small and selected

Teacher-authored, teacher-verifiable synthetic banks measure a particular distribution. Reusing the same author, APIs, and fixtures can create common artifacts even after text deduplication.

Repeated evaluations improve knowledge of serving variability; they do not provide new task diversity. Dev103 is now a tuning bank. F has already supplied aggregate evidence for further planning, so a fresh final bank is preferable for adaptive new methods.

## 8. Integrity and reporting issues to preserve

1. **Correct recovery uniqueness:** 45 strict rows, **42 unique task-step pairs**. [Runlog correction](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:98)

2. **Do not generalize collection failure into impossibility.** The yield applies to this collection and sampling budget.

3. **Keep the recovery-control confound explicit:** student-equivalent targets differ sharply between recovery and control candidates; the retention comparison does not isolate an error-state effect.

4. **Restore the declared subset analysis:** the plan specifies episodes error-free in both base runs, while the log reports a B1-defined split. [Plan](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND4_PLAN.md:70)

5. **Preserve contamination disclosure:** five rows from two training tasks share fixture 8-grams with three dev tasks. Later collection records exclusions; the earlier result still carries that caveat. [Provenance findings](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_NEXT_METHODS.md:86)

6. **Clean up duplicated and superseded narrative:** Round 3 findings repeats its Round 3b section; early runlog claims about backend parity and “mostly format” were later corrected.

7. **Avoid historical-base shortcuts:** the gate script freezes older base references. New precision, harness, or serving conditions need contemporary matched bases. [Gate implementation](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/distill_gate_eval.py:52)

8. **Keep oracle and observable verification separate.** Gold checker access is legitimate training supervision. Evaluation-time gold selection changes the experiment into oracle-assisted search.

9. **Budget engineering as well as GPU time.** Preference learning and GRPO are not validated turnkey paths in this repo. Their integration cost can exceed the training cost.

## 9. External method claims: verified interpretation

- **Fission-GRPO:** verified ACL 2026 paper. It reports Qwen3-8B BFCL v4 Multi-Turn accuracy increasing **42.75→46.75%**, using on-policy recovery resampling and a learned Error Simulator. This supports targeted corrective experience; it does not establish feasibility or expected gains for Gemma on one Mac. Its headline covers a broader Multi-Turn evaluation than this thesis’s base split. [Paper](https://aclanthology.org/2026.acl-long.1880/)

- **Guided-OPD:** verified. It mixes teacher and student turns and decays teacher intervention. Its reported environments differ from BFCL. The relevant transferable idea is controlled state visitation. [Paper](https://arxiv.org/abs/2606.15912)

- **PivotOPD:** verified. It combines prevention and recovery supervision around pivotal mistakes, including KL-based objectives. Text-only teacher actions permit an inspired intervention experiment, not a faithful reproduction. [Paper](https://arxiv.org/abs/2609.40285)

- **CacheRL:** the “SFT does most of the work” reading is supported within its own setup: its ablation reports **0.756 SFT versus 0.779 full-system validation reward**. That reward combines judge and process signals; it is not BFCL task accuracy, and the evolving configurations complicate causal interpretation. [Paper](https://arxiv.org/html/2606.14179v1)

The strongest next experiment is therefore a controlled comparison of learning signals on trustworthy student-state data. Another broad adapter sweep or an immediate RL implementation would leave the central uncertainty unresolved.