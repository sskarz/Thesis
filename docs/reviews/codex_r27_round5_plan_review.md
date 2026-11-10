**Verdict: REVISE before any Round 5 run.** The overall sequence is sensible, but the current plan has unresolved provenance failures, an ambiguous preference-label rule, and controls that do not yet support a one-variable interpretation.

I read the requested documents, current code/configs, and both historical DPO files at `34e0d5b`. No models, training, evaluations, or paid APIs were run; no files were written. F inspection and reporting stayed aggregate-only.

1. **Screening rule — OK in principle; REVISE its exact definition.**

   Combining historical F bases with two new K2 bases is coherent for a descriptive resource screen. Define the composite runs explicitly:

   \[
   B_j=35+K_j,\quad N=81+|K2|,\quad \mu_B=(B_1+B_2)/2.
   \]

   Replace [the screening paragraph](~/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND5_PLAN.md:9) with:

   > Run 1 advances iff \(C_1\ge\lceil\mu_B+0.06N\rceil\) and \(C_1>B_*\). After one restarted repeat, advance iff \((C_1+C_2)/2\ge\mu_B+0.06N\) and each \(C_i>B_*\). \(B_*\) is the higher-count composite base run; ties select B1. No additional repeats.

   This avoids ambiguity about separately rounding the improvement and baseline. With 70 K2 tasks, \(N=151\), and the unrounded improvement is **9.06 tasks**.

   “Paired net > 0” is mathematically just \(C_i-B_*>0\) when both runs contain the identical task set. Keep gains/losses as useful diagnostics, but do not describe net positivity as independent paired evidence.

   Reusing F does **not** put F into training. It does make FK an **adaptively informed screening bank**: F aggregates helped motivate this plan. Report F, K2, and pooled FK aggregates separately; freeze K2 membership before base outcomes; do not top up or change it afterward. Historical F bases also leave a temporal comparison limitation despite matching hashes and argv.

   Required wording corrections:

   - Replace “~±5 task noise” with the observed fact: **F base runs exchanged five gains and five losses while retaining equal totals**. That does not estimate a ±5 uncertainty interval.
   - `K2 < 60` implies `FK < 141`, not `<140`. State the threshold directly.
   - Specify whether an API-error retry replaces the **entire bank run**. I recommend one complete replacement run, preserving the failed attempt; never assemble a result from favorable episode retries.
   - The pair floor applies **after all filtering and the validation split**. Failure stops B, leaving A results intact.

2. **Leakage/provenance — REVISE; task disjointness currently overstates content cleanliness.**

   The exact source chain is:

   | Boundary | Files and implemented protection |
   |---|---|
   | Round 3 sources | `r3/labels_old/labels.jsonl`, `r3/labels_new/labels.jsonl`, sourced from `r3/pool_all.jsonl`; family/similarity separation lives in [distill_split.py](~/Documents/GitHub/Thesis-wt-distill/benchmark/distill_split.py:48). |
   | G sources | `g/labels/labels.jsonl`, sourced from **`g/tasks_F.jsonl`** despite that filename; `g/author_g.sh` supplies prior banks for code-side deduplication, and `g/chainG1.sh` filters against dev, confirm, r3, and F. |
   | F separation | [build_bank_f.py](~/Documents/GitHub/Thesis-wt-distill/sft/build_bank_f.py:30) checks supplied-bank IDs, function signatures, and turn similarity. Authoring separately checks shared 8-grams. |
   | Benchmark protection | [distill_exclusions.py](~/Documents/GitHub/Thesis-wt-distill/benchmark/distill_exclusions.py:112) loads all four BFCL multi-turn splits. Its explicit mid20/fresh-20 groups are subsets of the protected 200. |
   | Final row protection | [check_provenance.py](~/Documents/GitHub/Thesis-wt-distill/sft/check_provenance.py:59) requires explicitly supplied held-out banks and task sources; it is not automatic bank discovery. |

   My read-only aggregate checks found:

   - After the two declared task exclusions and prompt/target deduplication: **1,723 kept states from 224 tasks**, all found in the r3/G source files.
   - Those source tasks have **zero ID, function-signature, and source-user-turn 8-gram overlaps** with F, dev_all, and confirm.
   - Using effect-equivalence to define corrections gives **166 recovery + 296 non-error correction states = 462 eligible states**. The 600-state cap is currently inactive.
   - **Two eligible states from one source task hit F’s protected 8-gram guard.**
   - **One eligible recovery state hits a BFCL multi-turn 8-gram in historical assistant content.**

   These are mechanical guard hits, not proof of semantic task leakage. Nevertheless, they invalidate an unconditional cleanliness claim. Source-task checks do not cover everything copied into rollout histories.

   There are also two checker gaps:

   - BFCL `scan_rows()` compares benchmark text against **user content**, so the assistant-content hit above escapes it.
   - The content checks omit separate **`reasoning_content`** fields. Preference rows also need explicit expansion into **both** prompt+chosen and prompt+rejected message rows.

   **Exact edit:**

   > Before selecting S or sampling, audit eligible source histories against every protected bank. Drop affected source tasks under the registered strict rule, then recompute counts. Final provenance covers both preference branches, all serialized content/reasoning actually rendered, and train plus validation rows. Reports expose only aggregate protected-bank results.

   K2 cannot be certified before construction. Pin its path/hash and require the same checks before use. For synthetic banks supply `dev_all`, confirm, F, and K2 explicitly; BFCL protection must retain all-split text checks. Do not assume raw BFCL files satisfy the checker’s requirement for joined gold.

   Finally, **A2 contradicts the universal strict gate**. The saved AB provenance failure and its five flagged rows are current: I verified report input/code hashes. The F-versus-AB report is also current and passes, but it covers that narrower comparison.

   To preserve strict provenance and a clean one-variable experiment, change A2 to **remove the two flagged tasks from both arms and train a fresh ordinary-SFT control alongside weighted SFT**. The existing adapter can remain a historical reference; it cannot be the identical-data clean control.

3. **Pair design, weighting, and controls — REVISE.**

   Same-state student-versus-student samples are a useful design: they reduce teacher/student format confounding. They still select states where the student can produce **both** target-matching and nonmatching responses, so conclusions concern reachable, ambiguous states.

   The matching specification currently combines incompatible code:

   - [distill_rationalize.py](~/Documents/GitHub/Thesis-wt-distill/benchmark/distill_rationalize.py:54) requires **exact function-string equality**, nonempty reasoning, and a 3,500-character reasoning limit.
   - [same_effect()](~/Documents/GitHub/Thesis-wt-distill/benchmark/distill_onpolicy.py:122) compares **stringified execution returns**, not final backend states.
   - [check_label()](~/Documents/GitHub/Thesis-wt-distill/benchmark/distill_onpolicy.py:97) makes `act_ok` an execution/grounding check, not a continuation-success label.

   Equal returns can conceal different mutations; unequal returns can represent acceptable alternatives. A nonmatching action can be a useful lookup, a valid alternative order, or another successful plan. Even `[]` is context-dependent.

   **Exact edit:** define one matcher before collection. Canonicalize argument-key order while preserving call order. Separate:

   > exact target match; verified equivalent alternative; verified inferior action; unresolved alternative.

   Do not automatically train unresolved alternatives as rejected. If keeping a broader teacher-match experiment, name it **teacher-action preference learning**, not outcome-verified DPO, and limit its claims accordingly. A claim about better recovery requires a preregistered training-side continuation audit of chosen and rejected branches.

   Required diagnostics, fixed before pair outcomes:

   - Source/API, recovery status, teacher label kind, action/call count, and original student-target agreement.
   - Exact/equivalent matching and rejection reason: execution error, premature stop, missing lookup/action, harmless alternative, unresolved.
   - Prompt, reasoning, answer, and total lengths; chosen-minus-rejected length; sampling temperature/index; finish reason and truncation.
   - Unique tasks, states, chosen responses, and pair multiplicity; distributions before and after every filter.
   - Preference margin, chosen/rejected log-ratio changes, and preference accuracy by those strata.

   Full-response sequence-sum DPO can reward reasoning/style and length shortcuts. Reference subtraction does **not** eliminate these confounds. Do not silently shorten responses or normalize DPO scores mid-experiment.

   Also freeze deterministic pair selection, conflicting-label handling for identical prompts, and the exact task-disjoint validation rule. A repeated chosen response must appear with the **same multiplicity** in the SFT control. Exclude incomplete generations symmetrically. State whether parsing follows the permissive serving parser or Round 4’s stricter JSON filter.

   **90/10 weighting:** coherent as an objective ablation, but “90% loss mass” is underspecified. Token weights cannot guarantee 90% of observed numerical loss. Prefer:

   \[
   L=0.9\,\operatorname{mean}(CE_{\text{answer+termination}})
     +0.1\,\operatorname{mean}(CE_{\text{reasoning}}).
   \]

   Specify separator ownership and absent-segment handling. Alternatively register a training-only corpus-count weight ratio; acknowledge that it does not produce 90/10 weighting within every row. Do not derive weights from validation outcomes.

   **Chosen-only control:** currently **not one-variable** because its LR is `2e-5` versus DPO’s `1e-5`. Use `1e-5` for both if that interpretation matters, with identical chosen exposure, optimizer settings, order, and updates. Objective normalization still differs, so avoid claiming equal optimization dose.

   **B5:** conditionally skipping control evaluation after DPO fails is acceptable resource triage. It leaves the preference contribution unmeasured. For a passing DPO arm, evaluate the control **twice under matched conditions** and define the paired comparisons explicitly. Two selected DPO runs versus one control run is weak support for preference attribution.

   A2 also needs an ordinary-control **K2 evaluation**; only its F result already exists. Replace “must … (reported, not a gate)” with an unambiguous descriptive comparison or an actual gate.

4. **Technical feasibility and training dose — feasible direction; REVISE implementation prerequisites.**

   **Precision conversion/serving: OK conditionally.** The complete local HF snapshot exists at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, including a roughly 10.25-GB safetensors file. Local conversion code registers Gemma4 and supports `f16`/`bf16`. The LoRA loader checks architecture/tensor compatibility rather than requiring a Q4 base, so the existing FP16 adapter is not inherently Q4-specific.

   Required edits:

   - Choose **BF16 or F16 now**, with a declared fallback; do not select precision after results.
   - Pin snapshot, conversion source, server binary, tokenizer/template, and runtime LoRA scale. [The adapter converter](~/Documents/GitHub/Thesis-wt-distill/sft/mlx_adapter_to_gguf_lora.py:21) currently hardcodes the other checkout and takes the first snapshot glob.
   - Verify effective prompt token identity. Sending already `<|think|>`-prefixed training messages to a server that inserts the marker risks duplication.
   - Freeze the actual 35+35 prompt identities/hashes. [probe_action_change.py](~/Documents/GitHub/Thesis-wt-distill/sft/probe_action_change.py:31) otherwise selects first-N rows and compares files with unchecked `zip`.
   - Log HTTP failures and finish reasons. Its fixed 1,500-token generation cap can confound precision comparisons.

   Four additional held-out agreements is an **11.4-point difference-in-differences on 35 prompts**, clustered within few tasks. Fine as a diagnostic trigger; failure should read “no detected precision rescue on this probe.”

   **MLX weighting: implementation needed.** `train_mlx.py` only wraps ordinary `mlx_lm.lora`; YAML does not supply segment weights. The installed trainer exposes a custom `loss` callable, consistent with the [official trainer source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py). Require shifted-token mask tests and an all-weights-equal reproduction check before training.

   **DPO memory: not established at 6,000 tokens.** The recorded machine is 48 GB. A single full `6000 × 262144` logits tensor occupies approximately **3.15 GB in BF16 or 6.29 GB in FP32**, before gradients and activations. Two retained branch graphs matter; accumulation four does not make them cheap.

   Precompute frozen-reference scores with one base, unload serving, use checkpointing, and require a longest-pair forward/backward memory preflight. If necessary, compute branch gradients sequentially using the shared DPO coefficient. An OOM stops the arm; no silent truncation or changed window.

   The historical trainer also needs substantive adaptation:

   - `train_dpo_mlx.py` permits left-truncated prompts; `run_dpo_full_mlx.py` rejects overlength branches. Keep the latter no-truncation behavior.
   - Cast selected continuation logits to **FP32 before log-softmax**, not merely selected log-probabilities before summation.
   - At initialization require policy/reference agreement, loss near `ln(2)`, correct causal masks, and an actual numerical gradient-direction check.
   - Replace old task-specific imports, paths, scale, module selection, stopping, and update logic explicitly.

   **Dose: defensible pilot, neither predictably inert nor safe.** The actual adapter is **q-only**, 524,288 trainable parameters, because layers 19–34 share KV. Replace “q/v” throughout with that fact.

   Prior identical-family SFT at 80 updates changed behavior: training agreement increased **9→17/35**, held-out **25→24/35**, and F completion was **30 versus 35**. Narrow adaptation can therefore materially perturb behavior.

   For DPO, with \(z=\beta\Delta\):

   - At initialization, \(|\partial L/\partial\Delta|=0.05\).
   - At beta 0.1, a **10-nat** relative-margin change gives loss about **0.313**; 30 nats gives **0.049**.
   - Over a 300-token continuation, 10 nats is only about **0.033 nat/token** if concentrated on one branch.

   Thus long-response sums can move quickly despite small beta. Adam also prevents treating `beta × LR` as a simple dose predictor. The old run used scale **32**, rank 16, and stopped at **250 updates**; the proposal uses scale 2 and 80 accumulated updates. Its tenfold larger LR does not make those doses directly comparable. Keep the fixed pilot parameters, but monitor margins, segment log-ratios, adapter deltas, and finite gradients; do not tune them on FK.

5. **Standing constraints, budget, and scope — mostly OK; REVISE enforcement.**

   Model/task/hardware scope, text-level teacher, no hinted rescue, protected-bank separation, and one local GPU job at a time are consistent.

   Remaining edits:

   - Resolve the strict-provenance/A2 conflict and current source-history hits **before generation**.
   - Freeze B’s selection and diagnostic rules before E2 outcomes can influence them.
   - Repair `stageA_diag.py`: it currently reports a **B1-defined** error subset, not the declared error-free-in-both-bases subset.
   - Add local wall-time estimates/caps, generation limits, and memory preflights. A dollar cap alone does not bound thousands of local samples and DPO work.
   - Enforce the $2.50 total across calls, retries, and concurrency; “solve ~$0.3” needs an explicit command cap.
   - Replace “after telling Sans” with **“after explicit Sans approval and an estimate”** for protected-bank runs.

**Required plan edits, condensed:**

- Formalize composite bases, rounding, ties, retries, and minimum-bank arithmetic.
- Identify FK as adaptive screening; freeze membership and report K2 separately.
- Filter source-history guard hits; audit both branches and rendered reasoning.
- Use clean, identically filtered E2/control data.
- Define matching, inferior versus unresolved alternatives, parsing, and pair selection.
- Specify the 90/10 equation; match control LR/exposure and repeat counts.
- Correct q-only architecture and add numerical, memory, and template preflights.
- Restore subset logic and explicit budget/time/approval boundaries.