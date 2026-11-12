**My recommendation: stop fine-tuning for this thesis, write up the negative evidence, and retain the existing harness gains as the final result.** Return to harness work only for a concrete, testable mechanism—not because another improvement is needed to justify stopping.

The evidence supports that decision: repeated SFT variants failed to improve screening performance; Round 5’s valid S0 and Sw results were 59/155 and 64/155 versus base 68/155 and 67/155. Answer weighting did not rescue performance, and the precision probe found no detected precision rescue. Round 4’s retry nudge changed behaviour without improving total success. Another training run now has diminishing information value.

| Option | Expected information value versus cost/time |
|---|---|
| **Stop and synthesize** | Highest thesis value. No additional compute/API cost; makes the tested failures, controls, and stopping decisions reviewable. |
| Exploratory DPO/chosen-only SFT on 120 pairs | Low–moderate value: tests an objective distinction still unanswered here. Requires final provenance certification first. Registered caps allow up to 6 h DPO and 3.5 h SFT, plus evaluation. Any result is explicitly post-outcome exploration. |
| More sampling under a new protocol | Low–moderate value: resolves data availability before testing efficacy. About 3.5 h sampling, with no guarantee of reaching the audited floor, followed by training/evaluation. The 150-pair floor is a protocol threshold, not a demonstrated efficacy threshold. |
| Rejection-sampling SFT, GRPO-style training, larger LoRA targets | Low immediate value relative to implementation and validation effort. Verified student-response SFT was already represented in Round 3; exact teacher-action matches do not establish successful continuations. Broader adapters or a new reward objective introduce further unresolved variables. |

The remaining API allowance is not a reason to spend it: local compute, evaluation, and interpretability are the binding costs.

**A meaningful positive:** retain the registered FK screen—first run at least **77/155**, then a restarted repeat averaging at least **76.8**, with each run above **68**. A single gain of roughly five tasks is insufficient. The observed flips are descriptive variability, not a calibrated ±5 confidence interval. Passing FK would justify fresh protected-bank confirmation, not establish a thesis result; DPO attribution additionally requires the chosen-only control.

**Write regardless:** the hypotheses, registered rules, negative SFT screens, behaviour-versus-success distinction, and the data-yield stop. Keep training history in Appendix B under the current paper rules; summarize its implications in the discussion.

Flag as overclaiming: “fine-tuning cannot help,” “DPO failed” (untrained), “recovery training failed” (untrained), or “harness optimization universally wins.” FK is adaptively informed screening; existing harness claims retain their selection and multiple-comparison caveats.