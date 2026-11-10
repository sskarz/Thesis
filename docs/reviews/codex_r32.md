**Sw is a verified negative screen. A2 cannot yet be called fully complete under v2.3: S0’s K2 attempt failed integrity, and omitting its replacement deviated from the plan.** No model runs or file writes; F/K2 content remained aggregate-only.

- **VERIFIED — Training matched except for the objective.** Saved adapter configs differ only in output/config paths. Both used the same ordered clean data, seed 61, Adam lr 2e-5, batch 1, 320 microbatches/80 updates, accumulation 4, max sequence 9500, prompt masking and checkpointing. Both artifacts contain 524,288 parameters: q_proj only, layers 19–34, rank 8, scale 2, dropout 0. Clean data exactly matches the original rows with the two declared tasks removed: 315 train/35 validation.

- **VERIFIED — Sw’s registered loss was applied.** [The wrapper](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/train_mlx_weighted.py:31) implements the 0.9 answer/0.1 reasoning segment means, separator assigned to answer, absent-segment renormalization, and stock shifted supervision mask. Its monkeypatch reaches both gradient computation and validation in the installed trainer. Saved boundary/equivalence tests passed before training. Validation changed **0.267→0.195 for S0**, versus **1.625→0.558 for Sw**; training losses also differ substantially while reported token counts match throughout. **No sign that weighting was bypassed.** Cross-objective loss comparisons remain inappropriate. The shared extra supervised pad target is already disclosed in the runlog.

- **VERIFIED — Conversion and serving are consistent with the intended arms.** Each conversion used its own adapter and produced 32 F16 q-projection tensors, with training scale encoded correctly. Eval scripts select the corresponding GGUF; successful serving guards checked its basename and runtime scale 1.0. Both used bar Q4_K_M, build `b10894-d344123fe`, four workers, greedy sample 0 and `iffc-ooda-card`. Current bar SHA matches `923c4c86…`; F matches `7a45cd08…` and K2 matches `e2ac40e4…`. Candidate/base verdict membership is identical.

- **VERIFIED — Aggregate recount and threshold.**

  | Arm | F passes /81 | K2 passes /74 | Composite | API-error episodes |
  |---|---:|---:|---:|---:|
  | B1 | 35 | 33 | 68 | 0 |
  | B2 | 35 | 32 | 67 | 0 |
  | S0 | 29 | 30 | 59* | 1 |
  | Sw | 33 | 31 | 64 | 0 |

  Every bank has the expected row count and unique task/episode IDs. S0’s single API-error episode is flagged `context_overflow:turn_1` and did not pass. Threshold: `ceil(67.5 + 0.06×155) = 77`, additionally `>68`. Sw fails; **no Sw repeat is required**.

- **BUG — The S0 replacement was required.** [Plan line 11](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_ROUND5_PLAN.md:11) explicitly requires one complete replacement of an error-containing bank, preserving the failed attempt. The ≤60 calculation holds only with every other outcome fixed; a complete replacement can change other outcomes. It therefore does **not** justify waiving replacement.

  **Exact edits:** Replace [runlog line 133](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:133) with:
  > “S0’s observed composite is 59. Correcting only the flagged episode could raise it to at most 60, but the registered integrity rule requires a complete K2 replacement. No replacement was performed; this is a protocol deviation, and S0’s registered disposition remains incomplete.”

  Mark its table composite **“59, invalid attempt”**. Remove “S0 result is decided regardless” from [the continuation-script comment](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/data/distill/r5/chainA2eval_Sw.sh:2). Completion requires the registered replacement, or an explicitly acknowledged post-outcome exception—not a claim of protocol compliance.

- **CONCERN — Tighten the conclusion and uncertainty wording.** Replace [runlog lines 142–143](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/docs/DISTILLATION_RUNLOG.md:142) with:
  > “Observed totals are below both base composites. Sw exceeds the invalid S0 attempt by 5 passes; this is descriptive and includes S0’s flagged failure. Sw did not meet the registered screen, so no efficacy repeat is required. S0’s integrity replacement remains outstanding. FK variability is not estimated by these runs.”

  “Run-1 noise on FK is several tasks” overstates the evidence. The two base composites differ by one pass while exchanging 6 gains/7 losses; that is not a noise interval.

- **CONCERN — Finish provenance capture.** Both eval scripts assert K2’s hash but omit F’s runtime hash assertion. Add the same check for F using `7a45cd08b7b2226ebad135f1e7e9aa8b4fdd098801742ddd673df08262f13709`. Extend the manifest with A2 configs, clean-data hashes, adapter/GGUF hashes and chain scripts. [evallib](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/data/distill/r5/evallib.sh:10) should persist `/lora-adapters` and validate exact JSON path/scale; its current substring checks provide weaker evidence. **These are audit gaps, not evidence of a wrong adapter or bank.**