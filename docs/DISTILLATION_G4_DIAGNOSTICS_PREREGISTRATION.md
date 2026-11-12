# G4 diagnostics on the BFCL 200: pre-registration (EXPLORATORY; written and committed before any diagnostic run)

Written 2026-11-12 (Mac clock about 08:00 ET), before D1, D2 and D3 have run. These runs are **exploratory and outside the G4 registration** (`docs/DISTILLATION_G4_PREREGISTRATION_DRAFT.md`, registered result "arm worse"); every report of them says so.

## 1. Approval
Ledger A-011 (the advisor's approvals ledger), Sans 2026-11-12: "2. Yes", in answer to "Run the three diagnostics on the 200 (about 4.5 hours, $0, exploratory)? Yes/no." Conditions copied from the entry: exploratory, outside the G4 registration; a short pre-registration (this file) committed before the first run, containing the reading rules of section 2 of the advisor's plan file unchanged; render capture for each server mode before its run; errored tasks count as fails, a run is void above 10 errored tasks; one run each, no repeats without a new approval; no other model or setting on the 200; the G4 runs U1 (88) and U2 (89) are the untuned thinking-on reference and the session difference is stated as a limit; results backed up after the last run. Expiry Nov 25. Cost $0.
A-010 (same day, "1. Yes") stopped the combined retrain (G3c, attempt `20261112_030947_run_c_A`) at micro-batch 1,267 of 8,166 before these runs; the GPU is free.

## 2. The three runs (single runs, in this order)
| Run | Model | Thinking | Why |
|---|---|---|---|
| D1 | untuned GGUF, no adapter | OFF | how much of the G4 loss is answer style alone |
| D3 | arm A after one pass: `train/adapters_pass1` of attempt `20261110_062035_run_A.lBKtpA`, converted with the G3 converter and audited, identity-checked on the server | OFF | did the second pass make the benchmark worse |
| D2 | arm B, the screened GGUF `G3B.gguf` (sha256 `2c0bb899...1e80`, equal to the hash recorded at both of its screens) | ON | does a trained model that keeps its reasoning avoid the loss |

## 3. Setup (same as G4)
Compact-list full stack (`--fc-mode iffc-state-ooda-slim`), `benchmark.run_all --model gemma4-e2b --only-multi-turn --categories multi_turn_base`, harness worktree at commit `dfdbc2b` (clean), llama.cpp `b10894-d344123fe`, bar Q4_K_M GGUF (prefix `923c4c86177d2ee1`), server flags `--jinja -fa on -c 65536 -np 4 -cb -ub 1024 --temp 0.0 -ngl 99`, thinking OFF through `--chat-template-kwargs {"enable_thinking":false}` (same switch as the arm A server in G4), LoRA applied at scale 1.0 where there is one. Scorer: `bfcl_eval` 2026.3.23, checker hash as pinned in the G4 document; stored Runs 84 and 85 re-score to 88 and 72 in the preflight; each run has its own results folder and database and is scored in its own process. Render capture first for each server mode (D1 and D3 thinking off: 200 of 200 equal to the local render, 0 markers; D2 thinking on: 200 of 200, 200 markers, and equal to the stored 2026-11-09 reference). Errored, unparsed or overflow tasks count as fails; their count is reported per run; a run with more than 10 is void, the chain stops and reports (no automatic rerun). The chain script is `sft/g4d_chain.sh` (pieces of `sft/g4_chain.sh` and `sft/g4_lib.sh`); the readings below are applied mechanically by `sft/g4d_readings.py`, which has boundary tests. Time estimate (advisor, from the G4 trained-run timings): D1 and D3 about 50 min each, D2 about 2.5 h, about 4.5 h plus server restarts [ANALYSIS, not measured for these models]. Chain cap 8 h, per-run cap 4.5 h. No spend.

## 4. Reference and its limit
The reference is U1 = 88 and U2 = 89 (untuned, thinking ON, G4, 2026-11-11) and T1 = T2 = 69 (arm A, thinking OFF, G4). The diagnostics run in a different session, on a different day, with a freshly started server; with 0 and 1 flips between repeats in G4 this should matter little, but it is a stated limit.

## 5. Reading rules (copied unchanged from `reviews/2026-11-12-data-plan-v2.md` section 2, "Diagnostics on the 200")
Reference: untrained, thinking on, 88 and 89 of 200 (G4). Version A, thinking off, 69 and 69. One run each, so differences of a few tasks mean nothing; the bands below are wide on purpose and were written before any of the three runs.

| Run | Question | Reading |
|---|---|---|
| D1: untrained, thinking off | How much of the 19-task loss is the answer style alone? | 75 or below: answering without thinking explains most of the loss. 82 or above: the style costs little here and the training itself did the damage. Between: both contribute. |
| D3: version A after one pass, thinking off | Did the second pass (the one that took the screen from 37 to 64 on one set) make the benchmark worse? | At least 6 above 69: the second pass over-fitted; train shorter and pick the stopping point with the new screen. Within 5 of 69: over-fitting in the second pass is not the issue. |
| D2: version B (trained, thinking on) | Does a trained model that keeps its reasoning avoid the loss, with the same coverage gap in its data? | 82 or above: reasoning-first training does no harm even with the gap; strong case for this plan in that format. 76 to 81: partial. 75 or below: the data gap hurts in either format; this plan must fix coverage to have any chance, and the odds in section 9 drop. |

Use in this plan: the format (section 6) is reasoning-first unless D2 is 75 or below **and** D1 is 82 or above, in which case I come back to you before anything is generated. The diagnostics choose nothing else. They are post hoc runs on the test set and are reported as exploratory.

## 6. Limits stated with any result
The reading text for D2 at 75 or below ("the data gap hurts in either format") and for D1 at 82 or above ("the training itself did the damage") is the advisor's registered wording; no diagnostic changes the data coverage or isolates one training factor, so each such reading is a **hypothesis that needs a coverage or training control**, not an established cause, and is reported that way.
Single runs; 200 tasks that were also the tuning set of the harness (tuning leakage); one trained model per format and one seed; post hoc and exploratory, on the test set; D2 and D3 and D1 each differ from the G4 runs in one setting but the session also differs; the bands above were written before any of the three runs and choose nothing beyond what the last paragraph of section 5 says. No BFCL outcome changes a model, a retrain or data except through that paragraph, and nothing is generated before the advisor comes back to Sans.
