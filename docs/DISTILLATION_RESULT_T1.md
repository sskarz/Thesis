# Distillation null-scale result (distill-t1): gate FAILED

2026-11-06 night run on the Mac, executed while Sans slept, per his instruction to run the chain,
verify with Codex, document everything and stop only if needed. Full action log:
`docs/DISTILLATION_RUNLOG.md`. Per-stage Codex reviews: `docs/reviews/`.

## What was run
* Data: `sft/data/distill/mlx-t3-null154` (154 teacher-verified train rows from 25 tasks, 31 valid rows).
  Teacher `deepseek/deepseek-v4.1-flash-1017` via OpenRouter/DeepInfra, tasks authored over the real BFCL
  backends, verified by the stock BFCL checker, leak-checked (0 id / exact / 8-gram / >=0.85 hits), three
  Codex data reviews (docs/reviews/).
* Training: base `google/gemma-4-e2b-it` converted to bf16 MLX; LoRA rank 8, scale 2.0, 16 layers, q/v proj,
  lr 5e-6, batch 1 x accumulation 4, 156 micro-batches = 39 optimizer updates, max_seq 8192, mask_prompt,
  seed 61. Config identical to the Run 73 null test except adapter path (Codex-verified).
  Validation loss 1.139 (iter 1) -> 1.000 (iter 156); reported only, never used to screen.
  Final adapter sha256 `542655a417fab707ec9734d7b3d77ab78588f09c37f88546c20937d9b662ce57`.
* Fuse: mlx_lm fuse -> convert_hf_to_gguf -> llama-quantize Q4_K_M. GGUF
  `Thesis/sft/fused/gemma-4-e2b-it-distill-t1-Q4_K_M.gguf`, sha256
  `a5b1bf1e0b9dcd854579532533b3d9ace77a8249d8d5687d19a39dc1bd8f0c77`.
* Serve: R69 argv (`--jinja -fa on -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024`), llama.cpp
  b10894-d344123fe (the build recorded for Run 69). Served-model identity verified (/props + sha).
* Gate: `sft/distill_gate_eval.py --set mid20`, fc-mode `iffc-ooda-card`, one run.

## Result (source: `benchmark.db`, run 86, model_key `gemma4-e2b-distill-t1`, category multi_turn_base)
**mid20: 10/20.** Bar (Run 69, base + card): 17/20. **Gate not passed.**
* Passes: 45, 12, 8, 113, 57, 138, 100, 60, 83, 115.
* Regressions vs Run 69 (passed there, fail here): 23, 7, 14, 91, 150, 146, 112, 186 -> **8** (rule allows <=1).
* New passes vs Run 69: 115.
* The 10 failures by checker message (my own message matching, not the paper's labels): empty turn 4
  (112 t2, 146 t2, 150 t1, 186 t0), execution results missing 3 (14 t0, 195 t0, 7 t1), state mismatch 3
  (152 TravelAPI, 23 GorillaFileSystem, 91 MessageAPI).
* Same-harness references on the same 20 tasks (DB): Run 69 base 3 errors; Run 73 null test (self rows,
  same pipeline) 5 errors (15/20); Run 74 12 errors (8/20); Run 75 6 errors (14/20); this run 10 errors (10/20).

Not run, by rule: the fresh-20 bank. A failed mid20 fails the conjunctive pass rule; fail-fast means no
variants and no tuning (handoff section 6). The fresh-bank eval remains available if Sans wants it as a
no-harm measurement.

## How to read it (hedged)
* A gate result is a tripwire on 20 tasks, not an accuracy; nothing here supports a claim about the 200.
* This candidate failed this screen: the teacher-verified, in-world 154-row null-scale data did not clear
  the bar and scored below the self-distillation null test (15/20) on the same gate. The conversion
  confound, repeated exposure of these 20 tasks, and single-run variability mean this is not a broader
  causal conclusion about teacher distillation.
* Conversion confound (Codex): the bar's GGUF is a different conversion (bartowski, imatrix) from ours; the
  like-for-like reference is Run 73 (same pipeline). A base-through-our-pipeline control would isolate this
  and was NOT run (extra benchmark run, needs approval).
* Hypotheses, none tested: (1) the teacher's clean-JSON history format differs from the rows Gemma sees at
  test time (its own reasoning text is persisted in history); (2) teacher step style or task style (explicit,
  user-dictated text) does not transfer; (3) 154 rows across 8 backends is thin and Trading is 59% of rows;
  (4) stop rows at 22% still teach early stopping (4 of 10 failures are empty turns).
* Run variance: Run 69's own base score is one run; no repeat of this candidate was done (no variants rule).

## Teacher reference (separate from training)
Teacher on all 200 tasks, greedy, same loop: 155/200 (log under `sft/data/distill/teacher_ceiling_200_EVAL_ONLY/`,
not in benchmark.db, DeepInfra fp8). Gemma full-stack paper numbers are 61/200 (JSON text) and 77/200.

## Codex verification of this result (round 5: VERIFIED WITH CORRECTIONS)
Codex re-read the DB read-only: run 86 = 10/20 with the exact mid20 ids, per-task results and the 10 error
messages match this document; runs 1-85 are byte-identical to the pre-run backup (existing rows unchanged);
no infrastructure problem (155 API calls, 20 timing rows, no flags, no embedded ERROR results, server log
156 completed requests, truncated=0, max slot use 12,568/16,384 tokens, no overflow). Corrections applied:
* Serving provenance is only partly stored: the server log and GGUF sha confirm the candidate and path; the
  full launch argv and the `/props` answer (4 slots, build b10894-d344123fe) were checked live and recorded in
  the run log text, but no `/props` file was saved and run 86's `server_params` / `llama_cpp_version` DB fields
  are NULL. Treat "R69-equivalent runtime" as verified from the live check, not from a stored artifact.
* Times in the run log are Eastern (run 86 started 04:10 UTC = 23:10 Eastern, Nov 6); earlier "Los Angeles"
  wording in the run-log header is wrong.
* `truncated=0` does not prove every termination reason; no additional run is needed to report the failure.
