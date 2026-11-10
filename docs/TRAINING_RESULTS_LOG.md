# Training results log (running document, draft)

One entry per finished step, newest last. Each entry: question, setup, counts with source, registered rule and outcome, limits, review. Numbers are copied from the files named; nothing is recomputed unless marked [ANALYSIS]. Mac paths are relative to `~/Documents/GitHub/` and are gitignored data, not in this repo. Draft; not part of `paper/main.tex` or the CSULA build; nothing pushed.

Scope: Gemma-4-E2B-it (Q4_K_M GGUF, llama.cpp b10894), BFCL v4 `multi_turn_base`, one MacBook M4 Pro 48 GB. The BFCL 200 test tasks are never training data and have not been run in any step below.


## Index: from a result to its file (updated 2026-11-10 17:00 ET Mac clock)

Paths are relative to `~/Documents/GitHub/` on the Mac (gitignored data, not in this repo). `r6g3` = `Thesis-wt-r6/sft/data/distill/r6g3`; `yield` = `Thesis-wt-data/sft/data/distill/yield/attempts`. Worktree HEADs now: r6 `65169a6`, data `dc6c7a9`, g2 `8beee53`, g2b `90725fa`, g2p2 `063b6e1`, slim `dfdbc2b`, compact `8f62b8a`, distill `5187288` (read-only).

| Section | Result | Worktree (branch) @ commit | Attempt directory | Report / evidence file |
|---|---|---|---|---|
| 1 | Stage 1: six arms, base 64/65/68 | `Thesis-wt-r6` (`thesis/distill-r6-diagnostics`) | `Thesis-wt-r6/sft/data/distill/r6/attempts/20261109_121007_stage1.1ihURW`; extra base run `.../20261109_191001_extrabase.zf0OhZ` | `STAGE_REPORT.json`, `chain.log`; `docs/reviews/codex_s1_stage1_result.md`, `codex_x1_extrabase.md` |
| 2 | Factory fix, Part 1 samples | `Thesis-wt-data` (`thesis/distill-data-scale`) @ `a5a784a` | `yield/20261109_144037_part1.E11FEx` | `YIELD_REPORT.json`, `judge/`, `calibration.json`; `docs/reviews/codex_a1_sample_verdicts.md`, `codex_a1b_fresh_sample_verdicts.md`, `subagent_a1b_fresh_sample_verdicts.md` |
| 3 | G1 overfit gate, 11 to 32 of 40 | `Thesis-wt-r6` @ `4294d7c` (review `01aa6ae`) | `r6g1/attempts/20261109_201851_g1.P8Hx7x` | `G1_REPORT.json`, `rowcheck_*.json`; `docs/DISTILLATION_G1_PREREGISTRATION.md`, `docs/reviews/codex_g1_result.md` |
| 4, 4b | Batch 1 authoring, merge, state checks, judge | `Thesis-wt-g2` (`thesis/g2-scale`) @ `058c7dc` code, `8beee53` prereg | `yield/20261109_212657_g2full.AUhok0` | `author/report.json`, `merge_report.json`, `det_flags.json`, `judge/verdicts.jsonl`, `prefilter_report.json`, `judge_failed_keylimit_0334/`; `docs/DATA_YIELD_G2_PREREGISTRATION.md` (sections 10, 11); `docs/reviews/codex_g2_repair_sample_verdicts.md` |
| 4c, 8 | Batch 2 authoring, cap stop, judge | `Thesis-wt-g2b` (`thesis/g2b-scale`) @ `59cdece`, prereg `90725fa` | `yield/20261110_040120_g2full.wjjYRA` | `author/report.json`, `author/f/chunk_6/log/guard_flags.budget_hit_0702.json`, `merge_report.json`, `prefilter_report.json`; `docs/DATA_YIELD_G2B_PREREGISTRATION.md` (sections 5 to 7) |
| 5 | G3 base references 59 / 61 / 54, threshold 70 | `Thesis-wt-r6` @ `571c9e7` | `r6g3/attempts/20261110_033718_base.XtrzLh` | `G3_BASE_REPORT.json`; `docs/reviews/codex_g3b_base.md` |
| 6 | Batch 1 solve, rows, provenance | `Thesis-wt-g2p2` (`thesis/g2-phase2`) @ `0386403`, `3575e87` | `yield/20261109_212657_g2full.AUhok0` | `solve/pilot_report.json`, `G2_REPORT_final.json` (copy `G2_REPORT.json`), `rows_final_A/`, `rows_final_B/`, `provenance_final_A.json`, `provenance_final_B.json`; frozen copy `r6g3/final/{A,B}` |
| 7 | Arm A training | `Thesis-wt-r6` @ `b006989` | `r6g3/attempts/20261110_062035_run_A.lBKtpA` | `train/train_log.jsonl`, `chain.log`; `G3_A_RUN1.json` pending |
| 8b | Batch 2 solve, rows, provenance, 501 tasks | `Thesis-wt-g2p2` @ `063b6e1` | `yield/20261110_040120_g2full.wjjYRA` | `solve/pilot_report.json`, `rows_clean/`, `rows_B/`, `provenance_fast_A.json`, `provenance_fast_B.json` |
| 8c | Reconciled spend | `Thesis-wt-g2b` @ `48383ec` (ledger fix) | all `yield/` attempts | `sft/g2_tools.py spend`; `runlogs/spend_now.json` per attempt |
| 8d | Batch 2 accuracy review | `Thesis-wt-g2b` @ `90725fa` | `yield/20261110_040120_g2full.wjjYRA` | `review_sample.md`, `review_ids.json`, `codex_a1c_verdicts.md`, `subagent_a1c_part1.txt`, `subagent_a1c_part2.txt` |
| 8e | Pass-1 probe (exploratory) | `Thesis-wt-r6` @ `cf2b8a8`, `65169a6` | `r6g3/attempts/20261110_062035_run_A.lBKtpA/probe_pass1` (void), `.../probe_pass1_r2` | `PROBE_RESULT.json`, `evals/F_pass1/report.json`, `monitor.log`; `docs/DISTILLATION_G3_PREREGISTRATION.md` amendments |
| Plan | Combined retrain (approved, not launched) | `Thesis-wt-r6` @ `5cc0c92` | none yet | `docs/DISTILLATION_G3C_COMBINED_PREREGISTRATION.md` |
| Backup | Data backup 2026-11-10 | none | `~/thesis-backups/2026-11-10` | `STATUS.txt`, `MANIFEST_sha256.txt`, `EXCLUDED_still_being_written.txt`; section 9 below |

Round 2 to 5 results are in `Thesis-wt-distill/docs/DISTILLATION_ROUND{2,3,4,5}_FINDINGS.md` (read-only worktree, branch `thesis/distill-stage0` @ `5187288`); their data is in `~/thesis-backups/2026-11-09`.

---

## 1. Round 6 stage 1: is the old Round 5 adapter separable from the untuned model? (finished 2026-11-09)

- **Question.** On the 155-task screening bank FK (F 81 + K2 74), do six arms (untuned model, old adapter at three strengths, and so on) separate?
- **Setup.** Worktree `Thesis-wt-r6`, branch `thesis/distill-r6-diagnostics`, chain `sft/r6_chain.sh stage1`, mode iffc-ooda-card, greedy, server restarted between arms. Banks hashed: F `7a45cd08...`, K2 `e2ac40e4...`.
- **Counts** (source: attempt `sft/data/distill/r6/attempts/20261109_121007_stage1.1ihURW/chain.log`, `STAGE_REPORT.json`): untuned model composite 64 (B3) and 65 (B3b); old adapter S0 64; S0 at scale 0.5 about 66 and at 0.25 about 65 [ANALYSIS, inferred from the log, K2 = 65 - total]. Every run had 81 and about 74 valid tasks and 0 API errors.
- **Registered rule.** Base band 65-70, S0 band 56-62, screen >= 77 and > 68. Outcome: B3 = 64 and S0 = 64 fell outside their bands (drift flags).
- **Follow-up.** One extra untuned run (approved), attempt `20261109_191001_extrabase.zf0OhZ`: 36 + 32 = 68 of 155, valid (81/81, 74/74, 0 API errors). All attempts are listed.
- **Reading.** No arm is separable from the untuned model in this session, and the untuned model itself moved by 4 passes between runs (64, 65, 68). That is the single-run noise floor. It is not a statement that the old adapter is harmful or harmless.
- **Limits.** One session, greedy, FK shares its author with the training data so only the BFCL 200 settles claims.
- **Review.** Codex `docs/reviews/codex_s1_stage1_result.md` (Thesis-wt-r6); extra run reviewed in `codex_x1_extrabase.md` ("No major problems found").

## 2. Data factory fix and two samples (finished 2026-11-09)

- **Question.** Can deterministic state checks plus a request-coverage judge remove the wrong tasks found in the first 40-task review?
- **Setup.** Part 1 synthetic tasks (99 kept, `Thesis-wt-data`, attempt `20261109_144037_part1.E11FEx`). Checks: `id_collision`, `empty_read_with_state_key`, `quoted_suffix_not_in_state`; judge `sft/yield_judge.py`; pre-registration `docs/DATA_YIELD_FACTORY_FIX.md` (commit `a5a784a`).
- **Counts.** First review of 40 tasks: Codex 27 faithful / 6 ambiguous / 7 wrong; two subagents 33 / 4 / 3 (all 3 inside Codex's 7). Judge on the 99 tasks: WRONG 24, FAITHFUL 74, AMBIGUOUS 1 ($0.0128, 99 calls, `judge/`); 71 of 99 survive checks plus judge (`calibration.json`). Fresh sample of 40 survivor tasks not previously reviewed: Codex 31 faithful / 8 ambiguous / 1 wrong (`distill_y_0843`); subagents 35 / 5 / 0 (`docs/reviews/codex_a1b_fresh_sample_verdicts.md`, `subagent_a1b_fresh_sample_verdicts.md`).
- **Registered rule.** Catch at least 6 of the 7 wrong tasks and wrongly drop at most 5 of the 27 faithful; the fresh sample may hold at most 2 wrong. Outcome: calibration passed; fresh sample passed (1 wrong).
- **Limits.** The reviewers are models, not humans; the judge and the Codex labeller are not independent of the teacher family in every respect; 40-task samples.
- **Review.** Codex f1/f2 (`codex_f1_factory_fix.md`), labelling runs through `sft/codex_label.sh`.

## 3. G1 overfit gate (finished 2026-11-10 01:44 ET): PASS, pipeline check only

- **Question.** Can the adapter recipe plus the llama.cpp GGUF serving path learn its own training tasks?
- **Setup.** `Thesis-wt-r6` commit `4294d7c`, chain `sft/g1_chain.sh g1`, attempt `sft/data/distill/r6g1/attempts/20261109_201851_g1.P8Hx7x`. 40 tasks chosen by sha256 of the task id from the calibration survivors; 243 rows (8 rows over 6,999 tokens dropped); adapter layers 8-34, rank 16, lr 1e-4, 4 passes, thinking off, prompt mask; 972 micro-batches, 243 updates, 18,243 s, peak 47.43 GiB. Pre-registration `docs/DISTILLATION_G1_PREREGISTRATION.md`.
- **Counts** (`G1_REPORT.json`): base 11/40, adapter 32/40, both thinking off, 0 API-error episodes. Parse rate 1.0 (base) and 0.9968 (adapter, 1 of 315 outputs unparsed). Row check (`rowcheck_*.json`): base 165/243 exact, adapter 234/243; exact on stop rows 60/63 and 63/63; non-stop 105/180 and 171/180 [ANALYSIS]; tasks with all rows exact 7 and 32 of 40. GGUF audit error 2.8e-4; render parity 0 mismatches.
- **Registered rule.** PASS iff adapter >= 24/40 and adapter - base >= 8. Outcome: PASS.
- **Limits.** The same 40 tasks were used for training and scoring, so this is not generalization. Mode iffc-ooda-card, not the compact-list mode. Four tasks lost rows to the length cutoff. One run, one seed. It says nothing about the Round 2-5 adapters.
- **Review.** Codex `docs/reviews/codex_g1_result.md`: "No major problems found" (Thesis-wt-r6 `01aa6ae`).

## 4. G2 data scale-up, authoring and cleaning (authoring finished 2026-11-10 03:28 ET; judge pending)

- **Question.** Can the same factory give at least 250 clean multi-turn tasks (target 400) and 1,500 train rows?
- **Setup.** `Thesis-wt-g2` branch `thesis/g2-scale`, attempt `Thesis-wt-data/sft/data/distill/yield/attempts/20261109_212657_g2full.AUhok0`; three authoring processes (a, c and a replacement d for b), 10,302 slots; pre-registration `docs/DATA_YIELD_G2_PREREGISTRATION.md` (section 10 is the deviation log).
- **Counts** (`author/report.json`, `merge_report.json`): 824 valid tasks; API rejects 193 of 10,302 (1.87%); slots lost to the deadline 0; authoring spend a+c+d $9.1562, attempt ledger including the crashed process b $9.4033; 0 responses without cost. After the cross-process 8-gram dedupe 734 tasks; audit kept 734; state checks flagged 151 (id_collision 139).
- **Registered rules.** A6 (at most 2% of slots lost): passes narrowly (1.87%). Floors: 250 surviving tasks, 1,500 train rows: not yet decidable. Target 400: the Part 1 ratio projects about 308 survivors [projection, not a result].
- **Deviations.** Process b died in its first chunk (uncaught `TypeError`, 222 slots started, $0.2473, no tasks); d replaced it.
- **Key limit (what the API rejects were).** The OpenRouter key carries its own $20 monthly limit (`GET /api/v1/auth/key` on 2026-11-10: spend $20.002, remaining -$0.002, resets 2026-12-01). 189 of the 193 API rejects (1.87% of slots) fell in d's last chunk and are attributed to the key running out, not to a provider failure. The same limit made the judge return 734 SKIPPED with 0 calls and $0. The key's $20 includes spend from earlier rounds outside this attempt's ledger ($9.40 here, $2.14 earlier), so the key limit, not the project's $50 cap, is the binding one. The judge, the pre-solve filter and the phase 2 solve wait for Sans to raise it. No other key or provider is used.
- **id_collision repair (registered deviation, prereg section 11).** 134 of the 734 merged tasks start with a backend counter at or below an existing id. A deterministic repair (counter = largest existing id + 1; calls and user turns unchanged) kept 129 (R1 107 where no other gold result changes, R2 22 where one does); 5 dropped. Join the pool only after a 20-task Codex review (<= 2 wrong, <= 1 among 6 R2). **Result: FAILED, no repaired task joins the pool.** Codex on the 20-task sample (`docs/reviews/codex_g2_repair_sample_verdicts.md`; seed 9093): 13 faithful / 1 ambiguous / 6 wrong. By class: R1 3 wrong of 14 (two `follow` calls that returned false and the gold ignored it, one turn that omits a requested word count), R2 3 wrong of 6 (two where the gold acts on the old order id 12446 instead of the newly created 12447, i.e. the repair moved the id the gold relied on; one ignored `follow` result). Registered rule: at most 2 wrong of 20 and at most 1 among the 6 R2: failed on both counts, so the rule says none join. Reading: the R1 failures are generic wrong-gold defects that the judge exists to remove, not caused by the counter change (the sample was drawn before the judge, so it is not comparable with the post-judge reviews above); the R2 failures are caused by the repair. A post-judge re-review of repaired survivors would be a separate registered step and needs the key limit lifted first.
- **Limits.** The Part 1 ratios are borrowed; duplicates were 21.7% of slots.
- **Review.** Codex `codex_g2_package.md`, `codex_g2b_package.md`, `codex_g2c_package.md` (Thesis-wt-g2).

### 4b. G2 batch 1: judge and pre-solve filter (2026-11-10 03:50-03:53 ET)

- **Setup.** Key limit lifted by Sans (`/auth/key`: limit none, usage 20.002 then). Judge `sft/yield_judge.py`, cap $0.50, on the 734 merged and audited tasks; state checks as in section 2. Repaired tasks not used (repair review failed, section 4).
- **Counts** (`judge/verdicts.jsonl`, `prefilter_report.json`, attempt 20261109_212657_g2full.AUhok0): FAITHFUL 509, WRONG 206, AMBIGUOUS 19; $0.0974 over 734 calls, 0 without cost. State-check hard flags 141 tasks. **481 tasks pass both and enter the solve.** (The failed first judge run, key limit, is kept as `judge_failed_keylimit_0334`.)
- **Rule.** None registered for this step beyond the filters themselves; the floors (250 tasks, 1,500 train rows) are applied after the solve and the held-out filter.
- **Limits.** Judge is a model; Part 1 calibration (24 of 99 WRONG, catching 6+ of 7 known-wrong tasks) is the only evidence of its accuracy; the post-solve 40-task review is still to come.

### 4c. G2b: second authoring batch (launched 2026-11-10 04:01 ET)

- **Approval.** Sans 2026-11-10: "yes I agree let's do those things in parallel and keep track, need to maximize time and resource utilization". About 10,000 slots, API only, cap $12 including judge and solve.
- **Setup.** `Thesis-wt-g2b` branch `thesis/g2b-scale` commit `59cdece`, pre-registration `docs/DATA_YIELD_G2B_PREREGISTRATION.md`: 40 new themes; backend weights = BFCL co-occurrence share / batch 1 share (structure only, no BFCL text); prompt rule against id-counter collisions; dedupe against batch 1; per-slot exception catch; 6 h whole-chain cap; API-error gate 2%. Smoke: 12 slots, 0 API rejects, $0.0167. Authoring attempt `20261110_040120_g2full.wjjYRA`.
- **Review.** Codex: round 1 two blockers (whole-chain cap, API gate) fixed; round 2 "No major problems found" (`docs/reviews/codex_g2b_batch2.md`, `codex_g2b2_batch2.md` in Thesis-wt-g2b).
- **Outcome.** Pending.

## 5. G3 base references (running since 2026-11-10 03:37 ET)

- **Question.** What does the untuned model score on FK in compact-list mode with thinking on (twice) and off (once)? These fix the threshold any trained arm must beat.
- **Setup.** Approved by Sans 2026-11-10 ("Approve for all 3"). `Thesis-wt-r6` commit `571c9e7`, `sft/g3_base.sh`, attempt `sft/data/distill/r6g3/attempts/20261110_033718_base.XtrzLh`; mode iffc-state-ooda-slim from `Thesis-wt-slim` `dfdbc2b`; greedy; server restarted between the three runs.
- **Registered rule** (before any result): a candidate's first run needs at least ceil(mean of the two thinking-on runs + 0.06 x 155) and more than the higher of them; the restarted repeat uses the same mean rule.
- **Counts** (source: `sft/data/distill/r6g3/attempts/20261110_033718_base.XtrzLh/G3_BASE_REPORT.json`, Mac): thinking on, run 1: F 30 + K2 29 = **59**; thinking on, run 2: F 33 + K2 28 = **61**; thinking off: F 27 + K2 27 = **54** (parse rate 0.998, 2 unparsed of 1,017 outputs; thinking-on parse rate 1.0). All runs valid (81 and 74 unique tasks, 0 API errors). Finished 05:28 ET (1 h 51 min).
- **Registered threshold** (computed by the script from the two thinking-on runs, mean 60.0): an arm's run 1 needs at least max(ceil(60.0 + 0.06 x 155) = 70, 61 + 1) = **70** of 155; a restarted repeat needs at least ceil(mean + 9.3) = 70 under my reading of "same mean rule".
- **Reading.** The untuned model in compact-list mode scores 59 and 61 with thinking on (two runs, 2 apart) and 54 with thinking off, 6 to 7 below. Earlier rounds' untuned values in the other tool-list mode (64, 65, 68, Round 6 stage 1) are not comparable with these.
- **Limits.** One session, greedy, FK shares its author with the training data; the thinking-off value is a single run.
- **Review.** Codex static review of the script: "Keep running" (`docs/reviews/codex_g3b_base.md`, Thesis-wt-r6, not yet committed there).

## 6. G2 batch 1 solve, rows, provenance (finished 2026-11-10 05:42 ET)

- **Question.** After the judge, how many tasks and rows survive the teacher solve and the held-out filter, and do they pass the leak gate?
- **Setup.** `Thesis-wt-g2p2` branch `thesis/g2-phase2` `0386403`/`ab4747e`, `sft/g2_phase2.sh`: teacher solve in compact-list mode (`iffc-state-ooda-slim`, k = 4, 4 workers) with two or three plain sentences of reasoning (instruction only in the HTTP body; 0 of 318 logged smoke prompts contained it), 481 tasks in.
- **Counts** (attempt 20261109_212657_g2full.AUhok0, `solve/pilot_report.json`, `G2_REPORT.json`): 1,924 episodes, 1,152 verified, 344 tasks with at least one verified episode, solve cost $1.0591 (0 calls without cost); held-out filter kept **325 tasks** (target 400 not met, floor 250 met); conflict dedupe dropped 210 train rows (115 prompts with conflicting targets); rows arm A **2,944** (train 2,672, valid 272); arm B 2,751 (93.4%; excluded: leak phrase 109, one sentence 74, four sentences 10). Attempt spend $10.5598 by the ledger (authoring $9.1562, incl. crashed process b $0.2473; judge $0.0974; solve $1.0591), under the $15 step cap.
- **Registered rules.** Floors 250 tasks and 1,500 train rows: **met**. Provenance gate (`check_provenance.py`, now also on reasoning text): **FAILED on first run** for both arms and fixed by exclusions recorded before use: arm A one task (`distill_g2d5_0039`, 5 rows, a generic login 8-gram shared with a held-out task); arm B additionally 5 rows of other tasks whose teacher sentences share trade-phrase 8-grams with held-out tasks. Re-run on the final files: both pass, bound to the frozen rows by sha256.
- **Final training data.** Arm A 2,667 train + 272 valid rows; arm B 2,494 train + 249 valid rows (93.5% of A's train rows); 292 train + 32 valid tasks in both arms; no task in both splits; arm B is a subset of arm A.
- **Limits.** Yield report flags A3 (628 rows with consecutive assistant messages in the compact-list history) and A5 (ledger now includes batch 2) fail for the reasons in the G3 pre-registration section 3; the A1 accuracy review of the post-solve tasks (40-task sample) has not been run; the teacher sentences were scanned by a regex only, not read.
- **Review.** Codex on the package (two rounds, `docs/reviews/codex_g3_package.md`, `codex_g3b_package.md` in Thesis-wt-r6).

## 7. G3 arm A training (started 2026-11-10 06:20 ET)

- **Question.** Does an adapter trained on 2,667 clean rows (direct answer, thinking off) beat the untuned model with thinking on in compact-list mode on FK?
- **Setup.** `Thesis-wt-r6` `b006989`, `sft/g3_arm.sh run A`, attempt `r6g3/attempts/20261110_062035_run_A.lBKtpA`; adapter layers 8-34 rank 16 lr 1e-4, 2 passes, 5,334 micro-batches; smoke 13.32 s per micro-batch; estimated 19.7 h [ANALYSIS]. Registered threshold: run 1 at least 70 of 155.
- **Outcome.** Pending (training until about 03:00 ET Nov 11).

## 8. G2b batch 2: authoring, cleaning, judge (2026-11-10 04:01 to 07:44 ET Mac clock)

- **Question.** Does a second authoring batch with wider themes, backend weights toward Message/Travel and a prompt rule against id collisions add clean tasks at acceptable cost?
- **Setup.** `Thesis-wt-g2b` `59cdece` (+ pre-registration commits to `6eb80db`), attempt `20261110_040120_g2full.wjjYRA`; 3 processes (e, f, g) of 3,334 slots in chunks of 500; caps 3 x $3.30, monitor $9.90, total step cap $12.
- **Counts** (`author/report.json`, `merge_report.json`, `prefilter_report.json`): 393 valid tasks over 9,334 worked slots (4.2% of worked slots; batch 1: 8.0% of 10,302); dup_in_batch 13.1% (batch 1: 21.7%); authoring spend $9.6666 (ledger incl. killed chunks $9.8355). Stop: process f hit its $3.30 cap in its last chunk (cost guard; its 270 refused slots were counted as lost, 0 real API rejects); the spend monitor then ended e and g in their last chunk (668 slots lost). After merge against Part 1 and batch 1: 368 tasks; audit 368; state-check flags 41 (id_collision 19 = 5.2% of tasks, against 18.9% in batch 1: the prompt rule helped); judge FAITHFUL 271, WRONG 84, AMBIGUOUS 13 ($0.0474); **270 tasks pass both and enter the solve**.
- **Registered rules.** Cap stop (G2B prereg section 5): applies, no decision needed. API-error gate 2%: 0 real API rejects. Floors: not applied to batch 2 alone.
- **Reading.** The lower valid rate per slot is mostly the backend mix (Message and Travel slots yield about 2 to 3% in both batches and were 74% of batch 2's slots against 39.5% in batch 1); slots with neither backend yield 11.8% against 12.8%.
- **Limits.** Chunk 0 only for the by-backend analysis; the solve, the held-out filter, the provenance gate and an accuracy review of survivors are still to come; batch 2 is for a later combined retrain and does not enter arm A or B.
- **Review.** Codex on the package (two rounds, `docs/reviews/codex_g2b_batch2.md`, `codex_g2b2_batch2.md`, Thesis-wt-g2b); the stop handling was not separately reviewed.

### 8b. G2b batch 2: solve, rows, provenance, combined count (finished 2026-11-10 08:45 ET Mac clock)

- **Setup.** Solve as in batch 1 (`Thesis-wt-g2p2` `063b6e1`: slim mode, k = 4, teacher sentences) on the 270 tasks that passed judge and state checks; caps overridden to total $12 and solve $1.70 (Sans, 2026-11-10: "No need to stop on the spend cap, write as many tasks that we need to hit our target"; the $50 project cap holds).
- **Counts** (attempt 20261110_040120_g2full.wjjYRA): 1,080 episodes, 664 verified, 195 tasks with a verified episode, solve $0.5756; held-out filter kept **177 tasks**; rows arm A 1,541 (train 1,416, valid 125); arm B 1,430 (92.8%). Provenance: arm A passes; arm B fails on 4 rows (sentences share an 8-gram with a held-out task), to be dropped at retrain time. Gate status **FAILED-BY-CAP** (authoring stopped by the cost guard and the monitor; 270 guard refusals counted as lost slots, 0 real API rejects).
- **Combined with batch 1:** 324 + 177 = **501 surviving tasks**, at the 500 target; no batch 3. Attempt ledger $10.5055; key usage $31.68 (project total, limit none).
- **Valid rate by backend, chunk 0 (batch 1 -> batch 2)**, from regenerated slots [ANALYSIS]: slots with Message or Travel 17/592 = 2.9% -> 23/1109 = 2.1%; slots with neither 116/908 = 12.8% -> 46/391 = 11.8%; Message and Travel were 39.5% -> 73.9% of slots. Per backend: Trading 19.0 -> 9.7%, Ticket 14.7 -> 6.0, Vehicle 9.9 -> 4.8, Twitter 9.5 -> 6.3, GorillaFS 5.5 -> 5.8, Math 9.0 -> 11.3, Message 2.8 -> 2.0, Travel 3.3 -> 2.2. Survivors per planned slot: batch 1 325/10,302 = 3.2%, batch 2 177/10,002 = 1.8%. The weights over-sampled the two lowest-yield backends; ledger cost per surviving task [ANALYSIS]: batch 1 $10.56 / 325 = $0.032, batch 2 $10.51 / 177 = $0.059.
- **Limits.** Rows of batch 2 do not enter arm A or B; no accuracy review of batch 2 survivors yet (the 40-task A1 sample is still to be drawn); cap-stop deviations are in `docs/DATA_YIELD_G2B_PREREGISTRATION.md` sections 5 and 6 of Thesis-wt-g2b.

### 8c. Reconciled spend and combined row counts (2026-11-10 08:50 ET Mac clock)

- **Spend by attempt** (`g2_tools.ledger` over all timestamped attempts in `Thesis-wt-data/.../yield/attempts`, logged successful responses only): Part 1 $2.1215 (authoring, solve and the first judge, 13,686 calls); batch 1 $10.5598 (62,592 calls; authoring $9.1562 incl. the crashed process b $0.2473, judge $0.0974, solve $1.0591); batch 2 $10.5055 (53,422 calls; authoring ledger incl. killed chunks, judge $0.0474, solve $0.5756 plus the aborted first solve start); smokes $0.0159 and $0.0167. **Ledger total $23.2194, 0 calls without cost.** The key's own usage counter (`/auth/key`) reads **$31.68**, limit none. The difference of about $8.47 is outside these ledgers [ANALYSIS]: it was already about $8.3 at the 03:44 ET check before the judge, so it predates this session's batches (most likely earlier rounds' API use on the same key, plus requests that were billed but not logged, such as in-flight requests killed by the monitor); it is not explained further here. The advisor's $40 report mark has not been passed.
- **Combined rows:** arm A 4,083 train + 397 valid rows over 501 tasks (batch 1 2,667 + 272 over 324 tasks; batch 2 1,416 + 125 over 177 tasks, 159 train + 18 valid); arm B 3,803 train + 366 valid rows (batch 1 2,494 + 249; batch 2 1,309 + 117 after removing 4 rows of tasks `distill_g2e0_0152`, `distill_g2e3_0079`, `distill_g2g4_0096` whose sentences share an 8-gram with a held-out task). Task ids of the two batches are disjoint.
- **Combined retrain.** Draft pre-registration `docs/DISTILLATION_G3C_COMBINED_PREREGISTRATION_DRAFT.md` (Thesis-wt-r6): same adapter and screen; estimated 26 to 29 h of training [ANALYSIS]; runs only after both arms are screened and needs Sans's approval line; which arm is retrained is left open (two options).

### 8d. Batch 2 accuracy review (2026-11-10 ~10:55 ET Mac clock)
- Sample: chain's seed-9092 draw, 40 of 177 survivors. Codex: FAITHFUL 32, AMBIGUOUS 6, WRONG 2; subagents 18/2/0 and 14/6/0. Registered rule (at most 2 WRONG per set): **pass, at the limit for Codex**. The first Codex call exited 1 (prompt file not yet copied) and was re-run once on the same sample and prompt.
- Both WRONGs (`g2e5_0264`, `g2f5_0347`) share one pattern: `view_messages_sent` in the BFCL Message backend returns the whole inbox grouped by receiver, so it is a simulator convention, not a task error. 9 of 501 combined tasks contain that call (3 + 6).
- Known limitation: 4 subagent AMBIGUOUS (one with garbled wording) are Trading cancels by id without `get_order_details`; 46 of 501 tasks have `cancel_order` without `get_order_details` in gold.
- Limits: reviewers read tasks, not trained-model behaviour; one sample; no row changed.

### 8e. Extra, exploratory: arm A pass-1 checkpoint on bank F (2026-11-10)
- **Attempt 1 (void), 15:38 to 15:47 ET Mac clock.** Same approved run (Sans, 2026-11-10, quoted in the G3 pre-registration amendment). Checkpoint converted and audited; 8081 server identity ok. Result: **81 rows, 81 unique tasks, 81 API-error episodes, 0 passes; the server received no request.** Cause: the eval harness fixes the server address to port 8080 (`TeacherConfig.local`), and the probe's `BFCL_SERVER_URL` is not read on that path. Nothing listens on 8080 during training, so every call was a connection error. No model output exists; this is not a look at F.
- **Attempt 2** started 15:57:38 with a launcher that routes the eval to 8081 (pre-registration amendment 2, `Thesis-wt-r6` `65169a6`). Result below when finished.
- **Attempt 2 result (finished 16:23:43, 1,568 s; exploratory):** bank F, arm A pass-1 checkpoint (thinking off, slim, greedy, 2 workers): **81 rows, 81 unique tasks, 0 API-error episodes, 37 passes of 81**; 586 assistant outputs, 0 unparsed, stop share 0.389. Base F references on record: thinking on 30 and 33, thinking off 27. Headroom: arm A mean s/micro peaked at 18.1 against 13.06 before (1.38x); swap peaked at 7.93 GB (stop line 8 GB); arm A kept training. Limits: mid-schedule checkpoint, F shares its author with the training data, one run, one bank, not the registered screen; it selects nothing and replaces nothing. Files: `probe_pass1_r2/PROBE_RESULT.json`.

## 9. Data backup (2026-11-10 16:52 to 16:53 ET Mac clock)

- **What.** Copy (rsync -a at nice 15; nothing moved or deleted) of all gitignored training data made since the 2026-11-09 backup: `Thesis-wt-r6/sft/data/distill` (G1 adapter and evals, frozen G3 rows, base reference runs, arm A attempt including the pass-1 checkpoint and both probe attempts), `Thesis-wt-data/sft/data/distill` (Part 1, batch 1 and batch 2 authoring, judge, solve, rows, provenance reports, review samples), and the small data folders of `Thesis-wt-g2` and `Thesis-wt-g2p2`. `Thesis-wt-g2b`, `-compact` and `-slim` have no data folder. `Thesis-wt-distill` (read-only, 11 GB) is already in `~/thesis-backups/2026-11-09` and was not copied again.
- **Where.** `~/thesis-backups/2026-11-10` on the Mac; free disk before: 293 GiB (above the 50 GB mark), after: 287 GiB.
- **Sizes and counts** (`STATUS.txt`): 1,045 files, 6,856,864 KB (6.5 GiB) in total; `data-distill` 631 files, 5,603,372,727 bytes, byte-for-byte equal to its source (`diff -rq` empty); `r6-distill` 390 of 394 source files, 1,396,610,679 of 1,476,010,426 bytes; `wt-g2-data` 9 files, `wt-g2p2-data` 8 files. SHA-256 manifest of adapters, GGUFs, train/valid rows, task files, verdicts and reports: 164 lines (`MANIFEST_sha256.txt`).
- **Skipped (still being written by arm A training, `EXCLUDED_still_being_written.txt`).** `train/adapters_ckpt.safetensors` (rewritten every 50 updates), `train/train_log.jsonl`, `chain.log`, `runlogs/train.out` of the arm A attempt, and `train/adapters/` (the final adapter does not exist yet): 4 files, 79,399,747 bytes. The pass-1 checkpoint is complete and was copied.
- **To repeat** after arm A's screen and after every later run (the next copy must include the arm A final adapter and its screen reports). The one-time copy script was `/tmp/bk2.sh` on the Mac.
