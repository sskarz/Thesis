# Round 4 plan: error recovery (pre-registration draft v2, after Codex r20)

Status: DRAFT for Sans's approval. Nothing in this plan has run. Steps that spend API money, train, or run an evaluation wait for his OK at the gates in section 8; only the free reading and analysis in P0 is in progress. Written 2026-11-08. Numbers come from `sft/data/distill/r3/` artifacts as noted. Costs and times are estimates from logged runs and are labelled as such.

## 0. Approvals (Sans, 2026-11-08)
- API budget for Round 4: $15 (new; separate from the earlier $4 cap, which is spent). Spend is logged from per-call `cost_usd`; each authoring/solve command carries its own `--max-cost`.
- H scope approved as written in section 4 (generic nudge only).
- Go-ahead for Stage A and for carrying out the remaining phases. Conditional phases still follow the pre-registered rules in section 6. Single-use protected banks (confirm bank, mid20, fresh-20, full 200) are only touched if a candidate advances and repeats; I will tell Sans before that step starts rather than run it silently.

## 1. Why this target

Round 3 and 3b/3c: five adapters, 41 / 36 / 37 / 40 / 36 of 103 on dev_all (AB, A, A-think, AB-think, expanded targets) vs base 44 / 44. `DISTILLATION_NEXT_METHODS.md` section 4a showed the labelled never-pass failures are mostly argument-convention, lookup, partial-action and empty-turn errors. Looking at what happens right after a tool error gave one behavior that cuts across them: the model usually answers `[]` (gives up) instead of retrying.

Calls whose last message is an execution result containing `{"error"`, and what the model answered (literal `[]` / a new call; one think_AB response was prose that the IFFC parser reads as `[]`, so parsed-action counts are 28 of 38 for think_AB):

| Where | Calls after an error | Empty | New call |
|---|---:|---:|---:|
| Dev B1 | 42 | 33 | 9 |
| Dev B2 | 40 | 29 | 11 |
| Dev think_AB adapter | 38 | 27 literal (28 parsed) | 10 |
| Training-pool rollouts (T=0.7, 304 episodes) | 198 | 131 | not split out |

Episodes with at least one tool error pass less often (observational: an error also marks an already-wrong call): dev B1 4 of 29 vs 40 of 74 without; B2 4 of 26 vs 40 of 77; training-pool rollouts 13 of 115 vs 80 of 189.

Training data holds almost no recovery: 13 of 355 AB-think rows follow an error result (train 11 of 320: 10 retry, 1 empty; validation 2 of 35: both retry). These small, clustered counts do not establish that the adapter changed give-up behavior, and a new call is not the same as a successful recovery. The pattern is illustrated by `imperial_si_conversion(unit_out='pounds')` returning "Conversion from 'kg' to 'pounds' is not supported", after which the model answered `[]`.

Why so few recovery rows (audited from `sft/data/distill/r3/*.jsonl`, 2026-11-08): the pipeline stages hold few after-error rows to begin with. Arm A candidates 12 of 456; hybrid JSON rows 30 of 593; the 300 candidates sent to hint-free native sampling 25 of 300; kept after sampling 11 of 25 (44%, vs 161 of 300 = 54% overall; about 10 points lower, so the filter is somewhat harsher on these rows). Scarcity is already present before filtering: the main reason there are few is that few such states were put forward (rescue switched at one point per failed episode). That means recovery rows can likely be produced by deliberately collecting after-error states, without relaxing the hint-free rule; how many survive is unmeasured.

Hypothesis (not shown): much of the failure mass is "does not retry with a corrected call after an error", and either a harness change or recovery training could address it. Alternatives that fit the data: most errors are irrecoverable once the first call is wrong; the model cannot guess valid values without a lookup.

**Provenance of the idea.** The hypothesis came from reading dev_all failures and statistics (dev-informed hypothesis generation). The training-pool numbers above are independent of dev_all and show the same pattern, and the implementation below is specified from training-side evidence, but the dev reading cannot be undone. dev_all can no longer certify a new candidate; a fresh bank F is required and aggregate F results are firewalled (section 3).

## 2. Questions and arms (on a fresh bank F)

- Q1. Does a harness-side recovery change (H) help the untuned base?
- Q2. Does training on recovery-rich data (C) help, beyond what a matched amount of non-recovery data does (C_match)?
- Q3. Does the C adapter work under H, and does retraining with H-rendered rows (C_H) add anything?

| Arm | Model | Harness | Training data | Stage |
|---|---|---|---|---|
| B1, B2 | bar GGUF (bartowski Q4_K_M, sha prefix 923c4c86177d2ee1) | current | none | A |
| AB | bar GGUF + existing think_AB adapter | current | existing 320 rows | A |
| H | bar GGUF | current + H (section 4) | none | A |
| C | bar GGUF + recovery adapter | current | existing AB rows + recovery rows | B |
| C_match | bar GGUF + control adapter | current | existing AB rows + non-recovery rows matched to C on task count, row count and token count as closely as practical | B |
| C+H | the C adapter | H at inference only | as C | B |
| C_H | bar GGUF + adapter trained with H-rendered rows | H | as C, rendered with H | C, conditional |

If no C vs C_match difference can be built (not enough matched non-recovery rows), the C result is reported as a package comparison, not as a recovery-content effect.

## 3. Bank, firewall and integrity rules

- **F**: about 100 new teacher-authored, teacher-verified tasks (k=4, outcome-checked), same authoring pipeline as round 3. Authored before any curriculum task, hashed, and frozen. Curriculum authoring excludes F by code-side hash/text dedupe only; F text is never in an authoring prompt or review feedback.
- **Family separation** covers existing AB tasks, archived recovery sources, new curriculum, validation, and F (gold-function-sequence signature and similarity), checked by an extended `sft/check_provenance.py` (text, id, 8-gram, similarity and gold-function-signature checks against explicitly supplied banks; BFCL splits are covered by text checks only, and raw BFCL tasks without joined gold are rejected explicitly). Final checks run on the actual C, C_match and C_H rows against every protected bank, including mid20, fresh-20, confirm bank, dev_all, and all BFCL splits. Hashes and redacted exclusion reports are published with each phase.
- **Firewall**: aggregate F results (pass counts and the declared diagnostics) cannot change training rows, H, dose, checkpoint, or curriculum design; those are specified before F is evaluated. F is not read per task until all F evaluations and decisions are final; reading F failures afterwards retires F from future certification.
- **Training recipe** is the Round 3b one, data as the only change: think marker, rank 8, scale 2.0, `q_proj`/`v_proj` target keys over layers 19-34 (which yields q-only adapters, since layers 19-34 share KV and have no k/v), lr 2e-5, grad-accum 4, seed 61, 320 microbatches = 80 optimizer updates, same row mixing rule (rows sampled uniformly, no extra repetition rules beyond the Round 3 config), same validation split rule (task-disjoint). Validation loss is reported only; there is no checkpoint selection, no lr/rank/target/epoch search, and nothing tuned on dev_all or F. If recovery-row mixing needs a rule, it is written down before training.
- Dev, mid20, fresh-20, confirm bank and the 200 tasks are never training data. The Round 3 ladder (confirm bank, mid20, fresh-20, full 200) only starts if a candidate advances and repeats on F.
- Scope: Gemma-4-E2B-it only, BFCL multi_turn_base, one Mac. RAG remains Future Work.
- Codex (gpt-6.1-sol, medium, read-only) reviews: this plan; the H spec; bank and provenance reports; training row samples and counts; every result before it is reported. Wording corrections are applied before anything is presented as a finding.

- **Existing-data prerequisite.** The saved provenance report for the AB-think rows flags 5 rows from 2 training tasks (`distill_s2a_0016`, `distill_s2a_0285`) that share word 8-grams with 3 dev tasks. Any reuse of existing rows for C and C_match first drops those two tasks' rows (and the report is rerun), before any training. F authoring, F provenance and the H implementation are kept separate from curriculum authoring and row building (different steps, F text never passed to them).
- **Reference for ties.** The reference base for "the better of B1 and B2" is the higher pass count; if tied, the run with fewer API calls; paired gains and losses are computed against both runs and reported. A second API-error episode in an arm ends that arm's evaluation (report, no replacement). A recovery row is unique by (task id, step id); duplicates and near-duplicates (same state) count once.

## 4. The harness change H

Error text is already in the execution message, so H must change something a model can respond to. H is exactly these two prompt edits, applied only on the call that follows an execution result containing an error object:
1. The `next:` field of the OODA card says to fix and retry instead of `call`/`done`.
2. One fixed sentence is appended to the execution-results message: a call returned an error; read it and call again with corrected arguments, or call a lookup function to find valid values.

H does not add attempts, change termination, resolve arguments against state, or contain any benchmark-specific values or mappings (no `'pounds'`→`'lb'`, no appended `.txt`). The exact strings and trigger are written into a spec before F is evaluated and reviewed by Codex. Trigger: the most recent execution result contains an error and no later successful retry yet. Mixed success/error results use the same trigger. Because H changes prompts, any tokens, model requests and latency are reported per arm, and all regressions are assessed on the no-error subset defined by the baseline B1/B2 runs (episodes with no error result in either), not per-arm subsets.

This is benchmark-informed engineering; transfer beyond BFCL is not claimed. Decision for Sans: confirm this scope.


## 5. Recovery data for C and C_match

Sources:
1. **Existing, free.** The Round 3 labels contain 197 teacher-labelled states right after a tool error. In the labeller, the label is the teacher's action at the student's state: 125 were kept (123 `act_ok` where the teacher's action replays cleanly, plus 2 `stop_correct`) and 72 dropped (40 `stop_early` where the teacher stopped before the turn was complete, 25 `act_error` where the teacher's action errored on replay, 4 `act_when_done`, 2 `ungrounded`, 1 `unparsed`). Of the 123 `act_ok` states, 31 match the student's own action exactly, 2 are effect-equivalent and 90 differ. Only 13 rows after errors reached the training set, so the labelled material is largely unused. P0 audits why (candidate selection was one switch point per failed episode plus random continuation rows).
2. **New curriculum tasks** (about 100 verified) authored to require lookup-before-act, valid-value discovery, retry after error, finishing all requested actions, and navigation. Student rollouts at T=0.7, k=2; teacher labels and hybrid rescue at failed or erroneous states; native rows as in Round 3.

**Collection rule (recommended):** build candidate rows at every after-error state in the existing pools and curriculum rollouts where the labeller kept the teacher's action (`act_ok`, 123 existing states; `stop_correct` where the turn is complete), then run the same hint-free native sampling and strict JSON/leak filters as round 3. The observed 44% retention for after-error candidates is a hint about survival, not a forecast. Existing material is the 123 `act_ok` states above; the curriculum rollouts are what adds more. If recovery rows still number under 100 after filtering, the stopping rule in section 6 applies. The fallback of relaxing hint-free sampling (sample student reasoning with a hint, keep a row only if the reasoning never refers to the hint, train without the hint) is not proposed and would need Sans's explicit decision because it changes a round-3 rule.

## 6. Measurements and pre-registered rules

**Primary (screening tripwire, not a significance test).** On F, a candidate advances if it passes at least 3 more tasks than the better of B1 and B2 on its first run. To count as repeating, a second run of the same configuration (server restarted, interleaved with the base runs) must pass at least as many tasks as the better base run and have more paired gains than losses against it. Each of H, C, C+H, C_H is judged against the same two base runs, so multiple candidates sharing F raises selection risk: if more than one advances, one winner is chosen by the higher mean over its two runs, ties go to the simpler arm (H over C+H over C over C_H), and only that winner goes to the ladder. Results are reported as paired gains and losses per task; F near 100 tasks can show large effects only.

**Mechanism diagnostics** (declared now, reported for every arm, not used to pick a winner), computed on parsed actions: (a) share of calls after a tool-error result that are empty; (b) share that are a new call; (c) pass rate of the baseline-defined error and no-error subsets; (d) model requests, tool invocations and mean API calls per episode; (e) API-error episodes (must be 0); (f) token use and latency.

**C_H condition.** C_H is run only if H meets the primary rule; if H fails but C+H shows a gain over C, an interaction is possible and is reported as exploratory, not as a candidate.

**Stopping rules.** Stop and report if: (1) fewer than 80 F tasks survive audit and provenance; (2) the mean base pass rate on F is below 20% or above 80% (weak room to measure; report F as uninformative for that comparison, no re-authoring or top-up of F after outcomes are seen); (3) fewer than 60 verified curriculum tasks, or fewer than 100 unique recovery training rows after filtering, which stops the training arms only (H still runs); (4) API spend would pass the approved budget; (5) a provenance failure that cannot be fixed by dropping tasks before any outcome is seen; (6) any API-error episode in an arm (the arm is rerun once, not replaced).

F supports a pilot-scale statement about teacher-verifiable synthetic tasks, not certification; the external claim still comes from the protected-bank ladder and BFCL.

## 7. Cost and time estimates (not guarantees)

**API (correction to my earlier message).** Cumulative spend is about $2.18 in round 3 plus about $1.55 earlier, about $3.73 of the $4 cap in the run log, so roughly $0.27 remains, not the $1.82 I said earlier. This plan needs a new explicit budget from Sans.

Estimates from round 3 actuals: authoring $1.5614 and solves $0.3101 produced 157 verified tasks, about $0.012 per verified task. F (100 verified) is about $1.2; a 100-task curriculum about $1.2; labels and hybrid rescue cost $0.3125 in round 3. Stage A (F only) is about $1.2; Stage B (curriculum, labels, hybrid) about $1.5 more; total about $2.7 before any provenance or audit losses. Options: approve Stage A only ($1.2) and decide Stage B after its aggregates; or cut F to 80 verified (about $1.0).

**GPU time.** Logged: base dev_all evaluations took about 49 and 56 minutes, adapter evaluations about 37-38 minutes; pool rollouts 62 and 57 minutes per 76 tasks; Round 3b training 80-95 minutes for 304-320 rows. Stage A with F near 100 tasks: B1, B2, AB, H about 3 h (using those precedents, before any extra latency from H). Stage B: curriculum rollouts for 100 tasks about 1.5-2 h; native-row sampling, reviews and provenance not yet estimated; training C and C_match about 1.5-2 h each (more tokens than round 3b is possible); evaluating C, C_match, C+H on F about 2 h. Conditional C_H and repeats are extra. A total in the order of 12-16 h wall across stages is a rough planning figure, not a validated one.

## 8. Phases and approval gates

| Phase | Work | Needs |
|---|---|---|
| P0 (free, in progress) | recovery-row audit, H spec, extend provenance script to family separation, finalize plan, Codex review | nothing |
| P1 | Stage A: author and verify F (about 100), provenance, then B1, B2, AB, H on F | **Sans OK for an API budget (Stage A about $1.2) and for the GPU runs, with the H scope line and the H scope line settled** |
| P2 | Stage B: author curriculum, rollouts, labels/hybrid, rows, leak scans, Codex row review | **separate OK for API (about $1.5) and GPU** |
| P3 | train C and C_match; evaluate C, C_match, C+H on F | **separate OK for training and evaluation** |
| P4 conditional | C_H rows, training, evaluation; repeats; ladder | **separate OK each, only if the earlier rules are met** |

Nothing in P1 or later starts without a yes at its gate.
