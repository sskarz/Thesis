# Distillation round 4: error recovery (findings, 2026-11-08)

Setup, fixed before any result: docs/DISTILLATION_ROUND4_PLAN.md, docs/DISTILLATION_ROUND4_H_SPEC.md. Scope: Gemma-4-E2B-it, BFCL v4 multi_turn_base-style tasks, one Mac. Teacher DeepSeek V4.1 Flash. Serving: bar GGUF (bartowski Q4_K_M, sha256 prefix 923c4c86177d2ee1), greedy. All numbers below come from verdicts.jsonl / logs under sft/data/distill/ (gitignored) and the run log. Codex reviews: r22 (H spec), r23 (bank F), r24 (Stage A), r25 (Stage B data).

## Question
Round 3 adapters (five variants, 36-41 on a 103-task dev bank vs base 44/44) did not help. A diagnostic showed the base model replies with an empty action list after a tool error instead of retrying. Hypothesis: (a) a generic harness nudge ("that failed, retry") helps, and/or (b) training rows that demonstrate recovery help.

## Stage A: fresh bank F (81 teacher-verified tasks), four arms, 0 API-error episodes
| arm | passes /81 |
|---|---|
| base B1 | 35 |
| base B2 (server restarted) | 35 |
| H = harness retry nudge (mode iffc-ooda-card-recover) | 35 |
| AB (existing think_AB adapter) | 30 |
Pre-registered advance rule: >= max(B1,B2)+3 = 38. H does not advance; AB is 5 below either base run. Paired H vs B1: +7/-7; AB vs B1: +3/-8; B2 vs B1: +5/-5 (greedy run-to-run noise is about 5 tasks each way).
Mechanism (descriptive, sft/stageA_diag.py, detector agrees with the H note marker on 539/539 calls): model calls directly after a step containing an error: base B1 40 (31 empty replies), B2 40 (29 empty). Under H: 49 post-error calls, 30 are new calls (19 not). So the give-up behaviour reproduces on fresh data and H changes it, but pass count does not move. Retrying is not the bottleneck by itself.
Correction in the record: my first post-error counts read the wrong message (execution results sit in the last user message after the OODA card); Codex r24 caught it; fixed before any conclusion was used.

## Stage B: recovery training data (stopped by the pre-registered rule)
79 curriculum tasks (G), student rollouts (T=0.7, 2 samples), teacher labels, then hint-free native sampling (the base student must itself reproduce the teacher action). 166 recovery candidates (states right after a tool error where the teacher action was kept) from 77 tasks; after native sampling 60 rows pass the implemented filter, but only 45 pass strict JSON and the task-step uniqueness key (Codex r25). Rule: < 100 unique recovery rows stops training arms. C, C_match and C+H were not trained. Hinted sampling was not used (not approved).
Control (non-error states, same count and action mix): 142 of 166 pass the implemented filter, 135 strict. The gap (retention 36% vs 86% on the implemented filter) is not a controlled error effect: the control matches only count and action mix; student-equivalent targets are 37/166 in R vs 124/166 in control, and prompt lengths differ.

## Cost
About $3.2 of the $15 Round 4 budget (authoring F $1.27 + G $1.53, solve ~$0.34, labels ~$0.07).

## What this does and does not show
* The empty-reply-after-error behaviour is real and reproduces on a fresh bank.
* A generic retry nudge changes that behaviour without changing the pass count (35/81 vs 35/81, 7 gains / 7 losses).
* Strictly hint-free recovery rows cannot be produced at 100+ scale from this teacher/student pair with this pipeline (45 strict rows from 166 candidates).
* It does not show that recovery training would not help; that arm was not run.
* Cumulative record across rounds: no tested fine-tune or harness variant has beaten the untuned base on a held-out bank under the pre-registered rules.
