# Distillation pilot (Stage 1) results

Run 2026-11-06 on the Mac, branch `thesis/distill-stage0`. Teacher
`deepseek/deepseek-v4.1-flash-1017` via OpenRouter, reasoning off, pinned to
provider DeepInfra (DeepSeek's own endpoint is blocked by the account privacy
setting). Student-loop mode for the teacher: `iffc-ooda-card`.
All numbers below come from the logs in `sft/data/distill/pilot/` (gitignored):
`tasks.jsonl`, `b/tasks.jsonl`, `tasks_all.jsonl`, `calls.jsonl`, `b/calls.jsonl`,
`solve/verdicts.jsonl`, `solve/episodes.jsonl`, `solve/pilot_report.json`.

| Step | Count | Spend (API-reported `usage.cost`) |
|---|---|---|
| Author batch 1 (25 slots, seed 1) | 12 valid | $0.0296, 105 calls, 2:33 |
| Author batch 2 (25 slots, seed 2) | 10 valid, 1 dropped as cross-batch duplicate | $0.0310, 100 calls |
| Solve + verify (21 tasks, 4 workers) | 6 verified (28.6%) | $0.0137, 195 calls, 36.3 s wall |

Solve: 897,464 prompt tokens, 4,749 completion tokens, 0 reasoning chars
stripped, 0 API errors, 0 harness flags, 2 of 195 outputs unparsed.
Decision mix of the teacher's outputs: stop rows 76 of 193 parsed (39.4%),
look-up calls 83 of 167 (49.7%, name-prefix heuristic, not the autopsy regex).

Findings: see the report to Sans. Open issues: tasks over Ticket, Travel and
Vehicle verify 0 times; gold calls often use values the user never gave
(offline `ungrounded_literals` heuristic: 3 of 4 grounded tasks verified, 3 of
17 ungrounded). No training row has been written.
