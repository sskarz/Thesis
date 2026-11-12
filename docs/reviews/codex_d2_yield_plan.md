D1 dispositions:

1. **RESOLVED** — `sft/yield_guard.py:17–22`: missing cost blocks subsequent requests; both wrappers use the guard.
2. **RESOLVED** — `sft/data_yield_chain.sh:27,33`: external limits match the revised registered caps of 160/70 minutes, with 30-second kill grace.
3. **RESOLVED** — `sft/build_bank_f.py:34–36`: skipped verdicts without `task_id` are ignored and counted.
4. **RESOLVED** — `sft/yield_report.py:267–274`, `docs/DATA_YIELD_PLAN.md:56–57`: nonempty rows and explicit evidence minimums prevent vacuous certification.
5. **RESOLVED** — `docs/DATA_YIELD_PLAN.md:73`: distinguishes preliminary 60 from strict-filtered 45.

Remaining major findings:

- **BLOCKER — `sft/yield_report.py:272–273`:** Budget-truncated solving can pass A6. `generate_multi_turn.py:743–745` catches `BudgetExceeded` and converts it into an episode error, preventing `distill_solve.py:153–155` from recording the budget stop. Remaining jobs can become error episodes while `stopped_early` remains `None`, certifying an incomplete solve. **Fix:** Latch budget exhaustion in the guarded client and make A6 fail whenever that latch is set.

- **BLOCKER — `sft/data_yield_lib.sh:65–66`:** Historical responses lacking cost are counted as zero. After a missing-cost abort, relaunch resets the guard and admits further spending without reconciling those charges; the cross-attempt $5 guarantee therefore remains unproven. **Fix:** Refuse relaunch when historical costs are missing until reconciled, and carry cumulative accounting into A5.

Other requested checks: `pilot()` does instantiate the patched guarded class (`benchmark/distill_solve.py:132`). Both harness HTTP fallbacks read the modified `config.SERVER_URL`, so they target port 9. The chain’s producer/consumer paths and JSON keys match; no reachable part-1 operation starts GPU inference, controls llama-server, requests port 8080, or prints the API key. Inspection only; no workloads or tests run.

Not ready to launch part 1: budget-stop certification and cross-attempt missing-cost accounting still need fixes.