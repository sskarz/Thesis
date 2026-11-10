- **F1-1 RESOLVED** — `sft/yield_judge.py:103–122`: The deadline watchdog marks unfinished tasks SKIPPED and reaches `os._exit` without joining workers, bounding local execution.
- **F1-2 RESOLVED** — `sft/yield_calibrate.py:50–61`, `sft/yield_judge.py:118–121`: Missing, skipped, unparsed, or guarded runs cannot pass calibration and exit unsuccessfully.
- **F1-3 RESOLVED** — `sft/yield_judge.py:33–36`, `docs/DATA_YIELD_FACTORY_FIX.md:10`: Distinctive task examples are removed, and calibration-informed error categories are disclosed.
- **F1-4 PARTLY — BLOCKER** — `sft/yield_judge.py:39–41`: The verdict definitions still require every quoted value to exist in state, contradicting the exemption at line 34. Legitimate supplied tweet text can still be classified WRONG, distorting false-drop and survivor counts. **Fix:** restrict both verdict definitions’ grounding requirements to references to existing objects.

New major problem:

- **BLOCKER — `sft/yield_judge.py:108–121`; `benchmark/teacher_api.py:246–264,334`**: `os._exit` abandons in-flight requests whose provider charges may continue; those charges never reach the reported `spent_usd`. The cap checks only completed-response spending and reserves nothing for outstanding requests, so actual spending can exceed the registered cap while the reported total understates it. **Fix:** reserve worst-case request costs before dispatch, including retries, and reconcile abandoned requests before reporting final spending.

For distinct task IDs, verdict attribution remains tied to each submitted task; model-returned IDs are not used.

Not ready to run the judge until the value-rule contradiction and in-flight cost accounting are fixed.