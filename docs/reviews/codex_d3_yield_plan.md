1. **RESOLVED** — `sft/yield_guard.py:23`: worker-thread budget exceptions latch the shared class attribute. Workers join before `sft/yield_solve.py:20–23` writes flags and exits 3; `sft/data_yield_chain.sh:33` aborts with outputs preserved. `sft/yield_report.py:286` fails A6.

2. **RESOLVED** — `sft/data_yield_lib.sh:62–78`: earlier missing-cost calls or flags block relaunch, and earlier spend plus $3.65 must fit within $5. `sft/yield_report.py:272–285` includes earlier spend and missing-cost calls in A5.

No major problems found.

Ready to launch part 1: static inspection confirms paid stages abort on failure, and subsequent reporting and sampling spend nothing; no code or tests were run.