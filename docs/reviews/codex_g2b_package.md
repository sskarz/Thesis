1. **RESOLVED** — `sft/g2_chain.sh:62`: `RL` is initialized immediately after `init_attempt`.
2. **RESOLVED** — `sft/g2_chain.sh:99,183`; `sft/g2_author.py:50`: outer watchdogs limit overruns to short grace periods; authoring stops starting slots three minutes early.
3. **RESOLVED** — `sft/g2_author.py:63`; `sft/g2_tools.py:131`; `sft/g2_chain.sh:147`: cap exhaustion fails, stop reasons survive aggregation, and incomplete authoring is gated before judging or solving.
4. **BLOCKER — PARTLY** — `sft/yield_report.py:276`: judge spend and timestamped attempt discovery are fixed, but earlier authoring reads only the aggregate log. Cancelled or timed-out G2 attempts can leave paid chunk logs without that aggregate, understating earlier spend and making A5 use incomplete accounting. **Fix:** enumerate calls through `g2_tools.calls_files(other)`, which includes chunk logs while excluding aggregate duplicates.

No new major problems found.

Not ready to be proposed for approval because blocker 4 remains partly unresolved.