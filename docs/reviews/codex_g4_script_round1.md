- **BLOCKER — sft/g4_chain.sh:113:** `res=$ATT/results_$name` expands before the same `local` statement assigns `name`. Reproduced under `/bin/bash -u`: `name: unbound variable`. The chain exits before U1.
  **Fix:** Declare `name` first, then assign `res` in a separate statement.

- **BLOCKER — sft/g4_tools.py:62:** Error counting excludes the harness’s `iffc_unparseable:turn_*` flags. More than 10 unparseable tasks can therefore be accepted as a valid run, violating plan §2; these tasks also are not explicitly excluded from `passed_ids`.
  **Fix:** Include unparseable tasks in the distinct errored-task count and force their verdicts to fail.

- **BLOCKER — sft/g4_chain.sh:119:** The evaluator inherits `BFCL_SERVER_URL`, whereas server identity checks use port 8080. An existing override sends BFCL requests to another server, bypassing the checked model, adapter, scale and build.
  **Fix:** Explicitly set `BFCL_SERVER_URL=http://127.0.0.1:8080` for evaluation and bind parity capture to that endpoint.

- **BLOCKER — sft/g4_chain.sh:55:** The scorer preflight checks historical verdicts but never checks the registered checker version or SHA-256, although `g4_rescore.py` reports both. A changed checker that agrees on Runs 84/85 passes preflight and can score new outputs differently.
  **Fix:** Require the exact registered version and checker hash alongside the existing verdict checks.

Not ready for unattended launch until these blockers are fixed.