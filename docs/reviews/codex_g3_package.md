- **BLOCKER — sft/g3_arm.sh:35,61:** Smoke trains before checking provenance, and full runs check only the selected arm. The hash-matching `provenance_final_B.json` currently has `pass: false` with five held-out 8-gram hits. B smoke bypasses the hard provenance rule; A can launch despite the registration requiring both gates to pass. **Fix:** resolve B’s failures, refreeze, and require both hash-bound provenance passes before any training, including smoke.

- **BLOCKER — sft/g3lib.sh:20,75:** Sourcing this helper overwrites the requested 34-hour arm cap and 90-minute smoke cap with 16 hours. Its shared `chain_start_epoch` also charges the completed base run and intervening idle time against every subsequent launch, causing premature termination. **Fix:** preserve the caller’s cap and use a fresh clock per attempt.

- **BLOCKER — sft/g3_train.py:137–144:** The 30-hour deadline is checked only after the entire validation sweep. Crossing the deadline during validation permits all remaining validation rows to run; the shell timeout enforces the chain budget rather than the training budget. **Fix:** enforce the training deadline during validation and with an external timeout around the trainer.

- **BLOCKER — sft/g3lib.sh:88:** Preflight requires `thesis/distill-r6-diagnostics`, whereas handoff §4 explicitly requires `thesis/distill-stage0`; the chain therefore requires violating that hard rule. **Fix:** align the branch gate or obtain and record an explicit exception.

Not ready for the smoke and then arm A until these blockers are resolved.