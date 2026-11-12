1. **RESOLVED** — Both provenance reports pass, match the registered row hashes, and are required before smoke or full training.
2. **RESOLVED** — Each arm resets its clock; its cap is assigned after sourcing the helper.
3. **PARTLY — BLOCKER — sft/g3_arm.sh:89:** Validation checks the deadline, but the external timeout is 109,800 seconds: a stalled GPU operation can retain the GPU for 30 minutes beyond the registered 30-hour cap. **Fix:** use 108,000 seconds with the existing 60-second kill grace.
4. **RESOLVED** — The retained branch satisfies your clarified prefix rule.

No new major problems found.

Not ready to launch arm A until the remaining training-timeout gap is closed.