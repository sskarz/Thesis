# Codex review of results log section 19 (G4 diagnostics scored)

## Round 1
- **BLOCKER — [results_log_19.md:2](~/g2dev/g4/results_log_19.md:2):** Omits the pre-registration §6 limit that these 200 tasks also served as the harness tuning set. Without that caveat, the diagnostic readings can appear to support generalization beyond the evidence; the FK-screen caution covers a separate selection issue. **Fix:** Explicitly state that these are post hoc test-set diagnostics with harness tuning leakage.

Not ready until that required limitation is included.
## Round 2 (after the tuning-leakage limit was added)
No major problems found.

Ready: the revised limits address the prior blocker, and the scores and readings match the source files and pre-registration.