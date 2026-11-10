Read-only inspection of commits `af1b07e`, `5c0a46e`, and the working tree completed. All six listed files match HEAD. No commands, scripts, tests, or model runs were executed; no files were written.

1. **BLOCKER — r37 #12: PARTLY.** [sft/evallib6.sh:95](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:95): the startup loop allows 120 health requests plus sleeps without checking the remaining chain budget. `bt` bounds the major subprocesses, but startup and identity checks remain outside it. **Fix:** enforce the remaining deadline during startup and identity checks, retaining server ownership in the parent shell.

2. **BLOCKER — r37 #11: PARTLY.** [sft/evallib6.sh:33](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:33) now exits after signals, and background-plus-`wait` addresses deferred traps. However, [line 23](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:23) kills the timeout supervisor after two seconds, preventing its 30-second escalation from killing a TERM-resistant child. The cooperative `sleep` harness does not cover this case. **Fix:** terminate the bounded process group, or preserve the supervisor through escalation and reap it.

3. **FIX — r37 #8: PARTLY.** [sft/evallib6.sh:148](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:148) preserves the shared FK deadline; attempt uniqueness is fixed. But the inner FK `timeout` at [line 132](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:132) has no kill-after interval, so TERM-resistant evaluation can exceed the FK cap until the outer chain deadline. **Fix:** add `-k 30` to the inner timeout and close item 1’s deadline gap.

4. **FIX — r37 #18: PARTLY.** [sft/r6_chain.sh:34](~/Documents/GitHub/Thesis-wt-r6/sft/r6_chain.sh:34) correctly retains the summarizer status. However, [sft/r6_tools.py:74](~/Documents/GitHub/Thesis-wt-r6/sft/r6_tools.py:74) writes an invalid probe report and returns success. The unchanged probe runner also returns success for truncations ([sft/r5_precision_probe.py:77](~/Documents/GitHub/Thesis-wt-r6/sft/r5_precision_probe.py:77)). Thus the chain can announce COMPLETE with invalid probe comparisons. **Fix:** reject invalid probes immediately and make report acceptance require all six valid comparisons.

5. **NOTE — r37 #5: RESOLVED.** [sft/r6_random_adapter.py:71](~/Documents/GitHub/Thesis-wt-r6/sft/r6_random_adapter.py:71) and [the plan:36](~/Documents/GitHub/Thesis-wt-r6/docs/DISTILLATION_ROUND6_DIAGNOSTICS_PLAN.md:36) consistently register spectrum tolerance `1e-4`, describe rounding, and require converted-matrix auditing. **Fix:** none.

6. **NOTE — r37 #6: RESOLVED.** [sft/r6_tools.py:111](~/Documents/GitHub/Thesis-wt-r6/sft/r6_tools.py:111) explicitly requires an additional base run, all attempts, and a drift flag for S0 drift; [the plan:56](~/Documents/GitHub/Thesis-wt-r6/docs/DISTILLATION_ROUND6_DIAGNOSTICS_PLAN.md:56) agrees. **Fix:** none.

7. **NOTE — r37 #9: RESOLVED in the procedure.** [The plan:73](~/Documents/GitHub/Thesis-wt-r6/docs/DISTILLATION_ROUND6_DIAGNOSTICS_PLAN.md:73) explicitly assigns tracked-job completion notice plus chat reporting as the replacement for unavailable PushNotification. **Fix:** none.

8. **NOTE — r37 #14: RESOLVED.** [sft/evallib6.sh:40](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:40) atomically allocates attempts with trailing `${1}.XXXXXX`; directory and redirection failures are checked. This template satisfies BSD `mktemp`’s trailing-X requirement. **Fix:** none.

9. **NOTE — r37 #16: RESOLVED.** [sft/r6_tools.py:150](~/Documents/GitHub/Thesis-wt-r6/sft/r6_tools.py:150) checks the exact tensor-name set; [line 160](~/Documents/GitHub/Thesis-wt-r6/sft/r6_tools.py:160) checks compatible dimensions and compares the full scaled update matrices. **Fix:** none.

10. **NOTE — r37 #19: RESOLVED.** [sft/evallib6.sh:77](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:77) joins continued lines before tokenization and compares the complete serving flags. **Fix:** none.

11. **NOTE — r37 #20: RESOLVED for this source.** [sft/r6_tools.py:137](~/Documents/GitHub/Thesis-wt-r6/sft/r6_tools.py:137) rejects invalid alpha; [lines 154–169](~/Documents/GitHub/Thesis-wt-r6/sft/r6_tools.py:154) suppress floating-point warnings while explicitly rejecting nonfinite factors, products, and errors. The supplied numerical results were not rerun. **Fix:** none.

12. **NOTE — r37 #21: RESOLVED.** [sft/r6_tools.py:85](~/Documents/GitHub/Thesis-wt-r6/sft/r6_tools.py:85) requires all six arms, exact cardinalities, unique non-null task IDs, and zero API errors; invalid summaries exit nonzero. Missing-arm, duplicate-ID, and S0-drift tests are present at [sft/test_r6_tools.py:59](~/Documents/GitHub/Thesis-wt-r6/sft/test_r6_tools.py:59). **Fix:** none.

13. **NOTE — r37 #24 enumeration: RESOLVED.** [The plan:3](~/Documents/GitHub/Thesis-wt-r6/docs/DISTILLATION_ROUND6_DIAGNOSTICS_PLAN.md:3) explicitly defers r36 findings **8, 9, 10, 29, 32, 33, 34**. [sft/r6_chain.sh:9](~/Documents/GitHub/Thesis-wt-r6/sft/r6_chain.sh:9) accepts stage 1 only. **Fix:** none.

14. **NOTE — remaining requested shell checks pass by inspection.** [sft/evallib6.sh:50](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:50) captures `$!` immediately and preserves `wait`’s status. Redirections and `||` lists retain that status; current `bt` calls invoke external programs, so `die()` cannot recurse through `bt`. [The pushd/popd sequence:155](~/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:155) introduces no identified `set -u` hazard. The installed [/usr/share/man/man1/timeout.1:108](/usr/share/man/man1/timeout.1:108) explicitly supports `-k`. Failed FK arms abort; invalid probes remain the exception described in item 4. **Fix:** none beyond the findings above.

Automatic approval review rejected an optional memory read as outside repository scope; it was skipped without blocking this review.

**Not ready to launch stage 1: whole-chain deadline coverage, cancellation cleanup, and invalid-probe success propagation remain incomplete.**