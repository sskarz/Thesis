Reviewed HEAD `b655907`; scoped files match HEAD. No commands or tests ran; harness evidence is user-supplied.

1. **NOTE — r39 #1 resolved.** `sft/evallib6.sh:78–80,116–132`: every HTTP call uses the remaining-budget bound. **Fix:** none.
2. **NOTE — r39 #2 resolved.** `sft/evallib6.sh:29–34,62–66`: owned process-group TERM/KILL covers orphaned members; supplied harness evidence covers the prior gap. **Fix:** none.
3. **NOTE — r39 #4 resolved.** `sft/r6_tools.py:37–45,56–100`; `sft/r6_chain.sh:39`: shared validity includes think markers, missing files invalidate comparisons, and COMPLETE requires both checks. **Fix:** none.
4. **NOTE — r39 #5 resolved.** `sft/evallib6.sh:176–181`; `docs/DISTILLATION_ROUND6_DIAGNOSTICS_PLAN.md:41,65`: producer failure records invalidity, FK continues, and the plan matches. **Fix:** none.

Automatic approval review rejected an optional memory lookup outside the repository; it did not affect this review.

Ready to launch stage 1: no remaining launch-blocking findings in the requested scope.