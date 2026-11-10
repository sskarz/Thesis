Read-only file inspection completed. No shell commands, project scripts, tests, or model runs were executed; no files were written.

1. **NOTE — r36 #4: RESOLVED.** `docs/DISTILLATION_ROUND6_DIAGNOSTICS_PLAN.md:25,47,59` describes the observed exchanges and explicitly rejects a noise-interval interpretation. **Fix:** none.

2. **NOTE — r36 #5: RESOLVED.** `sft/evallib6.sh:3,134` exports `PYTHONDONTWRITEBYTECODE=1` and invokes the R5 probe with `-B`, an R5 working directory, and an absolute R6 output. **Fix:** none.

3. **NOTE — r36 #6: RESOLVED.** `sft/evallib6.sh:92–101` compares canonical model/adapter paths, exact build, four slots, adapter ID and scale; `:52` checks the registered model-hash prefix. **Fix:** none for the identity predicates.

4. **NOTE — r36 #7: RESOLVED.** `sft/r6_random_adapter.py:109` rewrites `adapter_path`; `sft/r6_chain.sh:20–23` provides concrete build, conversion and audit commands. **Fix:** none.

5. **FIX — r36 #11: PARTLY.** `sft/r6_random_adapter.py:69–88,111` audits the saved FP32 artifact and its simulated FP16 product error. However, the spectrum tolerance is `1e-4` at `:85`, whereas the audit docstring promises `1e-5`; the converted GGUF receives only a product-norm check. Plan `:36` also still says “exactly” despite serialization error. **Fix:** register one spectrum tolerance, describe approximate preservation, and audit the converted factors explicitly.

6. **FIX — r36 #15: PARTLY.** `sft/r6_chain.sh:26–31` now brackets stage 1 with B3/B3b, and `sft/r6_tools.py:102–105` flags base/S0 drift. But plan `:56` says the same additional-base disposition applies to S0 drift; the S0 flag merely says “drift flag.” **Fix:** make the S0 follow-up and mandatory post-stage procedure explicitly require that additional base run.

7. **NOTE — r36 #16: RESOLVED.** `sft/r6_tools.py:37–72` validates B3 cardinality, unique keys, errors and truncations, checks matching keys/rendered hashes, and suppresses difference counts when either side is invalid. **Fix:** none for reference-validity propagation.

8. **FIX — r36 #17: PARTLY.** `sft/evallib6.sh:123–129` correctly shares one FK deadline between F and K2. Whole-chain enforcement and attempt uniqueness remain incomplete, as detailed below. **Fix:** bound every phase by the remaining chain deadline and allocate attempts atomically.

9. **FIX — r36 #19: PARTLY.** R5 bytecode isolation is implemented, and plan `:7` records the branch exception. However, plan `:72–73` specifies chat messages, while handoff `:44` requires **PushNotification** for results, stopping rules and decisions. **Fix:** explicitly assign and require that notification mechanism in the launch/completion procedure.

10. **NOTE — r36 #22: RESOLVED.** Plan `:71` has correct arithmetic:
    - Stage 1: six × 55–60 minutes = **5h30m–6h**, plus the labelled one-hour guess = **6h30m–7h**.
    - Both stages: eight × 55–60 minutes = **7h20m–8h**; plus 1h30m training and 2–3h allowance = **10h50m–12h30m**, reasonably rounded to **11–12.5h**.
    
    The historical timings match `docs/DISTILLATION_ROUND5_FINDINGS.md:79` and `docs/DISTILLATION_RUNLOG.md:147`. The **6 gains/7 losses** citation matches `docs/reviews/codex_r32.md:30`. Plan `:24` now distinguishes registered choices from measurements. **Fix:** none.

11. **BLOCKER — r36 #23: PARTLY.** `sft/evallib6.sh:21–30` removes broad killing and tracks the launched PID, but `trap stop_server EXIT INT TERM` cleans up without terminating on INT/TERM. Bash can resume execution afterward, allowing subsequent commands or arms to start. **Fix:** use separate INT/TERM handlers that exit with signal-appropriate status, retaining ownership cleanup on EXIT.

12. **BLOCKER — r36 #24: PARTLY.** `sft/evallib6.sh:111,123–129` enforces the FK budget, but the **16-hour whole-chain cap is not a hard deadline**. Self-tests, adapter construction/conversion/audits, probes, and several HTTP requests are unbounded (`sft/r6_chain.sh:19–24`; `sft/evallib6.sh:76–92,134`). Checking remaining time before an arm and before each bank cannot stop those phases at the deadline. **Fix:** apply the remaining chain budget to every blocking phase, with bounded HTTP calls and forced termination after a grace period.

13. **NOTE — r36 #25: RESOLVED.** `sft/evallib6.sh:50–52,80,92–101` replaces substring identity checks with the registered hashes, listener ownership, canonical paths and exact structured identities. **Fix:** none for those checks.

14. **FIX — r36 #26: PARTLY.** `sft/r6_chain.sh:12–14` redirects before preflight, so ordinary preflight output and abort messages reach the attempt log. However, `sft/evallib6.sh:34–37` uses second-resolution timestamps and `mkdir -p`: two launches within one second reuse an attempt directory. Directory creation and redirection failures are also unchecked. **Fix:** create the attempt directory atomically with `mktemp -d` and abort if allocation or log redirection fails.

15. **NOTE — r36 #28: RESOLVED.** `sft/evallib6.sh:3,134` supplies both required bytecode protections for the R5 invocation. **Fix:** none.

16. **FIX — r36 #30: PARTLY.** Saved keys, shapes, FP32 dtype, finite factors, spectra and FP16 matrix error are checked at `sft/r6_random_adapter.py:69–88`. The GGUF audit at `sft/r6_tools.py:133–152` checks only source-derived pairs and scalar product norms; it does not require an exact GGUF tensor set or verify converted matrix error. **Fix:** validate the exact expected tensor names/shapes and compare converted products against the source, allowing any documented conversion permutation.

17. **NOTE — r36 #35: RESOLVED.** `sft/r6_tools.py:48,60–72` invalidates comparisons against invalid or mismatched B3 output. The self-tests cover invalid reference, truncation and rendered-hash mismatch (`sft/test_r6_tools.py:33–44`). **Fix:** none for the original reference bug; tests were inspected, not run.

18. **FIX — r36 #36: PARTLY.** Plan `:72–73` now assigns review, reporting and follow-up responsibilities, and stage 1 has a closing base. But report failures do not fail the chain: `sft/r6_chain.sh:33` merely prints a message, and `:34` replaces the summarizer’s exit status with `cat` before announcing completion. **Fix:** require successful generation and validation of both reports before declaring completion, and retain the explicit notification/follow-up obligations.

19. **BLOCKER — preflight always rejects the actual serving script.** [sft/evallib6.sh:58](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6/sft/evallib6.sh:58) selects only physical line 6 of `sft/serve_gguf.sh`. That line ends with a continuation backslash, so `shlex.split` raises “No escaped character”; even removing that backslash leaves the flags on line 7 absent. **Fix:** join the continued command before tokenizing, then compare the complete argument vector.

20. **FIX — the matmul warning cannot yet be dismissed safely.** `sft/r6_tools.py:136–148` uses the correct `blk.N.attn_q.weight.lora_a/b` names for the inspected S0 q-only layers **19–34**. The reader/converter orientation makes the first compatible product `gb @ ga` appropriate, and `alpha/r` supplies the expected scaling. However, only the GGUF product norm is checked for finiteness: nonfinite source products, alpha or scaled results can produce NaN relative errors that `max(worst, nan)` silently fails to propagate. A warning alone does not prove corrupted results, but this implementation cannot certify it harmless. **Fix:** reject nonfinite inputs, metadata, products and errors explicitly, and investigate any remaining warning before accepting the audit.

21. **FIX — summary completeness is weaker than run validity.** `sft/r6_tools.py:85–108` discovers whatever F directories exist, does not require all six registered arms, and does not check unique task IDs. An empty attempt can produce “no follow-up flags”; duplicate verdicts can appear complete. The normal `run6` path checks uniqueness, but the standalone review artifact does not preserve that guarantee. **Fix:** validate the expected arm set, both banks, unique IDs and API-error counts, and exit nonzero on incomplete/invalid reports.

22. **NOTE — remaining requested shell checks pass for the current calls.** `sft/r6_chain.sh:8–11` supplies arguments safely under `set -u` and sources the helper after the correct `cd`. Bash `$!` at `sft/evallib6.sh:73` captures the background `nohup` job PID; `nohup` executing llama-server under that PID is expected, so the listener equality check at `:80` is valid. `ps -o command= -p` is valid; `exec >>` preserves the EXIT trap, and `die` clears ownership before EXIT cleanup runs again. `min2`, `chain_left`, and FK remaining-time arithmetic are sound for valid numeric state. **Fix:** optionally use `ps -ww` for complete argv capture and validate persisted clock contents.

23. **NOTE — evaluator invocation and working-tree gate are understood correctly.** `sft/evallib6.sh:111–118` matches R5 `benchmark/distill_student_eval.py:44–58,65–78`; the R6 evaluator file is identical. All requested flags exist; `verdicts.jsonl` contains `task_id`, `valid`, and `api_error`. `passed` is a report aggregate, not the current per-verdict success field; its fallback is unnecessary but does not alter these verdicts. At `evallib6.sh:49`, untracked review files are ignored, while the stated uncommitted tracked v3 changes must be committed before launch. **Fix:** commit the reviewed tracked files after corrections; no exception is needed for untracked reviews.

24. **NOTE — stage-2 deferral is effective.** Plan `:3` withholds stage-2 authorization and names the outstanding provenance, exposure/truncation and prompt-parity work. This covers r36 **#8, #9, #10, #29, #32, #33 and #34**. `sft/r6_chain.sh:9,19–35` accepts only stage 1 and contains no self-target generation, provenance invocation, D3 training, S0-self evaluation or B4 execution. **Fix:** explicitly enumerate all deferred finding numbers in the plan for audit clarity; stage 1 cannot reach those operations.

**Not ready to launch stage 1: preflight is broken, signal handling can resume the chain after cancellation, and the registered whole-chain deadline remains unenforced; the audit and reporting gaps also need correction before results are accepted.**