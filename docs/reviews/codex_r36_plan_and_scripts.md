**Not ready to launch stage 1.** V2 fixes most source and interpretation problems, but server ownership, identity checks, runtime caps, and attempt preservation remain unenforced. Review was read-only; no tests or scripts were run and no files were written.

The 19 r35 findings, in their original order:

| # | Tag / status | Evidence and one-line fix |
|---|---|---|
| 1 | **NOTE — RESOLVED** | Plan `23–27,42` retains the verified Round 5 counts. **Fix:** none. |
| 2 | **NOTE — RESOLVED** | Plan `25,37`; `r6_chain.sh:53–56` explicitly preserves S0’s configuration and selects the final adapter at `57`. **Fix:** none for the configuration specification. |
| 3 | **NOTE — RESOLVED** | Plan `24` correctly cites `codex_r32.md:30`; `66` correctly attributes generation timing and labels its estimate `[analysis]`. The unsupported “15 of 15” assertion is gone. **Fix:** none for these attribution corrections. |
| 4 | **FIX — PARTLY** | Plan `46–48` removes the calibrated SD and calls the bands descriptive, but `58` still presents approximately five tasks as an observed noise-sized difference. **Fix:** describe the actual observed exchanges without implying a repeatability interval. |
| 5 | **FIX — PARTLY** | Plan `20`; `evallib6.sh:2–9,58`; chain `49–51` resolve inputs and explicit outputs correctly. Python imports can nevertheless create caches in R5. **Fix:** disable bytecode writes for every R5 invocation. |
| 6 | **FIX — PARTLY** | `evallib6.sh:27–37` checks adapter ID and scale, but matches only a filename substring and accepts either build substring. **Fix:** require the canonical adapter path and exact expected base/build identity. |
| 7 | **FIX — PARTLY** | Chain `21–23,53–59` supplies complete conversion/training commands. `r6_random_adapter.py:87` copies S0’s stale `adapter_path`. **Fix:** rewrite artifact paths in the generated config. |
| 8 | **BLOCKER — PARTLY** | Plan `38` and chain `48–51` add a real provenance gate, but its BFCL and similarity checks still omit generated assistant reasoning/answers. **Fix:** extend those checks to the new assistant spans before stage 2. |
| 9 | **FIX — PARTLY** | Plan `37,50,54` adds retention handling and weaker interpretation, but overlength handling and promised exposure reports remain incomplete. **Fix:** register truncation handling and implement the exposure audit before training. |
| 10 | **FIX — PARTLY** | Plan `37` correctly describes Q4-to-BF16 self-imitation and nonzero gradients; prompt/template parity is not verified beyond marker count. **Fix:** persist and compare rendered prompts/tokenization under the actual training and serving templates. |
| 11 | **FIX — PARTLY** | Plan `35`; random builder `51–66,88–94` implements the spectrum-based construction and norm gates. Actual serialized spectra and FP16 perturbation errors are not checked. **Fix:** audit singular values and matrix error on the real artifacts. |
| 12 | **NOTE — RESOLVED** | Plan `36,53` correctly limits D2 to inference-time attenuation and tested scales. **Fix:** none. |
| 13 | **NOTE — RESOLVED** | Plan `16,44–58` replaces causal conclusions with descriptive readings and retains alternative explanations. **Fix:** preserve those limitations in reporting. |
| 14 | **NOTE — RESOLVED** | Plan `48,56` supplies exhaustive bands and an inconclusive default. **Fix:** none. |
| 15 | **FIX — PARTLY** | Plan `34,39,55` adds contemporary S0 and bracketing bases, but the chain never evaluates the drift rule or schedules the additional base run. **Fix:** implement the registered disposition or explicitly make it a mandatory reviewed follow-up. |
| 16 | **FIX — PARTLY** | Plan `40`; `r6_tools.py:21–46` adds the reporter, fresh B3 reference, and validity flag, but comparisons can use invalid B3 results. **Fix:** validate the reference and propagate invalidity to comparisons. |
| 17 | **FIX — PARTLY** | Plan `62–66` separates conditional work, but the composite and whole-chain caps are not enforced, and attempts overwrite previous artifacts. **Fix:** enforce deadlines and use immutable attempt directories. |
| 18 | **NOTE — RESOLVED** | Plan `5,7,58,72` preserves screening limits, protected-bank approval, recipe scope, and no paper edits. **Fix:** none. |
| 19 | **FIX — PARTLY** | Plan `7` incorporates the hard rules and branch exception; `codex_review.sh:8–9` matches the review command. R5 write isolation and notifications remain incomplete. **Fix:** enforce isolation and explicitly arrange the required notifications. |

The numerical recheck:

20. **NOTE — Copied numerical facts now match their sources.** Plan `23–27,42,66` agrees with the following:

   - `DISTILLATION_ROUND5_FINDINGS.md:19,21,23,40–44`: **81+74=155**, K2 SHA, base **68/67**, mean **67.5**, S0 **59=29+30**, Sw **64=33+31**, first screen **≥77 and >68**, repeat mean **≥76.8** with each **>68**.
   - `codex_r32.md:30`: **6 gains + 7 losses = 13 discordant tasks out of 155**.
   - `DISTILLATION_RUNLOG.md:147`: **2,832 generations in 3h28m**.
   - `DISTILLATION_ROUND5_FINDINGS.md:79`: **55–60 minutes per FK arm**, **approximately 1.5 hours S0 training**.
   - R5 `adapter_config.json:3–26,38`: rank **8**, scale **2**, dropout **0**, layers **16**, batch **1**, accumulation **4**, **320** iterations, LR **2e-5**, seed **61**, maximum sequence **9500**, Adam, prompt masking and checkpointing. **320/4=80** updates. Effective q-only layers **19–34** and **524,288** parameters are supported by findings `35`.
   - Findings `27,35,60,73`: **70 prompts/36 tasks**, **315+35 rows**, **99/472 states**.
   
   **Fix:** retain these corrected citations.

21. **NOTE — Band arithmetic is sound.** Plan `46–48`: **56–62 = 59±3**; **65–70 = 67.5±2.5**. The bands cover every integer composite from **0 through 155**, without gaps or overlaps. The extra `>68` in band F is redundant. Seed **6061**, scales **0.5/0.25**, retention **95%**, and tolerances **1e-5/1e-2** are newly registered choices, rather than historical measurements. **Fix:** distinguish registered choices from copied measurements in line `5`.

22. **FIX — The `[analysis]` time total is an unvalidated planning estimate.** Plan `66`: seven FK runs take **385–420 minutes = 6h25m–7h**; adding training gives **7h55m–8h30m** before adapter construction, probes and generation. The proposed **10–12 hours** assumes an unmeasured allowance for those operations. **Fix:** state that allowance explicitly; do not present the historical generation rate as validation of this workload.

The shell review:

23. **BLOCKER — Cleanup can kill a server the chain does not own.** `evallib6.sh:11`; `serve_gguf.sh:4`; chain `31,40,64` use broad `pkill`. `abort()` itself kills nothing, but preflight is only a point-in-time check: another server can start afterward, including before the first build failure invokes `die()`. **Fix:** launch and track the server PID/process group, verify ownership, and use ownership-scoped cleanup traps throughout.

24. **BLOCKER — Registered runtime caps are false as implemented.** Plan `64`; `evallib6.sh:43,64–65`: each bank gets 9000 seconds, allowing **five hours per FK composite**. No code enforces **16 hours** across the two stages. Chain `56` correctly enforces **3.5 hours** training. **Fix:** share a composite deadline across F/K2 and persist a whole-chain deadline across stages.

25. **BLOCKER — Server identity checks are weaker than the registered checks.** `evallib6.sh:20,27–37` accepts a basename substring for the model/adapter and either `10894` or `d344123fe` anywhere in props; the base file’s named SHA is not verified. **Fix:** parse structured props, compare canonical paths and exact build identity, and verify the pinned model hash before evaluation.

26. **FIX — Attempts are overwritten rather than preserved.** Chain `22,32,39,56,65` and `evallib6.sh:18,43,58` reuse output names; evaluator `benchmark/distill_student_eval.py:44,61` overwrites verdicts/report. Preflight also precedes the main log redirection at chain `17`, leaving abort details outside that log. **Fix:** allocate a unique attempt directory before preflight and log every attempt there.

27. **NOTE — The requested argument/path cases are otherwise valid for the current constants.** Chain `25–27` supplies `serve6 NAME SCALE [LORA]` correctly; `${3:-}` is safe under `set -u`. `set -- $spec` currently splits the three fixed, space-free fields correctly, but permits word splitting/globbing. Relative LoRA paths work because `serve_gguf.sh` does not change directory; `sed -i ''` at chain `54` is valid macOS syntax. **Fix:** replace encoded specifications with explicit function calls or safe field parsing, quote expansions, and reject unknown stages before preflight.

28. **FIX — R5 working-directory commands resolve correctly, but do not guarantee read-only execution.** Chain `49–51` uses accepted provenance arguments: absolute R6 rows/output and valid R5-relative held-out/source paths. `evallib6.sh:58` correctly gives the probe its R5 working directory and absolute R6 output; the probe opens only its specified output. Imported R5 modules can create `__pycache__`. **Fix:** use `PYTHONDONTWRITEBYTECODE=1` or `python -B` for these invocations.

29. **FIX — The retention gate does not enforce the registered input dataset.** Chain `42–45` checks retention against whatever input count the manifest reports, without asserting **315 train/35 valid** or pinned source hashes. For the registered **350**, the gate requires at least **333 retained**. **Fix:** verify split counts and expected hashes before generation and before interpreting retention.

The Python review:

30. **FIX — D1 preserves spectra algebraically, approximately after serialization.** `r6_random_adapter.py:51–57` correctly constructs random orthonormal factors with S0’s nonzero singular values. Casting to FP32 and then FP16 prevents literal exact preservation. Lines `59–64` check only **relative Frobenius-norm change**, not `||P16−P32||/||P32||`; matching norms can conceal matrix error. The synthetic spectrum check at `test_r6_random_adapter.py:49–59` does not audit actual S0 output. **Fix:** enforce finite values, complete pair/key/shape preservation, singular-value tolerances, and matrix error on the real FP32/converted artifacts.

31. **NOTE — MLX convention and conversion agree.** Installed `mlx_lm/tuner/lora.py:88–98` implements **A[in,r], B[r,out]**, adding `scale*((x@A)@B)`. Converter `sft/mlx_adapter_to_gguf_lora.py:26–46` transposes the factors and sets **alpha=scale×rank**, so it supports the random builder’s output. **Fix:** only correct the copied config’s stale output path; no orientation change is needed.

32. **NOTE — Self-target marker and response handling match the stated policy.** `r6_self_targets.py:52–89` strips the system marker only for requests, asserts one rendered marker, sends no `max_tokens`, keeps legitimate `[]`, drops the registered response categories, and preserves the stored marker through copying. Prior assistant turns—including any `reasoning_content`—are preserved unchanged. The current source rows contain no prior assistant reasoning fields; future inputs could. **Fix:** register that history policy and verify rendered parity rather than silently stripping history.

33. **BLOCKER for stage 2 — Appending reasoning does not implement the promised provenance coverage.** `r6_tools.py:81–83` appends reasoning to assistant content, but R5 `check_provenance.py:154–183` applies similarity/BFCL task checks to user/source turns; `distill_exclusions.py:249–258` likewise scans user content. Held-out 8-gram checking at `check_provenance.py:150–151` already includes reasoning. Its output at `198` can also expose task-level failure identifiers rather than aggregate-only findings. **Fix:** scan generated answers/reasoning with all required text checks and emit only aggregate review output.

34. **FIX — Exposure and truncation remain uncontrolled.** Self-target builder `72–94` reports reasoning-character median/max, not the promised target totals for both datasets. Installed trainer `151–170` truncates overlength sequences at **9500**, and `112–145` sorts/shuffles using sequence lengths. Thus regenerated lengths can change supervised exposure and sampling order even with unchanged prompts/settings. **Fix:** audit prompt/reasoning/answer token lengths, effective supervision and truncations before training, then apply a registered disposition.

35. **FIX — Probe comparisons can remain interpretable-looking with an invalid reference.** `r6_tools.py:22–46` does not validate B3’s cardinality, unique keys, errors or truncations before comparisons; missing reference keys are silently skipped. A clean arm can consequently be marked valid while its comparison uses truncated or incomplete B3 output. **Fix:** validate B3 and matching prompt identities first, and invalidate comparisons whenever either side is invalid.

The remaining design/handoff concerns, in priority order:

36. **FIX — Session validity and review handoff need an executable disposition.** Plan `55` requires an additional base run after drift, but neither stage computes composites or applies that rule; stage 1 ends without its own closing base. Plan `71` and handoff `44` require step reviews and notifications, while the chain only prints completion messages. **Fix:** explicitly define the post-stage reviewer’s mandatory aggregate checks, drift follow-up and notification procedure before authorizing the next stage.

37. **NOTE — The design now supports limited descriptive conclusions.** Plan `50–58` appropriately retains single-seed, direction, exposure, precision and run-noise alternatives. D1 cannot establish general update harm, and D3 cannot isolate teacher content without stronger exposure matching. **Fix:** retain these limits even if an arm lands neatly in a registered band.

**Final verdict: not ready to launch stage 1, because server ownership, exact identity validation, registered runtime caps, and preservation of attempts still need enforcement; stage 2 separately requires repaired provenance coverage and exposure/truncation handling.**