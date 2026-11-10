**Verdict: REVISE. A0 is not complete with only the K2-inclusive re-audit outstanding.** The reported counts mostly reproduce, but token-mask accounting, guard coverage, selection semantics, and manifest completeness need fixes.

Read-only review completed; no model runs or file writes. F inspection and reporting remained aggregate-only.

1. **Token counts — VERIFIED for rendered continuations; WRONG as an exact account of MLX supervision.**

   Independent tokenizer recomputation gives:

   | Split | Rows | Reasoning | Separator | Answer JSON | Reported termination | Total |
   |---|---:|---:|---:|---:|---:|---:|
   | Train | 320 | 92,118 | 320 | 16,319 | 640 | 109,397 |
   | Valid | 35 | 11,538 | 35 | 1,618 | 70 | 13,261 |

   Every prompt is an exact prefix of its full render. Every answer segment decodes to the complete recorded assistant content, and all 355 answers parse as JSON. The answer delimiter is therefore correct for these files. Maximum full lengths are **8,928/8,950**, below 9,500.

   Two corrections are required:

   - **“Termination tokens” are actually `<turn|>` plus an ordinary newline:** 320 turn-end tokens + 320 newlines in train; 35 + 35 in valid. Rename this segment “template suffix” or separate the two. Replace the broad special-token heuristic with explicit expected boundaries. See [token_segments_audit.py:29](~/Documents/GitHub/Thesis-wt-distill/sft/token_segments_audit.py:29).
   - **The installed MLX trainer supervises one additional `<pad>` target per row.** Its batch iterator pads beyond the full sequence, while its shifted loss mask uses `steps <= length`. Consequently, current training masks cover **109,717 train / 13,296 valid targets**, rather than the reported continuation totals. See [trainer.py:94](~/Documents/GitHub/Thesis/.venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py:94). Audit this explicitly and resolve it before implementing Sw’s masks or claiming equal-loss reproduction.

   The required `diag_mlx` explanation is also incomplete. Its separator is **included in reasoning NLL and excluded from answer NLL**, while `n_reason` excludes it and `n_answer` includes it. It is not excluded from all NLL. Document that exact mismatch. See [diag_mlx.py:38](~/Documents/GitHub/Thesis-wt-distill/sft/diag_mlx.py:38).

2. **Declared Stage A subsets — VERIFIED.**

   Independent history/error detection and verdict aggregation reproduce:

   | Arm | Error in either base: 27 tasks | Error-free in both: 54 tasks |
   |---|---:|---:|
   | B1 | 4 | 31 |
   | B2 | 4 | 31 |
   | H | 8 | 27 |
   | AB | 4 | 26 |

   B1/B2 error-task counts are **26/25**, with union **27**. Each arm has 81 unique verdict tasks and zero API-error episodes. The saved subset output and Round 5 runlog entry are accurate. The [new tail logic](~/Documents/GitHub/Thesis-wt-distill/sft/stageA_diag.py:63) implements the declared union correctly.

3. **Manifest — VERIFIED hashes; INCOMPLETE pins.**

   All **18 listed file hashes** match. The bar GGUF, HF tokenizer/template, server binary hashes, and recorded package versions also match. The local llama.cpp checkout describes itself as **`b10894-0-gd344123fe`**.

   Required additions to [manifest.json](~/Documents/GitHub/Thesis-wt-distill/sft/data/distill/r5/manifest.json:1):

   - Explicit server build string and HF→GGUF conversion source/hash.
   - Local MLX tokenizer configuration and model/config provenance. Its `tokenizer_config.json` **differs from HF’s**, although tokenizer JSON and template match.
   - Resolved server/evaluation argv, harness mode, generation limits, and LoRA scale. The hashed serving script pins default server flags indirectly, but not the complete invocation.
   - The A adapter corresponding to the recorded A GGUF, plus audit input hashes: labels, source tasks, exclusion lists, banks, BFCL text, and guard dependencies.

   `worktree_head` is an earlier commit than current HEAD; label its snapshot context and distinguish it from the harness revision used for the historical runs.

4. **Eligible-state arithmetic — VERIFIED under the script’s first-occurrence dedupe order.**

   Raw recomputation reproduces:

   - **757** non-kept labels; **22** declared-task exclusions; **712** duplicates.
   - **1,723 kept states / 224 tasks** after those operations.
   - **497 eligible / 184 tasks = 166 recovery + 331 correction-only.**
   - Guard drops remove **9 states across 2 tasks**, leaving **488 / 182 = 161 recovery + 327 correction-only**.

   **Difference from r27 explained exactly:** effect-equivalence correction gives **462 / 167 tasks = 166 + 296**. Canonical correction adds **35 non-error states** whose actions differ canonically but whose labels mark them return-equivalent. This follows v2.1’s changed correction definition.

5. **Eligibility/order — predicate VERIFIED; operational selection needs clarification.**

   The code uses `keep AND (after_error OR canonical(label) != canonical(student_calls))`, preserves call order, and canonicalizes argument-key order.

   However, [dedupe happens before eligibility](~/Documents/GitHub/Thesis-wt-distill/sft/r5_state_audit.py:67), retaining the first old/new/G occurrence. This matters: applying eligibility first produces **500 / 185**, or **491 / 183 after the same drops**. Three qualifying duplicate occurrences are otherwise suppressed by earlier nonqualifying representatives.

   Explicitly register the representative/dedupe rule before freezing selection. Also sort states by the required `(task_id, episode_id, step)`; current iteration order is not sorted. Sorting before dedupe happens to retain the current 497 count.

   The kept pool contains **16 prompts with conflicting targets**. Enforce the registered conflicting-label exclusion before pair use. The provisional drop file contains only the **two newly hit tasks**; downstream consumers must also retain the original two exclusions.

6. **Guard findings — hits VERIFIED; “faithful superset” is WRONG.**

   Independent checks reproduce **1 BFCL history 8-gram hit + 2 held-out history 8-gram hits**, affecting exactly the saved two source tasks. Source-task ID/signature/8-gram/ratio/BFCL checks find no additional hits.

   Current content and tool-result text are scanned; `reasoning_content` is explicitly included. However, the implementation omits required protections:

   - Actual-row plain-user/OODA-goal similarity against held-out banks and BFCL.
   - Serialized excluded-BFCL-ID scanning beyond source task IDs.
   - Duplicate-ID rejection and several missing-text/gold/input-validity checks provided by `check_provenance`.
   - Complete rendered-field handling, including the `reasoning` alias and structured tool fields. Appending an empty assistant also does not audit the teacher target.

   Supplemental checks found **zero current row-ratio hits** and **zero serialized BFCL-ID hits**. Thus these gaps do not change the recomputed 488 today, but must be repaired before certification. Current eligible histories have only `role`/`content` fields, so separate reasoning-field handling is not exercised by this inventory.

   K2 must later enter **all** bank guards—including ID and signature checks—not just text grams. Final training provenance must additionally cover both preference branches, rendered reasoning, and train plus validation rows.

The Round 5 runlog accurately transcribes the saved outputs, but needs the token-accounting qualifications above. **488/182 is a reproducible provisional inventory, not yet a certified training freeze.** Fix these A0 issues, then perform the registered K2-inclusive source-history and final-row audits before A2 training.