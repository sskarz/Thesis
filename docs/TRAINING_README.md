# Training work: what this branch holds

Branch `thesis/training`. It collects, in one place, the work done after the main paper's experiments: the fine-tuning rounds that did not beat the untuned model, the compact tool-list stack, the synthetic-data pipeline, and the scaled training runs. Scope is unchanged: Gemma-4-E2B-it only, BFCL v4 `multi_turn_base`, one MacBook M4 Pro. The 200 BFCL test tasks have not been used in any step recorded here. `paper/main.tex` and the CSULA thesis sources are not changed by this branch.

**Status (2026-11-10, evening):** data pipeline finished (501 checked synthetic tasks over two batches). Arm A (direct-answer adapter, 2 passes) is training; arm B and a combined retrain are queued; nothing has been scored on the 200 test tasks. A first look at arm A after one pass (37 of 81 on one screening bank) is exploratory and is not the registered screen. Update this line with every merge into the branch.

## What is in the branch

The tree is the result of merging nine work branches in dependency order. The branch is published as a single commit on top of `main`; the detailed commit history of the work branches stays in the author's local repository. Heads merged:

| Source branch | Head merged | Contents |
|---|---|---|
| `thesis/distill-stage0` | `5187288` | Rounds 2 to 5: teacher-student training, harness, findings documents |
| `thesis/compact-list-r81` | `8f62b8a` | compact tool-list mode as first used for Runs 84 and 85 |
| `thesis/slim-port` | `dfdbc2b` | the same mode ported onto the distillation code (used for G2 and G3) |
| `thesis/distill-r6-diagnostics` | `65169a6` | Round 6 diagnostics, G1 overfit gate, G3 base references, arm A and combined-retrain pre-registrations, probe |
| `thesis/distill-data-scale` | `dc6c7a9` | data-yield measurement and factory fix |
| `thesis/g2-scale` | `8beee53` | batch 1 authoring chain and pre-registration |
| `thesis/g2-phase2` | `063b6e1` | batch 1 and 2 solve with reasoning sentences, provenance gate, id-collision repair |
| `thesis/g2b-scale` | `90725fa` | batch 2 authoring chain and pre-registration |
| `thesis/negative-result-draft` | `9ed2f1d` | `docs/TRAINING_RESULTS_LOG.md`, chapter draft `docs/drafts/ch6_appA_distillation_draft.tex` |

### Merge notes (from the local merge)

* `thesis/compact-list-r81` branches from the February base `7e5262c` and conflicted with the distillation lineage in 15 files (`.gitignore`, `CLAUDE.md`, `METHODS.md`, `PLAN.md`, `README.md`, `THESIS.md`, `benchmark/config.py`, `generate_multi_turn.py`, `iffc.py`, `ooda_card.py`, `run_all.py`, `paper/Makefile`, `paper/README.md`, `paper/main.tex`, `paper/references.bib`). All 15 were resolved by keeping the side already in the branch, so no paper file changes. Runs 84 (compact list, 88 of 200) and 85 (full list, 72 of 200) used commit `8f62b8a`; that commit is in the history and is the point to check out to reproduce them. The same mode in the distillation lineage is `dfdbc2b`.
* `thesis/g2-phase2`: one conflict in `docs/DATA_YIELD_G2_PREREGISTRATION.md` (both sides appended after section 8); both texts kept, the older placeholder section is renamed 9a.
* The other six merges were clean.
* Checks on the merged tree: every commit of the nine source branches is reachable; no tracked file under `sft/data`, no `.gguf` or `.safetensors`, nothing over 5 MB; offline render of the opening prompts of all 200 `multi_turn_base` tasks gives identical text and message hashes to the earlier renders in all three modes used (`iffc-state-ooda-slim`, `iffc-state-ooda`, `iffc-ooda-card`); the test suite gives 351 passed, 3 skipped, 2 failed. The 2 failures (`tests/sft/test_cap_nontool_v7.py::test_build_mix_cli_subprocess_nonzero`, `tests/sft/test_filter_empty_after_exec.py::test_train_mlx_prepare_recipe_v7`) also fail on the unmerged source branches because they expect a `/workspace` directory from the earlier cloud environment.

## From a result to its file

`docs/TRAINING_RESULTS_LOG.md` has the full table at its top (result, worktree, branch, commit, attempt directory, report file). In short:

| Result | Where |
|---|---|
| Rounds 2 to 5 (no trained variant beat the untuned model) | `docs/DISTILLATION_ROUND{2,3,4,5}_FINDINGS.md` |
| Round 6 stage 1 (no arm separable from the untuned model; it moved 64 to 68 across a session) | results log section 1 |
| Data factory fix and review samples | results log section 2; `docs/DATA_YIELD_FACTORY_FIX.md`; `docs/reviews/` |
| G1 overfit gate (11 to 32 of 40; pipeline check only) | results log section 3; `docs/DISTILLATION_G1_PREREGISTRATION.md` |
| Batch 1 and batch 2 funnels, 501 tasks | results log sections 4 to 8d; `docs/DATA_YIELD_G2_PREREGISTRATION.md`, `docs/DATA_YIELD_G2B_PREREGISTRATION.md` |
| G3 base references (59, 61 thinking on; 54 off; threshold 70 of 155) | results log section 5; `docs/DISTILLATION_G3_PREREGISTRATION.md` |
| Arm A training and pass-1 probe | results log sections 7 and 8e |
| Combined retrain (approved, not launched) | `docs/DISTILLATION_G3C_COMBINED_PREREGISTRATION.md` |
| Chapter and appendix draft | `docs/drafts/ch6_appA_distillation_draft.tex` |

## What is not in git

* Data, adapters, GGUF files, banks and run logs sit under the gitignored `sft/data/distill/` on the Mac. They are backed up in `~/thesis-backups/2026-11-09` and `~/thesis-backups/2026-11-10` with a SHA-256 manifest (`MANIFEST_sha256*.txt`).
* The screening banks (F, K2), the earlier dev and confirm banks and the BFCL test tasks are held-out material. Their per-task text is not in this branch. Their SHA-256 hashes are recorded in the pre-registrations (F `7a45cd08…`, K2 `e2ac40e4…`; full hashes in `sft/g3lib.sh`) and the frozen training rows have hashes in `sft/g3_expected.sha`.
* The OpenRouter API key and the commands that use it stay in the Mac's environment. Costs are in the results log (ledger total $23.22; the key's own counter reads $31.68, with the gap unexplained there).

## Limits that apply to every number here

Single seed; greedy decoding; the screening banks share their author with the training data, so only the 200 test tasks can settle a claim; the 200-task result needs the paired test used in Chapter 5 and a separate decision; statistics wording is never stronger than the record. Reviewers of the synthetic tasks are models, not people, on 40-task samples.
