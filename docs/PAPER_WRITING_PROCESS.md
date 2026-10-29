# How the thesis paper was written (handoff)

Handoff for any agent or person continuing work on `paper/main.tex`. Last updated 2026-10-19 (America/New_York). The latest compiled PDF is [`Thesis_Paper.pdf`](../Thesis_Paper.pdf) at the repo root (same file as on Drive).
Facts below come from the git history of this repo, PRs #66/#84/#88/#89/#90/#91, the review files listed in §5, and the agent workstream that produced the October rewrite. Anything not verifiable from those sources is marked **(uncertain)**.

---

## 1. Purpose and audience

- Master's thesis paper by Sanskar Thapa, written for his research professor (advisor review), with limitations stated plainly.
- Presentation: end of November 2026. Thesis due: December 2026.
- Readers include people unfamiliar with the project, so every term must be defined and figures must be readable on their own.

## 2. Current state (main @ `a445429`, 2026-10-14)

- File: `paper/main.tex`. Title: **"State, Not Syntax: Harness Optimization for Small-Model Multi-Turn Tool Calling"**.
- 16 pages after the shorter-list revision (main text through the conclusion about pp. 1–11, references about p. 12, appendices about pp. 13–16). The October 2026 snapshot this section used to describe was 15 pages, then 14 after the focus fixes, then 16 with this result. Builds with 0 overfull boxes and 0 undefined references.
- Scope: **Gemma-4-E2B-it (Q4_K_M) only**, BFCL v4 `multi_turn_base`, N = 200, llama.cpp `llama-server` on a MacBook Pro M4 Pro 48 GB.
- Core claim: multi-turn failure is about state tracking and the act-vs-abstain decision, not output syntax; harness-side control gains +8.0 pp over JSON-text calling, while forcing valid syntax and fine-tuning both lose ground. A shorter tool list on that harness scored 84/200 against a 71/200 rerun (13 more tasks, p=0.053): no detectable effect. The first-run 77/200 stays beside that rerun. State-not-syntax remains the main finding. The shorter list is not in the abstract.

### Headline numbers and provenance

"Run N" = internal run id (`runs.id` in `benchmark.db`); see the run-ID map in Appendix D (`tab:runs`) of the paper.

| Result | Value | Source |
|---|---|---|
| Full stack (IFFC + STATE + gated stop-on-repeat + status card), first run | 77/200 = 38.5% (Wilson 32.0–45.4) | Run 78, `scores.id` 51; log `sft/v13/paper_evals/runs/b_fullstack_mtb200.log`; harness code from PR #90 (`344ce45`, merged 2026-10-14 as `83f35ee`) |
| Full stack, fresh rerun, original tool list | 71/200 = 35.5% (Wilson 29.2–42.3) | Run 82, `scores.id` 61, log `R81_slim/full_off.log`. Same count as Run 42, so the paper uses the same one-decimal Wilson interval. An earlier line wrote 29.2–42.4; z=1.959964 gives 29.1986–42.3479, which rounds to 29.2–42.3. Against JSON text (Run 5): +5.0 pp, 27/17, p=0.174. |
| Shorter tool list on the full stack (keep every tool; each group's shared description once; schemas without indentation) | 84/200 = 42.0% (Wilson 35.4–48.9). **No detectable effect.** | Run 83, `scores.id` 62, log `R81_slim/full_on.log`. Paired vs Run 82: 26 / 13 / 58 / 103. Exact McNemar 0.053252, reported as p=0.053, above the 0.05 level set in the plan. 13 more tasks, not a gain and not a new best. Opening-prompt median 4,923.5 to 2,888 tokens (41%, exact 41.3%), from `R81_slim/*.metrics.jsonl`. Mean context per request 4,990 vs 7,107 (about 30%). Smoke: Run 80 (`scores.id` 59, `smoke_off.log`) 5/10; Run 81 (`scores.id` 60, `smoke_on.log`) 8/10; calls differed on tasks 55, 69, 26, 179, 59; gate required 0/10 (GATE FAIL), so the comparison run changed from Run 78 to Run 82. A second run of both lists is planned; pair-2 and pooled cells stay empty. |
| Minimal recipe (IFFC + stop-on-repeat + card) | 71/200 = 35.5% | Run 42 |
| JSON text (IFFC) | 61/200 = 30.5% | Run 5 |
| Native function calling (NFC) | 54/200 = 27.0% | Run 4 (used 8 server slots vs 4 elsewhere — disclosed) |
| Vote of five (BoN, N=5, on IFFC+STATE) | 73/200 = 36.5%, ≈7× generation wall-clock (21.3 h vs 3.0 h) | Run 11 |
| Constrained decoding (two-pass) | 21/200 = 10.5% | Run 8 |
| Tool-name automaton (PTC-A) | 27/200 = 13.5% | Run 46 |
| Best SFT (SFT-v4) | 30/200 = 15.0% | Run 15 |
| DPO | 53/200 = 26.5% | Run 51 |
| Single-call (native API), three categories | 772/840 = 91.9% (Wilson 89.9–93.6) | Run 79, `scores.id` 52–58. Rows kept in the paper: simple 375/400, multiple 183/200, irrelevance 214/240. Those three sum to 772/840. On 2026-10-19 Sanskar excluded the Java (37/100) and JavaScript (14/50) rows from the paper; they are not mentioned there. The previous five-category total was 823/990 = 83.1% (Wilson 80.7–85.3). Check: 823−37−14 = 772 and 990−100−50 = 840, and 375+183+214 = 772, 400+200+240 = 840. The new interval is `wilson(772, 840)` in `paper/figures/src/make_refocus_figs.py` (z = 1.959964), rounded to one decimal from 89.86–93.56. It is not a new run. Both parallel-call categories stay excluded because earlier runs on this server hit a tool-grammar bug that truncated multi-call output. `scores.id` 52–58 is the run's score-row range; this note does not assign a single id to one category. |
| Paired tests | Exact two-sided McNemar on per-task results (from `eval_errors`), 8 planned tests, Holm correction | Only **best SFT vs JSON text** (7 vs 38, p ≈ 3e-6) survives Holm. Full stack vs JSON text p = 0.023 is uncorrected only. |
| Failure labels | empty turn / under-call / state mismatch per failed task | Derived from checker messages in `eval_errors` ("list is empty", "does not contain all", "does not match the state"). Run 78: 20 / 35 / 68. Note: Run 4 rows are duplicated in `eval_errors` (counts appear doubled). |

### Prose numbers traced on 2026-10-19

These appear in `paper/main.tex` and were not in the table above. Sources are repo notes or code, not a new measurement. The Spearman interval −0.52 to 0.67 was not found in those notes and was removed from the paper.

| Claim in the paper | Source |
|---|---|
| 37 forced stops, all on tasks that failed | `THESIS.md` Run 27 (ungated stop-on-repeat): stops fired on 37/200, all among the failures. Scoreboard ID 27 is 54/200. This is not the gated under-call cell of 37. |
| First-mix empty turns 167, then 2 on the best fine-tune | `THESIS.md` / `sft/README.md`: Run 13 primary empty-turn count 167. The 2 is the best-SFT cell in `tab:failfull`. |
| Recovery examples | `sft/README.md` SFT v3/v4: `recovery_after_empty` rows (an earlier empty reply, then a new user question, then a real call). |
| Stop-target share 78.1% (2,121 versus 596) | `THESIS.md` and `METHODS.md`: after a tool-result user message, 2,121 empty targets versus 596 continue, all ToolACE. 2121/(2121+596) = 78.1%. |
| Guided template: 19 tasks each way; 79 state mismatches; seven steps | 61/200 is scoreboard ID 9. `THESIS.md` records a 19-row swap each way against JSON text and 79 state mismatches. The seven steps are `GST_STEPS` in `benchmark/gst.py` (comment: He et al. Table 5, step 7 extended for multi-turn abstain). |
| 6 of 12 self-distillation failures | `THESIS.md` Run 74. 12 = 20−8 (scoreboard ID 74). |
| Look-up 37.9% (67/177) to 19.7% (78/395); stop 23.1% (36/156) to 33.2% (146/440) | `THESIS.md` Run 74 autopsy. Name-matching heuristic. |
| Stop share capped 22.9% (86/376); look-up left at 20.6% (84/407) | `THESIS.md` Run 75. 14/20 is scoreboard ID 75. |
| Look-up restored 39.6%; lost 7 passes; empty turns 9 of 66 | `THESIS.md` Run 76. 12/20 is scoreboard ID 76. |
| Tripwire 3/10; Spearman 0.118 on 11 strong runs (≥50/200) | `METHODS.md` Run 68 verdict and `THESIS.md` Run 68. The approximate interval was not in those notes. |
| Rebuilt mix 30,000 / 1,500; 45/40/10/5; 22% empty-list targets; 41.2% multi-call rows | `THESIS.md` rebuilt-mix notes. |
| 39 updates; 156 logged answers | `THESIS.md` Run 73. 156 is also the denominator of 36/156. |
| Stopped at step 26,500 of 32,000; best validation loss 0.030 at step 22,000 | `THESIS.md` and `METHODS.md`. Step 22,000 is scoreboard ID 70. |
| Self-distillation about 2.8× the null test (110 versus 39 updates) | `THESIS.md` after Runs 75–77: "about 2.8× as many updates as R73 (110 vs 39)". The paper previously said "about 3×"; 110/39 = 2.82, so the paper now uses 2.8×. |
| DPO β = 0.2 | `THESIS.md` Runs 49 and 51. How preferred and rejected responses were paired is not in those notes. |
| Last two turns verbatim; up to eight card slots | `benchmark/iffc_state.py` ("Last ~2 user/assistant+tool turns stay verbatim"); `benchmark/ooda_card.py` `_MAX_SLOTS = 8`. |
| llama.cpp build 8111 (commit `11c325c6e`) for earlier runs in the paper; runs 78 and 80–83 used commit `d344123fe` with `--jinja -fa on -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024` | Reproducibility paragraph. 8111 remains the recorded build for the earlier runs (`THESIS.md` server note). This note does not re-audit every pre-78 log. 48 GB and the slot difference are in the headline table above. |

## 3. History of the paper

1. **Markdown `THESIS.md`** — informal working notes; still in the repo (do not delete).
2. **Google Doc draft (September 2026)** — now trashed. Content and exact dates **(uncertain; not in repo)**.
3. **LaTeX conversion (2026-09-20 → 09-23)** — `74a6046` "Add formal LaTeX research-paper draft under paper/" (#30, 09-20); `f73853d` figures next to prose (#45, 09-23).
4. **Readability rewrite — PR #66** (merged 2026-09-30, `fc666da`): reader-facing rewrite, review rounds R2/R3, "drop RAG from thesis scope; narrow edge claims" (`9a962c4`), outsider-readability rewrite (`4893093`).
5. **Run updates — PR #84** (merged 2026-10-07, `3347e7b`: soft10 ceiling, mid20 gate, paused SFT-v11) and **PR #88** (merged 2026-10-12, `a339032`: Runs 70–74 mid20, v11 stopped at 26,500, R74 autopsy).
6. **Refocus to a short paper — PR #89**, decided 2026-10-13: first refocus as "State, Not Syntax" empirical study (`1171e0e`), Codex/Antigravity and Claude Code review folds, overnight R78/R79 results added 10-14 (`64be7e8`).
7. **Focus rewrite (2026-10-14)** after Claude Code's keep/cut/appendix review (`69143bc`): harness-optimization framing, Gemma only, cut list applied, training history moved to the appendix, full stack added to the failure figure.
8. **QA rounds** by Codex and Antigravity: `d072e8d` (both NEEDS_CHANGES → fixes), `ae96ed8` (Fig 2 plain-language labels). Both approved; PR #89 merged 2026-10-14 as `cfe0068`.
9. **Outlook font fix — PR #91** (merged 2026-10-14, `a445429`): Outlook's preview dropped "a"/"A". Probable cause: subset Type 1 Latin Modern fonts with custom encoding (all fonts were already embedded with ToUnicode; no Type 3). Fix: `glyphtounicode` + `\pdfgentounicode=1` + `cmap`, and a `make dist` ghostscript pass that embeds full, unsubsetted fonts. Whether Outlook now renders correctly was **not** verified by the agent (no Outlook access) **(uncertain)**. Figure fonts (DejaVu TrueType from matplotlib) remain subset.

## 4. Editorial decisions and Sanskar's stated preferences

- **Focused paper, not a project history.** Cut anything that does not serve the abstract.
- **Gemma only.** Cut: other-model scores (Qwen, Llama, FunctionGemma, `tab:st`), Jetson/edge-hardware and "8 GB device" text, helper-model runs (58/61/64), xLAM results (kept only as related work/data source), shell-as-action, type coercion, harder-split numbers.
- **Fine-tuning version history goes in the appendix** (App. B "Training history"), as do the 10-task tripwire, the stop-target paragraph, the guided-template row and the Gemma-only single-turn table.
- **Harness-optimization framing**, honestly stated: the state summary (STATE) and the status card reach the model as text in its context, so it is "harness-side control delivered partly through the context", not "no prompting".
- **Guided-template tie addressed openly** (Discussion): a guided template on the native API also scores 61/200, so the NFC→IFFC channel gain alone is not claimed; whether a template adds to or substitutes for the harness levers is untested.
- **Headline:** R78 38.5% (it replaced R42 35.5% after R78 finished; R42 is the "minimal recipe"). Sanskar did not object to switching the headline to 38.5%. October 2026 update: a shorter tool list scored 84/200 against a 71/200 rerun. Sanskar's QA decision: no detectable effect (p=0.053). It stays out of the abstract. The first-run 77/200 stays beside the rerun.
- **Limitations stated**, including tuning leakage (headline chosen after seeing results on the same 200 tasks), gate selection leakage, one model/split/device, sample size.
- **RAG tool selection appears only as Future Work.**
- **Never invent numbers.** Every number must trace to `benchmark.db` or the logs.
- **No new runs without Sanskar's approval.** Long jobs (training, 200-task BFCL) are queued overnight on his Mac, with a time estimate given first (a 200-task run takes ≈3 h).
- Agents' claims are not Sanskar's approval; merges need his explicit OK.

## 5. Sources of truth

- **Database (Mac):** `~/Documents/GitHub/Thesis/benchmark/results/benchmark.db` — tables `runs`, `scores` (per run/category correct/total), `eval_errors` (per failed task checker message; used for McNemar and failure labels). Older copies elsewhere (e.g. on the agent box) may stop before `scores.id` 49–58.
- **Logs (Mac):** `~/Documents/sft_logs_mirror/` (the old `/tmp/sft_logs` is gone).
- **Repo docs:** `METHODS.md`, `THESIS.md`, `PLAN.md`; figure script `paper/figures/src/make_refocus_figs.py` (all figure counts are hard-coded from the DB; no new data).
- **Vault:** `sskarz/thesis-vault` Obsidian notes (e.g. Runs 70–74 mid20 note; Runs 75–79 note from vault PR #49).
- **Review files (paths only):**
  - Mac: `~/Documents/decision-probe/docs/claude_PR89_FOCUS_REVIEW.md` (Claude Code focus review, keep/cut/appendix table)
  - Agent box: `/workspace/paper-refocus2/claude_review.md` (copy of the above), `/workspace/paper-refocus2/codex_QA_focus.md`, `/workspace/paper-refocus/OUTLINE.md`, `/workspace/paper-refocus/codex_QA_PR89.md`, `codex_QA_PR89_r2.md`, `codex_QA_PR89_r3.md`, `/workspace/sft_logs/agy_PR89_REFOCUS_REVIEW.md` (Antigravity rounds 1–4), `/workspace/sft_logs/codex_paper_outsider_readability_REVIEW.md`, `/workspace/paper-review/` (PR #66-era review rounds R1–R4 and consolidated summaries)
  - The earlier "is this strong enough for a master's?" review (verdict BORDERLINE, fixable) — location **(uncertain)**; that file states it was actually written by Grok Bot because Claude credits had run out.

## 6. Process and workflow

> **Agent workflow (updated 2026-10-19):** Claude Code is not used for paper work as of 2026-10-18. Cursor, Codex, and Antigravity run paper work through `.agents/skills/thesis-paper/SKILL.md` (hard rules in that file; see also `AGENTS.md`): load context → diagnose with the vendored reviewer agents + a fresh-context number tracer → save findings to `docs/reviews/` → Sanskar picks fixes → edit on a branch → build/check → verify the diff → draft PR → the Codex + Antigravity loop below. Point Codex and Antigravity at that file and the review file. A subagent running a reviewer role reads `.claude/agents/<role>.md` and the thesis-paper skill as its prompt. Where the vendored ml-paper-writing says "Claude should be proactive" or tells you to deliver drafts without waiting, the thesis-paper rules override it.

1. **Edits:** branch off `main`, push, open a **draft PR**. Never commit straight to main.
2. **Review loop:**
   - Content/focus review through `.agents/skills/thesis-paper/SKILL.md` and `.claude/agents/<role>.md`; Sanskar agrees the cut list before any rewrite and picks which fixes are applied.
   - QA by **Codex** and **Antigravity**: accuracy against the DB, readability for non-experts, figure fit. Repeat until **both APPROVE**.
   - **Sanskar approves the merge.** He decides all merges.
3. **Build:** in `paper/`: `latexmk -pdf -interaction=nonstopmode main.tex`, then `make dist` (ghostscript re-embeds all fonts fully into `paper/Thesis_Paper.pdf`). Check: no `Overfull` / undefined refs in `main.log`, no `??` in `pdftotext` output, `pdffonts` shows all fonts embedded and no Type 3. **`make dist` needs Ghostscript ≥ 10.05.** The flags are unchanged (`-dEmbedAllFonts=true -dSubsetFonts=false -dPDFSETTINGS=/prepress`). Ghostscript 10.02.1 embeds Latin Modern in full but drops its ToUnicode maps. Ghostscript 10.05.1 keeps those maps on the fully embedded Latin Modern fonts. The 2026-10-19 QA rebuild used 10.05.1.
4. **Canonical compiled PDF (two places).** The current compiled paper lives on **Google Drive** (`Thesis_Paper.pdf`, My Drive root of sskarz1016@gmail.com; id as of 2026-10-14: `1CLOoeqWVeZP83eM7QPyFs1IT66Jrnf4r`) **and** at the **repo root** as [`Thesis_Paper.pdf`](../Thesis_Paper.pdf). Both are the `make dist` output (full fonts embedded).
   - **Sync routine:** after every merge to main that touches `paper/`, the routine "Thesis paper Drive PDF sync" rebuilds the PDF (latexmk + `make dist`), uploads it to Drive and trashes the previous Drive file, then commits the same PDF to the repo root through a small PR titled `chore: update compiled Thesis_Paper.pdf`. It skips that PR if the file is identical to the one already committed.
   - The repo copy is kept **outside `paper/` on purpose**, so committing it does not retrigger the sync. Changes under `docs/` or the root PDF do not trigger it.
   - If you sync manually, first check Drive for a fresh upload so you do not create a duplicate; exactly one current `Thesis_Paper.pdf` should exist on Drive.
5. Name the agent that did a review in the review file.

## 7. How to continue

### Checklist for adding a new result
- [ ] Get Sanskar's approval for the run (with a time estimate); queue it overnight on the Mac.
- [ ] Confirm the score in `benchmark.db` (`scores`, note the `scores.id` and `runs.id`) and the log path.
- [ ] Recompute Wilson intervals and, if paired, McNemar from `eval_errors`; update Holm over the planned set.
- [ ] Update `tab:ladder` / `tab:board` / `tab:failfull` / `tab:runs`, the counts in `make_refocus_figs.py`, and regenerate figures.
- [ ] Update `THESIS.md` / `METHODS.md` and a vault note.
- [ ] Rebuild (latexmk + `make dist`), check overfull/`??`/fonts, open a draft PR, run the QA loop, get Sanskar's merge approval.

### Open items
- **Leakage caveat:** the full stack was assembled after seeing results on the same 200 tasks; its +8.0 pp is not significant after Holm.
- **Confirmatory runs** (held-out tasks, harder splits, a second model, matched serving) — not run; need approval.
- **Length:** target was ~13 pages total (~8.5 main text); currently 15.
- **DB copies:** some older DB copies do not contain R76, R78 or R79 rows (`scores.id` 49–58); always verify against the Mac DB.
- Figure fonts remain subset TrueType; Outlook rendering after PR #91 not confirmed.

## 8. Glossary

- **IFFC** — instruction-followed function calling: the model writes calls as a JSON list in plain text; the harness parses them.
- **NFC** — native function calling: structured `tool_calls` via the server's tool API.
- **STATE** — harness-written `[STATE]` summary (current dir, known files, last calls) inserted after each executed call; called "state summary" in the paper.
- **OODA / status card** — four-line card (goal, slots, last call + outcome, advisory next: call/ask/final/refuse) prefixed to the latest user message; filled deterministically, never sees references. Acts mainly as an act-vs-abstain control.
- **Stop on repeats** — harness ends a turn when the model re-proposes already-executed calls, gated to ≥6 tool steps (MIN=6).
- **BoN** — Best-of-N; here a majority vote over five sampled calls per step.
- **PTC-A** — tool-name automaton constraining only the tool name (after Yu et al.); 13.5%.
- **mid20 gate** — fixed 20-task early check from the same split; bar 17/20 (untuned + card, Run 69). Screening heuristic with selection leakage, not an accuracy. (The older 10-task "soft10" tripwire could not rank runs.)
- **Run-ID map** — `tab:runs` in Appendix D of `paper/main.tex`.
