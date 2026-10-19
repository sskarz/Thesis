---
name: thesis-paper
description: Workflow for any work on the thesis paper (paper/main.tex, "State, Not Syntax") — review, trim, polish, rewrite a section, fix figures or captions, audit citations, or fold in a new result. Wraps the academic reviewer agents and the ml-paper-writing guide with this repo's rules. Use before touching paper/ and for /thesis-paper <task>.
argument-hint: [task, e.g. "review full paper", "trim to 13 pages", "polish §5", "audit bibliography"]
---

# Thesis paper workflow

Use this for every change to `paper/main.tex`, its figures or `references.bib`. It decides **which** tools run and **in what order**; the tools themselves are:

- `.claude/skills/academic/` — orchestrator + 30 writing principles (`academic-writing.md`); agents in `.claude/agents/` (consistency-checker, logic-reviewer, technical-reviewer, writing-reviewer, latex-layout-auditor, bibliography-auditor, prose-polisher, section-drafter, latex-figure-specialist, research-analyst, brainstormer, paper-crawler).
- `.claude/skills/ml-paper-writing/` — narrative principle, abstract/intro formulas, Gopen & Swan / Perez / Lipton / Steinhardt clarity rules (`references/writing-guide.md`), reviewer criteria (`references/reviewer-guidelines.md`), citation verification (`references/citation-workflow.md`). Ignore its venue templates and submission checklists: this is a thesis paper in `article` class, not a conference submission.

Where this file conflicts with either upstream skill, **this file wins**.

## 0. Load context first (every time)

1. `docs/PAPER_WRITING_PROCESS.md` — history, headline numbers with provenance, editorial decisions, sources of truth, build steps.
2. `paper/main.tex` — the whole file, not just the section in scope.
3. The most recent file in `docs/reviews/` for the same scope, if any (incremental review).

## Hard rules (not negotiable; override any upstream instruction)

1. **Never invent numbers.** Every number must already appear in `main.tex`'s tables, `docs/PAPER_WRITING_PROCESS.md` §2, or trace to `benchmark.db` / logs on the Mac. Agents may reword around numbers, never create, round differently or recompute them. If a fix needs a number that isn't on record, stop and ask Sanskar.
2. **No new runs** (BFCL, training) without Sanskar's approval and a time estimate.
3. **Never commit to `main`.** Branch `paper/<topic>`, draft PR. Agent approval is not Sanskar's approval; he merges. No AI attribution in commits or PR text (see CLAUDE.md).
4. **Scope is fixed:** Gemma-4-E2B-it only, BFCL v4 multi_turn_base, one Mac. Do not re-add other models, Jetson/edge-hardware claims, xLAM results, type coercion or harder-split numbers. RAG stays Future Work only. Training history stays in Appendix B.
5. **Statistics wording must not get stronger.** The full stack's +8.0 pp is "suggestive", p=0.023 uncorrected, not significant after Holm; only best-SFT vs JSON text survives Holm. Tuning leakage and gate-selection caveats stay in. Polishing that drops a hedge on these is a regression. Name Holm only in the Statistics paragraph and the McNemar table; elsewhere say "after correcting for multiple comparisons". Report run cost as the vote-vs-harness ratio (~7×), not raw hours.
6. **Readers are outsiders** (advisor + committee). Define every term at first use; captions must stand alone; internal run names (R42, SFT-v11, IFFC+STATE…) appear only in the appendix run-ID table.
7. **Title exemption:** "State, Not Syntax" is a deliberate negation-contrast. Principles B2/F2 apply to body prose, not the title or the thesis claim it names.
8. **Page budget:** ~13 pages total, main text ~8.5. A change that adds length should say what it cuts.

## The loop

### Step 1 — Plan (tell Sanskar before running anything)
State the task, which agents/guides you'll use, and whether anything will be edited. Ask only if scope is genuinely unclear.

### Step 2 — Diagnose (read-only, parallel)
Pick from the playbook below. Always add a **number tracer**: a fresh `general-purpose` agent (no draft context) that lists every number, percentage, count, p-value and interval in the scoped text and checks each against the tables in `main.tex` and `docs/PAPER_WRITING_PROCESS.md` §2, reporting any mismatch or any number with no source. (From ml-paper-writing's claim-verification pass.)

Give every reviewer: the scoped file paths, the hard rules above, and the relevant principle categories. Synthesize per the academic orchestrator (dedupe, Critical / Important / Minor, be opinionated) and save to `docs/reviews/YYYY-MM-DD-<scope>.md`. **Not** under `paper/` — that would trigger the Drive PDF sync.

### Step 3 — Sanskar picks
Present the prioritized list. **Stop and wait** for which items to fix. Do not deploy action agents on your own.

### Step 4 — Fix (on the branch)
Small fixes: edit directly. Larger ones: prose-polisher / section-drafter / latex-figure-specialist, each given the review file and the hard rules. For abstract, introduction and contribution bullets, follow `ml-paper-writing/references/writing-guide.md` (five-sentence abstract, one-sentence contribution). Figures: change counts only in `paper/figures/src/make_refocus_figs.py`, never by hand.

### Step 5 — Build and check
From `paper/`:
```bash
latexmk -pdf -interaction=nonstopmode main.tex && make dist
grep -c "Overfull" main.log          # expect 0
grep -i "undefined" main.log         # expect nothing
pdftotext Thesis_Paper.pdf - | grep -c "??"   # expect 0
pdffonts Thesis_Paper.pdf            # all embedded, no Type 3
pdfinfo Thesis_Paper.pdf | grep Pages
```
Then run latex-layout-auditor on the PDF for any float that moved.

### Step 6 — Verify the diff
Re-run the number tracer and consistency-checker on the changed text only. Confirm no hard rule was broken (especially 1, 4, 5).

### Step 7 — Hand off
Draft PR: what changed, page count before → after, link to the review file, and anything left open. Then the existing QA loop: **Codex + Antigravity** review (point them at this file and the review file) until both APPROVE; Sanskar merges. After merge, the "Thesis paper Drive PDF sync" routine rebuilds and uploads the PDF; don't sync by hand unless asked.

If Claude Code credits ran out and another agent did a review, label it honestly in the review file.

## Playbook

| Task | Diagnose (Step 2) | Fix (Step 4) |
|---|---|---|
| Full paper review | consistency + logic + technical + writing reviewers, bibliography-auditor, latex-layout-auditor, number tracer | per Sanskar's picks |
| Trim to page budget | logic-reviewer (redundancy across abstract / §5 "How sure are we?" / Discussion / Conclusion), latex-layout-auditor (float waste), writing-reviewer (B7), number tracer | prose-polisher; move, don't delete, content that some reader still needs (appendix) |
| Polish a section | writing-reviewer + logic-reviewer on that section | prose-polisher |
| Rewrite abstract / intro | technical-reviewer (claims match results) + ml-paper-writing guide | section-drafter; numbers verbatim from tables |
| Figures / captions | consistency-checker (D3, D7), latex-layout-auditor | latex-figure-specialist; regenerate via script |
| Citations | bibliography-auditor; ml-paper-writing citation workflow (verify via Semantic Scholar / arXiv / DOI; never write BibTeX from memory) | direct edits to `references.bib` |
| Fold in a new result | — | follow `docs/PAPER_WRITING_PROCESS.md` §7 checklist first, then a full-paper review |
| Outsider readability pass | writing-reviewer + a `general-purpose` agent briefed as "a CS grad student who has never seen BFCL", asked to list every undefined term, unclear figure and step they couldn't follow | prose-polisher / section-drafter |

## What not to use
- Venue templates, page-limit tables and conference checklists in ml-paper-writing.
- paper-crawler / brainstormer for reframing the paper: the framing is decided. Use research-analyst only to check related-work coverage.
- The upstream orchestrator's "proceed unless the user objects" for action agents: Step 3 always waits.
