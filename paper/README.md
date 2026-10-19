# LaTeX research-paper draft

Formal arXiv-style draft of Sanskar Thapa's master's thesis, refocused (Oct 2026) as an empirical study:
**State, Not Syntax: Why Small Language Models Fail at Multi-Turn Tool Calling, and What Helps.**

This directory is the **formal paper**. Informal working notes stay in [`../THESIS.md`](../THESIS.md) and [`../METHODS.md`](../METHODS.md).

The paper has about 9.5 pages of main text plus an appendix, and is written for a research professor and a general reader. Every term is defined on first use, and run codenames are kept out of the main text; the run-ID map is Appendix E.

- **Headline:** the greedy full stack, IFFC + STATE + gated stop-on-repeat + status card (Run 78, draft PR #90), scores **77/200 = 38.5%** (Wilson 32.0–45.4). It is +8.0 pp over IFFC (McNemar p=0.023 uncorrected; not significant under Holm), and the config was selected after earlier results on the same tasks. The minimal recipe, R42, scores 71/200 = 35.5%. BoN scores 36.5% at about 7× the cost. Gemma's single-call score (Run 79; simple, multiple, and irrelevance) is 772/840 = 91.9% (Wilson 89.9–93.6, recomputed from those rows).
- **Failure finding:** constrained decoding (10.5%) and the tool-name automaton PTC-A (13.5%) lower accuracy, because errors are about state, not syntax.
- **Explanatory negative evidence:** SFT and DPO. The best SFT reached 15.0%. The rebuilt v11 mix scored 1–2/20 against the 20-task bar of 17/20. The Run 73 null test scored 15/20 (the pipeline is OK). Run 74 scored 8/20. The retrains that rebalanced its mix also failed: R75 (stop share capped) scored 14/20, R76 (lookup episodes added) 12/20, and R77, a fresh 20-task bank, 2/20 vs the base model's 3/20. The mix shift (lookups falling from 37.9% to 19.7%) is therefore only a **candidate** cause.
- **Statistics:** Wilson intervals are computed from counts. No McNemar p-values are quoted, because no source for them was found. No new tests were run, because per-task results are not in the repo.
- **Future Work:** retrieval-augmented (RAG) tool selection is listed as future work and is not evaluated.

All numbers come from measured MacBook Pro M4 Pro / llama.cpp Metal runs. Do not invent BFCL `multi_turn_base` scores. 10-task and 20-task checks are not 200-task accuracies.

Figures `ladder_n200.pdf`, `gate20.pdf` and `failure_anatomy.pdf` are regenerated from scoreboard counts with `python3 figures/src/make_refocus_figs.py` (matplotlib). The older figures in `figures/` are no longer referenced by `main.tex`.

## Build

Dependencies: a TeX Live (or MacTeX) installation with `pdflatex`, `bibtex`, and `latexmk`. Standard packages used: `lmodern`, `geometry`, `booktabs`, `graphicx`, `hyperref`, `natbib`, `xcolor`, `tikz`, `caption`, `tabularx`, `placeins`, `microtype`.

From this directory:

```bash
latexmk -pdf main.tex
```

Equivalent:

```bash
make
```

Output: `main.pdf`.

Clean auxiliaries:

```bash
latexmk -c    # or: make clean
latexmk -C    # or: make distclean  (also removes main.pdf)
```

Without `latexmk`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Layout

| Path | Role |
|------|------|
| `main.tex` | Article-class paper (title through appendices) |
| `references.bib` | BibTeX (BFCL, Gorilla, IFFC, XGrammar-2, GST, DPO, xLAM, ToolACE, \ldots) |
| `figures/` | Measured charts copied from the Mac comparison / multi-turn scoreboard |
| `Makefile` | `latexmk -pdf` wrapper |

The complete multi-turn scoreboard is the table in the paper body. The `multiturn_scoreboard.pdf` figure covers the earlier Gemma set (through SFT-v6) and is captioned as such; later rows are table-only. Single-turn accuracy / performance / memory PDFs are **three-model** plots (FunctionGemma is in the tables). `failure_buckets.pdf` is SFT-v4–v6 only.

## Caveats (defined in Appendix A)

Short version, in ordinary scientific language:

- Constrained decoding $\neq$ full XGrammar-2.
- GST is a prompt overlay, not a grammar and not IFFC.
- STATE is harness-written memory from executed tools, not an LLM summary.
- Best-of-N is majority vote, not a verifier (~7× generate cost).
- SFT-v4 15.0% recovers from 0.5% collapse and still sits below IFFC 30.5% / NFC 27.0%.
- xLAM 0% / 3% are this harness, not published overall BFCL ~78.94%.
- PTC-A 13.5% is a closed failure; token-level forced-name vs loop is not fully audited.
- Decision sidecar 25.0% is a scored failure on plain IFFC (treatment-only; helper not dead). Decision+OODA 26.0% (Run 61; 52/200) is also a scored failure versus IFFC--OODA (only +1.0pp over Run 58). Choice-SLM Decision 5.5% (Run 64; 11/200) is a scored failure versus Decision / Decision+OODA (−19.5pp / −20.5pp). Ranking is not Acc. Compose/hygiene smoke PASS is not an Acc claim. Runs 59/60 (1/10 vs 0/10) and Runs 62/63 (0/10 vs 0/10) are 10-ID trails, not $N{=}200$.
- DPO 10-ID is a health-gate pass. Full DPO on plain IFFC is scored 26.5% (Run 51) and is not a win over IFFC or NFC.
- Soft10 (Runs 65–68) cannot discriminate SFT vs PE (all 3/10 on the same pass set except the undertrained smoke 0/10); tripwire only. Mid20 (Run 69 17/20) is the replacement gate, not N=200 and not the PE bar. SFT-v11 train stopped at 26500 (best val 0.030 @22000) and failed mid20 (Runs 70–72: 2/20, 1/20, 2/20). Run 73 null test 15/20 (pipeline fine); Run 74 filtered self-distill 8/20 (likely cause: lookup-call share fell 38%→20%; unproven); rebalanced retrain in progress, no result.
- RAG is out of scope (earlier title framing; not a deferred near-term experiment).
- Title “edge devices” is branding; measured setting is Mac M4 Pro 48 GB.
