# LaTeX research-paper draft

Formal arXiv-style draft of Sanskar Thapa's master's thesis:
**Optimizing Consecutive Tool Calling for Small Language Models on Edge Devices.**

This directory is the **formal paper**. Informal working notes remain in [`../THESIS.md`](../THESIS.md) and [`../METHODS.md`](../METHODS.md). The paper is written for a general ML/systems reader: method names are defined before use, results are grouped by what moved accuracy, and chronological evaluation-log IDs live in Appendix B.

All own experimental numbers in `main.tex` are taken from measured MacBook Pro M4 Pro / llama.cpp Metal runs. Do not invent BFCL / `multi_turn_base` scores. Health-gate probes (SFT v9.2, DPO smoke) are not $N{=}200$ accuracies. As of 2026-09-29 a capped full DPO train is in flight; this draft reports no DPO BFCL percentage. Retrieval-augmented generation (RAG) appeared in an earlier registered-title framing and is **out of scope** for this paper (not deferred). Body claims are local / on-device SLM serving on the Mac M4 Pro (48 GB); Jetson 8 GB is unmeasured.

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

The complete multi-turn scoreboard is the table in the paper body. The `multiturn_scoreboard.pdf` figure covers the earlier Gemma set (through SFT-v6) and is captioned as such; later rows are table-only.

## Caveats (defined in Appendix A)

Short version, in ordinary scientific language:

- Constrained decoding $\neq$ full XGrammar-2.
- GST is a prompt overlay, not a grammar and not IFFC.
- STATE is harness-written memory from executed tools, not an LLM summary.
- Best-of-N is majority vote, not a verifier (~7× generate cost).
- SFT-v4 15.0% recovers from 0.5% collapse and still sits below IFFC 30.5% / NFC 27.0%.
- xLAM 0% / 3% are this harness, not published overall BFCL ~78.94%.
- PTC-A 13.5% is a closed failure; token-level forced-name vs loop is not fully audited.
- DPO 10-ID is a health-gate pass; full-train BFCL results are pending (in flight as of 2026-09-29; no $N{=}200$ Acc).
- RAG is out of scope (earlier title framing; not a deferred near-term experiment).
- Title “edge devices” is branding; measured setting is Mac M4 Pro 48 GB.
