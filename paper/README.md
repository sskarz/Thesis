# LaTeX research-paper draft

Formal arXiv-style draft of Sanskar Thapa's master's thesis:
**Optimizing Consecutive Tool Calling for Small Language Models on Edge Devices.**

This directory is the **formal paper**. Informal working notes remain in [`../THESIS.md`](../THESIS.md) and [`../METHODS.md`](../METHODS.md). The paper is written for a general ML/systems reader: method names are defined before use, results are grouped by what moved accuracy, and chronological evaluation-log IDs live in Appendix B.

All own experimental numbers in `main.tex` are taken from measured MacBook Pro M4 Pro / llama.cpp Metal runs. Do not invent BFCL / `multi_turn_base` scores. Health-gate probes (SFT v9.2, DPO smoke, Decision 10-ID, Decision+OODA 10-ID trail, Choice-SLM 10-ID trail, compose/hygiene smoke, SFT-v10/v11 soft10 and mid20) are not $N{=}200$ accuracies. Soft10 is a 10-task tripwire only; mid20 is a 20-task go/no-go gate (fresh bar 17/20), not the 200-task PE bar (R42 71/200 = 35.5%). SFT-v11 full CE training is paused mid-run (11500/32000; best val 0.059@11250) with **no Acc**. DPO-full is scored at 53/200 = 26.5% on plain IFFC (not a method win). Decision sidecar is scored FAIL at 50/200 = 25.0%. Decision+OODA (`iffc-decision-ooda` + tev1) is scored FAIL at 52/200 = 26.0% (Run 61; only +1.0pp over Run 58; does not clear IFFC 30.5% / Run 30 32.5% / Run 42 35.5%). Choice-SLM Decision (`choice_slm_chat` + menu_ll) is scored FAIL at 11/200 = 5.5% (Run 64; −19.5pp vs Run 58, −20.5pp vs Run 61; ranking is not Acc). Runs 59/60 are a 10-ID trail (1/10 vs 0/10), not $N{=}200$. Runs 62/63 are a 10-ID trail (0/10 vs 0/10), not $N{=}200$. Compose smoke and turn-0 timeout hygiene are PASS (Thesis `ee4d877` / probe `98f1b56`); those are not Acc claims. Retrieval-augmented generation (RAG) appeared in an earlier registered-title framing and is **out of scope** for this paper (not deferred). Body claims are local / on-device SLM serving on the Mac M4 Pro (48 GB); Jetson 8 GB is unmeasured.

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
- Soft10 (Runs 65–68) cannot discriminate SFT vs PE (all 3/10 on the same pass set except the undertrained smoke 0/10); tripwire only. Mid20 (Run 69 17/20) is the replacement gate, not N=200 and not the PE bar. SFT-v11 train is paused; no Acc.
- RAG is out of scope (earlier title framing; not a deferred near-term experiment).
- Title “edge devices” is branding; measured setting is Mac M4 Pro 48 GB.
