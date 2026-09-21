# LaTeX research-paper draft

Formal arXiv-style draft of Sanskar Thapa's master's thesis:
**Optimizing Consecutive Tool Calling for Small Language Models on Edge Devices.**

This directory is the **formal paper**. Informal working notes remain in [`../THESIS.md`](../THESIS.md) (do not delete). All own experimental numbers in `main.tex` are taken from measured MacBook Pro M4 Pro / llama.cpp Metal runs; do not invent scores.

## Build

Dependencies: a TeX Live (or MacTeX) installation with `pdflatex`, `bibtex`, and `latexmk`. Standard packages used: `geometry`, `booktabs`, `graphicx`, `hyperref`, `natbib`, `xcolor`, `tikz`, `caption`, `tabularx`.

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
| `main.tex` | Article-class paper (title through references) |
| `references.bib` | BibTeX (BFCL, Gorilla, IFFC, XGrammar-2, GST, BoN, QLoRA, Hammer, xLAM, ToolACE, \ldots) |
| `figures/` | Measured charts copied from the Mac comparison / multi-turn scoreboard |
| `Makefile` | `latexmk -pdf` wrapper |

## Honesty labels (do not drop in revisions)

- **CD ≠ full XGrammar-2** (TagDispatch-style two-pass on stock llama-server).
- **GST ≠ GBNF / not IFFC** (prompt overlay on NFC; checker ignores GST drafts).
- **STATE ≠ LLM summarization** (harness-written snapshot from executed tool results).
- **BoN = MV@N majority vote**, not verifier BoN; ~7× wall-clock cost.
- **Run 14 strip-empty = DIAGNOSTIC history bookkeeping, not a method win** (4/200 = 2.0%).
- **SFT Runs 12–13 scored negatives** (0.5%); ToolACE `[]` stop-after-tools is the documented root cause.
- **SFT Run 15 (v4) = scored recovery 15.0%**, still below IFFC 30.5% and NFC 27.0%; **not a win over IFFC**.
- **SFT-v4+STATE Run 16 = scored negative/null overlay 13.0%** vs Run 15 15.0%; STATE helps base Gemma, not this ckpt. **Do not BoN this checkpoint as the next win.**

## TODOs still open in the draft

Parallel / `parallel_multiple` re-run after the llama-server tools GBNF truncation fix; multi-turn evaluation for Qwen / Llama / FunctionGemma; NVIDIA Jetson Orin Nano Super replication; dual-model IFFC (paper Algorithm 1); SFT recipe v5 (turn-0 completeness + env-state consistency; keep v4 empty-recovery). IFFC+STATE on the SFT-v4 GGUF is scored (Run 16, 13.0%, null/negative).
