# LaTeX research-paper draft

Formal arXiv-style draft of Sanskar Thapa's master's thesis:
**Optimizing Consecutive Tool Calling for Small Language Models on Edge Devices.**

This directory is the **formal paper**. Informal working notes remain in [`../THESIS.md`](../THESIS.md) (do not delete). All own experimental numbers in `main.tex` are taken from measured MacBook Pro M4 Pro / llama.cpp Metal runs; do not invent scores.

## Build

Dependencies: a TeX Live (or MacTeX) installation with `pdflatex`, `bibtex`, and `latexmk`. Standard packages used: `geometry`, `booktabs`, `graphicx`, `hyperref`, `natbib`, `xcolor`, `tikz`, `caption`, `tabularx`, `placeins`.

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
- **SFT-v5 Run 17 = scored regression 11.5%** vs Run 15 15.0% (−3.5 pp / −7). Empty still 0. **Do not present as a method win.** No BoN / no more epochs on this ckpt.
- **SFT-v6 Run 18 = scored miss 13.0%** vs Run 15 15.0% (−2.0 pp / −4); modest lift vs Run 17 (+1.5 pp). Empty still 1. **Do not present as a method win.** No BoN / no more epochs on this ckpt. v4 IFFC 15% remains best SFT recovery; BoN 36.5% best inference on base Gemma.
- **xLAM-1b NFC Run 19 = scored format-channel collapse 0.0%** on this multi_turn NFC harness (NFC never parses `content`; not published overall BFCL ~78.94%; not a method win). Gemma NFC contrast is Run 4 27.0%.
- **xLAM-1b `--fc-mode xlam` Run 20 = format-fixed fair-channel 3.0%** after PR #42 (tools actually ran, avg `api_calls` 50.47; still **very low**; **not** a method win; **not** published ~78.94%; still below CD 10.5%).

## TODOs still open in the draft

Parallel / `parallel_multiple` re-run after the llama-server tools GBNF truncation fix; multi-turn evaluation for Qwen / Llama / FunctionGemma; NVIDIA Jetson Orin Nano Super replication; dual-model IFFC (paper Algorithm 1). SFT recipe v6 is scored (Run 18, 13.0%, miss vs v4 15.0%); next SFT lever TBD — not BoN on v6, not more epochs on the v6 mix. Thin filter ablation on this family is likely exhausted. IFFC+STATE on the SFT-v4 GGUF is scored (Run 16, 13.0%, null/negative). SFT-v5 is scored (Run 17, 11.5%, regression vs v4). xLAM-1b NFC is scored (Run 19, 0.0%, format-channel collapse on this harness). xLAM `--fc-mode xlam` is scored (Run 20, 3.0%, fair channel, still very low — not published overall BFCL ~78.94%; not a method win). Figure regen for a combined Gemma+xLAM scoreboard is **TODO** (needs Mac `benchmark.db`; current PDF is Gemma-only); text tables already include Runs 19–20.
