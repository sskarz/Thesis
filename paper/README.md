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
- **IFFC-stop-repeat Run 27 = scored miss 27.0%** vs Run 5 IFFC 30.5% (−3.5 pp) and vs BoN 36.5% (−9.5 pp). Ungated observe-same; `scores.id=20`; **200/200 responses**. Same accuracy level as Run 4 NFC 27.0% — **not** an identical method. Stop flags **37/200**, all failures (**0 stopped and correct**). VehicleControlAPI **22→33** vs Run 5. **Thursday lock on ungated:** tradeoff for **loop-dominated cases only**; **not** a method improvement.
- **IFFC-stop-repeat Run 30 = first scored win vs IFFC 32.5%** on BASE (tighten MIN=6 @ `78c22a0`, PR #54). BASE bartowski Q4_K_M; `fc_mode=iffc-stop-repeat`. `scores.id=21`; `multi_turn_base` only **65/200 = 0.325**; **200/200 responses + timings**; `entry_id` identical to Run 5/Run 27. **+2.0 percentage points vs Run 5 (65−61 correct)** — **not** “+4pp”; **+5.5 pp vs Run 27**. Still **below STATE 34.0% and BoN 36.5%**. Stop flags **37→3**, all FAIL. VC state **22** (recovered). `state_mismatch` **60** (=Run 5). **Thursday:** BASE + iffc-stop-repeat (`observe_same` gated after 6 tool-turns) → **32.5%** vs plain IFFC **30.5%** / ungated **27%**. **Celebrate / park SFT-v9: YES.** **Caveats:** (1) harness/decode guardrail on BASE — **not** an SFT/method win; (2) `finished_at` NULL — metrics 4/5 soft bookkeeping; accuracy locked in `scores`; (3) status JSON stale — trust DB+log; (4) `--resume-run-id 30` after orphan; N=200 complete. Do **not** overwrite BoN **36.5%** / IFFC **30.5%** / SFT-v4 **15%**. SFT v9.2 10-ID smoke-gate **FAILed** (Run 36 avg api **25.7**, hit **6/10**) — park; **not** a `multi_turn_base` %; **do not invent** a v9 BFCL %. **Codex CONFIRM_PASS.** Best still BoN **36.5%** / IFFC Run 5 **30.5%**. **Not** smoke81 / v7 / v8 / v9 SFT / BoN. Do **not** invent tok/s.

## TODOs still open in the draft

Parallel / `parallel_multiple` re-run after the llama-server tools GBNF truncation fix; multi-turn evaluation for Qwen / Llama / FunctionGemma; NVIDIA Jetson Orin Nano Super replication; dual-model IFFC (paper Algorithm 1). SFT recipe v6 is scored (Run 18, 13.0%, miss vs v4 15.0%); next SFT lever TBD — not BoN on v6, not more epochs on the v6 mix. Thin filter ablation on this family is likely exhausted. IFFC+STATE on the SFT-v4 GGUF is scored (Run 16, 13.0%, null/negative). SFT-v5 is scored (Run 17, 11.5%, regression vs v4). xLAM-1b NFC is scored (Run 19, 0.0%, format-channel collapse on this harness). xLAM `--fc-mode xlam` is scored (Run 20, 3.0%, fair channel, still very low — not published overall BFCL ~78.94%; not a method win). IFFC-stop-repeat ungated is scored (Run 27, 27.0%, miss vs IFFC 30.5% on base Gemma — not smoke81/v7/v8; not a method win). IFFC-stop-repeat tighten is scored (Run 30, 65/200=0.325, first scored win vs IFFC on BASE — +2.0 percentage points vs Run 5 (65−61 correct), not “+4pp”; +5.5 pp vs Run 27; still below STATE 34.0% / BoN 36.5%; not a new overall best; Codex CONFIRM_PASS; harness/decode guardrail on BASE — not an SFT/method win; finished_at NULL / metrics 4/5 soft bookkeeping; status JSON stale — trust DB+log; --resume-run-id 30 after orphan; N=200 complete). SFT v9.2 10-ID smoke-gate **FAILed** (Run 36 avg api **25.7**, hit **6/10**) — park; **not** a `multi_turn_base` %; **do not invent** a v9 BFCL %. Figure regen for a combined Gemma+xLAM scoreboard is **TODO** (needs Mac `benchmark.db`; current PDF is Gemma-only); text tables already include Runs 19–20, 27, and 30.
