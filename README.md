# Optimizing Consecutive Tool Calling for Small Language Models on Edge Devices

**Thesis:** *Optimized Tool Calling for On-Device LLMs in Resource Constraint Environments.*

An earlier registered-title framing mentioned RAG; **retrieval-augmented generation is out of scope** for this thesis paper (not deferred as a near-term experiment). Body claims are local / on-device SLM serving on a MacBook Pro M4 Pro (48 GB); Jetson 8 GB deployment is unmeasured.

Benchmark LLM tool/function-calling accuracy and inference performance on local hardware using [llama.cpp](https://github.com/ggml-org/llama.cpp) and the [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html).

The **formal LaTeX research-paper draft** is in [`paper/`](paper/) (`paper/main.tex`). Informal working notes remain in [`THESIS.md`](THESIS.md) (do not delete). Build with `latexmk -pdf` from `paper/` — see [`paper/README.md`](paper/README.md).

How the paper was written (process, sources, decisions; handoff for agents): [`docs/PAPER_WRITING_PROCESS.md`](docs/PAPER_WRITING_PROCESS.md).

## Prerequisites

| Requirement | Tested version | Notes |
|---|---|---|
| macOS (Apple Silicon) | 26.3 (Tahoe), M4 Pro | Metal GPU backend |
| Ubuntu + NVIDIA GPU | RTX 4090, CUDA 13.0 | CUDA GPU backend |
| cmake | 4.2.3+ | `brew install cmake` / `apt install cmake` |
| Xcode CLI tools | — | Pre-installed on macOS |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager |

## Quick Start

### 1. Clone and build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# macOS (Apple Silicon)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Linux (NVIDIA GPU)
# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release -j$(nproc)
cd ..
```

The `llama.cpp/` directory is git-ignored. It must be built locally on each machine.

### 2. Start the inference server

```bash
./llama.cpp/build/bin/llama-server \
  --jinja -fa on \
  -hf bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 8192 --temp 0.0
```

The first launch downloads the GGUF model (~1.8 GB) to `~/Library/Caches/llama.cpp/` (macOS) or `~/.cache/llama.cpp/` (Linux). Subsequent runs use the cache.

Verify the server is up:

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

### 3. Install Python dependencies

```bash
uv sync
```

This creates a `.venv/` and installs all dependencies from `pyproject.toml`.

### 4. Run the tests

The repository includes a `pytest` suite for benchmark validity and answer-format checks.

```bash
# Hermetic suite: fixture-driven, no local llama-server required
uv run pytest -q
```

This covers:

1. BFCL function-to-tool conversion and response parsing
2. Multi-turn tool-loop behavior and timing aggregation
3. Single-turn and multi-turn evaluator error accounting
4. Metrics sampling/normalization
5. Visualization input handling
6. `benchmark.run_all` mode wiring for single-turn vs multi-turn runs

Optional live smoke tests exercise the real local `llama-server` with structural assertions only:

```bash
BENCHMARK_LIVE=1 uv run pytest -q -m live
```

The live smoke layer:

1. Skips cleanly when `BENCHMARK_LIVE` is unset
2. Skips cleanly when `http://localhost:8080` is unavailable
3. Checks that single-turn and multi-turn responses are structurally valid for the pipeline

MLX LoRA SFT (Gemma-4-E2B, IFFC mix) is documented in [`sft/README.md`](sft/README.md). Unit tests: `uv run pytest -q tests/sft`. Do not run overnight training in CI. **Run 12 scored v1 as a negative: 1/200 = 0.5%**. **Run 13 scored the v2 filtered-[] mix as the same collapse: 1/200 = 0.5%** (filter insufficient — see `THESIS.md`). **Run 14** (`--iffc-strip-empty-assistant` on the same v2 GGUF) is a **DIAGNOSTIC** history-bookkeeping ablation: **4/200 = 2.0%** — did **not** rescue; **do not present 2% as a method win**. **Run 15** SFT-IFFC-v4 LoRA best@1600 is a **scored recovery: 30/200 = 15.0%**, still **below IFFC 30.5% and below NFC 27.0%** — **do not present as beating IFFC**. **Run 16** IFFC+STATE on that same v4 GGUF is a **scored negative/null overlay: 26/200 = 13.0%** (−2.0 pp vs Run 15) — **do not propose BoN as the next win on this ckpt**. **Run 17** SFT-IFFC-v5 LoRA best@1800 is a **scored regression: 23/200 = 11.5%** (−3.5 pp / −7 vs Run 15) — **do not present as a method win**; **no BoN / no more epochs on this mix**. **Run 18** SFT-IFFC-v6 LoRA best@1800 is a **scored miss: 26/200 = 13.0%** (−2.0 pp / −4 vs Run 15; +1.5 pp vs Run 17) — **do not present as a method win**; **no BoN / no more epochs on v6**. **Run 19** xLAM-1b-fc-r NFC (`--model xlam-1b --fc-mode nfc`) is a **scored format-channel collapse: 0/200 = 0.0%** on this harness (NFC never parses `content`) — **not** published overall BFCL ~78.94%; **not** a method win. **Run 20** `--fc-mode xlam` is the **format-fixed fair-channel: 6/200 = 3.0%** (tools actually ran; still **very low**; **not** a method win; **not** published ~78.94%; still **below CD 10.5%**). v4 IFFC 15% remains best SFT recovery; BoN 36.5% remains best inference on **base** Gemma.

### 5. Run the benchmark

```bash
uv run python -m benchmark.run_all
```

Multi-turn function-calling mode and category filter:

- `--fc-mode nfc` (default): native OpenAI `tools` / `tool_calls` via llama-server `--jinja`
- `--fc-mode iffc`: Instruction-Followed Function Calling ([arXiv:2608.22472](https://arxiv.org/abs/2608.22472) Appendix B) — JSON list in assistant text; the API request does **not** include `tools`/`tool_choice`
- `--fc-mode cd`: TagDispatch-style two-pass constrained decoding ([arXiv:2601.04426](https://arxiv.org/abs/2601.04426) approximation on stock llama-server). **Not** native `tools` GBNF. Pass 1: GBNF over live tool names; pass 2: `json_schema` for the selected tool’s arguments. See METHODS.md.
- `--fc-mode gst`: Guided-Structured Templates ([arXiv:2509.18076](https://arxiv.org/abs/2509.18076)) on the NFC channel — 7-step `[GST]` scaffold in content, native `tool_calls` still executed. Prompt-only; not GBNF. See METHODS.md.
- `--fc-mode iffc-state`: IFFC JSON-in-text plus a harness-written `[STATE]` snapshot from **actual** tool execution results (METHODS §5 working memory). Same checker as IFFC. **Not** GST, **not** CD/GBNF, **not** LLM summarization. See METHODS.md.
- `--fc-mode iffc-state-bon`: Best-of-N majority voting overlay on **IFFC+STATE** (METHODS §4; Snell et al. arXiv:2408.03314). Sample N=5 (configurable `--bon-n`) at `temperature=0.7`, vote per step, execute the winner once. **~N× latency.** **Not** NFC/GST/CD. See METHODS.md.
- `--fc-mode xlam`: Salesforce xLAM JSON-in-text (`{"tool_calls":[{name, arguments}]}` in assistant content). Does **not** send `tools`/`tool_choice`. Empty `{"tool_calls": []}` ends the turn. Fair channel for `--model xlam-1b` (Run 19 NFC 0/200 was format-channel collapse; Run 20 `--fc-mode xlam` scored **6/200 = 3.0%** — still **very low**; **not** a method win). **Do not claim published ~78.94% parity.** See METHODS.md.
- `--iffc-strip-empty-assistant`: **diagnostic-only** IFFC history bookkeeping — omit trailing assistant `[]` stop turns from `chat_history` before the next BFCL user turn (and right after the within-turn stop). Default **off** (Run 5–13 IFFC unchanged). **Not** a training method and **not** a claim that standard IFFC improved. Notes record `iffc_strip_empty_assistant=1`; figures nest under `iffc-strip-empty-assistant/`. **Run 14 already scored this on `gemma4-e2b-sft-iffc-v2`: 4/200 = 2.0%** — did **not** rescue the never-start collapse; **do not present 2% as a method win** (see `THESIS.md`).
- `--categories multi_turn_base`: run only that multi-turn split (skip miss_func / miss_param / long_context)

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc
```

xLAM-1b-fc-r JSON-in-text adapter (serve command in `CLAUDE.md`). Run 19 NFC scored **0/200** (format-channel collapse). Run 20 `--fc-mode xlam` scored **6/200 = 3.0%** (fair channel; still **very low**; **not** a method win). Do **not** overwrite either run. **Do not claim published ~78.94% parity.**

```bash
uv run python -m benchmark.run_all --model xlam-1b --fc-mode xlam \
  --only-multi-turn --categories multi_turn_base
```

Gemma CD vs the scored NFC 27.0% / IFFC 30.5% `multi_turn_base` baselines (do not update THESIS.md tables until this run is scored):

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode cd \
  --notes "TagDispatch-style CD vs NFC 27% / IFFC 30.5%"
```

Long multi-turn jobs (especially `--fc-mode cd`) checkpoint each finished entry into `benchmark/results/benchmark.db`. Kill the process anytime; rerun the **same** `--model` / `--fc-mode` / `--categories` with `--resume` to continue that unfinished run (skips entry IDs already in `responses`). Pin a run with `--resume-run-id N`. See METHODS.md.

Gemma GST vs the scored NFC 27.0% / IFFC 30.5% / CD 10.5% `multi_turn_base` baselines (Run 9 scored **61/200 = 30.5%**, ties IFFC; see `THESIS.md`):

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode gst \
  --notes "GST+NFC vs NFC 27% / IFFC 30.5% / CD 10.5%"
```

Optional `--resume` if an unfinished gst run exists.

Gemma IFFC+STATE vs the scored NFC 27.0% / IFFC 30.5% / GST 30.5% / CD 10.5% `multi_turn_base` baselines (Run 10 scored **68/200 = 34.0%**, first method above the 30.5% plateau; see `THESIS.md`):

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state \
  --notes "IFFC+STATE vs NFC 27% / IFFC 30.5% / GST 30.5% / CD 10.5%"
```

Optional `--resume` if an unfinished iffc-state run exists.

Gemma Best-of-N on IFFC+STATE vs the scored NFC 27.0% / IFFC 30.5% / GST 30.5% / CD 10.5% / IFFC+STATE 34.0% `multi_turn_base` baselines (Run 11 scored **73/200 = 36.5%**, beats STATE by +2.5 pp at ~7× wall; see `THESIS.md`):

```bash
uv run python -m benchmark.smoke_bon
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state-bon \
  --notes "BoN N=5 on IFFC+STATE vs 34% / IFFC 30.5% / GST 30.5% / NFC 27% / CD 10.5%"
```

Optional `--bon-n N` and `--resume` if an unfinished iffc-state-bon run exists.

Gemma SFT-IFFC vs the scored NFC 27.0% / IFFC 30.5% / GST 30.5% / CD 10.5% / IFFC+STATE 34.0% / BoN 36.5% `multi_turn_base` baselines. Run 12 (`--model gemma4-e2b-sft-iffc`, LoRA best@1000) scored **1/200 = 0.5%**. Run 13 (`--model gemma4-e2b-sft-iffc-v2`, filtered-[] LoRA best@2000) scored **1/200 = 0.5%** — same collapse; the non-terminal-`[]` filter did **not** recover (see `THESIS.md`). Do **not** overwrite Run 12 or Run 13; do **not** BoN/STATE on these checkpoints; do **not** claim the `[]` filter fixed SFT.

**Run 14 already scored** the strip-trailing-`[]` history-bookkeeping diagnostic on the same v2 GGUF (`--fc-mode iffc --iffc-strip-empty-assistant`): **4/200 = 2.0%**. **DIAGNOSTIC — not a training method.** Did **not** rescue the never-start collapse. **Do not present 2% as a method win.** Do **not** overwrite Run 14. Figures: `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/iffc-strip-empty-assistant/figures`.

**Run 15 already scored** SFT-IFFC-v4 (`--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc`, LoRA best@1600): **30/200 = 15.0%**. **Scored recovery** from the 0.5–2% collapse, still **below IFFC 30.5% and below NFC 27.0%**. **Do not present 15% as beating IFFC.** Do **not** overwrite Run 15.

**Run 16 already scored** IFFC+STATE on that **same** v4 GGUF (`--model gemma4-e2b-sft-iffc-v4 --fc-mode iffc-state`): **26/200 = 13.0%**. **Scored negative/null overlay** vs Run 15 (−4 / −2.0 pp). STATE helps **base** Gemma, not this checkpoint. **Do not propose BoN as the next win on this ckpt.** Do **not** overwrite Run 16.

**Run 18 already scored** SFT-IFFC-v6 (`--model gemma4-e2b-sft-iffc-v6 --fc-mode iffc`, LoRA best@1800): **26/200 = 13.0%**. **Scored miss vs v4** (−4 / −2.0 pp vs Run 15 15.0%; modest +3 / +1.5 pp vs Run 17; ties Run 16 13.0%). Empty still essentially 1 (`_43`); residual state_mismatch 118 + under_call 55; avg api_calls/entry **13.34** (fixed v5’s 40.3 churn). **Do not present 13.0% as a method win.** **No BoN on this ckpt; no more epochs on the v6 mix.** Do **not** overwrite Run 18. v4 IFFC 15% remains best SFT recovery; BoN 36.5% remains best inference on **base** Gemma. Next SFT lever TBD (likely not another thin filter on the same family).

**Run 19 already scored** xLAM-1b-fc-r NFC (`--model xlam-1b --fc-mode nfc`, published `Salesforce/xLAM-1b-fc-r-gguf` Q4_K_M, registry [PR #41](https://github.com/sskarz/Thesis/pull/41)): **0/200 = 0.0%**. **Scored format-channel collapse** on this multi_turn NFC harness (all 200 `Model response list is empty for turn 0`). Wall **13.55 min**; finished `2026-09-22T14:26:42Z`. Avg `api_calls` **3.63**; gen_tps_mean ≈ **196.7** — **not** a dead server. Same-harness contrast: Gemma NFC Run 4 = **27.0%**. **Do not** claim xLAM scores 0 on BFCL overall; **do not** equate to published ~78.94%; **do not** present as a method win. Do **not** overwrite Run 19.

**Run 20 already scored** xLAM-1b-fc-r `--fc-mode xlam` (same GGUF, adapter [PR #42](https://github.com/sskarz/Thesis/pull/42)): **6/200 = 3.0%**. **Format-fixed fair-channel** after Run 19 NFC collapse. Smoke extracted calls; full score still **very low**. Wall **~13.28 h** (`2026-09-22T15:28:35Z` → `2026-09-23T04:45:30Z`). 194/200 eval errors; `state_mismatch` 128; under_call/missing exec 46; empty_turn 20. Flags: `hit_step_limit` 171/200; `xlam_unparseable` 111/200; `context_overflow` 42/200. Avg `api_calls` **50.47** (min 2, max 101) — tools actually ran. gen_tps_mean ≈ **91.6** (n=50). Correct: `multi_turn_base_{50,119,133,139,144,161}`. **Not** a method win; **not** published ~78.94%; still **below CD 10.5%**. Do **not** overwrite Run 20.

```bash
uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc-v2 --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc --iffc-strip-empty-assistant \
  --notes "DIAGNOSTIC: strip trailing IFFC assistant [] vs Run 13 0.5% (history bookkeeping, not a training method)"
```

Offline IFFC parser + mock chat-completions smoke (no llama-server):

```bash
uv run python -m benchmark.smoke_iffc
```

Offline CD constraint + two-pass mock loop (no llama-server):

```bash
uv run python -m benchmark.smoke_cd
```

Offline GST curriculum + mock NFC-channel loop (no llama-server):

```bash
uv run python -m benchmark.smoke_gst
```

Offline IFFC+STATE tracker + mock IFFC loop (no llama-server):

```bash
uv run python -m benchmark.smoke_iffc_state
```

Offline Best-of-N majority vote on IFFC+STATE (no llama-server):

```bash
uv run python -m benchmark.smoke_bon
```

The pipeline runs five steps in sequence:

1. **Server check** — verifies llama-server is reachable
2. **Generate responses** — sends all BFCL test entries to the model, saves responses + per-request timings
3. **Evaluate responses** — scores each response using BFCL's AST checker
4. **Collect metrics** — dedicated metrics run (3 warmup + 50 controlled requests) for throughput, TTFT, and memory
5. **Visualize** — generates thesis-quality charts (PNG + PDF at 300 DPI)

Total runtime is ~20-25 minutes on an M4 Pro.

### Server flags reference

| Flag | Purpose |
|---|---|
| `--jinja` | Activate the model's chat template (required for tool calling) |
| `-fa on` | Enable flash attention (`on`/`off`/`auto` — bare `-fa` no longer works) |
| `-hf` | Download GGUF from HuggingFace (cached after first run) |
| `-ngl 99` | Offload all layers to GPU (Metal on Mac, CUDA on Linux) |
| `-c 8192` | Context window size (8192 needed for multi-turn long_context) |
| `--temp 0.0` | Deterministic output (recommended by BFCL) |

## Results

Results are organized per model under `benchmark/results/<model-slug>/`, with a cross-model comparison dashboard:

```
benchmark/results/
├── Qwen2.5-3B-Instruct-Q4_K_M/
│   ├── responses/          # Raw model responses (JSONL) + timing CSV
│   ├── scores/             # accuracy.csv — per-category accuracy
│   ├── metrics/            # inference_metrics.json + per-request CSV
│   └── figures/            # Charts (PNG + PDF)
├── Llama-3.2-3B-Instruct-Q4_K_M/
│   └── ...                 # Same structure
└── comparison/
    ├── index.html          # Interactive comparison dashboard
    └── *.png / *.pdf       # Static comparison charts
```

### Comparison Dashboard

An interactive Chart.js dashboard compares all benchmarked models side-by-side:

```bash
# Option A: Open directly (works in most browsers)
open benchmark/results/comparison/index.html

# Option B: Serve locally (if your browser blocks file:// CDN loads)
cd benchmark/results/comparison && python3 -m http.server 8765
# Then visit http://localhost:8765
```

Features: category filters, model visibility toggles, mean/median metric switch, accuracy + throughput + latency + memory charts, ranked category detail, and a dynamic summary table. The dashboard uses an N-model data architecture — to add a new model, append an entry to the `MODELS[]` array in `index.html` (zero code changes needed).

### Accuracy (BFCL v4) — Apple M4 Pro

| Category | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|
| simple | **93.2%** (373/400) | 51.8% (207/400) |
| multiple | **92.5%** (185/200) | 33.5% (67/200) |
| irrelevance | **58.8%** (141/240) | 0.8% (2/240) |
| java | 37.0% (37/100) | **54.0%** (54/100) |
| javascript | 34.0% (17/50) | **70.0%** (35/50) |
| parallel | 0.0% | 0.0% |
| parallel_multiple | 0.0% | 0.0% |

> **Note:** parallel/parallel_multiple score 0% on both models — a known limitation of 3B-parameter models at Q4_K_M quantization (single tool call returned instead of multiple).

### Inference Performance — Apple M4 Pro

| Metric | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|
| Prompt throughput (mean) | 192 tok/s | **825 tok/s** |
| Generation throughput (mean) | **93 tok/s** | 89 tok/s |
| TTFT (mean) | **35 ms** | 249 ms |
| TTFT (median) | **12 ms** | 218 ms |
| Memory baseline (RSS) | **8,513 MB** | 10,778 MB |
| Memory peak (RSS) | **8,547 MB** | 12,384 MB |

## Testing a Different Model

The pipeline includes a model registry in `benchmark/config.py`. Use the `--model` flag to select which model to benchmark:

```bash
# Available models:
#   qwen2.5-3b   — Qwen2.5-3B-Instruct (Q4_K_M)  [default]
#   llama-3.2-3b  — Llama-3.2-3B-Instruct (Q4_K_M)

# 1. Start llama-server with the desired model
./llama.cpp/build/bin/llama-server \
  --jinja -fa on \
  -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 8192 --temp 0.0

# 2. Run the benchmark
uv run python -m benchmark.run_all --model llama-3.2-3b
```

Results land in `benchmark/results/Llama-3.2-3B-Instruct-Q4_K_M/`.

Mac-local SFT GGUF (not HuggingFace). Start llama-server with `-m`, then pass a distinct `--model` so scores do not overwrite base Gemma-4-E2B. Run 12 already scored the v1 checkpoint (**1/200 = 0.5%**, scored negative — do not overwrite). Run 13 already scored the v2 filtered-[] checkpoint (**1/200 = 0.5%**, same collapse — do not overwrite; figures `benchmark/results/Gemma-4-E2B-it-SFT-IFFC-v2-Q4_K_M/figures`). Run 14 already scored `--iffc-strip-empty-assistant` on that same v2 GGUF (**4/200 = 2.0%**, **DIAGNOSTIC** history bookkeeping — **not** a method win; do not overwrite; figures `.../iffc-strip-empty-assistant/figures`). Run 15 already scored SFT-IFFC-v4 best@1600 (**30/200 = 15.0%**, scored recovery, still below IFFC/NFC — do not overwrite; do not present as beating IFFC). Run 16 already scored `--fc-mode iffc-state` on that same v4 GGUF (**26/200 = 13.0%**, scored negative/null vs Run 15 — do not overwrite; do not BoN this ckpt). Run 17 already scored SFT-IFFC-v5 best@1800 (**23/200 = 11.5%**, scored regression vs Run 15 — do not overwrite; do not present as a method win; do not BoN / more-epochs this mix). Run 18 already scored SFT-IFFC-v6 best@1800 (**26/200 = 13.0%**, scored miss vs Run 15 — do not overwrite; do not present as a method win; do not BoN / more-epochs this mix). Run 19 already scored xLAM-1b-fc-r NFC (**0/200 = 0.0%**, scored format-channel collapse on this harness — do not overwrite; do not claim published-BFCL parity). Run 20 already scored `--fc-mode xlam` (**6/200 = 3.0%**, fair channel, still very low — do not overwrite; do not present as a method win; do not claim published ~78.94%):

```bash
# v1 (Run 12 already scored 0.5% — do not overwrite)
./llama.cpp/build/bin/llama-server --jinja -fa on \
  -m sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4
uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc

# v4 IFFC (Run 15 already scored 15.0% recovery, still below IFFC/NFC — do not overwrite)
# ./llama.cpp/build/bin/llama-server --jinja -fa on \
#   -m sft/fused/gemma-4-e2b-it-iffc-v4-best1600-Q4_K_M.gguf \
#   --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4
# uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc-v4 --only-multi-turn \
#   --categories multi_turn_base --fc-mode iffc

# v4 + STATE (Run 16 already scored 13.0% null/negative vs Run 15 — do not overwrite;
# do not BoN this ckpt)
# uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc-v4 --only-multi-turn \
#   --categories multi_turn_base --fc-mode iffc-state

# v5 IFFC (Run 17 already scored 11.5% regression vs Run 15 — do not overwrite;
# do not present as a method win; do not BoN / more-epochs this mix)
# ./llama.cpp/build/bin/llama-server --jinja -fa on \
#   -m sft/fused/gemma-4-e2b-it-iffc-v5-best1800-Q4_K_M.gguf \
#   --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4
# uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc-v5 --only-multi-turn \
#   --categories multi_turn_base --fc-mode iffc

# v6 IFFC (Run 18 already scored 13.0% miss vs Run 15 — do not overwrite;
# do not present as a method win; do not BoN / more-epochs this mix)
# ./llama.cpp/build/bin/llama-server --jinja -fa on \
#   -m sft/fused/gemma-4-e2b-it-iffc-v6-best1800-Q4_K_M.gguf \
#   --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4
# uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc-v6 --only-multi-turn \
#   --categories multi_turn_base --fc-mode iffc

# xLAM-1b NFC (Run 19 already scored 0.0% format-channel collapse — do not overwrite).
# xLAM-1b --fc-mode xlam (Run 20 already scored 3.0% fair channel — still very low;
# do not overwrite; do not present as a method win; do not claim published ~78.94%).
# ./llama.cpp/build/bin/llama-server --jinja -fa on \
#   -hf Salesforce/xLAM-1b-fc-r-gguf:Q4_K_M \
#   --host 0.0.0.0 --port 8080 -ngl 99 -c 16384 --temp 0.0 -np 4
# uv run python -m benchmark.run_all --model xlam-1b --fc-mode xlam \
#   --only-multi-turn --categories multi_turn_base
```

To add a new model, add an entry to the `MODELS` dict in `benchmark/config.py`.

## Project Structure

```
Thesis/
├── README.md
├── THESIS.md                          # Informal project notes (not the paper)
├── paper/                             # Formal LaTeX draft (arXiv-style)
│   ├── main.tex
│   ├── references.bib
│   ├── Makefile
│   └── figures/
├── CLAUDE.md                          # Development notes & reproducibility reference
├── pyproject.toml                     # Python dependencies (managed by uv)
├── llama-cpp-bfcl-setup-guide.md      # Full setup guide
├── llama.cpp/                         # Git-ignored; built locally
│   └── build/bin/llama-server
├── sft/                               # MLX LoRA SFT v1 (IFFC; see sft/README.md)
└── benchmark/
    ├── __init__.py
    ├── config.py                      # Server URL, model registry, paths, BFCL categories
    ├── generate_responses.py          # Send BFCL tests to llama-server, save responses + timings
    ├── evaluate_responses.py          # Score responses using BFCL's AST checker
    ├── collect_metrics.py             # Dedicated metrics collection (warmup + controlled requests)
    ├── visualize.py                   # Generate thesis-quality charts
    ├── run_all.py                     # Orchestrator (runs all steps in sequence)
    └── results/
        ├── <model-slug>/             # Per-model results
        │   ├── responses/
        │   ├── scores/
        │   ├── metrics/
        │   └── figures/
        └── comparison/
            ├── index.html            # Interactive N-model comparison dashboard
            └── *.png / *.pdf         # Static comparison charts
```

## Known Issues

- **`-fa` flag:** llama.cpp now requires `-fa on` (not bare `-fa`). This is a recent change.
- **bfcl-eval requires Python 3.13:** The `qwen_agent` dependency (pulled in by bfcl-eval) has compatibility issues with Python 3.14.
- **`soundfile` dependency:** Required by bfcl-eval's import chain (`ast_checker` -> `model_config` -> `qwen_agent` -> `soundfile`), even though it's not used at runtime.
- **`powermetrics` needs sudo:** Power draw measurement is optional and skipped automatically when not running as root.
