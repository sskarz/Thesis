# Optimizing Tool Calling on Edge Devices Using RAG and Structured Output

**Thesis:** *Optimized Tool Calling for On-Device LLMs in Resource Constraint Environments.*

Benchmark LLM tool/function-calling accuracy and inference performance on local hardware using [llama.cpp](https://github.com/ggml-org/llama.cpp) and the [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html).

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
- `--categories multi_turn_base`: run only that multi-turn split (skip miss_func / miss_param / long_context)

```bash
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc
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

To add a new model, add an entry to the `MODELS` dict in `benchmark/config.py`.

## Project Structure

```
Thesis/
├── README.md
├── CLAUDE.md                          # Development notes & reproducibility reference
├── pyproject.toml                     # Python dependencies (managed by uv)
├── llama-cpp-bfcl-setup-guide.md      # Full setup guide
├── llama.cpp/                         # Git-ignored; built locally
│   └── build/bin/llama-server
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
