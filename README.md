# Optimizing Tool Calling on Edge Devices Using RAG and Structured Output

Benchmark LLM tool/function-calling accuracy and inference performance on local hardware using [llama.cpp](https://github.com/ggml-org/llama.cpp) and the [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html).

## Prerequisites

| Requirement | Tested version | Notes |
|---|---|---|
| macOS (Apple Silicon) | 26.3 (Tahoe), M4 Pro | Metal GPU backend |
| cmake | 4.2.3 | `brew install cmake` |
| Xcode CLI tools | — | Pre-installed on macOS |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager |

> Future target: NVIDIA Jetson Orin Nano Super (CUDA backend).

## Quick Start

### 1. Clone and build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
cd ..
```

The `llama.cpp/` directory is git-ignored. It must be built locally on each machine.

### 2. Start the inference server

```bash
./llama.cpp/build/bin/llama-server \
  --jinja -fa on \
  -hf bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 4096 --temp 0.0
```

The first launch downloads the GGUF model (~1.8 GB) to `~/Library/Caches/llama.cpp/`. Subsequent runs use the cache.

Verify the server is up:

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

### 3. Install Python dependencies

```bash
uv sync
```

This creates a `.venv/` and installs all dependencies from `pyproject.toml`.

### 4. Run the benchmark

```bash
uv run python -m benchmark.run_all
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
| `-ngl 99` | Offload all layers to GPU (Metal on Mac, CUDA on Jetson) |
| `-c 4096` | Context window size |
| `--temp 0.0` | Deterministic output (recommended by BFCL) |

## Results

Results are organized per model under `benchmark/results/<model-slug>/`:

```
benchmark/results/
└── Qwen2.5-3B-Instruct-Q4_K_M/
    ├── responses/          # Raw model responses (JSONL) + timing CSV
    ├── scores/             # accuracy.csv — per-category accuracy
    ├── metrics/            # inference_metrics.json + per-request CSV
    └── figures/            # Charts (PNG + PDF)
```

### Qwen2.5-3B-Instruct (Q4_K_M) on Apple M4 Pro

#### Accuracy (BFCL v3)

| Category | Correct | Total | Accuracy |
|---|---|---|---|
| simple | 373 | 400 | 93.2% |
| multiple | 185 | 200 | 92.5% |
| irrelevance | 141 | 240 | 58.8% |
| java | 37 | 100 | 37.0% |
| javascript | 17 | 50 | 34.0% |
| parallel | 0 | 200 | 0.0% |
| parallel_multiple | 0 | 200 | 0.0% |

> **Note:** parallel/parallel_multiple score 0% because the model consistently returns a single tool call instead of multiple. This is a known limitation of Qwen2.5-3B, not a pipeline bug.

#### Inference Performance

| Metric | Value |
|---|---|
| Prompt throughput (mean) | 192 tok/s |
| Generation throughput (mean) | 93 tok/s |
| TTFT (median) | 12 ms |
| TTFT (mean) | 35 ms |
| Memory baseline (RSS) | 8,513 MB |
| Memory peak (RSS) | 8,547 MB |

## Testing a Different Model

1. Edit `benchmark/config.py` — update the model identity constants:

   ```python
   MODEL_HF_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
   MODEL_QUANT = "Q4_K_M"
   MODEL_NAME = "qwen2.5-7b"          # Must match what the server reports in /v1/models
   MODEL_DISPLAY = f"Qwen2.5-7B-Instruct ({MODEL_QUANT})"
   MODEL_SLUG = f"Qwen2.5-7B-Instruct-{MODEL_QUANT}"
   ```

2. Restart `llama-server` with the new `-hf` argument.

3. Run the benchmark again — results go into a new directory automatically:

   ```bash
   uv run python -m benchmark.run_all
   ```

   Output lands in `benchmark/results/Qwen2.5-7B-Instruct-Q4_K_M/`.

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
    ├── config.py                      # Server URL, model identity, paths, BFCL categories
    ├── generate_responses.py          # Send BFCL tests to llama-server, save responses + timings
    ├── evaluate_responses.py          # Score responses using BFCL's AST checker
    ├── collect_metrics.py             # Dedicated metrics collection (warmup + controlled requests)
    ├── visualize.py                   # Generate thesis-quality charts
    ├── run_all.py                     # Orchestrator (runs all steps in sequence)
    └── results/                       # Git-ignored output
        └── <model-slug>/
            ├── responses/
            ├── scores/
            ├── metrics/
            └── figures/
```

## Known Issues

- **`-fa` flag:** llama.cpp now requires `-fa on` (not bare `-fa`). This is a recent change.
- **bfcl-eval requires Python 3.13:** The `qwen_agent` dependency (pulled in by bfcl-eval) has compatibility issues with Python 3.14.
- **`soundfile` dependency:** Required by bfcl-eval's import chain (`ast_checker` -> `model_config` -> `qwen_agent` -> `soundfile`), even though it's not used at runtime.
- **`powermetrics` needs sudo:** Power draw measurement is optional and skipped automatically when not running as root.
