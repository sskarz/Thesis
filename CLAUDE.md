# Thesis: Optimizing Tool Calling on Edge Devices Using RAG and Structured Output

## Project Overview

Evaluate LLM tool/function calling on local hardware using llama.cpp + BFCL v4 (Berkeley Function Calling Leaderboard). Developed on Mac (Apple Silicon), now running on Ubuntu desktop with NVIDIA RTX 4090. Future target: NVIDIA Jetson Orin Nano Super.

## Repository Structure

```
Thesis/
├── CLAUDE.md                          # This file
├── README.md
├── THESIS.md                          # Full thesis writeup
├── pyproject.toml                     # Python dependencies (managed by uv)
├── .gitignore
├── llama.cpp/                         # Git-ignored; built locally
│   └── build/bin/llama-server         # The server binary
├── benchmark/
│   ├── config.py                      # Model registry, server URL, paths
│   ├── generate_responses.py          # Send BFCL tests → save responses + timings
│   ├── generate_multi_turn.py         # Multi-turn tool execution loop
│   ├── parse_functiongemma.py         # Custom output parser for FunctionGemma
│   ├── evaluate_responses.py          # Score responses using BFCL AST checker
│   ├── evaluate_multi_turn.py         # Score multi-turn using execution checker
│   ├── collect_metrics.py             # Throughput, TTFT, memory measurement
│   ├── visualize.py                   # Per-model thesis-quality charts
│   ├── run_all.py                     # Orchestrator (all steps in sequence)
│   └── results/
│       ├── Qwen2.5-3B-Instruct-Q4_K_M/    # Per-model results
│       ├── Llama-3.2-3B-Instruct-Q4_K_M/
│       ├── FunctionGemma-270M-IT/
│       └── comparison/
│           ├── index.html             # Interactive comparison dashboard (Chart.js)
│           └── *.png / *.pdf          # Static comparison charts
└── tests/
    ├── conftest.py                    # Shared fixtures
    ├── helpers.py                     # Test utilities
    ├── benchmark/                     # Unit tests for benchmark modules
    └── live/                          # Optional live smoke tests
```

## Development Environments

### Current: Ubuntu Desktop (NVIDIA RTX 4090)
- **Machine:** Ubuntu desktop with NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **OS:** Linux 6.17.0-19-generic
- **GPU backend:** CUDA 13.0
- **Build:** `cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENSSL=ON`

### Previous: Mac (Apple M4 Pro)
- **Machine:** Mac with Apple M4 Pro
- **OS:** macOS 26.3 (Tahoe)
- **GPU backend:** Metal (Apple Silicon)
- **Build:** `cmake -B build -DGGML_METAL=ON`

## BFCL Version

**Using BFCL v4** (bfcl-eval package version 2026.3.11). Upgraded from v3.

Key v3 → v4 changes:
- Data files renamed: `BFCL_v3_*` → `BFCL_v4_*`
- `simple` split into `simple_python`, `simple_java`, `simple_javascript`
- `MULTI_TURN_FUNC_DOC_FILE_MAPPING` moved to `bfcl_eval.constants.executable_backend_config`
- `ast_checker` `language` parameter now requires `bfcl_eval.constants.enums.Language` enum (not strings)
- New categories added (web_search, memory, format_sensitivity) — not yet integrated

## Quick Reference

### Build llama.cpp (CUDA)

```bash
cd /home/sanskar/Documents/Github/Thesis
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENSSL=ON
cmake --build build --config Release -j$(nproc)
```

### Start the server

```bash
cd /home/sanskar/Documents/Github/Thesis/llama.cpp
./build/bin/llama-server --jinja -fa on \
  -hf bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --temp 0.0
```

### Check if server is running

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

### Run tests

```bash
uv run pytest -q
BENCHMARK_LIVE=1 uv run pytest -q -m live
```

### Run the benchmark

```bash
uv run python -m benchmark.run_all --model qwen2.5-3b
```

### Server flag reference

| Flag | Purpose |
|------|---------|
| `--jinja` | Activate model's chat template (required for tool calling) |
| `-fa on` | Enable flash attention (note: requires explicit `on`/`off`/`auto`) |
| `-hf` | Download GGUF from HuggingFace (cached after first run) |
| `-ngl 99` | Offload all layers to GPU (Metal on Mac, CUDA on Linux) |
| `-c 8192` | Context window size (8192 needed for multi-turn long_context) |
| `--temp 0.0` | Deterministic output (BFCL recommends this) |

### Start FunctionGemma server

```bash
cd /home/sanskar/Documents/Github/Thesis/llama.cpp
./build/bin/llama-server --jinja -fa on \
  -hf ggml-org/functiongemma-270m-it-GGUF \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --temp 0.0
```

**Note:** FunctionGemma uses a non-standard output format (`<start_function_call>call:name{args}<end_function_call>`) that llama-server cannot parse into OpenAI-style `tool_calls`. The custom parser in `benchmark/parse_functiongemma.py` post-processes the raw `content` field into the `[{"func": {args}}]` format expected by the BFCL evaluator. This is called automatically by `generate_responses.py`.

## BFCL Categories

### Single-turn (AST-evaluated)
| Category | File | Test Cases |
|----------|------|-----------|
| simple | BFCL_v4_simple_python.json | 400 |
| multiple | BFCL_v4_multiple.json | 200 |
| parallel | BFCL_v4_parallel.json | 200 |
| parallel_multiple | BFCL_v4_parallel_multiple.json | 200 |
| irrelevance | BFCL_v4_irrelevance.json | 240 |
| java | BFCL_v4_simple_java.json | 100 |
| javascript | BFCL_v4_simple_javascript.json | 50 |

### Multi-turn (execution-evaluated)
| Category | File | Test Cases |
|----------|------|-----------|
| multi_turn_base | BFCL_v4_multi_turn_base.json | 200 |
| multi_turn_miss_func | BFCL_v4_multi_turn_miss_func.json | 200 |
| multi_turn_miss_param | BFCL_v4_multi_turn_miss_param.json | 200 |
| multi_turn_long_context | BFCL_v4_multi_turn_long_context.json | 200 |

## Known Issues / Gotchas

1. **`-fa` flag change:** The `-fa` (flash attention) flag now requires an explicit value. Use `-fa on`, not bare `-fa`. This is a recent llama.cpp change.
2. **Model download on first run:** First launch takes several minutes to download the GGUF (~1.8 GB for Q4_K_M). Subsequent runs use the cached file at `~/Library/Caches/llama.cpp/` (macOS) or `~/.cache/llama.cpp/` (Linux).
3. **`llama.cpp/` is git-ignored:** The entire llama.cpp directory is in `.gitignore`. It must be cloned and built locally on each machine.
4. **FunctionGemma output format:** FunctionGemma does not produce OpenAI-style `tool_calls`. Its output arrives as raw text in the `content` field using custom control tokens. The `parse_functiongemma.py` module handles this transparently.
5. **HTTPS support for `-hf` flag:** On Linux, llama.cpp must be built with `-DLLAMA_OPENSSL=ON` (requires `libssl-dev`) to use the `-hf` flag for downloading models.
6. **Context window:** Use `-c 8192` (not 4096) to accommodate multi-turn long_context entries where two large tool doc sets can exceed 4096 tokens.
