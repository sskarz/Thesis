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
│   ├── config.py                      # Model registry, server URL, paths, RUN_ID
│   ├── db.py                          # SQLite data-access layer (8 tables, all CRUD)
│   ├── generate_responses.py          # Send BFCL tests → save responses + timings to DB
│   ├── generate_multi_turn.py         # Multi-turn tool execution loop → DB
│   ├── iffc.py                        # IFFC Appendix B prompt + JSON-in-text parser
│   ├── cd.py                          # TagDispatch-style CD (name GBNF + per-tool json_schema)
│   ├── gst.py                         # GST 7-step curriculum (arXiv:2509.18076) on NFC tools
│   ├── iffc_state.py                  # IFFC+STATE overlay (harness-written working memory)
│   ├── bon.py                         # Best-of-N majority vote overlay on IFFC+STATE
│   ├── smoke_iffc.py                  # Offline IFFC smoke (parser + mock loop)
│   ├── smoke_cd.py                    # Offline CD smoke (constraint checks + mock two-pass)
│   ├── smoke_gst.py                   # Offline GST smoke (template in payload + mock NFC loop)
│   ├── smoke_iffc_state.py            # Offline IFFC+STATE smoke (STATE + last-2-turns + mock IFFC)
│   ├── smoke_bon.py                   # Offline BoN smoke (majority vote + mock IFFC+STATE loop)
│   ├── coerce_types.py                # Schema-aware arg type coercion (fixes Llama string serialization)
│   ├── parse_functiongemma.py         # Custom output parser for FunctionGemma
│   ├── evaluate_responses.py          # Score responses using BFCL AST checker → DB
│   ├── evaluate_multi_turn.py         # Score multi-turn using execution checker → DB
│   ├── collect_metrics.py             # Throughput, TTFT, memory measurement → DB
│   ├── visualize.py                   # Per-model thesis-quality charts (reads from DB)
│   ├── compare.py                     # Cross-model comparison charts (reads from DB)
│   ├── run_all.py                     # Orchestrator (creates run, all steps, finishes run)
│   ├── api.py                        # FastAPI backend wrapping db.py (uvicorn :8000)
│   ├── migrate_to_db.py              # One-time import of flat-file results into DB
│   └── results/
│       └── benchmark.db               # SQLite database (all runs, git-ignored)
├── sft/                               # MLX LoRA SFT v1 (IFFC mix; Apple Silicon)
│   ├── download_datasets.py           # ToolACE + xLAM-60k → sft/data/raw/
│   ├── format_iffc.py                 # IFFC chat JSONL (reuses benchmark.iffc)
│   ├── split_data.py                  # Frozen train/val/corpus_test + MANIFEST.json
│   ├── train_mlx.py / train_mlx.sh    # mlx_lm.lora wrapper (smoke vs full)
│   └── README.md                      # Mac copy-paste commands
├── frontend/                          # React + Vite dashboard (git-ignored: node_modules, dist)
│   ├── src/
│   │   ├── api/                       # API client, types, SWR hooks
│   │   ├── components/                # Reusable UI components
│   │   ├── pages/                     # Route pages (Comparison, ModelDetail, Runs, RunDetail)
│   │   ├── hooks/                     # Custom React hooks
│   │   └── lib/                       # Colors, formatters, Chart.js setup
│   ├── index.html
│   └── vite.config.ts                 # Proxies /api to :8000
└── tests/
    ├── conftest.py                    # Shared fixtures
    ├── helpers.py                     # Test utilities
    ├── benchmark/                     # Unit tests for benchmark modules
    ├── sft/                           # Unit tests for MLX SFT formatters/splits
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
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 8 -cb -ub 1024
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

### Start the frontend dashboard

```bash
# Terminal 1: API backend
uv run uvicorn benchmark.api:app --reload --port 8000

# Terminal 2: Vite dev server
cd frontend && npm run dev
```

Open `http://localhost:5173` for the interactive dashboard.

### Run the benchmark

```bash
uv run python -m benchmark.run_all --model qwen2.5-3b
uv run python -m benchmark.run_all --model qwen2.5-3b --notes "RTX 4090 baseline"
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode gst \
  --notes "GST+NFC vs NFC 27% / IFFC 30.5% / CD 10.5%"
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state \
  --notes "IFFC+STATE vs NFC 27% / IFFC 30.5% / GST 30.5% / CD 10.5%"
uv run python -m benchmark.run_all --model gemma4-e2b --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc-state-bon \
  --notes "BoN N=5 on IFFC+STATE vs 34% / IFFC 30.5% / GST 30.5% / NFC 27% / CD 10.5%"
uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc
uv run python -m benchmark.run_all --model gemma4-e2b-sft-iffc-v2 --only-multi-turn \
  --categories multi_turn_base --fc-mode iffc --iffc-strip-empty-assistant \
  --notes "DIAGNOSTIC: strip trailing IFFC assistant [] vs Run 13 0.5% (history bookkeeping, not a training method)"
# Run 14 already scored this diagnostic: 4/200 = 2.0% — do not overwrite;
# do not present 2% as a method win.
```

### Query the database

```python
from benchmark import db
db.list_runs()                          # all runs
db.get_latest_run("qwen2.5-3b")        # most recent finished run
db.get_scores(run_id)                   # accuracy by category
db.get_inference_metrics(run_id)        # throughput, TTFT, memory
db.get_all_model_results()              # latest run per model (for comparison)
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
| `-np 8` | 8 parallel request slots (matches `config.PARALLEL_SLOTS`) |
| `-cb` | Continuous batching — lets new requests fill freed slots immediately |
| `-ub 1024` | Larger micro-batch for faster prompt processing (default 512) |

### Start Qwen3-4B server

```bash
cd /home/sanskar/Documents/Github/Thesis/llama.cpp
./build/bin/llama-server --jinja -fa on \
  -hf bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 8 -cb -ub 1024
```

**Note:** Qwen3 models have a built-in thinking mode that produces `reasoning_content` alongside `content`. Tool calling still works correctly (proper `tool_calls` field). The thinking tokens consume extra generation tokens and time. To disable thinking, prepend `/no_think` to the system prompt.

### Start FunctionGemma server

```bash
cd /home/sanskar/Documents/Github/Thesis/llama.cpp
./build/bin/llama-server --jinja -fa on \
  -hf ggml-org/functiongemma-270m-it-GGUF \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 32768 --temp 0.0 \
  -np 4 -cb -ub 1024
```

**Note:** FunctionGemma uses a non-standard output format (`<start_function_call>call:name{args}<end_function_call>`) that llama-server cannot parse into OpenAI-style `tool_calls`. The custom parser in `benchmark/parse_functiongemma.py` post-processes the raw `content` field into the `[{"func": {args}}]` format expected by the BFCL evaluator. This is called automatically by `generate_responses.py`.

### Start SFT Gemma (Mac-local GGUF)

Local fused Q4_K_M — **not** `-hf`. Path may vary slightly. Use `--model gemma4-e2b-sft-iffc` for scoring identity so results do not overwrite base `gemma4-e2b`.

```bash
cd /path/to/Thesis/llama.cpp
./build/bin/llama-server --jinja -fa on \
  -m ../sft/fused/gemma-4-e2b-it-iffc-best1000-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --temp 0.0 \
  -np 4 -cb -ub 1024
```

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

## Storage Architecture

All benchmark results are stored in a SQLite database at `benchmark/results/benchmark.db`. Each pipeline execution creates a new **run** with auto-detected metadata (GPU, CPU, RAM, llama.cpp version, bfcl version). Multiple runs of the same model coexist — no data is ever overwritten.

**Tables:** `runs`, `responses`, `single_turn_timings`, `multi_turn_timings`, `scores`, `eval_errors`, `inference_metrics`, `metrics_per_request`. All linked by `run_id` foreign key.

**Data flow:** `run_all.py` creates a run → sets `config.RUN_ID` → all modules write to DB via `db.py` → `run_all.py` finishes the run. Visualization and comparison modules read from DB using `config.RUN_ID` or `db.get_latest_run()`.

**Frontend:** The React dashboard at `frontend/` reads from the API (`benchmark/api.py`) which wraps `db.py`. Run both the API server and Vite dev server for development.

## Known Issues / Gotchas

1. **`-fa` flag change:** The `-fa` (flash attention) flag now requires an explicit value. Use `-fa on`, not bare `-fa`. This is a recent llama.cpp change.
2. **Model download on first run:** First launch takes several minutes to download the GGUF (~1.8 GB for Q4_K_M). Subsequent runs use the cached file at `~/Library/Caches/llama.cpp/` (macOS) or `~/.cache/llama.cpp/` (Linux).
3. **`llama.cpp/` is git-ignored:** The entire llama.cpp directory is in `.gitignore`. It must be cloned and built locally on each machine.
4. **FunctionGemma output format:** FunctionGemma does not produce OpenAI-style `tool_calls`. Its output arrives as raw text in the `content` field using custom control tokens. The `parse_functiongemma.py` module handles this transparently.
5. **HTTPS support for `-hf` flag:** On Linux, llama.cpp must be built with `-DLLAMA_OPENSSL=ON` (requires `libssl-dev`) to use the `-hf` flag for downloading models.
6. **Context window:** Use `-c 65536` with `-np 8` (8192 per slot) to accommodate multi-turn long_context entries and maximize parallel request processing. The RTX 4090 has ample VRAM for this (~14.5 GB used out of 24 GB).
7. **Llama argument type serialization:** Llama 3.2 wraps all tool-call arguments as JSON strings (e.g., `"5"` instead of `5`). The `coerce_types.py` module post-processes arguments using the BFCL function schema to fix this. A tool-calling system prompt is also injected by `generate_responses.py` when tools are present. Both are applied to all models uniformly (no-op for Qwen which produces correct types natively).
8. **Database file is git-ignored:** `benchmark/results/benchmark.db` is under the `benchmark/results/` gitignore pattern. After cloning on a new machine, run benchmarks to populate the DB.
