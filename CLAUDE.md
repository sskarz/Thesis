# Thesis: Optimizing Tool Calling on Edge Devices Using RAG and Structured Output

## Project Overview

Evaluate LLM tool/function calling on local hardware using llama.cpp + BFCL (Berkeley Function Calling Leaderboard). Develop on Mac (Apple Silicon), then migrate to NVIDIA Jetson Orin Nano Super.

## Repository Structure

```
Thesis/
├── CLAUDE.md                          # This file
├── README.md
├── .gitignore
├── llama-cpp-bfcl-setup-guide.md      # Full setup guide (Steps 1-7)
└── llama.cpp/                         # Git-ignored; built locally
    └── build/bin/llama-server         # The server binary
```

## Development Environment

- **Machine:** Mac with Apple M4 Pro
- **OS:** macOS 26.3 (Tahoe)
- **GPU backend:** Metal (Apple Silicon)
- **cmake:** 4.2.3 (installed via Homebrew)
- **Xcode CLI tools:** Required (pre-installed)

## What Has Been Done (Steps 1-3)

### Step 1: Prerequisites

```bash
brew install cmake
```

Only cmake is needed for building llama.cpp. Python/conda are needed later for BFCL (Step 4+).

### Step 2: Build llama.cpp

```bash
cd /Users/sanskarthapa/Documents/GitHub/Thesis
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

- **llama.cpp version:** 8111 (commit `11c325c6e`)
- **Binary location:** `llama.cpp/build/bin/llama-server`
- The `llama.cpp/` directory is git-ignored; it must be cloned and built locally

### Step 3: Launch the Server

```bash
cd /Users/sanskarthapa/Documents/GitHub/Thesis/llama.cpp

./build/bin/llama-server \
  --jinja -fa on \
  -hf bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 4096 \
  --temp 0.0
```

**IMPORTANT:** The `-fa` flag requires an explicit value (`on`, `off`, or `auto`). The original guide uses bare `-fa` which no longer works — use `-fa on` instead.

- **Model:** Qwen2.5-3B-Instruct at Q4_K_M quantization (~1.8 GB)
- **Model cache:** `~/Library/Caches/llama.cpp/` (auto-downloaded on first run from HuggingFace)
- **API endpoint:** `http://localhost:8080` (OpenAI-compatible)
- **Slots:** 4 concurrent inference slots

### Verification (confirmed working)

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "The city name"}
          },
          "required": ["city"]
        }
      }
    }]
  }'
```

**Result:** Returns `finish_reason: "tool_calls"` with `get_weather(city: "San Francisco")`.

**Observed performance (M4 Pro):**
- Prompt processing: ~591 tokens/sec
- Generation: ~100 tokens/sec

## What Remains (Steps 4-7)

- **Step 4:** Install BFCL (`conda create -n BFCL python=3.10`, `pip install bfcl-eval`)
- **Step 5:** Run BFCL evaluation (`bfcl generate` + `bfcl evaluate`)
- **Step 6:** Collect inference metrics (tokens/sec, memory, power draw)
- **Step 7:** Test additional models and quantization levels

See `llama-cpp-bfcl-setup-guide.md` for full details on Steps 4-7.

## Quick Reference

### Rebuild llama.cpp from scratch

```bash
cd /Users/sanskarthapa/Documents/GitHub/Thesis
rm -rf llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

### Start the server

```bash
cd /Users/sanskarthapa/Documents/GitHub/Thesis/llama.cpp
./build/bin/llama-server --jinja -fa on \
  -hf bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 4096 --temp 0.0
```

### Check if server is running

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

### Server flag reference

| Flag | Purpose |
|------|---------|
| `--jinja` | Activate model's chat template (required for tool calling) |
| `-fa on` | Enable flash attention (note: requires explicit `on`/`off`/`auto`) |
| `-hf` | Download GGUF from HuggingFace (cached after first run) |
| `-ngl 99` | Offload all layers to GPU (Metal on Mac, CUDA on Jetson) |
| `-c 4096` | Context window size |
| `--temp 0.0` | Deterministic output (BFCL recommends this) |

## Known Issues / Gotchas

1. **`-fa` flag change:** The `-fa` (flash attention) flag now requires an explicit value. Use `-fa on`, not bare `-fa`. This is a recent llama.cpp change.
2. **Model download on first run:** First launch takes several minutes to download the GGUF (~1.8 GB for Q4_K_M). Subsequent runs use the cached file at `~/Library/Caches/llama.cpp/`.
3. **`llama.cpp/` is git-ignored:** The entire llama.cpp directory is in `.gitignore`. It must be cloned and built locally on each machine.
