# Tool Calling Baseline Setup Guide

**Thesis:** Optimizing Tool Calling on Edge Devices Using RAG and Structured Output
**Control Device (Target):** NVIDIA Jetson Orin Nano Super Developer Kit
**Development Device (Current):** Mac (Apple Silicon)

---

## Overview

This guide walks through setting up a reproducible baseline for evaluating **LLM tool/function calling** on local hardware. The pipeline is:

1. **llama.cpp** serves a quantized model with an OpenAI-compatible API
2. **BFCL** (Berkeley Function Calling Leaderboard) sends tool-calling prompts to that API
3. You collect accuracy scores and inference metrics

We develop and validate on Mac first, then migrate to Jetson later (only the build step changes).

---

## Prerequisites

### macOS (Apple Silicon)

```bash
# Xcode command line tools (if not already installed)
xcode-select --install

# Homebrew packages
brew install cmake python@3.10 conda git

# Verify Metal support (should show your Apple GPU)
system_profiler SPDisplaysDataType | grep "Chipset Model"
```

> **Note:** On Apple Silicon, llama.cpp uses the **Metal** backend for GPU acceleration instead of CUDA. This is handled automatically at build time.

---

## Step 1 — Build llama.cpp

### macOS (Apple Silicon — Metal backend)

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

The binary lands at `build/bin/llama-server`. Metal is enabled by default on macOS, but the explicit flag makes intent clear.

### Jetson Orin Nano Super (for later — CUDA backend)

```bash
# Same repo, different GPU backend
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

---

## Step 2 — Choose a Model

For an 8 GB Apple Silicon Mac (or the 8 GB Jetson), these models all fit comfortably at Q4_K_M quantization:

| Model | ~RAM (Q4_K_M) | Tool Calling Support | Notes |
|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | ~1.2 GB | Native (Jinja template) | Smallest viable tool caller |
| **Qwen2.5-3B-Instruct** | ~2.1 GB | Native (Jinja template) | **Recommended control model** |
| **Llama-3.2-3B-Instruct** | ~2.1 GB | Native | Meta's edge-optimized model |
| **Hermes-2-Pro-Llama-3-8B** | ~5.0 GB | Native (ChatML + Hermes format) | Best tool-calling accuracy in class |
| **Functionary-small-v3.2** | ~5.0 GB | Native (purpose-built) | Supports parallel function calls |

**Recommended starting point:** `Qwen2.5-3B-Instruct` at `Q4_K_M` — good balance of size, speed, and tool-calling quality. It uses a Jinja chat template with built-in tool-use formatting.

---

## Step 3 — Launch the Server

```bash
cd llama.cpp

./build/bin/llama-server \
  --jinja -fa \
  -hf bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 4096 \
  --temp 0.0
```

### Flag Reference

| Flag | Purpose |
|---|---|
| `--jinja` | Activates the model's built-in chat template (required for tool calling) |
| `-fa` | Enables flash attention for faster inference |
| `-hf` | Downloads the GGUF directly from Hugging Face (cached after first run) |
| `-ngl 99` | Offloads all layers to GPU (Metal on Mac, CUDA on Jetson) |
| `-c 4096` | Context window size — 4K is sufficient for BFCL prompts |
| `--temp 0.0` | Deterministic output — BFCL recommends this (accuracy can vary up to 10% between temp 0.0 and 0.7) |
| `--host 0.0.0.0` | Listens on all interfaces (change to `127.0.0.1` if you only need local access) |
| `--port 8080` | Port for the OpenAI-compatible API |

### Verify It Works

```bash
curl http://localhost:8080/v1/chat/completions \
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

You should see a response with a `tool_calls` array containing a call to `get_weather` with `city: "San Francisco"`. If you see plain text instead, the `--jinja` flag may not be taking effect — check that the model's chat template supports tools.

---

## Step 4 — Install BFCL

```bash
# Create a clean Python environment
conda create -n BFCL python=3.10 -y
conda activate BFCL

# Install the evaluation harness
pip install bfcl-eval

# Set up project root for results storage
export BFCL_PROJECT_ROOT=~/bfcl-results
mkdir -p $BFCL_PROJECT_ROOT
```

### Configure the Endpoint

Create a `.env` file so BFCL knows where your llama-server is:

```bash
cat > $BFCL_PROJECT_ROOT/.env << 'EOF'
LOCAL_SERVER_ENDPOINT=localhost
LOCAL_SERVER_PORT=8080
EOF
```

---

## Step 5 — Run the Evaluation

### Generate Model Responses

The `--skip-server-setup` flag tells BFCL not to try launching its own vLLM/sglang instance — it will just talk to your already-running llama-server.

```bash
# Run all single-turn categories
bfcl generate \
  --model Qwen2.5-3B-Instruct \
  --test-category all \
  --skip-server-setup
```

### Score the Results

```bash
bfcl evaluate \
  --model Qwen2.5-3B-Instruct \
  --test-category all
```

### BFCL Test Categories

You can run individual categories instead of `all`:

| Category | What It Tests |
|---|---|
| `simple` | One function available, one call expected |
| `multiple` | Several functions available, pick the right one |
| `parallel` | Multiple independent calls needed at once |
| `parallel_multiple` | Parallel calls across multiple available functions |
| `relevance` | Model must recognize when NO function is appropriate |
| `java` / `javascript` | Language-specific type handling |
| `multi_turn_base` | Multi-step function calling with state |
| `multi_turn_miss_func` | Multi-turn where a needed function is missing |
| `multi_turn_miss_param` | Multi-turn where required parameters are missing |
| `multi_turn_long_context` | Multi-turn under long-context pressure |

**Suggested order:** Start with `simple` to verify the pipeline works, then run `all`.

---

## Step 6 — Collect Inference Metrics

Beyond BFCL accuracy, capture these performance numbers for your thesis:

### Tokens Per Second

llama.cpp logs throughput to the terminal during inference. You can also query the Prometheus-format metrics endpoint:

```bash
curl http://localhost:8080/metrics
```

### macOS-Specific System Metrics

```bash
# GPU utilization (Metal)
sudo powermetrics --samplers gpu_power -i 1000 -n 10

# Memory pressure
memory_pressure

# Overall system power
sudo powermetrics --samplers all -i 1000 -n 10 | grep "Combined Power"
```

### Jetson-Specific System Metrics (for later)

```bash
# GPU/CPU/memory utilization and power draw
tegrastats --interval 1000
```

### Metrics to Record Per Model Per Quantization

| Metric | How to Capture |
|---|---|
| **BFCL Accuracy** (per category) | `bfcl evaluate` output |
| **Tokens/sec** (prompt eval) | llama-server logs or `/metrics` |
| **Tokens/sec** (generation) | llama-server logs or `/metrics` |
| **Time to First Token (TTFT)** | llama-server logs |
| **Peak Memory Usage** | `memory_pressure` (Mac) / `tegrastats` (Jetson) |
| **Power Draw** | `powermetrics` (Mac) / `tegrastats` (Jetson) |
| **Quantization Level** | Record which GGUF variant you used |

---

## Step 7 — Run Multiple Models (Building a Comparison Table)

Repeat Steps 3–6 for each model. A suggested test matrix:

```bash
# Models to test (in order of size)
MODELS=(
  "bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
  "bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
  "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
  "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M"
)
```

For each model: stop the current llama-server, relaunch with the new model, wait for it to load, then run `bfcl generate` and `bfcl evaluate`.

To also test quantization impact, run your control model at multiple levels:

```
Qwen2.5-3B-Instruct → Q2_K, Q4_K_M, Q5_K_M, Q8_0
```

This gives you a quantization-vs-accuracy curve for tool calling — likely novel data for your thesis.

---

## Troubleshooting

**"No tool_calls in response, just plain text"**
→ Make sure `--jinja` is passed to llama-server. Without it, the model's tool-calling chat template isn't used.

**"BFCL can't connect to the server"**
→ Verify with `curl http://localhost:8080/v1/models`. Check your `.env` has the right port.

**"Out of memory on 8 GB Mac"**
→ Drop to a smaller model or more aggressive quantization (Q2_K). Reduce context with `-c 2048`.

**"Model downloads are slow"**
→ Pre-download with `huggingface-cli download bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q4_K_M.gguf` then point llama-server at the local path with `-m /path/to/model.gguf` instead of `-hf`.

---

## References

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama.cpp Function Calling Docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [BFCL V4 Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [BFCL GitHub (Gorilla)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [bfcl-eval on PyPI](https://pypi.org/project/bfcl-eval/)
- [BFCL Dataset on Hugging Face](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- [bartowski GGUF Models on Hugging Face](https://huggingface.co/bartowski)
- [Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [Edge-First LLM Inference (IEEE ICDCS 2025)](https://arxiv.org/html/2505.16508v1)
