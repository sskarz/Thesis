"""Dedicated metrics collection: warmup, controlled requests, memory."""

import csv
import json
import random
import subprocess
from pathlib import Path

import psutil
import requests
from tqdm import tqdm

from . import config
from . import generate_multi_turn
from .generate_responses import bfcl_to_openai_tools, load_jsonl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP_REQUESTS = 3
BENCHMARK_REQUESTS = 50
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_llama_server_pid() -> int | None:
    """Return the PID of the running llama-server binary, or None."""
    best_pid = None
    best_rss = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            # Match the actual binary, not shell wrappers
            if any("llama-server" in str(c) for c in cmdline):
                mem = proc.info.get("memory_info")
                rss = mem.rss if mem else 0
                # Pick the process with the highest RSS (the real server)
                if rss > best_rss:
                    best_rss = rss
                    best_pid = proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return best_pid


def sample_requests(
    n: int,
    include_single: bool = True,
    include_multi: bool = False,
) -> list[dict]:
    """Sample *n* test entries across all categories, reproducibly."""
    rng = random.Random(RANDOM_SEED)
    category_pools: list[list[dict]] = []
    if include_single:
        for cat, info in config.CATEGORIES.items():
            data_file = config.BFCL_DATA_DIR / info["file"]
            category_entries: list[dict] = []
            for entry in load_jsonl(data_file):
                entry["_category"] = cat
                entry["_mode"] = "single"
                category_entries.append(entry)
            category_pools.append(category_entries)
    if include_multi:
        for cat, info in config.MULTI_TURN_CATEGORIES.items():
            data_file = config.BFCL_DATA_DIR / info["file"]
            category_entries = []
            for entry in load_jsonl(data_file):
                entry["_category"] = cat
                entry["_mode"] = "multi"
                category_entries.append(entry)
            category_pools.append(category_entries)
    if not category_pools:
        return []

    selected: list[dict] = []
    if n >= len(category_pools):
        for entries in category_pools:
            selected.append(rng.choice(entries))
            entries.remove(selected[-1])

    remaining_pool = [entry for entries in category_pools for entry in entries]
    remaining_n = min(n - len(selected), len(remaining_pool))
    if remaining_n > 0:
        selected.extend(rng.sample(remaining_pool, remaining_n))

    return selected


def build_payload(entry: dict) -> dict:
    """Convert a BFCL test entry to an API request payload."""
    messages = entry["question"][0]
    bfcl_functions = entry.get("function", [])
    tools = bfcl_to_openai_tools(bfcl_functions) if bfcl_functions else None
    payload: dict = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def send_request(payload: dict) -> dict:
    """Send one request and return the full response JSON."""
    resp = requests.post(
        f"{config.SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _to_tokens_per_second(token_count: int, ms: float) -> float:
    if ms <= 0:
        return 0
    return token_count / (ms / 1000.0)


def measure_request(entry: dict) -> dict:
    """Execute one sampled entry and return a normalized timing row."""
    if entry.get("_mode") == "multi":
        all_tools = generate_multi_turn._load_func_docs(
            entry.get("involved_classes", [])
        )
        _, timing = generate_multi_turn.generate_for_entry(
            entry,
            entry["_category"],
            all_tools,
            execution_model_name=f"{config.MODEL_NAME}_metrics",
        )
        prompt_n = timing.get("total_prompt_n", 0)
        prompt_ms = timing.get("total_prompt_ms", 0)
        predicted_n = timing.get("total_predicted_n", 0)
        predicted_ms = timing.get("total_predicted_ms", 0)
        return {
            "id": entry["id"],
            "category": entry["_category"],
            "prompt_n": prompt_n,
            "prompt_ms": timing.get("first_prompt_ms", 0),
            "prompt_per_second": _to_tokens_per_second(prompt_n, prompt_ms),
            "predicted_n": predicted_n,
            "predicted_ms": predicted_ms,
            "predicted_per_second": _to_tokens_per_second(predicted_n, predicted_ms),
        }

    data = send_request(build_payload(entry))
    t = data.get("timings", {})
    return {
        "id": entry["id"],
        "category": entry["_category"],
        "prompt_n": t.get("prompt_n", 0),
        "prompt_ms": t.get("prompt_ms", 0),
        "prompt_per_second": t.get("prompt_per_second", 0),
        "predicted_n": t.get("predicted_n", 0),
        "predicted_ms": t.get("predicted_ms", 0),
        "predicted_per_second": t.get("predicted_per_second", 0),
    }


def get_gpu_memory_mb() -> float:
    """Return GPU VRAM usage in MB via nvidia-smi, or 0 if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0.0


def get_memory_rss(pid: int) -> int:
    """Return RSS in bytes for the given PID."""
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def try_power_draw() -> float | None:
    """Try to get instantaneous power draw (NVIDIA GPU first, then macOS powermetrics)."""
    # NVIDIA GPU (Linux/CUDA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            val = result.stdout.strip().split("\n")[0]
            return float(val)
    except Exception:
        pass
    # macOS (Apple Silicon)
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "1000", "--samplers", "cpu_power"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "Combined Power" in line or "CPU Power" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    val = parts[1].strip().split()[0]
                    return float(val) / 1000.0  # mW → W
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(include_single: bool = True, include_multi: bool = False) -> Path:
    """Run controlled metrics collection. Returns path to metrics JSON."""
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pid = find_llama_server_pid()
    if pid:
        print(f"[metrics] Found llama-server PID: {pid}")
    else:
        print("[metrics] Warning: could not find llama-server PID; memory metrics will be 0")

    # Baseline memory
    baseline_rss = get_memory_rss(pid) if pid else 0
    baseline_gpu_mb = get_gpu_memory_mb()

    # Sample requests
    entries = sample_requests(
        WARMUP_REQUESTS + BENCHMARK_REQUESTS,
        include_single=include_single,
        include_multi=include_multi,
    )
    if not entries:
        raise ValueError("No benchmark categories selected for metrics collection.")
    warmup_entries = entries[:WARMUP_REQUESTS]
    bench_entries = entries[WARMUP_REQUESTS:]

    # Warmup
    print(f"[metrics] Warmup: {WARMUP_REQUESTS} requests (discarded)")
    for entry in warmup_entries:
        try:
            measure_request(entry)
        except Exception:
            pass

    # Benchmark
    print(f"[metrics] Benchmark: {len(bench_entries)} requests")
    timing_rows: list[dict] = []
    peak_rss = baseline_rss
    peak_gpu_mb = baseline_gpu_mb

    for entry in tqdm(bench_entries, desc="  metrics", unit="req"):
        try:
            timing_rows.append(measure_request(entry))
        except Exception as e:
            print(f"  [warn] Request failed for {entry['id']}: {e}")
            continue

        # Track peak memory
        if pid:
            rss = get_memory_rss(pid)
            if rss > peak_rss:
                peak_rss = rss
        gpu_mb = get_gpu_memory_mb()
        if gpu_mb > peak_gpu_mb:
            peak_gpu_mb = gpu_mb

    # Optional power draw
    power_w = try_power_draw()

    # Compute summary statistics
    prompt_speeds = [r["prompt_per_second"] for r in timing_rows if r["prompt_per_second"] > 0]
    gen_speeds = [r["predicted_per_second"] for r in timing_rows if r["predicted_per_second"] > 0]
    ttfts = [r["prompt_ms"] for r in timing_rows if r["prompt_ms"] > 0]

    def percentile(data, p):
        if not data:
            return 0
        s = sorted(data)
        k = (len(s) - 1) * p / 100
        f, c = int(k), int(k) + 1
        if c >= len(s):
            return s[f]
        return s[f] + (k - f) * (s[c] - s[f])

    summary = {
        "model": config.MODEL_NAME,
        "num_requests": len(timing_rows),
        "prompt_tokens_per_sec": {
            "mean": sum(prompt_speeds) / len(prompt_speeds) if prompt_speeds else 0,
            "median": percentile(prompt_speeds, 50),
            "p95": percentile(prompt_speeds, 95),
            "p99": percentile(prompt_speeds, 99),
        },
        "generation_tokens_per_sec": {
            "mean": sum(gen_speeds) / len(gen_speeds) if gen_speeds else 0,
            "median": percentile(gen_speeds, 50),
            "p95": percentile(gen_speeds, 95),
            "p99": percentile(gen_speeds, 99),
        },
        "ttft_ms": {
            "mean": sum(ttfts) / len(ttfts) if ttfts else 0,
            "median": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
        },
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "peak_rss_bytes": peak_rss,
            "baseline_rss_mb": round(baseline_rss / 1024 / 1024, 1),
            "peak_rss_mb": round(peak_rss / 1024 / 1024, 1),
            "baseline_gpu_vram_mb": round(baseline_gpu_mb, 1),
            "peak_gpu_vram_mb": round(peak_gpu_mb, 1),
        },
        "power_watts": power_w,
    }

    # Save JSON summary
    json_path = config.METRICS_DIR / "inference_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[metrics] Summary saved to {json_path.name}")

    # Save per-request CSV
    csv_path = config.METRICS_DIR / "inference_metrics.csv"
    fields = [
        "id", "category",
        "prompt_n", "prompt_ms", "prompt_per_second",
        "predicted_n", "predicted_ms", "predicted_per_second",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(timing_rows)
    print(f"[metrics] Per-request data saved to {csv_path.name}")

    # Print summary
    mem = summary["memory"]
    print(f"[metrics] Memory: baseline={mem['baseline_rss_mb']}MB, peak={mem['peak_rss_mb']}MB")
    print(f"[metrics] GPU VRAM: baseline={mem['baseline_gpu_vram_mb']}MB, peak={mem['peak_gpu_vram_mb']}MB")
    if power_w is not None:
        print(f"[metrics] Power draw: {power_w:.1f} W")
    else:
        print("[metrics] Power draw: unavailable (needs sudo)")

    return json_path


if __name__ == "__main__":
    run()
