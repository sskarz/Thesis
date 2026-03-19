"""Send BFCL test entries to llama-server and save responses + timings."""

import asyncio
import copy
import csv
import json
from pathlib import Path

import aiohttp
from tqdm import tqdm

from . import config
from .coerce_types import coerce_tool_calls


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_CALLING_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. When calling tools:\n"
    '- Pass numeric values as numbers, not strings (e.g., 5 not "5")\n'
    "- Pass arrays as JSON arrays, not strings (e.g., [1,2] not \"[1,2]\")\n"
    "- Pass objects as JSON objects, not strings\n"
    "- Always use the correct JSON types matching the parameter schema."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _convert_types(obj):
    """Recursively map BFCL/Gorilla types to JSON-Schema types."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str):
                mapped = config.GORILLA_TO_OPENAPI.get(v, v)
                if mapped is not None:
                    result[k] = mapped
            else:
                result[k] = _convert_types(v)
        return result
    if isinstance(obj, list):
        return [_convert_types(item) for item in obj]
    return obj


def bfcl_to_openai_tools(bfcl_functions: list[dict]) -> list[dict]:
    """Convert a list of BFCL function defs to OpenAI ``tools`` format."""
    tools = []
    for func in bfcl_functions:
        func = copy.deepcopy(func)
        converted = _convert_types(func)
        # Keep only standard fields
        openai_func = {
            "name": converted["name"],
            "description": converted.get("description", ""),
        }
        if "parameters" in converted:
            openai_func["parameters"] = converted["parameters"]
        else:
            openai_func["parameters"] = {"type": "object", "properties": {}}
        tools.append({"type": "function", "function": openai_func})
    return tools


def build_payload(entry: dict) -> dict:
    """Convert a single-turn BFCL test entry to an API request payload."""
    messages = list(entry["question"][0])
    bfcl_functions = entry.get("function", [])
    tools = bfcl_to_openai_tools(bfcl_functions) if bfcl_functions else None

    if tools:
        messages = [
            {"role": "system", "content": TOOL_CALLING_SYSTEM_PROMPT},
            *messages,
        ]

    payload: dict = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def parse_tool_calls(
    response_json: dict,
    func_defs: list[dict] | None = None,
) -> list[dict] | str:
    """Return list-of-dicts for tool calls, or text content when none.

    When *func_defs* is provided, argument values are coerced to the types
    declared in the BFCL function schema (fixes Llama's string-for-everything
    serialization).
    """
    choice = response_json["choices"][0]
    message = choice["message"]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        result = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            result.append({name: arguments})
        if func_defs:
            result = coerce_tool_calls(result, func_defs)
        return result
    content = message.get("content", "")
    from .parse_functiongemma import parse_functiongemma_content
    parsed = parse_functiongemma_content(content)
    if parsed is not None:
        if func_defs:
            parsed = coerce_tool_calls(parsed, func_defs)
        return parsed
    return content


# ---------------------------------------------------------------------------
# Per-category generation (async)
# ---------------------------------------------------------------------------

TIMING_FIELDS = [
    "id", "category",
    "prompt_n", "prompt_ms", "prompt_per_second",
    "predicted_n", "predicted_ms", "predicted_per_second",
]


async def _generate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    entry: dict,
    category: str,
    pbar: tqdm,
) -> tuple[dict, dict]:
    """Send one request with concurrency bounded by *semaphore*."""
    payload = build_payload(entry)
    func_defs = entry.get("function", [])

    async with semaphore:
        try:
            async with session.post(
                f"{config.SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            result = parse_tool_calls(data, func_defs=func_defs)

            t = data.get("timings", {})
            timing = {
                "id": entry["id"],
                "category": category,
                "prompt_n": t.get("prompt_n", 0),
                "prompt_ms": t.get("prompt_ms", 0),
                "prompt_per_second": t.get("prompt_per_second", 0),
                "predicted_n": t.get("predicted_n", 0),
                "predicted_ms": t.get("predicted_ms", 0),
                "predicted_per_second": t.get("predicted_per_second", 0),
            }

        except Exception as e:
            result = f"ERROR: {e}"
            timing = {
                "id": entry["id"],
                "category": category,
                "prompt_n": 0, "prompt_ms": 0, "prompt_per_second": 0,
                "predicted_n": 0, "predicted_ms": 0, "predicted_per_second": 0,
            }

    pbar.update(1)

    result_row = {
        "id": entry["id"],
        "result": json.dumps(result) if isinstance(result, list) else result,
    }
    return result_row, timing


async def _generate_for_category_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    category: str,
    category_info: dict,
) -> tuple[list[dict], list[dict]]:
    """Generate responses for every entry in one BFCL category concurrently."""
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    entries = load_jsonl(data_file)

    pbar = tqdm(total=len(entries), desc=f"  {category}", unit="req")

    tasks = [
        _generate_one(session, semaphore, entry, category, pbar)
        for entry in entries
    ]
    pairs = await asyncio.gather(*tasks)
    pbar.close()

    results = [p[0] for p in pairs]
    timings = [p[1] for p in pairs]
    return results, timings


# ---------------------------------------------------------------------------
# Sync wrappers (keep the same public API)
# ---------------------------------------------------------------------------

def generate_for_category(
    category: str,
    category_info: dict,
) -> tuple[list[dict], list[dict]]:
    """Generate responses for every entry in one BFCL category.

    Returns (results, timings) where each is a list of dicts.
    """
    # Fallback for callers outside the async run() path (e.g. tests)
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    entries = load_jsonl(data_file)

    results: list[dict] = []
    timings: list[dict] = []

    import requests as req_lib

    for entry in tqdm(entries, desc=f"  {category}", unit="req"):
        payload = build_payload(entry)
        func_defs = entry.get("function", [])

        try:
            resp = req_lib.post(
                f"{config.SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            result = parse_tool_calls(data, func_defs=func_defs)

            t = data.get("timings", {})
            timings.append({
                "id": entry["id"],
                "category": category,
                "prompt_n": t.get("prompt_n", 0),
                "prompt_ms": t.get("prompt_ms", 0),
                "prompt_per_second": t.get("prompt_per_second", 0),
                "predicted_n": t.get("predicted_n", 0),
                "predicted_ms": t.get("predicted_ms", 0),
                "predicted_per_second": t.get("predicted_per_second", 0),
            })

        except Exception as e:
            result = f"ERROR: {e}"
            timings.append({
                "id": entry["id"],
                "category": category,
                "prompt_n": 0, "prompt_ms": 0, "prompt_per_second": 0,
                "predicted_n": 0, "predicted_ms": 0, "predicted_per_second": 0,
            })

        results.append({
            "id": entry["id"],
            "result": json.dumps(result) if isinstance(result, list) else result,
        })

    return results, timings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def _run_async() -> Path:
    """Async implementation of run()."""
    config.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(config.PARALLEL_SLOTS)
    all_timings: list[dict] = []

    async with aiohttp.ClientSession() as session:
        for category, info in config.CATEGORIES.items():
            print(f"[generate] Category: {category}")
            results, timings = await _generate_for_category_async(
                session, semaphore, category, info
            )
            all_timings.extend(timings)

            out_file = config.RESPONSES_DIR / f"{category}_responses.jsonl"
            with open(out_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"  → {len(results)} responses saved to {out_file.name}")

    timings_csv = config.RESPONSES_DIR / "all_timings.csv"
    with open(timings_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(all_timings)
    print(f"[generate] Timings saved to {timings_csv.name}")

    return timings_csv


def run() -> Path:
    """Generate responses for all categories. Returns path to timings CSV."""
    return asyncio.run(_run_async())


if __name__ == "__main__":
    run()
