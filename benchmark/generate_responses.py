"""Send BFCL test entries to llama-server and save responses + timings."""

import copy
import csv
import json
from pathlib import Path

import requests
from tqdm import tqdm

from . import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _convert_types(obj):
    """Recursively map BFCL/Gorilla types to JSON-Schema types."""
    if isinstance(obj, dict):
        return {
            k: (config.GORILLA_TO_OPENAPI.get(v, v) if k == "type" and isinstance(v, str)
                 else _convert_types(v))
            for k, v in obj.items()
        }
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


def parse_tool_calls(response_json: dict) -> list[dict] | str:
    """Return list-of-dicts for tool calls, or text content when none."""
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
        return result
    return message.get("content", "")


# ---------------------------------------------------------------------------
# Per-category generation
# ---------------------------------------------------------------------------

TIMING_FIELDS = [
    "id", "category",
    "prompt_n", "prompt_ms", "prompt_per_second",
    "predicted_n", "predicted_ms", "predicted_per_second",
]


def generate_for_category(
    category: str,
    category_info: dict,
) -> tuple[list[dict], list[dict]]:
    """Generate responses for every entry in one BFCL category.

    Returns (results, timings) where each is a list of dicts.
    """
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    entries = load_jsonl(data_file)

    results: list[dict] = []
    timings: list[dict] = []

    for entry in tqdm(entries, desc=f"  {category}", unit="req"):
        messages = entry["question"][0]  # single-turn → first (only) turn
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

        try:
            resp = requests.post(
                f"{config.SERVER_URL}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            result = parse_tool_calls(data)

            # Capture timings
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

def run() -> Path:
    """Generate responses for all categories. Returns path to timings CSV."""
    config.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    all_timings: list[dict] = []

    for category, info in config.CATEGORIES.items():
        print(f"[generate] Category: {category}")
        results, timings = generate_for_category(category, info)
        all_timings.extend(timings)

        # Write responses JSONL
        out_file = config.RESPONSES_DIR / f"{category}_responses.jsonl"
        with open(out_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  → {len(results)} responses saved to {out_file.name}")

    # Write combined timings CSV
    timings_csv = config.RESPONSES_DIR / "all_timings.csv"
    with open(timings_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(all_timings)
    print(f"[generate] Timings saved to {timings_csv.name}")

    return timings_csv


if __name__ == "__main__":
    run()
