"""Multi-turn generation: model ↔ tool execution ↔ model loop for BFCL v3."""

import csv
import json
from pathlib import Path

import requests
from tqdm import tqdm

from . import config
from .generate_responses import bfcl_to_openai_tools, load_jsonl

# Mapping from class name → func-doc JSON filename
from bfcl_eval.constants.category_mapping import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_func_docs(involved_classes: list[str]) -> list[dict]:
    """Load BFCL function docs for the given classes and return OpenAI tools."""
    all_funcs: list[dict] = []
    for cls_name in involved_classes:
        doc_file = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls_name)
        if not doc_file:
            continue
        path = config.MULTI_TURN_FUNC_DOC_DIR / doc_file
        if not path.exists():
            continue
        funcs = load_jsonl(path)
        all_funcs.extend(funcs)
    return bfcl_to_openai_tools(all_funcs)


def _tool_calls_to_func_strings(tool_calls: list[dict]) -> list[str]:
    """Convert OpenAI tool_calls to func-call strings for bfcl_eval execution.

    E.g. ``[{"id": "...", "function": {"name": "cd", "arguments": '{"folder": "doc"}'}}]``
    → ``["cd(folder='doc')"]``
    """
    result = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        raw_args = tc["function"].get("arguments")
        try:
            # If arguments is a JSON string, parse it; if it's already a mapping,
            # use it directly; otherwise default to empty dict.
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except (json.JSONDecodeError, TypeError):
            # Preserve the raw argument string so malformed tool calls are not
            # silently converted into a different (e.g., empty-arg) call.
            parts = [f"__arg_parse_error__={repr(raw_args)}"]
            result.append(f"{name}({', '.join(parts)})")
            continue
        parts = [f"{k}={repr(v)}" for k, v in args.items()]
        result.append(f"{name}({', '.join(parts)})")
    return result


def _build_miss_func_sets(entry: dict) -> tuple[set[str], dict[int, list[str]]]:
    """Parse ``missed_function`` field.

    Returns:
        withheld_names: set of function names to remove initially
        restore_map: {turn_idx: [func_names]} — turns where withheld funcs come back
    """
    mf = entry.get("missed_function", {})
    withheld: set[str] = set()
    restore_map: dict[int, list[str]] = {}
    for turn_idx_str, func_names in mf.items():
        turn_idx = int(turn_idx_str)
        withheld.update(func_names)
        restore_map[turn_idx] = func_names
    return withheld, restore_map


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def generate_for_entry(
    entry: dict,
    category: str,
    all_tools: list[dict],
    execution_model_name: str | None = None,
) -> tuple[list[list[list[str]]], dict]:
    """Run the multi-turn inference loop for a single test entry.

    Returns:
        (turn_results, timing_summary)
        turn_results: list[turns] of list[steps] of list[func_call_strings]
        timing_summary: dict with aggregated timing info
    """
    entry_id = entry["id"]
    initial_config = entry.get("initial_config", {})
    involved_classes = entry.get("involved_classes", [])
    questions = entry["question"]  # list of turns; each turn is list of messages

    # Handle miss_func: remove withheld tools initially
    is_miss_func = category == "multi_turn_miss_func"
    withheld_names: set[str] = set()
    restore_map: dict[int, list[str]] = {}
    withheld_tools: list[dict] = []  # stashed tools to restore later

    if is_miss_func:
        withheld_names, restore_map = _build_miss_func_sets(entry)

    # Build active tool list (possibly with some removed)
    active_tools = []
    for tool in all_tools:
        if is_miss_func and tool["function"]["name"] in withheld_names:
            withheld_tools.append(tool)
        else:
            active_tools.append(tool)

    chat_history: list[dict] = []
    turn_results: list[list[list[str]]] = []
    total_prompt_ms = 0.0
    total_predicted_ms = 0.0
    total_prompt_n = 0
    total_predicted_n = 0
    api_calls = 0

    exec_model_name = execution_model_name or config.MODEL_NAME

    for turn_idx, turn_messages in enumerate(questions):
        # miss_func: if this turn's question is [], add back withheld functions
        if is_miss_func and not turn_messages:
            # Restore withheld functions
            if turn_idx in restore_map:
                for tool in withheld_tools:
                    if tool["function"]["name"] in restore_map[turn_idx]:
                        active_tools.append(tool)
            chat_history.append({
                "role": "user",
                "content": (
                    "I have updated some more functions you can choose from. "
                    "What about now?"
                ),
            })
        else:
            # Append user messages for this turn
            for msg in turn_messages:
                chat_history.append(msg)
            if not turn_messages:
                # No user message this turn — skip model call
                turn_results.append([])
                continue

        # Inner step loop: model calls tools until it gives a text response
        step_results: list[list[str]] = []
        for _step in range(config.MULTI_TURN_MAX_STEPS):
            payload = {
                "model": config.MODEL_NAME,
                "messages": chat_history,
                "temperature": 0.0,
            }
            if active_tools:
                payload["tools"] = active_tools
                payload["tool_choice"] = "auto"

            try:
                resp = requests.post(
                    f"{config.SERVER_URL}/v1/chat/completions",
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                api_calls += 1
            except Exception as e:
                # On error, record empty step and break
                step_results.append([f"ERROR: {e}"])
                break

            # Capture timings
            t = data.get("timings", {})
            total_prompt_n += t.get("prompt_n", 0)
            total_prompt_ms += t.get("prompt_ms", 0)
            total_predicted_n += t.get("predicted_n", 0)
            total_predicted_ms += t.get("predicted_ms", 0)

            if not data.get("choices"):
                step_results.append(["ERROR: empty choices in response"])
                break

            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "")

            tool_calls_raw = message.get("tool_calls")

            if tool_calls_raw:
                # Model wants to call tools
                func_strings = _tool_calls_to_func_strings(tool_calls_raw)
                step_results.append(func_strings)

                # Add assistant message with tool_calls to history
                chat_history.append(message)

                # Execute function calls via bfcl_eval
                try:
                    exec_results, _ = execute_multi_turn_func_call(
                        func_call_list=func_strings,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name=exec_model_name,
                        test_entry_id=entry_id,
                        long_context=(category == "multi_turn_long_context"),
                        is_evaL_run=False,  # sic: upstream parameter name
                    )
                except Exception as e:
                    exec_results = [str(e)] * len(func_strings)

                # Feed tool results back to model
                for tc, exec_result in zip(tool_calls_raw, exec_results):
                    chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(exec_result),
                    })
            else:
                # Text response — turn is done
                content = message.get("content", "")
                chat_history.append({"role": "assistant", "content": content})
                break
        else:
            # Loop completed without break — model never gave a text response
            print(
                f"  [warn] {entry_id} turn {turn_idx}: "
                f"hit {config.MULTI_TURN_MAX_STEPS}-step limit without text response"
            )

        turn_results.append(step_results)

    timing_summary = {
        "id": entry_id,
        "category": category,
        "turns": len(questions),
        "api_calls": api_calls,
        "total_prompt_n": total_prompt_n,
        "total_prompt_ms": round(total_prompt_ms, 2),
        "total_predicted_n": total_predicted_n,
        "total_predicted_ms": round(total_predicted_ms, 2),
    }
    return turn_results, timing_summary


# ---------------------------------------------------------------------------
# Per-category driver
# ---------------------------------------------------------------------------

def generate_for_category(
    category: str,
    category_info: dict,
) -> tuple[list[dict], list[dict]]:
    """Generate multi-turn responses for all entries in one category."""
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    entries = load_jsonl(data_file)

    results: list[dict] = []
    timings: list[dict] = []

    for entry in tqdm(entries, desc=f"  {category}", unit="entry"):
        involved_classes = entry.get("involved_classes", [])
        all_tools = _load_func_docs(involved_classes)

        turn_results, timing = generate_for_entry(entry, category, all_tools)

        results.append({
            "id": entry["id"],
            "result": turn_results,
        })
        timings.append(timing)

    return results, timings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

TIMING_FIELDS = [
    "id", "category", "turns", "api_calls",
    "total_prompt_n", "total_prompt_ms",
    "total_predicted_n", "total_predicted_ms",
]


def run() -> Path:
    """Generate multi-turn responses for all categories. Returns timings CSV path."""
    config.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    all_timings: list[dict] = []

    for category, info in config.MULTI_TURN_CATEGORIES.items():
        print(f"[multi-turn generate] Category: {category}")
        results, timings = generate_for_category(category, info)
        all_timings.extend(timings)

        out_file = config.RESPONSES_DIR / f"{category}_responses.jsonl"
        with open(out_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  → {len(results)} entries saved to {out_file.name}")

    timings_csv = config.RESPONSES_DIR / "multi_turn_timings.csv"
    with open(timings_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(all_timings)
    print(f"[multi-turn generate] Timings saved to {timings_csv.name}")

    return timings_csv


if __name__ == "__main__":
    run()
