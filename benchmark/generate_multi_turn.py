"""Multi-turn generation: model ↔ tool execution ↔ model loop for BFCL v4.

Function-calling modes (``--fc-mode`` / ``config.FC_MODE``):

- ``nfc`` (default): native OpenAI ``tools`` / ``tool_calls`` via llama-server --jinja
- ``iffc``: Instruction-Followed Function Calling (arXiv:2608.22472 Appendix B) —
  JSON-in-text; do not send ``tools``/``tool_choice``
- ``cd``: TagDispatch-style two-pass constrained decoding (arXiv:2601.04426
  approximation). Never sends ``tools``/``tool_choice``. Pass 1 constrains the
  tool *name* with a GBNF grammar; pass 2 constrains arguments with
  ``json_schema``. Not llama-server native tools GBNF; not full XGrammar-2.
- ``gst``: Guided-Structured Templates (arXiv:2509.18076) on the NFC channel.
  Still sends ``tools``/``tool_choice``; injects a 7-step curriculum. Native
  ``tool_calls`` are executed; ``[GST]`` text is ignored by the checker.
"""

import argparse
import asyncio
import json

import aiohttp
import requests
from tqdm import tqdm

from . import cd as cd_mode
from . import config, db
from . import gst as gst_mode
from .generate_responses import bfcl_to_openai_tools, load_jsonl
from .iffc import (
    IFFC_PARSE_RETRIES,
    IFFC_RETRY_USER_MESSAGE,
    assistant_text,
    ensure_iffc_system_message,
    format_iffc_execution_user_message,
    iffc_calls_to_func_strings,
    try_parse_iffc_response,
)

# Mapping from class name → func-doc JSON filename
from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
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


def _resolve_fc_mode(fc_mode: str | None) -> str:
    mode = fc_mode or config.FC_MODE
    if mode not in config.FC_MODES:
        raise ValueError(
            f"Unknown fc-mode {mode!r}. Choose from: {', '.join(config.FC_MODES)}"
        )
    return mode


def _is_json_text_mode(fc_mode: str) -> bool:
    return fc_mode in config.FC_JSON_TEXT_MODES


def _build_chat_payload(
    chat_history: list[dict],
    active_tools: list[dict],
    fc_mode: str,
) -> dict:
    """Build a chat-completions payload. IFFC/CD never send tools/tool_choice.

    GST uses the NFC tools channel (``tools`` + ``tool_choice=auto``).
    """
    payload: dict = {
        "model": config.MODEL_NAME,
        "messages": chat_history,
        "temperature": 0.0,
    }
    if _is_json_text_mode(fc_mode):
        return payload
    if active_tools:
        payload["tools"] = active_tools
        payload["tool_choice"] = "auto"
    return payload


def _prepare_step_messages(
    chat_history: list[dict],
    active_tools: list[dict],
    fc_mode: str,
) -> None:
    if _is_json_text_mode(fc_mode):
        # Persistent transcript is IFFC-shaped for both IFFC and CD.
        # CD applies grammar/json_schema on a working copy inside complete_step.
        ensure_iffc_system_message(chat_history, active_tools)
        return
    if fc_mode == config.FC_GST:
        gst_mode.ensure_gst_system_message(chat_history)


def _post_chat_sync(payload: dict) -> dict:
    resp = requests.post(
        f"{config.SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def _complete_chat_sync(
    chat_history: list[dict],
    active_tools: list[dict],
    fc_mode: str,
) -> tuple[dict, int]:
    """One model step. CD may issue two constrained requests."""
    if fc_mode == config.FC_CD:
        return cd_mode.complete_step(
            chat_history,
            active_tools,
            post=_post_chat_sync,
            model=config.MODEL_NAME,
        )
    payload = _build_chat_payload(chat_history, active_tools, fc_mode)
    return _post_chat_sync(payload), 1


async def _post_chat_async(session: aiohttp.ClientSession, payload: dict) -> dict:
    async with session.post(
        f"{config.SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


def _decide_assistant_step(
    message: dict,
    chat_history: list[dict],
    fc_mode: str,
    parse_retries_left: int,
) -> tuple[str, list[str] | None, list | None, int, str | None]:
    """Interpret an assistant message.

    Returns ``(action, func_strings, tool_calls_raw, parse_retries_left, flag)``
    where *action* is ``execute``, ``end_turn``, or ``retry``.
    """
    if _is_json_text_mode(fc_mode):
        content = assistant_text(message)
        parsed = try_parse_iffc_response(content)
        if parsed:
            func_strings = iffc_calls_to_func_strings(parsed)
            chat_history.append({"role": "assistant", "content": content})
            return "execute", func_strings, None, IFFC_PARSE_RETRIES, None
        if parsed is None and parse_retries_left > 0:
            chat_history.append({"role": "assistant", "content": content})
            chat_history.append({"role": "user", "content": IFFC_RETRY_USER_MESSAGE})
            return "retry", None, None, parse_retries_left - 1, None
        chat_history.append({"role": "assistant", "content": content})
        flag = None if parsed is not None else (
            "cd_unparseable" if fc_mode == config.FC_CD else "iffc_unparseable"
        )
        return "end_turn", None, None, parse_retries_left, flag

    tool_calls_raw = message.get("tool_calls")
    if tool_calls_raw:
        func_strings = _tool_calls_to_func_strings(tool_calls_raw)
        chat_history.append(message)
        return "execute", func_strings, tool_calls_raw, parse_retries_left, None

    content = message.get("content", "")
    chat_history.append({"role": "assistant", "content": content})
    return "end_turn", None, None, parse_retries_left, None


def _append_execution_feedback(
    chat_history: list[dict],
    fc_mode: str,
    func_strings: list[str],
    tool_calls_raw: list | None,
    exec_results: list,
) -> None:
    if _is_json_text_mode(fc_mode):
        chat_history.append({
            "role": "user",
            "content": format_iffc_execution_user_message(func_strings, exec_results),
        })
        return
    for tc, exec_result in zip(tool_calls_raw or [], exec_results):
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": str(exec_result),
        })


# ---------------------------------------------------------------------------
# Core generation loop (sync — used by collect_metrics and tests)
# ---------------------------------------------------------------------------

def generate_for_entry(
    entry: dict,
    category: str,
    all_tools: list[dict],
    execution_model_name: str | None = None,
    fc_mode: str | None = None,
) -> tuple[list[list[list[str]]], dict]:
    """Run the multi-turn inference loop for a single test entry (sync).

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
    flags: list[str] = []
    total_prompt_ms = 0.0
    total_predicted_ms = 0.0
    total_prompt_n = 0
    total_predicted_n = 0
    api_calls = 0
    first_prompt_ms: float = 0.0

    exec_model_name = execution_model_name or f"{config.MODEL_NAME}_gen"
    fc_mode = _resolve_fc_mode(fc_mode)

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
        parse_retries_left = IFFC_PARSE_RETRIES
        for _step in range(config.MULTI_TURN_MAX_STEPS):
            _prepare_step_messages(chat_history, active_tools, fc_mode)

            try:
                data, n_calls = _complete_chat_sync(
                    chat_history, active_tools, fc_mode
                )
                api_calls += n_calls
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 400:
                    flags.append(f"context_overflow:turn_{turn_idx}")
                else:
                    flags.append(f"http_error_{status}:turn_{turn_idx}")
                step_results.append([f"ERROR: {e}"])
                break
            except Exception as e:
                step_results.append([f"ERROR: {e}"])
                break

            # Capture timings
            t = data.get("timings", {})
            total_prompt_n += t.get("prompt_n", 0)
            total_prompt_ms += t.get("prompt_ms", 0)
            total_predicted_n += t.get("predicted_n", 0)
            total_predicted_ms += t.get("predicted_ms", 0)
            if api_calls == n_calls:
                first_prompt_ms = t.get("prompt_ms", 0)

            if not data.get("choices"):
                step_results.append(["ERROR: empty choices in response"])
                break

            choice = data["choices"][0]
            message = choice["message"]

            action, func_strings, tool_calls_raw, parse_retries_left, flag = (
                _decide_assistant_step(
                    message, chat_history, fc_mode, parse_retries_left
                )
            )
            if flag:
                flags.append(f"{flag}:turn_{turn_idx}")
            if action == "retry":
                continue
            if action == "end_turn":
                break

            step_results.append(func_strings)

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

            _append_execution_feedback(
                chat_history, fc_mode, func_strings, tool_calls_raw, exec_results
            )
        else:
            # Loop completed without break — model never gave a text response
            flags.append(f"hit_step_limit:turn_{turn_idx}")
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
        "first_prompt_ms": round(first_prompt_ms, 2),
        "flags": json.dumps(flags) if flags else "",
    }
    return turn_results, timing_summary


# ---------------------------------------------------------------------------
# Async generation loop
# ---------------------------------------------------------------------------

async def _generate_entry_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    entry: dict,
    category: str,
    all_tools: list[dict],
    pbar: tqdm,
    fc_mode: str | None = None,
) -> tuple[dict, dict]:
    """Run one multi-turn entry asynchronously.

    Steps within the entry are sequential (each depends on the previous).
    The semaphore limits how many entries run concurrently.
    """
    entry_id = entry["id"]
    initial_config = entry.get("initial_config", {})
    involved_classes = entry.get("involved_classes", [])
    questions = entry["question"]
    fc_mode = _resolve_fc_mode(fc_mode)

    is_miss_func = category == "multi_turn_miss_func"
    withheld_names: set[str] = set()
    restore_map: dict[int, list[str]] = {}
    withheld_tools: list[dict] = []

    if is_miss_func:
        withheld_names, restore_map = _build_miss_func_sets(entry)

    active_tools = []
    for tool in all_tools:
        if is_miss_func and tool["function"]["name"] in withheld_names:
            withheld_tools.append(tool)
        else:
            active_tools.append(tool)

    chat_history: list[dict] = []
    turn_results: list[list[list[str]]] = []
    flags: list[str] = []
    total_prompt_ms = 0.0
    total_predicted_ms = 0.0
    total_prompt_n = 0
    total_predicted_n = 0
    api_calls = 0
    first_prompt_ms: float = 0.0

    exec_model_name = f"{config.MODEL_NAME}_gen"

    for turn_idx, turn_messages in enumerate(questions):
        if is_miss_func and not turn_messages:
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
            for msg in turn_messages:
                chat_history.append(msg)
            if not turn_messages:
                turn_results.append([])
                continue

        step_results: list[list[str]] = []
        parse_retries_left = IFFC_PARSE_RETRIES
        for _step in range(config.MULTI_TURN_MAX_STEPS):
            _prepare_step_messages(chat_history, active_tools, fc_mode)

            async def _gated_post(payload: dict) -> dict:
                async with semaphore:
                    return await _post_chat_async(session, payload)

            try:
                if fc_mode == config.FC_CD:
                    data, n_calls = await cd_mode.complete_step_async(
                        chat_history,
                        active_tools,
                        post=_gated_post,
                        model=config.MODEL_NAME,
                    )
                else:
                    payload = _build_chat_payload(
                        chat_history, active_tools, fc_mode
                    )
                    data = await _gated_post(payload)
                    n_calls = 1
                api_calls += n_calls
            except aiohttp.ClientResponseError as e:
                if e.status == 400:
                    flags.append(f"context_overflow:turn_{turn_idx}")
                else:
                    flags.append(f"http_error_{e.status}:turn_{turn_idx}")
                step_results.append([f"ERROR: {e}"])
                break
            except Exception as e:
                step_results.append([f"ERROR: {e}"])
                break

            t = data.get("timings", {})
            total_prompt_n += t.get("prompt_n", 0)
            total_prompt_ms += t.get("prompt_ms", 0)
            total_predicted_n += t.get("predicted_n", 0)
            total_predicted_ms += t.get("predicted_ms", 0)
            if api_calls == n_calls:
                first_prompt_ms = t.get("prompt_ms", 0)

            if not data.get("choices"):
                step_results.append(["ERROR: empty choices in response"])
                break

            choice = data["choices"][0]
            message = choice["message"]

            action, func_strings, tool_calls_raw, parse_retries_left, flag = (
                _decide_assistant_step(
                    message, chat_history, fc_mode, parse_retries_left
                )
            )
            if flag:
                flags.append(f"{flag}:turn_{turn_idx}")
            if action == "retry":
                continue
            if action == "end_turn":
                break

            step_results.append(func_strings)

            # Tool execution is CPU-bound — run in thread to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                exec_results, _ = await loop.run_in_executor(
                    None,
                    lambda fs=func_strings: execute_multi_turn_func_call(
                        func_call_list=fs,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name=exec_model_name,
                        test_entry_id=entry_id,
                        long_context=(category == "multi_turn_long_context"),
                        is_evaL_run=False,
                    ),
                )
            except Exception as e:
                exec_results = [str(e)] * len(func_strings)

            _append_execution_feedback(
                chat_history, fc_mode, func_strings, tool_calls_raw, exec_results
            )
        else:
            flags.append(f"hit_step_limit:turn_{turn_idx}")

        turn_results.append(step_results)

    pbar.update(1)

    result_row = {
        "id": entry_id,
        "result": turn_results,
        "flags": json.dumps(flags) if flags else "",
    }
    timing = {
        "id": entry_id,
        "category": category,
        "turns": len(questions),
        "api_calls": api_calls,
        "total_prompt_n": total_prompt_n,
        "total_prompt_ms": round(total_prompt_ms, 2),
        "total_predicted_n": total_predicted_n,
        "total_predicted_ms": round(total_predicted_ms, 2),
        "first_prompt_ms": round(first_prompt_ms, 2),
        "flags": json.dumps(flags) if flags else "",
    }
    return result_row, timing


# ---------------------------------------------------------------------------
# Checkpoint / resume (same runs + responses schema)
# ---------------------------------------------------------------------------

def pending_multi_turn_entries(
    entries: list[dict],
    done_ids: set[str],
) -> list[dict]:
    """Return entries whose ids are not already saved for this run/category."""
    return [e for e in entries if e["id"] not in done_ids]


def _done_entry_ids(run_id: int | None, category: str) -> set[str]:
    if run_id is None:
        return set()
    return db.get_response_entry_ids(run_id, category, mode="multi")


def _checkpoint_entry(
    run_id: int | None,
    category: str,
    result_row: dict,
    timing: dict,
) -> None:
    """Flush one completed entry to SQLite. No-op when run_id is unset."""
    if run_id is None:
        return
    db.insert_multi_turn_entry(run_id, category, result_row, timing)


def _log_resume_skip(category: str, n_total: int, n_pending: int) -> None:
    n_skip = n_total - n_pending
    if n_skip:
        print(
            f"  resume: skipping {n_skip} saved {category} entries, "
            f"{n_pending} remaining"
        )


# ---------------------------------------------------------------------------
# Per-category driver
# ---------------------------------------------------------------------------

def generate_for_category(
    category: str,
    category_info: dict,
    fc_mode: str | None = None,
    run_id: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate multi-turn responses for all entries in one category (sync)."""
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    entries = load_jsonl(data_file)
    fc_mode = _resolve_fc_mode(fc_mode)
    if run_id is None:
        run_id = config.RUN_ID

    done_ids = _done_entry_ids(run_id, category)
    pending = pending_multi_turn_entries(entries, done_ids)
    _log_resume_skip(category, len(entries), len(pending))

    results: list[dict] = []
    timings: list[dict] = []

    for entry in tqdm(pending, desc=f"  {category}", unit="entry"):
        involved_classes = entry.get("involved_classes", [])
        all_tools = _load_func_docs(involved_classes)

        turn_results, timing = generate_for_entry(
            entry, category, all_tools, fc_mode=fc_mode
        )

        result_row = {
            "id": entry["id"],
            "result": turn_results,
            "flags": timing.get("flags", ""),
        }
        _checkpoint_entry(run_id, category, result_row, timing)
        results.append(result_row)
        timings.append(timing)

    return results, timings


async def _generate_and_checkpoint(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    entry: dict,
    category: str,
    all_tools: list[dict],
    pbar: tqdm,
    fc_mode: str | None,
    run_id: int | None,
) -> tuple[dict, dict]:
    """Run one entry, then persist it before the caller waits on gather()."""
    result_row, timing = await _generate_entry_async(
        session, semaphore, entry, category, all_tools, pbar, fc_mode=fc_mode
    )
    _checkpoint_entry(run_id, category, result_row, timing)
    return result_row, timing


async def _generate_for_category_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    category: str,
    category_info: dict,
    fc_mode: str | None = None,
    run_id: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate multi-turn responses for all entries concurrently."""
    data_file = config.BFCL_DATA_DIR / category_info["file"]
    entries = load_jsonl(data_file)
    fc_mode = _resolve_fc_mode(fc_mode)
    if run_id is None:
        run_id = config.RUN_ID

    done_ids = _done_entry_ids(run_id, category)
    pending = pending_multi_turn_entries(entries, done_ids)
    _log_resume_skip(category, len(entries), len(pending))

    if not pending:
        return [], []

    pbar = tqdm(total=len(pending), desc=f"  {category}", unit="entry")

    tasks = []
    for entry in pending:
        involved_classes = entry.get("involved_classes", [])
        all_tools = _load_func_docs(involved_classes)
        tasks.append(
            _generate_and_checkpoint(
                session,
                semaphore,
                entry,
                category,
                all_tools,
                pbar,
                fc_mode,
                run_id,
            )
        )

    pairs = await asyncio.gather(*tasks)
    pbar.close()

    results = [p[0] for p in pairs]
    timings = [p[1] for p in pairs]
    return results, timings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def _run_async(
    categories: dict[str, dict] | None = None,
    fc_mode: str | None = None,
) -> None:
    """Async implementation of run()."""
    run_id = config.RUN_ID
    semaphore = asyncio.Semaphore(config.PARALLEL_SLOTS)
    all_timings: list[dict] = []
    cats = categories or config.resolve_multi_turn_categories(None)
    fc_mode = _resolve_fc_mode(fc_mode)

    print(
        f"[multi-turn generate] fc-mode={fc_mode}  categories={list(cats)}"
        f"  run_id={run_id}"
    )
    if run_id is None:
        print(
            "[multi-turn generate] WARNING: RUN_ID is unset; completed entries "
            "will not be checkpointed. Pass --run-id or use "
            "benchmark.run_all --resume."
        )

    async with aiohttp.ClientSession() as session:
        for category, info in cats.items():
            print(f"[multi-turn generate] Category: {category}")
            results, timings = await _generate_for_category_async(
                session,
                semaphore,
                category,
                info,
                fc_mode=fc_mode,
                run_id=run_id,
            )
            all_timings.extend(timings)
            dest = "DB" if run_id is not None else "memory (not persisted)"
            print(f"  → {len(results)} new entries checkpointed to {dest}")

    print(
        f"[multi-turn generate] {len(all_timings)} new timings "
        f"{'saved to DB' if run_id is not None else 'not persisted'}"
    )


def run(
    categories: dict[str, dict] | None = None,
    fc_mode: str | None = None,
) -> None:
    """Generate multi-turn responses for selected categories."""
    asyncio.run(_run_async(categories=categories, fc_mode=fc_mode))


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BFCL multi-turn responses (NFC, IFFC, CD, or GST)."
    )
    parser.add_argument(
        "--fc-mode",
        choices=config.FC_MODES,
        default=config.FC_NFC,
        help=(
            "nfc = native OpenAI tools/tool_calls (default). "
            "iffc = Instruction-Followed Function Calling "
            "(arXiv:2608.22472 Appendix B JSON-in-text). "
            "cd = TagDispatch-style two-pass constrained decoding "
            "(not native tools GBNF; not full XGrammar-2). "
            "gst = Guided-Structured Templates on the NFC tools channel "
            "(arXiv:2509.18076; prompt-only, not GBNF)."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CATEGORY",
        default=None,
        help=(
            "Subset of multi-turn categories to run "
            f"(choices: {', '.join(config.MULTI_TURN_CATEGORIES)}). "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Existing runs.id to checkpoint into and resume from. "
            "Skips entry IDs already in responses for each category. "
            "Prefer benchmark.run_all --resume when running the full pipeline."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    config.set_fc_mode(args.fc_mode)
    if args.run_id is not None:
        config.RUN_ID = args.run_id
    run(
        categories=config.resolve_multi_turn_categories(args.categories),
        fc_mode=args.fc_mode,
    )
