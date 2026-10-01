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
- ``iffc-state``: IFFC JSON-in-text plus a harness-written ``[STATE]`` snapshot
  from actual tool execution results. Same checker as IFFC. Not GST / CD / GBNF.
- ``iffc-state-bon``: Best-of-N majority vote overlay on IFFC+STATE (MV@N,
  default N=5, temperature=0.7). Vote per step, execute the winner once.
- ``xlam``: Salesforce xLAM JSON-in-text (``{"tool_calls":[{name, arguments}]}``).
  Never sends ``tools``/``tool_choice``. Parse assistant text; empty
  ``{"tool_calls": []}`` ends the turn. Fair channel for ``xlam-1b``
  (Run 19 NFC 0/200 was format-channel collapse; Run 20 scored 6/200 = 3.0%).
- ``iffc-stop-repeat``: IFFC JSON-in-text plus harness force-stop after a tool
  observation when the proposed call set equals the previous assistant set
  (immediate), or is a subset of already-executed calls **after**
  ``OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS`` (6) assistant tool-turns in the
  current BFCL ask. Short VehicleControl re-calls are not truncated.
  Baseline ``iffc`` is unchanged. Decode/harness lever only.
- ``iffc-decision``: IFFC JSON-in-text plus Decision sidecar (Ollama System
  One Choice) before each Gemma step. Gemma still fills parameters. Mass-based
  noop + ``is_exec_observation`` position guard; Noul report-only; identical
  veto OFF by default. Plain mode preserves the R58 archaeology recipe
  (no OODA / no stop-repeat).
- ``iffc-ooda-card``: IFFC JSON-in-text plus a one-pass harness-written
  ``[OODA]`` card (goal / slots / last / next) prefixed on each decision
  turn, stacked on the same stop-repeat MIN=6 gate. **Not** ``iffc-state``.
  **Not** a Specialized 4-pass loop.
- ``iffc-decision-ooda``: Compose tev1 Decision sidecar + R42 OODA card +
  stop-repeat MIN=6. Precedence: Decision hard-noop pre-Gemma = sole stop
  that step; soft_once first noop ≠ ``[]`` (bypass); working copy = OODA
  card then Decision hint (non-persisted); stop-repeat post-Gemma only;
  OODA ``next`` advisory; ≤1 harness ``[]`` / step.

Diagnostic (not a new fc-mode): ``--iffc-strip-empty-assistant`` omits trailing
IFFC assistant ``[]`` stop turns from ``chat_history`` before the next BFCL
user turn. Default off. History-bookkeeping ablation, not a training method.
"""

import argparse
import asyncio
import json

import aiohttp
import requests
from tqdm import tqdm

from . import bon as bon_mode
from . import cd as cd_mode
from . import config, db
from . import gst as gst_mode
from . import iffc_state as iffc_state_mode
from . import ooda_card as ooda_card_mode
from . import stop_on_repeat
from . import decision_sidecar
from . import xlam as xlam_mode
from .generate_responses import bfcl_to_openai_tools, load_jsonl
from .iffc import (
    IFFC_PARSE_RETRIES,
    IFFC_RETRY_USER_MESSAGE,
    assistant_text,
    ensure_iffc_system_message,
    format_iffc_execution_user_message,
    iffc_calls_to_func_strings,
    strip_trailing_empty_assistant,
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


def _name_to_class_map(involved_classes: list[str]) -> dict[str, str]:
    """Map tool name → BFCL involved class (for Decision relevance clamp)."""
    mapping: dict[str, str] = {}
    for cls_name in involved_classes or []:
        doc_file = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls_name)
        if not doc_file:
            continue
        path = config.MULTI_TURN_FUNC_DOC_DIR / doc_file
        if not path.exists():
            continue
        for func in load_jsonl(path):
            name = func.get("name") if isinstance(func, dict) else None
            if name:
                mapping[str(name)] = cls_name
    return mapping


def _consume_decision_noop(
    *,
    soft_noop_used: bool,
    turn_idx: int,
    flags: list[str],
) -> tuple[str, bool]:
    """Decide soft vs hard noop for iffc-decision.

    Returns ``(action, soft_noop_used)`` where action is:
      - ``"soft"``: convert to bypass this step (Gemma free-acts; re-probe later)
      - ``"hard"``: append [] and break the step loop
    """
    mode = decision_sidecar.noop_consumer_mode()
    if mode == "hard":
        flags.append(f"decision_noop_hard:turn_{turn_idx}")
        return "hard", soft_noop_used
    if mode == "noop_as_bypass":
        flags.append(f"decision_noop_soft_bypass:turn_{turn_idx}")
        return "soft", soft_noop_used
    # soft_once (default): first noop → bypass; later Decision noop → hard
    if not soft_noop_used:
        flags.append(f"decision_noop_soft:turn_{turn_idx}")
        return "soft", True
    flags.append(f"decision_noop_hard:turn_{turn_idx}")
    return "hard", soft_noop_used

def _uses_decision(fc_mode: str) -> bool:
    """True for plain ``iffc-decision`` and compose ``iffc-decision-ooda``."""
    return fc_mode in config.FC_IFFC_DECISION_MODES


def _uses_ooda(fc_mode: str) -> bool:
    """True for plain ``iffc-ooda-card`` and compose ``iffc-decision-ooda``."""
    return fc_mode in config.FC_IFFC_OODA_MODES

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


def _resolve_strip_empty_assistant(
    strip_empty_assistant: bool | None,
) -> bool:
    if strip_empty_assistant is None:
        return bool(config.IFFC_STRIP_EMPTY_ASSISTANT)
    return bool(strip_empty_assistant)


def _iffc_family_mode(fc_mode: str) -> bool:
    """Plain IFFC plus iffc-state / stop-repeat / ooda / decision overlays."""
    return (
        fc_mode == config.FC_IFFC
        or fc_mode in config.FC_IFFC_STOP_REPEAT_MODES
        or _uses_decision(fc_mode)
        or fc_mode in config.FC_IFFC_STATE_MODES
    )


def _maybe_strip_trailing_empty_assistant(
    chat_history: list[dict],
    fc_mode: str,
    enabled: bool,
) -> None:
    """Drop trailing assistant ``[]`` from IFFC *chat_history* when *enabled*.

    Diagnostic-only (``--iffc-strip-empty-assistant``). Default IFFC keeps
    the within-turn stop message so Run 5–13 bookkeeping is unchanged.
    When on, history must not contain assistant ``[]`` when the next BFCL
    user turn is appended (SFT v2 self-conditioning probe). No-op for
    NFC / GST / CD.
    """
    if not enabled or not _iffc_family_mode(fc_mode):
        return
    strip_trailing_empty_assistant(chat_history)


def _build_chat_payload(
    chat_history: list[dict],
    active_tools: list[dict],
    fc_mode: str,
    ooda_tracker: ooda_card_mode.OodaCardTracker | None = None,
) -> dict:
    """Build a chat-completions payload. IFFC family never send tools/tool_choice.

    GST uses the NFC tools channel (``tools`` + ``tool_choice=auto``).
    BoN sampling uses temperature=0.7 via :mod:`benchmark.bon`; this helper
    still emits 0.7 for ``iffc-state-bon`` so a stray single-sample path
    would not silently go greedy.

    OODA modes inject the ``[OODA]`` card on a **copy** of messages so the
    persistent transcript stays IFFC-clean for stop-on-repeat + Decision.
    """
    temperature = (
        config.BON_TEMPERATURE if fc_mode == config.FC_IFFC_STATE_BON else 0.0
    )
    messages: list[dict] = chat_history
    if _uses_ooda(fc_mode):
        tracker = ooda_tracker or ooda_card_mode.OodaCardTracker()
        messages = ooda_card_mode.inject_card(
            chat_history, tracker, has_tools=bool(active_tools)
        )
    payload: dict = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
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
        # Persistent transcript is JSON-in-text for IFFC, CD, iffc-state,
        # iffc-state-bon, and xlam. CD applies grammar/json_schema on a
        # working copy inside complete_step. iffc-state / iffc-state-bon
        # keep Appendix B + a short STATE-trust addendum. xlam uses the
        # Salesforce {"tool_calls":[...]} contract, not Appendix B.
        if fc_mode in config.FC_IFFC_STATE_MODES:
            iffc_state_mode.ensure_iffc_state_system_message(chat_history, active_tools)
        elif fc_mode == config.FC_XLAM:
            xlam_mode.ensure_xlam_system_message(chat_history, active_tools)
        elif _uses_ooda(fc_mode):
            ooda_card_mode.ensure_ooda_system_message(chat_history, active_tools)
        else:
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
    *,
    name_hint: str | None = None,
    strong_name_hint: bool = False,
    ooda_tracker: ooda_card_mode.OodaCardTracker | None = None,
) -> tuple[dict, int]:
    """One model step. CD may issue two constrained requests; BoN samples N.

    Working-copy order for compose (``iffc-decision-ooda``):
    history → OODA ``inject_card`` → Decision name-hint last.
    Neither card nor hint mutates persistent ``chat_history``.
    """
    if fc_mode == config.FC_CD:
        return cd_mode.complete_step(
            chat_history,
            active_tools,
            post=_post_chat_sync,
            model=config.MODEL_NAME,
        )
    if fc_mode == config.FC_IFFC_STATE_BON:
        return bon_mode.complete_step(
            chat_history,
            active_tools,
            post=_post_chat_sync,
            model=config.MODEL_NAME,
            n=config.BON_N,
        )
    payload = _build_chat_payload(
        chat_history, active_tools, fc_mode, ooda_tracker=ooda_tracker
    )
    if name_hint:
        # Append after OODA inject (payload messages may already be a copy).
        working = list(payload["messages"])
        working.append(
            decision_sidecar.name_hint_user_message(
                name_hint, strong=strong_name_hint
            )
        )
        payload = {**payload, "messages": working}
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
    *,
    decision_name_hint: str | None = None,
    name_bind_retries_left: int = 0,
) -> tuple[str, list[str] | None, list | None, int, str | None, int]:
    """Interpret an assistant message.

    Returns
    ``(action, func_strings, tool_calls_raw, parse_retries_left, flag,
    name_bind_retries_left)`` where *action* is ``execute``, ``end_turn``,
    ``retry``, or ``retry_name_hint``.

    When *decision_name_hint* is set (Decision execute path), name-bind
    inside this function: keep-all calls matching ``T``; on any mismatch retry
    once (non-persisted stronger hint), then bypass keeping Gemma's call with
    ``name_mismatch`` flag.
    """
    if _is_json_text_mode(fc_mode):
        content = assistant_text(message)
        parsed = try_parse_iffc_response(content)
        if parsed:
            # stop-repeat post-Gemma only (R42 / compose). Skipped when
            # Decision hard-noop already ended the step pre-Gemma.
            if fc_mode in config.FC_IFFC_STOP_REPEAT_MODES:
                reason = stop_on_repeat.force_stop_reason(chat_history, content)
                if reason is not None:
                    # Force a clean [] stop; do not re-execute the looped calls.
                    chat_history.append({"role": "assistant", "content": "[]"})
                    return (
                        "end_turn",
                        None,
                        None,
                        IFFC_PARSE_RETRIES,
                        f"stop_on_repeat_{reason}",
                        name_bind_retries_left,
                    )
            if decision_name_hint and _uses_decision(fc_mode):
                def _call_name(call: dict) -> str:
                    return str(
                        call.get("function_name") or call.get("name") or ""
                    )

                names = [_call_name(c) for c in parsed]
                if any(n != decision_name_hint for n in names):
                    if name_bind_retries_left > 0:
                        # Do not append mismatched content; retry with stronger hint.
                        return (
                            "retry_name_hint",
                            None,
                            None,
                            parse_retries_left,
                            None,
                            name_bind_retries_left - 1,
                        )
                    # Bypass: keep Gemma call as-is + name_mismatch flag.
                    func_strings = iffc_calls_to_func_strings(parsed)
                    chat_history.append({"role": "assistant", "content": content})
                    return (
                        "execute",
                        func_strings,
                        None,
                        IFFC_PARSE_RETRIES,
                        "name_mismatch",
                        0,
                    )
                # All names match T — keep-all (same-name multi-call OK).
                func_strings = iffc_calls_to_func_strings(parsed)
                chat_history.append({"role": "assistant", "content": content})
                return (
                    "execute",
                    func_strings,
                    None,
                    IFFC_PARSE_RETRIES,
                    None,
                    name_bind_retries_left,
                )
            func_strings = iffc_calls_to_func_strings(parsed)
            chat_history.append({"role": "assistant", "content": content})
            return (
                "execute",
                func_strings,
                None,
                IFFC_PARSE_RETRIES,
                None,
                name_bind_retries_left,
            )
        if parsed is None and parse_retries_left > 0:
            chat_history.append({"role": "assistant", "content": content})
            retry_msg = (
                xlam_mode.XLAM_RETRY_USER_MESSAGE
                if fc_mode == config.FC_XLAM
                else IFFC_RETRY_USER_MESSAGE
            )
            chat_history.append({"role": "user", "content": retry_msg})
            return (
                "retry",
                None,
                None,
                parse_retries_left - 1,
                None,
                name_bind_retries_left,
            )
        chat_history.append({"role": "assistant", "content": content})
        if parsed is not None:
            flag = None
        elif fc_mode == config.FC_CD:
            flag = "cd_unparseable"
        elif fc_mode == config.FC_XLAM:
            flag = "xlam_unparseable"
        else:
            flag = "iffc_unparseable"
        return (
            "end_turn",
            None,
            None,
            parse_retries_left,
            flag,
            name_bind_retries_left,
        )

    tool_calls_raw = message.get("tool_calls")
    if tool_calls_raw:
        func_strings = _tool_calls_to_func_strings(tool_calls_raw)
        chat_history.append(message)
        return (
            "execute",
            func_strings,
            tool_calls_raw,
            parse_retries_left,
            None,
            name_bind_retries_left,
        )

    content = message.get("content", "")
    chat_history.append({"role": "assistant", "content": content})
    return (
        "end_turn",
        None,
        None,
        parse_retries_left,
        None,
        name_bind_retries_left,
    )


def _append_execution_feedback(
    chat_history: list[dict],
    fc_mode: str,
    func_strings: list[str],
    tool_calls_raw: list | None,
    exec_results: list,
    state_tracker: iffc_state_mode.StateTracker | None = None,
    ooda_tracker: ooda_card_mode.OodaCardTracker | None = None,
) -> None:
    if _is_json_text_mode(fc_mode):
        if fc_mode in config.FC_IFFC_STATE_MODES and state_tracker is not None:
            content = iffc_state_mode.record_and_format(
                state_tracker, func_strings, exec_results
            )
        elif fc_mode == config.FC_XLAM:
            content = xlam_mode.format_xlam_execution_user_message(
                func_strings, exec_results
            )
        else:
            content = format_iffc_execution_user_message(func_strings, exec_results)
            if _uses_ooda(fc_mode) and ooda_tracker is not None:
                ooda_card_mode.record_execution(
                    ooda_tracker, func_strings, exec_results
                )
        chat_history.append({
            "role": "user",
            "content": content,
        })
        if fc_mode in config.FC_IFFC_STATE_MODES:
            iffc_state_mode.compress_older_dumps(chat_history)
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
    strip_empty_assistant: bool | None = None,
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
    name_to_class = _name_to_class_map(involved_classes)
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
    strip_empty_assistant = _resolve_strip_empty_assistant(strip_empty_assistant)
    state_tracker = (
        iffc_state_mode.StateTracker()
        if fc_mode in config.FC_IFFC_STATE_MODES
        else None
    )
    ooda_tracker = (
        ooda_card_mode.OodaCardTracker() if _uses_ooda(fc_mode) else None
    )

    for turn_idx, turn_messages in enumerate(questions):
        # Before the next BFCL user turn: drop the previous turn's IFFC ``[]``
        # stop so it cannot self-condition the next question (flag default off).
        _maybe_strip_trailing_empty_assistant(
            chat_history, fc_mode, strip_empty_assistant
        )
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
        soft_noop_used = False  # Decision soft_once: one Gemma-act chance per turn
        for _step in range(config.MULTI_TURN_MAX_STEPS):
            _prepare_step_messages(chat_history, active_tools, fc_mode)

            decision_name_hint: str | None = None
            name_bind_retries_left = 0
            strong_name_hint = False
            soft_noop_this_step = False
            if _uses_decision(fc_mode):
                probe = decision_sidecar.probe_next_tool(
                    chat_history,
                    active_tools,
                    entry_id=entry.get("id"),
                    turn=turn_idx,
                    step=_step,
                    name_to_class=name_to_class,
                )
                pol = probe.policy
                if pol.action == decision_sidecar.ACTION_NOOP:
                    noop_act, soft_noop_used = _consume_decision_noop(
                        soft_noop_used=soft_noop_used,
                        turn_idx=turn_idx,
                        flags=flags,
                    )
                    if noop_act == "hard":
                        chat_history.append({"role": "assistant", "content": "[]"})
                        _maybe_strip_trailing_empty_assistant(
                            chat_history, fc_mode, strip_empty_assistant
                        )
                        break
                    # soft: fall through as bypass so Gemma acts → new obs → re-probe
                    soft_noop_this_step = True
                elif pol.action == decision_sidecar.ACTION_BYPASS:
                    reason = pol.bypass_reason or "client_exception"
                    flags.append(f"decision_bypass_{reason}:turn_{turn_idx}")
                elif pol.action == decision_sidecar.ACTION_EXECUTE and pol.tool_name:
                    decision_name_hint = pol.tool_name
                    name_bind_retries_left = 1

            try:
                data, n_calls = _complete_chat_sync(
                    chat_history,
                    active_tools,
                    fc_mode,
                    name_hint=decision_name_hint,
                    strong_name_hint=strong_name_hint,
                    ooda_tracker=ooda_tracker,
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

            (
                action,
                func_strings,
                tool_calls_raw,
                parse_retries_left,
                flag,
                name_bind_retries_left,
            ) = _decide_assistant_step(
                message,
                chat_history,
                fc_mode,
                parse_retries_left,
                decision_name_hint=decision_name_hint,
                name_bind_retries_left=name_bind_retries_left,
            )
            if flag:
                if flag == "name_mismatch":
                    flags.append(f"decision_bypass_name_mismatch:turn_{turn_idx}")
                else:
                    flags.append(f"{flag}:turn_{turn_idx}")
            if action == "retry_name_hint" and decision_name_hint:
                # Re-call Gemma with stronger non-persisted hint; no history append.
                strong_name_hint = True
                try:
                    data, n_calls = _complete_chat_sync(
                        chat_history,
                        active_tools,
                        fc_mode,
                        name_hint=decision_name_hint,
                        strong_name_hint=True,
                        ooda_tracker=ooda_tracker,
                    )
                    api_calls += n_calls
                except Exception as e:
                    step_results.append([f"ERROR: {e}"])
                    break
                if not data.get("choices"):
                    step_results.append(["ERROR: empty choices in response"])
                    break
                message = data["choices"][0]["message"]
                (
                    action,
                    func_strings,
                    tool_calls_raw,
                    parse_retries_left,
                    flag,
                    name_bind_retries_left,
                ) = _decide_assistant_step(
                    message,
                    chat_history,
                    fc_mode,
                    parse_retries_left,
                    decision_name_hint=decision_name_hint,
                    name_bind_retries_left=name_bind_retries_left,
                )
                if flag:
                    if flag == "name_mismatch":
                        flags.append(
                            f"decision_bypass_name_mismatch:turn_{turn_idx}"
                        )
                    else:
                        flags.append(f"{flag}:turn_{turn_idx}")
            if action == "retry":
                continue
            if action == "end_turn":
                if soft_noop_this_step:
                    flags.append(f"decision_noop_soft_noact:turn_{turn_idx}")
                # Optional: strip immediately after the within-turn ``[]`` stop
                # as well as before the next BFCL user turn.
                _maybe_strip_trailing_empty_assistant(
                    chat_history, fc_mode, strip_empty_assistant
                )
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
                chat_history,
                fc_mode,
                func_strings,
                tool_calls_raw,
                exec_results,
                state_tracker=state_tracker,
                ooda_tracker=ooda_tracker,
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
    name_to_class = _name_to_class_map(involved_classes)
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
    state_tracker = (
        iffc_state_mode.StateTracker()
        if fc_mode in config.FC_IFFC_STATE_MODES
        else None
    )
    ooda_tracker = (
        ooda_card_mode.OodaCardTracker() if _uses_ooda(fc_mode) else None
    )
    strip_empty_assistant = _resolve_strip_empty_assistant(None)

    for turn_idx, turn_messages in enumerate(questions):
        _maybe_strip_trailing_empty_assistant(
            chat_history, fc_mode, strip_empty_assistant
        )
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
        soft_noop_used = False  # Decision soft_once: one Gemma-act chance per turn
        for _step in range(config.MULTI_TURN_MAX_STEPS):
            _prepare_step_messages(chat_history, active_tools, fc_mode)

            async def _gated_post(payload: dict) -> dict:
                async with semaphore:
                    return await _post_chat_async(session, payload)

            decision_name_hint: str | None = None
            name_bind_retries_left = 0
            soft_noop_this_step = False
            if _uses_decision(fc_mode):
                loop = asyncio.get_running_loop()
                _ch, _tools, _eid, _t, _s, _n2c = (
                    chat_history,
                    active_tools,
                    entry.get("id"),
                    turn_idx,
                    _step,
                    name_to_class,
                )
                probe = await loop.run_in_executor(
                    None,
                    lambda ch=_ch, tools=_tools, eid=_eid, t=_t, s=_s, n2c=_n2c: (
                        decision_sidecar.probe_next_tool(
                            ch,
                            tools,
                            entry_id=eid,
                            turn=t,
                            step=s,
                            name_to_class=n2c,
                        )
                    ),
                )
                pol = probe.policy
                if pol.action == decision_sidecar.ACTION_NOOP:
                    noop_act, soft_noop_used = _consume_decision_noop(
                        soft_noop_used=soft_noop_used,
                        turn_idx=turn_idx,
                        flags=flags,
                    )
                    if noop_act == "hard":
                        chat_history.append({"role": "assistant", "content": "[]"})
                        _maybe_strip_trailing_empty_assistant(
                            chat_history, fc_mode, strip_empty_assistant
                        )
                        break
                    # soft: fall through as bypass so Gemma acts → new obs → re-probe
                    soft_noop_this_step = True
                elif pol.action == decision_sidecar.ACTION_BYPASS:
                    reason = pol.bypass_reason or "client_exception"
                    flags.append(f"decision_bypass_{reason}:turn_{turn_idx}")
                elif pol.action == decision_sidecar.ACTION_EXECUTE and pol.tool_name:
                    decision_name_hint = pol.tool_name
                    name_bind_retries_left = 1

            try:
                if fc_mode == config.FC_CD:
                    data, n_calls = await cd_mode.complete_step_async(
                        chat_history,
                        active_tools,
                        post=_gated_post,
                        model=config.MODEL_NAME,
                    )
                elif fc_mode == config.FC_IFFC_STATE_BON:
                    data, n_calls = await bon_mode.complete_step_async(
                        chat_history,
                        active_tools,
                        post=_gated_post,
                        model=config.MODEL_NAME,
                        n=config.BON_N,
                    )
                else:
                    payload = _build_chat_payload(
                        chat_history,
                        active_tools,
                        fc_mode,
                        ooda_tracker=ooda_tracker,
                    )
                    if decision_name_hint:
                        # OODA card first (in payload), Decision hint last.
                        working = list(payload["messages"])
                        working.append(
                            decision_sidecar.name_hint_user_message(
                                decision_name_hint, strong=False
                            )
                        )
                        payload = {**payload, "messages": working}
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

            (
                action,
                func_strings,
                tool_calls_raw,
                parse_retries_left,
                flag,
                name_bind_retries_left,
            ) = _decide_assistant_step(
                message,
                chat_history,
                fc_mode,
                parse_retries_left,
                decision_name_hint=decision_name_hint,
                name_bind_retries_left=name_bind_retries_left,
            )
            if flag:
                if flag == "name_mismatch":
                    flags.append(f"decision_bypass_name_mismatch:turn_{turn_idx}")
                else:
                    flags.append(f"{flag}:turn_{turn_idx}")
            if action == "retry_name_hint" and decision_name_hint:
                try:
                    payload = _build_chat_payload(
                        chat_history,
                        active_tools,
                        fc_mode,
                        ooda_tracker=ooda_tracker,
                    )
                    working = list(payload["messages"])
                    working.append(
                        decision_sidecar.name_hint_user_message(
                            decision_name_hint, strong=True
                        )
                    )
                    payload = {**payload, "messages": working}
                    data = await _gated_post(payload)
                    api_calls += 1
                except Exception as e:
                    step_results.append([f"ERROR: {e}"])
                    break
                if not data.get("choices"):
                    step_results.append(["ERROR: empty choices in response"])
                    break
                message = data["choices"][0]["message"]
                (
                    action,
                    func_strings,
                    tool_calls_raw,
                    parse_retries_left,
                    flag,
                    name_bind_retries_left,
                ) = _decide_assistant_step(
                    message,
                    chat_history,
                    fc_mode,
                    parse_retries_left,
                    decision_name_hint=decision_name_hint,
                    name_bind_retries_left=name_bind_retries_left,
                )
                if flag:
                    if flag == "name_mismatch":
                        flags.append(
                            f"decision_bypass_name_mismatch:turn_{turn_idx}"
                        )
                    else:
                        flags.append(f"{flag}:turn_{turn_idx}")
            if action == "retry":
                continue
            if action == "end_turn":
                if soft_noop_this_step:
                    flags.append(f"decision_noop_soft_noact:turn_{turn_idx}")
                _maybe_strip_trailing_empty_assistant(
                    chat_history, fc_mode, strip_empty_assistant
                )
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
                chat_history,
                fc_mode,
                func_strings,
                tool_calls_raw,
                exec_results,
                state_tracker=state_tracker,
                ooda_tracker=ooda_tracker,
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
    if _uses_decision(fc_mode):
        decision_sidecar.prewarm()

    done_ids = _done_entry_ids(run_id, category)
    pending = pending_multi_turn_entries(entries, done_ids)
    _log_resume_skip(category, len(entries), len(pending))

    results: list[dict] = []
    timings: list[dict] = []

    for entry in tqdm(pending, desc=f"  {category}", unit="entry"):
        involved_classes = entry.get("involved_classes", [])
        all_tools = _load_func_docs(involved_classes)
        name_to_class = _name_to_class_map(involved_classes)

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
    if _uses_decision(fc_mode):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, decision_sidecar.prewarm)

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
        name_to_class = _name_to_class_map(involved_classes)
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

    extra = ""
    if fc_mode == config.FC_IFFC_STATE_BON:
        extra = f"  bon_n={config.BON_N}  temp={config.BON_TEMPERATURE}"
    if config.IFFC_STRIP_EMPTY_ASSISTANT:
        extra += "  iffc_strip_empty_assistant=1 (DIAGNOSTIC)"
    print(
        f"[multi-turn generate] fc-mode={fc_mode}  categories={list(cats)}"
        f"  run_id={run_id}{extra}"
    )
    if _uses_decision(fc_mode):
        knobs = decision_sidecar.hygiene_knobs()
        print(
            f"[multi-turn generate] Decision sidecar prewarm "
            f"model={decision_sidecar.DECISION_MODEL} "
            f"endpoint={decision_sidecar.DECISION_ENDPOINT} "
            f"timeout_s={knobs['DECISION_TIMEOUT_S']} "
            f"retries={knobs['DECISION_TIMEOUT_RETRIES']} "
            f"max_inflight={knobs['DECISION_MAX_INFLIGHT']} "
            f"prewarm_slots={knobs['DECISION_PREWARM_SLOTS']} "
            f"OLLAMA_NUM_PARALLEL={knobs['OLLAMA_NUM_PARALLEL']}"
        )
    if run_id is None:
        print(
            "[multi-turn generate] WARNING: RUN_ID is unset; completed entries "
            "will not be checkpointed. Pass --run-id or use "
            "benchmark.run_all --resume."
        )

    async with aiohttp.ClientSession() as session:
        if _uses_decision(fc_mode):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, decision_sidecar.prewarm)
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
        description="Generate BFCL multi-turn responses (NFC, IFFC, CD, GST, IFFC+STATE, IFFC+STATE+BoN, or xLAM)."
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
            "(arXiv:2509.18076; prompt-only, not GBNF). "
            "iffc-state = IFFC JSON-in-text plus harness-written STATE "
            "(not GST; not CD/GBNF). "
            "iffc-state-bon = Best-of-N majority vote on IFFC+STATE "
            "(default N=5, temperature=0.7; not NFC/GST/CD). "
            "xlam = Salesforce xLAM JSON-in-text "
            '({"tool_calls":[{name, arguments}]}; no native tools).'
        ),
    )
    parser.add_argument(
        "--bon-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Best-of-N sample count for --fc-mode iffc-state-bon "
            f"(default: {config.BON_N_DEFAULT}). Ignored by other modes."
        ),
    )
    parser.add_argument(
        "--iffc-strip-empty-assistant",
        action="store_true",
        help=(
            "Diagnostic-only: after an IFFC within-turn stop that parses as [], "
            "drop that trailing assistant [] from chat_history before the next "
            "BFCL user turn (and right after the stop). Default off — Run 5–13 "
            "IFFC bookkeeping unchanged. Not a training method."
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
    if args.bon_n is not None:
        config.set_bon_n(args.bon_n)
    config.set_iffc_strip_empty_assistant(args.iffc_strip_empty_assistant)
    if args.run_id is not None:
        config.RUN_ID = args.run_id
    run(
        categories=config.resolve_multi_turn_categories(args.categories),
        fc_mode=args.fc_mode,
    )
