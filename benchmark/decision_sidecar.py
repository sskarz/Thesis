"""Thin adapter: Thesis harness → Documents/decision-probe sidecar.

Mirrors ``stop_on_repeat.py`` style: logic lives outside generate loop;
this module is only imported when ``fc_mode=iffc-decision``.

Kill-switches (HARNESS_FIX_PLAN levers 1+2+3)::

    DECISION_DESC_MODE          off|short|shrunk|template_fallback  (default short)
    DECISION_DESC_MAX_CHARS     int (default 80)
    DECISION_NOOP_ARGMAX        0|1 (default 1)
    DECISION_NOOP_SOFT          soft_once|noop_as_bypass|hard (default soft_once)
    DECISION_NOOP_HARD_BREAK    0|1 (default 0; 1 forces hard break)
    DECISION_CLAMP_STRATEGY     lex|relevant (default relevant)

Turn-0 timeout hygiene (R58 ~87% first-call timeout_warm under load)::

    DECISION_TIMEOUT_S          float (default 45; was 20)
    DECISION_TIMEOUT_RETRIES    int (default 1; retry once on Timeout*)
    DECISION_MAX_INFLIGHT       int (default 2; 0=unlimited; caps concurrent tev1 POSTs)
    DECISION_PREWARM_SLOTS      int (default = MAX_INFLIGHT or 2; parallel tiny prewarms)
    DECISION_KEEP_ALIVE         str (default 30m)
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import stop_on_repeat

_DECISION_PROBE = Path.home() / "Documents" / "decision-probe"
if str(_DECISION_PROBE) not in sys.path:
    sys.path.insert(0, str(_DECISION_PROBE))

from sidecar import (  # noqa: E402
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
)
from sidecar.client import SystemOneResult, post_systemone  # noqa: E402
from sidecar.log import append_jsonl  # noqa: E402
from sidecar.policy import (  # noqa: E402
    ACTION_BYPASS,
    ACTION_EXECUTE,
    ACTION_NOOP,
    PolicyDecision,
    decide,
)
from sidecar.state import (  # noqa: E402
    DEFAULT_DESC_MAX_CHARS,
    MAX_ACTIVE_TOOLS,
    MAX_CHOICE_CRITERIA,
    active_tool_names,
    build_questions,
    chat_history_to_state,
    clamp_tools,
    desc_map_from_tools,
    estimate_tokens,
    latest_user_ask,
)


# Env knobs
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DECISION_ENDPOINT = _env_str("DECISION_ENDPOINT", DEFAULT_ENDPOINT)
DECISION_MODEL = _env_str("DECISION_MODEL", DEFAULT_MODEL)
DECISION_API_KEY = _env_str("DECISION_API_KEY", "ollama")
# Default 45s: R58 first-call timeout_warm storm hit the old 20s wall under
# PARALLEL_SLOTS gather stampede (Decision was ungated vs OLLAMA_NUM_PARALLEL).
DECISION_TIMEOUT_S = _env_float("DECISION_TIMEOUT_S", DEFAULT_TIMEOUT_S)
DECISION_CHOICE_MIN = _env_float("DECISION_CHOICE_MIN", 0.35)
DECISION_NOOP_MIN = _env_float("DECISION_NOOP_MIN", 0.30)  # placeholder
DECISION_IDENTICAL_VETO = _env_int("DECISION_IDENTICAL_VETO", 0)  # OFF
DECISION_KEEP_ALIVE = _env_str("DECISION_KEEP_ALIVE", "30m")
DECISION_RECENCY_MASK = _env_int("DECISION_RECENCY_MASK", 0)
DECISION_TIMEOUT_RETRIES = _env_int("DECISION_TIMEOUT_RETRIES", 1)
DECISION_MAX_INFLIGHT = _env_int("DECISION_MAX_INFLIGHT", 2)  # 0 = unlimited
_prewarm_slots_env = os.environ.get("DECISION_PREWARM_SLOTS")
if _prewarm_slots_env is None or _prewarm_slots_env == "":
    DECISION_PREWARM_SLOTS = (
        DECISION_MAX_INFLIGHT if DECISION_MAX_INFLIGHT > 0 else 2
    )
else:
    try:
        DECISION_PREWARM_SLOTS = max(1, int(_prewarm_slots_env))
    except ValueError:
        DECISION_PREWARM_SLOTS = 2

# Lever kill-switches (defaults ON for shipped levers)
DECISION_DESC_MODE = _env_str("DECISION_DESC_MODE", "short")
DECISION_DESC_MAX_CHARS = _env_int("DECISION_DESC_MAX_CHARS", DEFAULT_DESC_MAX_CHARS)
DECISION_NOOP_ARGMAX = _env_int("DECISION_NOOP_ARGMAX", 1)
DECISION_NOOP_SOFT = _env_str("DECISION_NOOP_SOFT", "soft_once")
DECISION_NOOP_HARD_BREAK = _env_int("DECISION_NOOP_HARD_BREAK", 0)
DECISION_CLAMP_STRATEGY = _env_str("DECISION_CLAMP_STRATEGY", "relevant")

# Legal prewarm Choice: need 2–26 criteria (noop alone is illegal).
_PREWARM_TOOL = "__prewarm__"

_prewarm_lock = threading.Lock()
_prewarmed = False
_prewarm_was_cold = True
_inflight_lock = threading.Lock()
_inflight_sem: threading.Semaphore | None = None


def _get_inflight_sem() -> threading.Semaphore | None:
    """Lazy semaphore so tests can monkeypatch DECISION_MAX_INFLIGHT."""
    global _inflight_sem
    n = int(DECISION_MAX_INFLIGHT or 0)
    if n <= 0:
        return None
    with _inflight_lock:
        if _inflight_sem is None:
            _inflight_sem = threading.Semaphore(n)
        return _inflight_sem


def reset_inflight_gate() -> None:
    """Test helper: drop cached semaphore after monkeypatching MAX_INFLIGHT."""
    global _inflight_sem
    with _inflight_lock:
        _inflight_sem = None


def _is_timeout_client_error(error: str | None) -> bool:
    if not error:
        return False
    if error in ("timeout_cold", "timeout_warm"):
        return True
    if "Timeout" in error or "timeout" in error.lower():
        return True
    return False


def _post_systemone_gated(**kwargs) -> SystemOneResult:
    """POST System One under DECISION_MAX_INFLIGHT (queue wait ≠ HTTP timeout)."""
    sem = _get_inflight_sem()
    if sem is None:
        return post_systemone(**kwargs)
    sem.acquire()
    try:
        return post_systemone(**kwargs)
    finally:
        sem.release()


def _post_systemone_with_retries(**kwargs) -> tuple[SystemOneResult, int]:
    """Gate + retry once (default) on Timeout*. Returns (result, attempts_used)."""
    attempts = 1 + max(0, int(DECISION_TIMEOUT_RETRIES or 0))
    last: SystemOneResult | None = None
    used = 0
    for i in range(attempts):
        used = i + 1
        last = _post_systemone_gated(**kwargs)
        if last.ok or not _is_timeout_client_error(last.error):
            return last, used
        # timeout → retry (if attempts remain)
    assert last is not None
    return last, used


def _ollama_num_parallel() -> str:
    """Best-effort: process env, then macOS launchctl getenv."""
    env = os.environ.get("OLLAMA_NUM_PARALLEL")
    if env:
        return env
    try:
        import subprocess

        out = subprocess.run(
            ["launchctl", "getenv", "OLLAMA_NUM_PARALLEL"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        val = (out.stdout or "").strip()
        if val:
            return val
    except Exception:  # noqa: BLE001
        pass
    return "(unset)"


def hygiene_knobs() -> dict[str, object]:
    """Snapshot of turn0 timeout hygiene knobs (for logs / STATUS)."""
    return {
        "DECISION_TIMEOUT_S": DECISION_TIMEOUT_S,
        "DECISION_TIMEOUT_RETRIES": DECISION_TIMEOUT_RETRIES,
        "DECISION_MAX_INFLIGHT": DECISION_MAX_INFLIGHT,
        "DECISION_PREWARM_SLOTS": DECISION_PREWARM_SLOTS,
        "DECISION_KEEP_ALIVE": DECISION_KEEP_ALIVE,
        "OLLAMA_NUM_PARALLEL": _ollama_num_parallel(),
    }


@dataclass
class ProbeResult:
    policy: PolicyDecision
    client: SystemOneResult
    tool_names: list[str] = field(default_factory=list)
    cold_or_warm: str = "warm"
    log_record: dict[str, Any] = field(default_factory=dict)


def is_noop_allowed(chat_history: list[dict]) -> bool:
    """Position guard: noop only after an exec observation (never ask step 0)."""
    if not chat_history:
        return False
    prev = chat_history[-1]
    if prev.get("role") != "user":
        return False
    return stop_on_repeat.is_exec_observation(prev.get("content"))


def noop_consumer_mode() -> str:
    """Effective soft/hard consumer mode for generate_multi_turn.

    ``DECISION_NOOP_HARD_BREAK=1`` forces ``hard`` (status-quo break).
    """
    if DECISION_NOOP_HARD_BREAK:
        return "hard"
    mode = (DECISION_NOOP_SOFT or "soft_once").strip().lower()
    if mode not in ("soft_once", "noop_as_bypass", "hard"):
        return "soft_once"
    return mode


def _one_prewarm_post() -> SystemOneResult:
    """Single legal prewarm Choice POST (dummy tool + noop)."""
    state = (
        f"available_tools:\n  - {_PREWARM_TOOL}\n"
        "transcript:\n  [user] warmup"
    )
    questions = build_questions(
        [_PREWARM_TOOL], include_noul=False, desc_mode="off"
    )
    questions.pop("_meta", None)
    criteria = list(questions["next_tool"]["criteria"].keys())
    assert 2 <= len(criteria) <= MAX_CHOICE_CRITERIA, (
        f"prewarm criteria out of range: {len(criteria)}"
    )
    result, _attempts = _post_systemone_with_retries(
        state=state,
        questions=questions,
        model=DECISION_MODEL,
        endpoint=DECISION_ENDPOINT,
        api_key=DECISION_API_KEY,
        timeout_s=DECISION_TIMEOUT_S,
        criteria_names=criteria,
        keep_alive=DECISION_KEEP_ALIVE or None,
    )
    return result


def prewarm(*, force: bool = False) -> SystemOneResult | None:
    """Tiny System One call(s) so cold load is not the first real step.

    Must be a **legal** Choice question: 2–26 criteria. Uses a dummy tool
    ``__prewarm__`` plus deploy ``noop`` (never noop-only).

    Fires ``DECISION_PREWARM_SLOTS`` parallel tiny calls (gated by
    ``DECISION_MAX_INFLIGHT``) so multiple Ollama slots get a warm KV touch
    before the gather stampede.
    """
    global _prewarmed, _prewarm_was_cold
    with _prewarm_lock:
        if _prewarmed and not force:
            return None
        n_slots = max(1, int(DECISION_PREWARM_SLOTS or 1))
        results: list[SystemOneResult] = []
        if n_slots == 1:
            results.append(_one_prewarm_post())
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_slots) as pool:
                futs = [pool.submit(_one_prewarm_post) for _ in range(n_slots)]
                for fut in as_completed(futs):
                    results.append(fut.result())
        # Prefer any ok result for cold/warm heuristic; mark prewarmed if any ok.
        ok_results = [r for r in results if r.ok]
        if ok_results:
            _prewarmed = True
            # Cold if *any* slot looked cold (slow first touch).
            _prewarm_was_cold = any(r.latency_ms >= 5000.0 for r in ok_results)
            return ok_results[0]
        # All failed — do not fake warm.
        worst = max(results, key=lambda r: r.latency_ms) if results else None
        if worst is not None:
            _prewarm_was_cold = worst.latency_ms >= 5000.0
        return worst


def probe_next_tool(
    chat_history: list[dict],
    active_tools: list[dict],
    *,
    entry_id: str | None = None,
    turn: int | None = None,
    step: int | None = None,
    log: bool = True,
    name_to_class: dict[str, str] | None = None,
    user_ask: str | None = None,
) -> ProbeResult:
    """Call Decision + policy. Never raises — fail-open to bypass."""
    global _prewarm_was_cold
    names = active_tool_names(active_tools)
    descs = desc_map_from_tools(active_tools)
    ask = user_ask if user_ask is not None else latest_user_ask(chat_history)
    clamp_strategy = (DECISION_CLAMP_STRATEGY or "relevant").strip().lower()
    if clamp_strategy not in ("lex", "relevant"):
        clamp_strategy = "relevant"
    kept, dropped = clamp_tools(
        names,
        strategy=clamp_strategy,
        max_active=MAX_ACTIVE_TOOLS,
        user_ask=ask,
        name_to_class=name_to_class,
        desc_by_name=descs,
    )
    desc_mode = (DECISION_DESC_MODE or "short").strip().lower()
    meta: dict[str, Any] = {}
    try:
        state = chat_history_to_state(
            chat_history, active_tools, kept_names=kept
        )
        questions = build_questions(
            kept_names=kept,
            include_noul=True,
            active_tools=active_tools,
            desc_by_name=descs,
            desc_mode=desc_mode,
            desc_max_chars=DECISION_DESC_MAX_CHARS,
            state_tokens_est=estimate_tokens(state),
        )
        meta = dict(questions.pop("_meta", {}) or {})
        criteria = list(questions["next_tool"]["criteria"].keys())
        cold_or_warm = "cold" if (not _prewarmed) else "warm"
        client, attempts_used = _post_systemone_with_retries(
            state=state,
            questions=questions,
            model=DECISION_MODEL,
            endpoint=DECISION_ENDPOINT,
            api_key=DECISION_API_KEY,
            timeout_s=DECISION_TIMEOUT_S,
            criteria_names=criteria,
            keep_alive=DECISION_KEEP_ALIVE or None,
        )
        # Refine timeout bypass reason for policy mapping
        if not client.ok and _is_timeout_client_error(client.error):
            tag = "timeout_cold" if cold_or_warm == "cold" else "timeout_warm"
            client = SystemOneResult(
                ok=False,
                choice=None,
                probabilities={},
                confidence=None,
                noul=None,
                latency_ms=client.latency_ms,
                raw=None,
                error=tag,
                http_status=client.http_status,
                input_tokens=None,
            )
        policy = decide(
            client,
            active_tool_names=kept,
            chat_history=chat_history,
            choice_min=DECISION_CHOICE_MIN,
            noop_min=DECISION_NOOP_MIN,
            is_exec_observation_fn=stop_on_repeat.is_exec_observation,
            argmax_respecting=bool(DECISION_NOOP_ARGMAX),
        )
        # Map timeout_* through decide's generic path
        if not client.ok and client.error in ("timeout_cold", "timeout_warm"):
            policy.bypass_reason = client.error
    except Exception as exc:  # noqa: BLE001 — absolute fail-open
        client = SystemOneResult(
            ok=False,
            choice=None,
            probabilities={},
            confidence=None,
            noul=None,
            latency_ms=0.0,
            raw=None,
            error=f"client_exception:{type(exc).__name__}:{exc}",
            http_status=None,
            input_tokens=None,
        )
        policy = PolicyDecision(
            action=ACTION_BYPASS,
            bypass_reason="client_exception",
            is_exec_observation=is_noop_allowed(chat_history),
        )
        cold_or_warm = "warm"
        state = ""
        criteria = []
        attempts_used = 0

    record = {
        "entry_id": entry_id,
        "turn": turn,
        "step": step,
        "model_decision": DECISION_MODEL,
        "choice": policy.choice,
        "choice_probs": dict(client.probabilities or {}),
        "confidence": policy.confidence,
        "noul": policy.noul,
        "p_noop": policy.p_noop,
        "policy_action": policy.action,
        "decision_latency_ms": client.latency_ms,
        "bypass_reason": policy.bypass_reason,
        "client_error": client.error,
        "http_status": client.http_status,
        "is_exec_observation": policy.is_exec_observation,
        "input_tokens": client.input_tokens,
        "state_chars": len(state) if isinstance(state, str) else None,
        "state_tokens_est": (
            estimate_tokens(state) if isinstance(state, str) and state else None
        ),
        "n_criteria": len(criteria) if criteria else None,
        "tools_dropped": list(dropped) if dropped else [],
        "active_tools_kept": list(kept),
        "cold_or_warm": cold_or_warm,
        "desc_mode": meta.get("desc_mode"),  # achieved
        "desc_mode_requested": meta.get("desc_mode_requested", desc_mode),
        "desc_chars_total": meta.get("desc_chars_total"),
        "desc_max_chars": DECISION_DESC_MAX_CHARS,
        "clamp_strategy": clamp_strategy,
        "noop_argmax": bool(DECISION_NOOP_ARGMAX),
        "noop_soft_mode": noop_consumer_mode(),
        "noop_min": DECISION_NOOP_MIN,
        "choice_min": DECISION_CHOICE_MIN,
        "decision_timeout_s": DECISION_TIMEOUT_S,
        "decision_max_inflight": DECISION_MAX_INFLIGHT,
        "decision_attempts": attempts_used,
        "flags": (
            [f"decision_timeout_retry:{attempts_used}"]
            if attempts_used > 1
            else []
        ),
    }
    if log:
        try:
            append_jsonl(record)
        except OSError:
            pass
    return ProbeResult(
        policy=policy,
        client=client,
        tool_names=kept,
        cold_or_warm=cold_or_warm,
        log_record=record,
    )


def name_hint_user_message(tool_name: str, *, strong: bool = False) -> dict:
    """Non-persisted user-role hint (append on a working copy only)."""
    if strong:
        text = (
            f'Decision sidecar (retry): call ONLY tool "{tool_name}" this step. '
            f'Every function_name must be "{tool_name}" '
            f"(or return [] if impossible)."
        )
    else:
        text = (
            f'Decision sidecar: call only tool "{tool_name}" this step '
            f"(or [] if impossible)."
        )
    return {"role": "user", "content": text}


__all__ = [
    "ACTION_BYPASS",
    "ACTION_EXECUTE",
    "ACTION_NOOP",
    "DECISION_CLAMP_STRATEGY",
    "DECISION_DESC_MAX_CHARS",
    "DECISION_DESC_MODE",
    "DECISION_ENDPOINT",
    "DECISION_IDENTICAL_VETO",
    "DECISION_MAX_INFLIGHT",
    "DECISION_MODEL",
    "DECISION_NOOP_ARGMAX",
    "DECISION_NOOP_HARD_BREAK",
    "DECISION_NOOP_MIN",
    "DECISION_NOOP_SOFT",
    "DECISION_PREWARM_SLOTS",
    "DECISION_TIMEOUT_RETRIES",
    "DECISION_TIMEOUT_S",
    "MAX_ACTIVE_TOOLS",
    "MAX_CHOICE_CRITERIA",
    "ProbeResult",
    "hygiene_knobs",
    "is_noop_allowed",
    "name_hint_user_message",
    "noop_consumer_mode",
    "prewarm",
    "probe_next_tool",
    "reset_inflight_gate",
]
