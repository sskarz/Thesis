"""One-pass OODA-inspired state card for ``--fc-mode iffc-ooda-card`` (also used by ``iffc-decision-ooda`` compose).

Inference-only overlay on the IFFC JSON-in-text channel. Each model decision
turn is prefixed with a compact harness-written card:

* **goal** — latest BFCL user question (not an exec dump / retry)
* **slots** — key=value facts from actual tool execution results
* **last** — last executed call plus a compact outcome
* **next** — advisory mode hint: ``call`` | ``ask`` | ``final`` | ``refuse``

Stacked on the same stop-on-repeat gate as ``iffc-stop-repeat``
(``OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS=6``). **Not** ``iffc-state`` (no
``[STATE]`` dump, no last-2-turn compression, no STATE-trust addendum).
**Not** OODA-Tool training (Guo et al., arXiv:2608.24368). **Not** a
Specialized 4-pass Observe→Orient→Decide→Act loop — one chat completion
per step, same as IFFC.

Why a dedicated fc-mode rather than a flag on ``iffc-stop-repeat``:

* Run 30 already scored ``iffc-stop-repeat`` at 65/200. A distinct
  ``fc_mode=iffc-ooda-card`` tag keeps resume / scoring identity off that
  row.
* The card is a prompt prefix, not a change to the stop heuristic.
* Default ``nfc`` is unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from .iffc import (
    IFFC_RETRY_USER_MESSAGE,
    build_iffc_system_prompt,
)
from .iffc_state import (
    coerce_exec_result,
    is_error_result,
    parse_func_string,
)
from .stop_on_repeat import is_exec_observation

CARD_OPEN = "[OODA]"
CARD_CLOSE = "[/OODA]"
NEXT_MODES = ("call", "ask", "final", "refuse")

_MAX_GOAL_CHARS = 160
_MAX_SLOTS = 8
_MAX_SLOT_CHARS = 48
_MAX_LAST_CHARS = 120
_MAX_CARD_CHARS = 500
_MAX_VALUE_CHARS = 40

_SKIP_SLOT_KEYS = frozenset(
    {
        "current_working_directory",
        "cwd",
        "working_directory",
        "pwd",
        "current_directory_content",
        "matches",
        "file_content",
        "terminal_output",
        "matching_lines",
        "error",
        "log",
    }
)
_CWD_KEYS = ("current_working_directory", "cwd", "working_directory", "pwd")
_LISTING_KEYS = ("current_directory_content", "matches")
_ARG_SLOT_KEYS = (
    "folder",
    "file_name",
    "filename",
    "path",
    "city",
    "query",
    "destination",
)

_MISSING_PARAM_RE = re.compile(
    r"(missing|required|need(?:s|ed)? (?:an? )?(?:arg|param)|invalid argument|"
    r"parameter .+ (?:required|missing))",
    re.IGNORECASE,
)
_FINAL_UTTERANCE_RE = re.compile(
    r"\b(thanks|thank you|that'?s (?:all|it|enough)|never mind|i'?m done)\b",
    re.IGNORECASE,
)

OODA_ADDENDUM = (
    f"A {CARD_OPEN}...{CARD_CLOSE} block prefixed on the latest user message "
    "is a harness-written one-pass snapshot (goal, known slots, last execution "
    "fact, next-mode hint). It is not model text and is not a second OODA pass. "
    "next is one of call, ask, final, refuse: call = emit function calls; "
    "ask / final / refuse = return []. Continue to answer only with a JSON "
    'list of dictionaries with "function_name" and "parameters", or [].'
)


# Compose (``iffc-state-ooda``) wording: the card sits after the [STATE]
# block rather than being prefixed. Same contract otherwise.
OODA_ADDENDUM_AFTER_STATE = (
    f"A {CARD_OPEN}...{CARD_CLOSE} block on the latest user message (placed "
    "after any [STATE] block) is a harness-written one-pass snapshot (goal, "
    "known slots, last execution fact, next-mode hint). It is not model text "
    "and is not a second OODA pass. next is one of call, ask, final, refuse: "
    "call = emit function calls; ask / final / refuse = return []. Continue "
    'to answer only with a JSON list of dictionaries with "function_name" '
    'and "parameters", or [].'
)


class OodaCardTracker:
    """Accumulates compact slots + last-exec fact from tool observations."""

    def __init__(self) -> None:
        self.slots: dict[str, str] = {}
        self.last_fact: str | None = None
        self.last_error: str | None = None
        self.last_name: str | None = None

    def update(self, func_strings: list[str], exec_results: list[Any]) -> None:
        for call, raw in zip(func_strings, exec_results):
            name, args = parse_func_string(call)
            parsed = coerce_exec_result(raw)
            error = is_error_result(raw)
            self.last_name = name
            self.last_fact = _compact_fact(name, args, parsed, error)
            if error:
                self.last_error = _compact_value(parsed)
                continue
            self.last_error = None
            self._apply_slots(args, parsed)

    def _apply_slots(self, args: dict[str, Any], parsed: Any) -> None:
        for key in _ARG_SLOT_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                self.slots[key] = _compact_value(val)
        if isinstance(parsed, dict):
            for key in _CWD_KEYS:
                val = parsed.get(key)
                if isinstance(val, str) and val.strip():
                    self.slots["cwd"] = _compact_value(val)
            for key in _LISTING_KEYS:
                val = parsed.get(key)
                if isinstance(val, list) and val:
                    self.slots["listing"] = _compact_value(val)
            for key, value in parsed.items():
                if not isinstance(key, str) or key.startswith("_"):
                    continue
                if key in _SKIP_SLOT_KEYS:
                    continue
                self.slots[key] = _compact_value(value)
        while len(self.slots) > _MAX_SLOTS:
            self.slots.pop(next(iter(self.slots)))


def build_ooda_system_prompt(tools_or_bfcl_docs: list[dict] | None) -> str:
    """Appendix B IFFC prompt plus a short one-pass card addendum (no STATE)."""
    return build_iffc_system_prompt(tools_or_bfcl_docs) + "\n\n" + OODA_ADDENDUM


def ensure_ooda_system_message(
    chat_history: list[dict],
    tools_or_bfcl_docs: list[dict] | None,
) -> None:
    """Insert or refresh the IFFC+OODA system message when tools change."""
    content = build_ooda_system_prompt(tools_or_bfcl_docs)
    if chat_history and chat_history[0].get("role") == "system":
        if chat_history[0].get("content") != content:
            chat_history[0] = {"role": "system", "content": content}
        return
    chat_history.insert(0, {"role": "system", "content": content})


def build_state_ooda_system_prompt(tools_or_bfcl_docs: list[dict] | None) -> str:
    """Appendix B IFFC prompt + STATE-trust addendum + card addendum.

    Used by ``iffc-state-ooda``. Both addenda are appended (STATE first,
    matching the in-message order); neither replaces the other.
    """
    from .iffc_state import STATE_ADDENDUM

    return (
        build_iffc_system_prompt(tools_or_bfcl_docs)
        + "\n\n"
        + STATE_ADDENDUM
        + "\n\n"
        + OODA_ADDENDUM_AFTER_STATE
    )


def ensure_state_ooda_system_message(
    chat_history: list[dict],
    tools_or_bfcl_docs: list[dict] | None,
) -> None:
    """Insert or refresh the IFFC+STATE+OODA system message when tools change."""
    content = build_state_ooda_system_prompt(tools_or_bfcl_docs)
    if chat_history and chat_history[0].get("role") == "system":
        if chat_history[0].get("content") != content:
            chat_history[0] = {"role": "system", "content": content}
        return
    chat_history.insert(0, {"role": "system", "content": content})


def current_goal(chat_history: list[dict]) -> str:
    """Latest BFCL user question, truncated; empty if none yet."""
    for message in reversed(chat_history):
        if message.get("role") != "user":
            continue
        text = message.get("content") or ""
        if not isinstance(text, str):
            continue
        if text == IFFC_RETRY_USER_MESSAGE:
            continue
        if is_exec_observation(text):
            continue
        stripped = _strip_existing_card(text).strip()
        if stripped:
            return _one_line(stripped, _MAX_GOAL_CHARS)
    return ""


def hint_next_mode(
    tracker: OodaCardTracker,
    chat_history: list[dict],
    *,
    has_tools: bool,
) -> str:
    """Deterministic When2Call-style hint. Advisory — does not force decode.

    Conservative on BASE so VehicleControl re-calls stay ``call``:

    * no live tools → ``refuse``
    * last exec looks like a missing/invalid argument → ``ask``
    * latest BFCL user utterance is a closing thanks/done → ``final``
    * otherwise → ``call``
    """
    if not has_tools:
        return "refuse"
    if tracker.last_error and _MISSING_PARAM_RE.search(tracker.last_error):
        return "ask"
    goal = current_goal(chat_history)
    if goal and _FINAL_UTTERANCE_RE.search(goal):
        return "final"
    return "call"


def render_card(
    tracker: OodaCardTracker,
    chat_history: list[dict],
    *,
    has_tools: bool,
) -> str:
    """Format the four-field card. Always includes goal / slots / last / next."""
    goal = current_goal(chat_history) or "(none)"
    slots = "; ".join(
        f"{k}={_compact_value(v, _MAX_SLOT_CHARS)}"
        for k, v in list(tracker.slots.items())[:_MAX_SLOTS]
    ) or "(none)"
    last = tracker.last_fact or "(none)"
    nxt = hint_next_mode(tracker, chat_history, has_tools=has_tools)
    if nxt not in NEXT_MODES:
        nxt = "call"
    lines = [
        CARD_OPEN,
        f"goal: {_one_line(goal, _MAX_GOAL_CHARS)}",
        f"slots: {_one_line(slots, _MAX_GOAL_CHARS)}",
        f"last: {_one_line(last, _MAX_LAST_CHARS)}",
        f"next: {nxt}",
        CARD_CLOSE,
    ]
    block = "\n".join(lines)
    if len(block) > _MAX_CARD_CHARS:
        block = block[: _MAX_CARD_CHARS - len(CARD_CLOSE) - 1].rstrip()
        block = block + "\n" + CARD_CLOSE
    return block


def inject_card(
    chat_history: list[dict],
    tracker: OodaCardTracker,
    *,
    has_tools: bool,
) -> list[dict]:
    """Return a **copy** of *chat_history* with the card prefixed on the last user.

    Persistent transcripts stay IFFC-clean so stop-on-repeat still sees
    unmodified exec dumps. No-op (still a shallow-copied list) when there
    is no user message to prefix.
    """
    messages = [dict(m) for m in chat_history]
    block = render_card(tracker, chat_history, has_tools=has_tools)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        original = messages[i].get("content") or ""
        if not isinstance(original, str):
            original = str(original)
        body = _strip_existing_card(original).lstrip()
        messages[i] = {
            **messages[i],
            "content": f"{block}\n\n{body}" if body else block,
        }
        break
    return messages


def inject_card_after_state(
    chat_history: list[dict],
    tracker: OodaCardTracker,
    *,
    has_tools: bool,
) -> list[dict]:
    """Like :func:`inject_card`, but STATE comes first, then the card.

    For ``iffc-state-ooda``: when the latest user message carries a
    ``[STATE]...[/STATE]`` block, the card is inserted right after
    ``[/STATE]`` (before the IFFC "Respond with..." footer). Without a STATE
    block (e.g. the BFCL question at step 0) the card is prefixed, as in
    plain ``iffc-ooda-card``. Returns a copy; the transcript is untouched.
    """
    from .iffc_state import STATE_CLOSE

    messages = [dict(m) for m in chat_history]
    block = render_card(tracker, chat_history, has_tools=has_tools)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        original = messages[i].get("content") or ""
        if not isinstance(original, str):
            original = str(original)
        body = _strip_existing_card(original).strip()
        idx = body.rfind(STATE_CLOSE)
        if idx >= 0:
            cut = idx + len(STATE_CLOSE)
            head, tail = body[:cut], body[cut:].lstrip()
            content = f"{head}\n\n{block}" + (f"\n\n{tail}" if tail else "")
        else:
            content = f"{block}\n\n{body}" if body else block
        messages[i] = {**messages[i], "content": content}
        break
    return messages


def record_execution(
    tracker: OodaCardTracker,
    func_strings: list[str],
    exec_results: list[Any],
) -> None:
    """Update the tracker from a just-executed IFFC step."""
    tracker.update(func_strings, exec_results)


def payload_omits_tools(payload: dict) -> bool:
    return "tools" not in payload and "tool_choice" not in payload


def payload_has_ooda_card(payload: dict) -> bool:
    for message in payload.get("messages") or []:
        text = message.get("content") or ""
        if isinstance(text, str) and CARD_OPEN in text and CARD_CLOSE in text:
            return True
    return False


def payload_has_state_block(payload: dict) -> bool:
    for message in payload.get("messages") or []:
        text = message.get("content") or ""
        if isinstance(text, str) and "[STATE]" in text and "[/STATE]" in text:
            return True
    return False


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_CARD_RE = re.compile(
    re.escape(CARD_OPEN) + r".*?" + re.escape(CARD_CLOSE),
    re.DOTALL,
)


def _strip_existing_card(text: str) -> str:
    return _CARD_RE.sub("", text).strip()


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) > limit:
        return compact[: limit - 3] + "..."
    return compact


def _compact_value(value: Any, limit: int = _MAX_VALUE_CHARS) -> str:
    if value is None:
        return "ok"
    if isinstance(value, dict):
        if "error" in value:
            value = value["error"]
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            value = ",".join(
                f"{k}={_compact_value(v, 16)}" for k, v in list(value.items())[:4]
            )
    elif isinstance(value, list):
        value = ",".join(_compact_value(v, 16) for v in value[:6])
    text = str(value).replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _compact_fact(
    name: str,
    args: dict[str, Any],
    parsed: Any,
    error: bool,
) -> str:
    outcome = "error" if error else _outcome(parsed)
    arg_hint = ", ".join(
        f"{k}={_compact_value(v, 24)}" for k, v in list(args.items())[:3]
    )
    piece = f"{name}({arg_hint})->{outcome}" if arg_hint else f"{name}()->{outcome}"
    return _one_line(piece, _MAX_LAST_CHARS)


def _outcome(parsed: Any) -> str:
    if parsed is None:
        return "ok"
    if isinstance(parsed, dict):
        if parsed.get("error") not in (None, "", False):
            return f"error:{_compact_value(parsed.get('error'), 32)}"
        keys = [k for k in parsed if not str(k).startswith("_")][:3]
        if keys:
            return ",".join(f"{k}={_compact_value(parsed[k], 16)}" for k in keys)
        return "ok"
    return _compact_value(parsed, 32)
