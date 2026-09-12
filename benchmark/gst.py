"""Guided-Structured Templates (GST) — He et al., arXiv:2509.18076.

GST is a **prompt-only** overlay on the NFC channel. Requests still send
OpenAI ``tools`` / ``tool_choice=auto`` (same scoring plumbing as Run 4).
The model is instructed to emit a short ``[GST]...[/GST]`` reasoning scaffold
(paper Detail template, steps 1–7) in assistant *content* before native
``tool_calls``. Executable actions remain ``tool_calls``; GST text is ignored
by the BFCL checker.

This is **not** GBNF, XGrammar, TagDispatch, or IFFC JSON-in-text. No
llama.cpp rebuild. Paper was single-turn; this harness adapts the curriculum
to multi-turn (prior tool observations, ABSTAIN, terse bullets).
"""

from __future__ import annotations

import re

# Paper Appendix A.2 "Detail" template (primary design for Template-Prompting),
# with step 7 extended for multi-turn ABSTAIN. Keep wording aligned with
# He et al. Table 5 so tests can assert the curriculum is in the payload.
GST_STEPS: tuple[str, ...] = (
    "Identify which function(s) best fit the user query from descriptions",
    "Pick that function/set to fulfill the request",
    "Examine selected function documentation",
    "Analyze parameters (types, required vs optional)",
    "Extract values from query (type conversions if needed)",
    "Draft the function call(s)",
    "Revalidate against docs + user request (or abstain / note missing params)",
)

GST_OPEN = "[GST]"
GST_CLOSE = "[/GST]"

# Prompt-side cap on the *model's* GST block (not the system curriculum).
# One short bullet per step plus a possible ABSTAIN line.
GST_MAX_GUIDANCE_LINES = 8

_CURRICULUM = "\n".join(
    f"{i}. {step}" for i, step in enumerate(GST_STEPS, start=1)
)

GST_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools.\n"
    "Before every native tool call (or decision not to call), write a short "
    "Guided-Structured Template block, then emit tool calls via the tools API.\n"
    "\n"
    "Output format:\n"
    f"1. A terse {GST_OPEN}...{GST_CLOSE} block with numbered steps 1-7 "
    f"(one short bullet each, max {GST_MAX_GUIDANCE_LINES} lines). "
    "Do not paste function schemas or long quotes from the docs.\n"
    "2. Then invoke tools via native tool_calls. Do not treat the GST draft "
    "as the executable call — only tool_calls are executed.\n"
    "3. If you should not call a tool this step, set step 7 to ABSTAIN "
    "(note missing params if that is why) and emit no tool_calls.\n"
    "\n"
    "Curriculum:\n"
    f"{_CURRICULUM}\n"
    "\n"
    "Multi-turn: reason over the full transcript, including prior tool "
    "observations and environment feedback, not only the latest user utterance.\n"
    "Keep GST bullets short to limit context growth.\n"
    "When calling tools, pass JSON types that match the schema "
    "(numbers as numbers, arrays as arrays, objects as objects)."
)

_GST_BLOCK_RE = re.compile(
    r"\[GST\](.*?)\[/GST\]",
    re.DOTALL | re.IGNORECASE,
)
_STEP_RE = re.compile(
    r"^\s*(?:\*\*)?(?:step\s*)?(\d{1,2})(?:\*\*)?[.)\]:]+\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def build_gst_system_prompt() -> str:
    """Return the GST system prompt (curriculum; tool schemas stay on NFC)."""
    return GST_SYSTEM_PROMPT


def ensure_gst_system_message(chat_history: list[dict]) -> None:
    """Insert the GST system message once; do not duplicate or dump schemas."""
    content = build_gst_system_prompt()
    if chat_history and chat_history[0].get("role") == "system":
        existing = chat_history[0].get("content") or ""
        if existing == content:
            return
        if GST_OPEN in existing and "Curriculum:" in existing:
            chat_history[0] = {"role": "system", "content": content}
            return
        chat_history.insert(0, {"role": "system", "content": content})
        return
    chat_history.insert(0, {"role": "system", "content": content})


def extract_gst_block(content: str | None) -> str | None:
    """Return the ``[GST]`` body, or unlabeled steps 1–7 if tags are missing."""
    if not content:
        return None
    match = _GST_BLOCK_RE.search(content)
    if match:
        return match.group(1).strip() or None
    steps = parse_gst_steps(content)
    if set(range(1, 8)).issubset(steps):
        return content.strip()
    return None


def parse_gst_steps(content: str | None) -> dict[int, str]:
    """Parse numbered curriculum lines (1–7) from GST or plain assistant text."""
    if not content:
        return {}
    block = _GST_BLOCK_RE.search(content)
    text = block.group(1) if block else content
    steps: dict[int, str] = {}
    for match in _STEP_RE.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= 7:
            steps[n] = match.group(2).strip()
    return steps


def has_gst_scaffold(content: str | None) -> bool:
    """True if assistant text has a GST wrapper or all seven labeled steps."""
    if not content:
        return False
    if _GST_BLOCK_RE.search(content):
        return True
    return set(range(1, 8)).issubset(parse_gst_steps(content))


def strip_gst_block(content: str | None) -> str:
    """Remove ``[GST]...[/GST]`` so checkers/parsers ignore the scaffold."""
    if not content:
        return ""
    return _GST_BLOCK_RE.sub("", content).strip()


def gst_payload_has_nfc_tools(payload: dict) -> bool:
    """True when the chat payload still uses the NFC tools channel."""
    tools = payload.get("tools")
    return bool(tools) and payload.get("tool_choice") == "auto"


def gst_payload_has_curriculum(payload: dict) -> bool:
    """True when some message carries the GST curriculum / wrapper tags."""
    for message in payload.get("messages") or []:
        text = message.get("content") or ""
        if not isinstance(text, str):
            continue
        if GST_OPEN in text and all(step in text for step in GST_STEPS):
            return True
    return False
