"""Salesforce xLAM JSON-in-text function-calling channel.

xLAM-1b-fc-r is trained to emit assistant *text* of the form::

    {"tool_calls": [{"name": "...", "arguments": {...}}, ...]}

or ``{"tool_calls": []}`` when no (further) call is needed. It does **not**
populate OpenAI ``message.tool_calls``. Sending ``tools`` / ``tool_choice``
makes llama-server / the chat template ask for native tools; the model then
returns prose and the NFC harness stores empty turn lists (BFCL Run 19:
0/200, all "Model response list is empty for turn 0").

``--fc-mode xlam`` therefore:

* never sends ``tools`` / ``tool_choice``
* injects a short system prompt matching xLAM's JSON-in-text contract
* parses ``content`` via the shared IFFC normalizer (which accepts
  ``tool_calls`` the same way it already accepts ``function_calls``)
* executes mapped calls; empty list ends the turn

This is a format adapter, **not** a claim of published BFCL ~78.94% parity.
Run 19 NFC on this harness scored 0/200 (format-channel collapse).
Run 20 ``--fc-mode xlam`` scored 6/200 = 3.0% (fair channel; still very
low; not a method win).
"""

from __future__ import annotations

from typing import Any

from .iffc import format_available_functions

# Task + format wording follows Salesforce's xLAM-1b-fc-r model card, adapted
# to a system message (BFCL user turns stay in chat_history). The "refuse to
# answer" sentence is omitted: with native tools it produced prose instead of
# JSON; empty ``{"tool_calls": []}`` is the stop signal this harness executes.
XLAM_SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a "
    "set of possible functions. Based on the question, you will need to make "
    "one or more function/tool calls to achieve the purpose.\n"
    "Here are the available functions:\n"
    "{available_functions}\n"
    "The output MUST strictly adhere to the following JSON format, and NO other "
    "text MUST be included. If no function call is needed, return "
    '{"tool_calls": []}. Example:\n'
    '{"tool_calls": [{"name": "func_name1", "arguments": '
    '{"argument1": "value1", "argument2": "value2"}}]}'
)

XLAM_RETRY_USER_MESSAGE = (
    "Your previous reply was not valid JSON in the required format. "
    'Reply with {"tool_calls": [{"name": "...", "arguments": {...}}]}, '
    'or {"tool_calls": []} if no function calling is needed. '
    "Do not include any other text."
)


def build_xlam_system_prompt(tools_or_bfcl_docs: list[dict] | None) -> str:
    """Fill ``XLAM_SYSTEM_PROMPT`` with serialized function schemas."""
    return XLAM_SYSTEM_PROMPT.replace(
        "{available_functions}",
        format_available_functions(tools_or_bfcl_docs),
    )


def ensure_xlam_system_message(
    chat_history: list[dict],
    tools_or_bfcl_docs: list[dict] | None,
) -> None:
    """Insert or refresh the xLAM system message when the active tool set changes."""
    content = build_xlam_system_prompt(tools_or_bfcl_docs)
    if chat_history and chat_history[0].get("role") == "system":
        if chat_history[0].get("content") != content:
            chat_history[0] = {"role": "system", "content": content}
        return
    chat_history.insert(0, {"role": "system", "content": content})


def format_xlam_execution_user_message(
    func_strings: list[str],
    exec_results: list[Any],
) -> str:
    """User-role observation after tool execution (xLAM stays JSON-in-text)."""
    lines = ["Function execution results:"]
    for call, exec_result in zip(func_strings, exec_results):
        lines.append(f"- {call} -> {exec_result}")
    if not func_strings:
        lines.append("- (no functions were executed)")
    lines.append("")
    lines.append(
        'Respond with {"tool_calls": [{"name": "...", "arguments": {...}}, ...]} '
        'or {"tool_calls": []} if no further function calling is needed.'
    )
    return "\n".join(lines)
