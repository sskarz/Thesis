"""BFCL GorillaFileSystem eval: shell dialect → execute_multi_turn_func_call.

Does not change NFC/IFFC defaults. Opt-in CLI only
(``experiments/vfs-shell/run_bfcl_fs.py``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .bfcl_fs import MappingError, func_string_to_shell, to_bfcl_func_string
from .dialect import GFS_SYSTEM_PROMPT, gfs_repair_message
from .parser import parse_command
from .types import ParseError, ToolCall

GenerateFn = Callable[[list[dict[str, str]]], str]
ExecuteFn = Callable[[list[str], dict, str, str], list[str]]

_THINK_RE = re.compile(
    r"<(think|reasoning|thought)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_DENIED_REPAIR = frozenset({"python", "python3", "bash", "sh", "curl", "wget", "sudo"})


def flatten_assistant_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned


def interpret_shell_step(text: str) -> tuple[str, list[str] | None, str | None]:
    """Return ``(action, func_strings, error)``.

    action: execute | end_turn | repair
    """
    parsed = parse_command(flatten_assistant_text(text))
    if isinstance(parsed, ToolCall):
        if parsed.name == "done":
            return "end_turn", None, None
        try:
            return "execute", [to_bfcl_func_string(parsed)], None
        except MappingError as exc:
            return "repair", None, str(exc)

    assert isinstance(parsed, ParseError)
    if parsed.kind == "unsafe":
        return "repair", None, parsed.message
    if parsed.kind == "deny":
        verb = (parsed.raw or "").strip().split(None, 1)
        token = verb[0] if verb else ""
        if token.lower() in _DENIED_REPAIR or token.lower().rstrip(":") in _DENIED_REPAIR:
            return "repair", None, parsed.message
        return "end_turn", None, None
    # parse: empty / no command / tokenize error → end the turn (NFC-style text)
    if parsed.kind == "parse" and "no command found" in parsed.message:
        return "end_turn", None, None
    return "repair", None, parsed.message


def flatten_recorded_emissions(emissions: list[list[str]]) -> list[str]:
    flat: list[str] = []
    for cmds in emissions:
        flat.extend(cmds)
        if not cmds or cmds[-1] not in {"done", "finish", "submit"}:
            flat.append("done")
    return flat


def generate_shell_for_entry(
    entry: dict,
    category: str,
    generate_fn: GenerateFn,
    execute_fn: ExecuteFn,
    *,
    max_steps: int = 20,
    max_repairs: int = 3,
    exec_model_name: str = "shell_fs_gen",
) -> tuple[list[list[list[str]]], dict[str, Any]]:
    """Multi-turn loop: command text → GFS func string → BFCL execute."""
    entry_id = entry["id"]
    questions = entry["question"]
    chat: list[dict[str, str]] = [
        {"role": "system", "content": GFS_SYSTEM_PROMPT},
    ]
    extra = involved_note(entry)
    if extra:
        chat[0]["content"] += extra

    turn_results: list[list[list[str]]] = []
    turn_shells: list[list[list[str]]] = []
    flags: list[str] = []

    for turn_idx, turn_messages in enumerate(questions):
        if not turn_messages:
            turn_results.append([])
            turn_shells.append([])
            continue
        for msg in turn_messages:
            chat.append({"role": msg["role"], "content": msg.get("content") or ""})

        step_results: list[list[str]] = []
        shell_steps: list[list[str]] = []
        repairs = 0
        for _step in range(max_steps):
            text = generate_fn(chat)
            action, func_strings, err = interpret_shell_step(text)
            chat.append({"role": "assistant", "content": text})
            if action == "end_turn":
                break
            if action == "repair":
                repairs += 1
                flags.append(f"repair:turn_{turn_idx}")
                chat.append({"role": "user", "content": gfs_repair_message(err or "invalid")})
                if repairs >= max_repairs:
                    flags.append(f"repair_limit:turn_{turn_idx}")
                    break
                continue
            assert func_strings is not None
            step_results.append(func_strings)
            shell_steps.append([_raw_shell_line(text)])
            try:
                exec_results = execute_fn(
                    func_strings, entry, category, exec_model_name
                )
            except Exception as exc:  # noqa: BLE001 — surface backend errors as observations
                exec_results = [str(exc)] * len(func_strings)
            obs = format_gfs_observation(func_strings, exec_results)
            chat.append({"role": "user", "content": obs})
        else:
            flags.append(f"hit_step_limit:turn_{turn_idx}")
        turn_results.append(step_results)
        turn_shells.append(shell_steps)

    timing = {
        "id": entry_id,
        "category": category,
        "turns": len(questions),
        "flags": json.dumps(flags) if flags else "",
        "shell_emissions": turn_shells,
    }
    return turn_results, timing


def involved_note(entry: dict) -> str:
    classes = [c for c in (entry.get("involved_classes") or []) if c != "GorillaFileSystem"]
    if not classes:
        return ""
    return (
        "\nThis test also involves non-file-system APIs: "
        + ", ".join(classes)
        + ". This ablation only exposes file-system commands; emit done if the "
        "user request cannot be served with those commands.\n"
    )


def _raw_shell_line(text: str) -> str:
    """Extract the allowlisted command line the assistant emitted."""
    parsed = parse_command(flatten_assistant_text(text))
    if isinstance(parsed, ToolCall) and parsed.raw:
        return parsed.raw
    return flatten_assistant_text(text)


def format_gfs_observation(func_strings: list[str], exec_results: list[Any]) -> str:
    lines = ["Command result:"]
    for call, result in zip(func_strings, exec_results):
        shown = func_string_to_shell(call)
        lines.append(f"- {shown} -> {result}")
    lines.append("")
    lines.append("Emit the next command, or done if this user request is finished.")
    return "\n".join(lines)


def bfcl_execute(
    func_strings: list[str],
    entry: dict,
    category: str,
    exec_model_name: str,
) -> list[str]:
    """Call bfcl_eval's executor (same path NFC/IFFC use)."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    results, _ = execute_multi_turn_func_call(
        func_call_list=func_strings,
        initial_config=entry.get("initial_config", {}),
        involved_classes=entry.get("involved_classes", []),
        model_name=exec_model_name,
        test_entry_id=entry["id"],
        long_context=(category == "multi_turn_long_context"),
        is_evaL_run=False,
    )
    return [str(r) for r in results]


def score_shell_results(
    *,
    category: str,
    test_entries: dict[str, dict],
    answers: dict[str, dict],
    results: list[dict],
    model_name: str,
) -> dict[str, Any]:
    """Score mapped traces with BFCL ``multi_turn_checker``."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )

    correct = 0
    total = 0
    errors: list[dict] = []
    for row in results:
        total += 1
        entry_id = row["id"]
        test_entry = test_entries.get(entry_id)
        answer_entry = answers.get(entry_id)
        if not test_entry or not answer_entry:
            errors.append({"id": entry_id, "error": "Missing test or answer data"})
            continue
        try:
            checked = multi_turn_checker(
                multi_turn_model_result_list_decoded=row["result"],
                multi_turn_ground_truth_list=answer_entry["ground_truth"],
                test_entry=test_entry,
                test_category=category,
                model_name=model_name,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": entry_id, "error": str(exc)})
            continue
        if checked.get("valid"):
            correct += 1
        else:
            errors.append({
                "id": entry_id,
                "error": checked.get("error_message", checked.get("error", [])),
            })
    accuracy = correct / total if total else 0.0
    return {
        "category": category,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "errors": errors,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_map(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in load_jsonl(path)}


def llama_generate_fn(server_url: str, model: str, timeout_s: float = 180.0):
    """OpenAI-compatible completion; never sends ``tools``."""
    import urllib.error
    import urllib.request

    raw = server_url.rstrip("/")
    base = raw if raw.endswith("/v1") else raw + "/v1"

    def generate(messages: list[dict[str, str]]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"llama-server not reachable at {base} ({exc})"
            ) from exc
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text") or "")
            content = "".join(parts)
        if not content:
            content = message.get("reasoning_content") or ""
        return flatten_assistant_text(content if isinstance(content, str) else "")

    return generate
