"""Token-budget unrolled SFT chats for mlx-lm prompt-masked training.

mlx-lm's ``ChatDataset`` computes the prompt offset before its batch iterator
right-truncates tokens.  This module mirrors that accounting and makes sure at
least one token from the final assistant target survives the configured limit.
"""

from __future__ import annotations

import copy
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models/gemma-4-e2b-it-mlx"
FUNCTIONS_START = "Here are the available functions:\n"
FUNCTIONS_END = "\nFor every user query"
POLICY_MARKER = "\n\nDomain policy:\n"
_TRUNCATED = "…[truncated]"


class BudgetGateError(RuntimeError):
    """Raised when one or more rows would have an empty masked-loss target."""

    def __init__(self, failures: list[dict[str, Any]], max_seq_length: int):
        self.failures = failures
        self.max_seq_length = max_seq_length
        details = "\n".join(
            "  - {id}: source={source} full_len={full_len} offset={offset} "
            "target_tokens={target_tokens} actions={actions}".format(**failure)
            for failure in failures
        )
        super().__init__(
            f"token-budget hard gate failed for {len(failures)} row(s) at "
            f"max_seq_length={max_seq_length}:\n{details}"
        )


def _cap_text(text: str, max_chars: int) -> str:
    if max_chars < 0:
        raise ValueError("policy_max_chars must be non-negative")
    if len(text) <= max_chars:
        return text
    if max_chars == 0:
        return ""
    if max_chars <= len(_TRUNCATED):
        return _TRUNCATED[:max_chars]
    return text[: max_chars - len(_TRUNCATED)].rstrip() + _TRUNCATED


def compact_system_content(text: str, policy_max_chars: int = 512) -> str:
    """Compact an already-built IFFC system prompt without changing schemas.

    Only JSON whitespace is removed from the available-functions payload.
    Function names, parameter names, types, required lists, enum values, and
    all other schema data remain unchanged.  An existing domain-policy suffix
    retains its marker and is capped to ``policy_max_chars`` characters.
    """
    compacted = str(text)

    before, start, rest = compacted.partition(FUNCTIONS_START)
    if start:
        schema_text, end, after = rest.partition(FUNCTIONS_END)
        if end:
            try:
                schemas = json.loads(schema_text)
            except (TypeError, ValueError):
                pass
            else:
                schema_text = json.dumps(
                    schemas, ensure_ascii=False, separators=(",", ":")
                )
                compacted = before + start + schema_text + end + after

    prompt, marker, policy = compacted.partition(POLICY_MARKER)
    if marker:
        compacted = prompt + marker + _cap_text(policy.strip(), policy_max_chars)
    return compacted


def _token_lengths(messages: list[dict[str, Any]], tokenizer: Any) -> dict[str, int]:
    if not messages:
        return {"full_len": 0, "offset": 0, "target_tokens": 0}
    full = tokenizer.apply_chat_template(messages, return_dict=False)
    offset_tokens = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=messages[-1].get("role") == "assistant",
        return_dict=False,
    )
    full_len = len(full)
    offset = len(offset_tokens)
    return {
        "full_len": full_len,
        "offset": offset,
        "target_tokens": max(0, full_len - offset),
    }


def _is_observation(message: dict[str, Any]) -> bool:
    role = str(message.get("role", "")).lower()
    if role in {"tool", "observation", "function"}:
        return True
    content = str(message.get("content", "")).lstrip().lower()
    return role == "user" and content.startswith("function execution results:")


def _oldest_history_boundary(messages: list[dict[str, Any]]) -> int | None:
    """Return an exclusive index for the oldest complete history group."""
    # Index 0 is the protected system message; the last item is the protected
    # assistant target.  Attach execution observations to the assistant call
    # that produced them so trimming never leaves an orphaned observation.
    for index in range(1, len(messages) - 1):
        if messages[index].get("role") != "assistant":
            continue
        boundary = index + 1
        while boundary < len(messages) - 1 and _is_observation(messages[boundary]):
            boundary += 1
        return boundary
    return None


def _gate_failed(lengths: dict[str, int], max_seq_length: int) -> bool:
    return (
        lengths["offset"] >= max_seq_length
        or lengths["target_tokens"] <= 0
        or lengths["full_len"] <= lengths["offset"]
    )


def budget_messages(
    messages: list[dict[str, Any]], tokenizer: Any, max_seq_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compact and message-trim one unrolled chat to the MLX token budget."""
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    if len(messages) < 2:
        raise ValueError("budgeted chat must contain at least two messages")
    if messages[0].get("role") != "system":
        raise ValueError("budgeted chat must begin with a system message")
    if messages[-1].get("role") != "assistant":
        raise ValueError("budgeted chat must end with an assistant target")

    result = copy.deepcopy(messages)
    before = _token_lengths(result, tokenizer)
    actions: list[str] = []
    turns_dropped = 0

    original_system = str(result[0].get("content", ""))
    compacted_system = compact_system_content(original_system)
    if compacted_system != original_system:
        result[0]["content"] = compacted_system
        actions.append("compact_system")

    current = _token_lengths(result, tokenizer)
    while current["full_len"] > max_seq_length and len(result) > 2:
        boundary = _oldest_history_boundary(result)
        if boundary is None:
            # The only remaining middle messages form the current prompt.
            # The hard-gate contract permits the irreducible system+target
            # form, so use it as the final history-trim fallback.
            result = [result[0], result[-1]]
        else:
            result = [result[0], *result[boundary:]]
        turns_dropped += 1
        current = _token_lengths(result, tokenizer)

    # A 512-character policy is normally enough.  If the protected system
    # itself still consumes the target boundary, tighten only that optional
    # suffix before declaring the row irreducible.
    if _gate_failed(current, max_seq_length):
        for policy_limit in (256, 128, 0):
            tighter = compact_system_content(original_system, policy_limit)
            if tighter == result[0].get("content"):
                continue
            result[0]["content"] = tighter
            current = _token_lengths(result, tokenizer)
            actions.append(f"policy_cap_{policy_limit}")
            if not _gate_failed(current, max_seq_length):
                break

    if turns_dropped:
        actions.append("trim_history")
    after = _token_lengths(result, tokenizer)
    return result, {
        "before": before,
        "after": after,
        "actions": actions,
        "turns_dropped": turns_dropped,
        "gate_failed": _gate_failed(after, max_seq_length),
    }


def _distribution(values: Iterable[int]) -> dict[str, int | float]:
    data = list(values)
    if not data:
        return {"min": 0, "median": 0, "max": 0}
    return {
        "min": min(data),
        "median": statistics.median(data),
        "max": max(data),
    }


def _summary(records: list[dict[str, int]], max_seq_length: int) -> dict[str, Any]:
    return {
        "full_len": _distribution(record["full_len"] for record in records),
        "offset": _distribution(record["offset"] for record in records),
        "target_tokens": _distribution(
            record["target_tokens"] for record in records
        ),
        "empty_mask_count": sum(
            _gate_failed(record, max_seq_length) for record in records
        ),
    }


def load_tokenizer(model_dir: str | Path = DEFAULT_MODEL_DIR) -> Any:
    """Load the local tokenizer without consulting the network."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )


def budget_unrolled_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_seq_length: int = 8192,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    hard_gate: bool = True,
    tokenizer: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Budget every unrolled row and optionally abort on any unsafe target."""
    tok = tokenizer if tokenizer is not None else load_tokenizer(model_dir)
    output: list[dict[str, Any]] = []
    before_records: list[dict[str, int]] = []
    after_records: list[dict[str, int]] = []
    action_counts: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        budgeted_messages, stats = budget_messages(
            row.get("messages") or [], tok, max_seq_length
        )
        budgeted = copy.deepcopy(row)
        budgeted["messages"] = budgeted_messages
        output.append(budgeted)
        before_records.append(stats["before"])
        after_records.append(stats["after"])

        source = str(row.get("source") or "unknown")
        per_source[source]["rows"] += 1
        if _gate_failed(stats["before"], max_seq_length):
            per_source[source]["empty_mask_before"] += 1
        if stats["gate_failed"]:
            per_source[source]["empty_mask_after"] += 1
        if stats["turns_dropped"]:
            per_source[source]["rows_trimmed"] += 1
            per_source[source]["turns_dropped"] += stats["turns_dropped"]
        for action in stats["actions"]:
            action_counts[action] += 1
            if action == "compact_system":
                per_source[source]["rows_compacted"] += 1

        if stats["gate_failed"]:
            failures.append(
                {
                    "id": str(row.get("id") or f"row-{index}"),
                    "source": source,
                    **stats["after"],
                    "actions": ",".join(stats["actions"]) or "none",
                }
            )

    before_summary = _summary(before_records, max_seq_length)
    after_summary = _summary(after_records, max_seq_length)
    report = {
        "rows": len(output),
        "max_seq_length": max_seq_length,
        "before": before_summary,
        "after": after_summary,
        "empty_mask_before": before_summary["empty_mask_count"],
        "empty_mask_after": after_summary["empty_mask_count"],
        "actions": dict(action_counts),
        "rows_compacted": action_counts["compact_system"],
        "rows_trimmed": action_counts["trim_history"],
        "turns_dropped": sum(
            source_counts["turns_dropped"] for source_counts in per_source.values()
        ),
        "per_source": {
            source: dict(counts) for source, counts in sorted(per_source.items())
        },
        "failures": failures,
    }
    if failures and hard_gate:
        raise BudgetGateError(failures, max_seq_length)
    return output, report


def format_budget_report(report: dict[str, Any], label: str = "v7") -> str:
    """Return a short prepare-log summary for a budget report."""
    return (
        f"token budget ({label}): rows={report['rows']} "
        f"compacted={report['rows_compacted']} trimmed={report['rows_trimmed']} "
        f"turns_dropped={report['turns_dropped']} empty_mask="
        f"{report['empty_mask_before']}→{report['empty_mask_after']} "
        f"max_offset={report['after']['offset']['max']} "
        f"max_seq_length={report['max_seq_length']}"
    )
