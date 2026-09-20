"""Drop non-terminal IFFC ``[]`` targets after tool-execution observations.

SFT v1 unrolled every assistant turn, including ToolACE replies that the
formatter maps to ``[]`` immediately after a ``format_iffc_execution_user_message``
observation. On the Mac mix that was ~5,520 empty ``[]`` targets (all ToolACE;
xLAM had 0). After an execution-result user turn, 78% of assistant targets
were ``[]`` (2121 empty vs 596 continue). That taught the adapter to stop as
soon as tools returned — BFCL Run 12 hit 70% empty turns vs ~13% on base IFFC
and collapsed ``multi_turn_base`` to 0.5%.

Policy (drop, not downweight):

* Drop an unrolled row iff the assistant target is exactly ``[]``, the previous
  user message looks like an IFFC tool-result observation, **and** the parent
  conversation still has a later assistant turn with a non-empty tool call.
* Keep true terminal ``[]`` (no later tool-calling assistant turn).
* Keep every non-empty tool-call target.
* Keep user → ``[]`` clarifying / irrelevance turns (previous user is not an
  execution observation).
* xLAM is single-turn and has no execution observations; those rows are
  unchanged unless they somehow gain this pattern.

mlx-lm chat JSONL has no sample-weight field, so v2 **drops** the bad rows
instead of downweighting.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from benchmark.iffc import parse_iffc_response
from sft.format_iffc import unroll_conversation
from sft.io import read_jsonl, write_jsonl

KIND_NONEMPTY = "nonempty"
KIND_EMPTY = "empty"
KIND_EMPTY_AFTER_EXEC_TERMINAL = "empty_after_exec_terminal"
KIND_EMPTY_AFTER_EXEC_NONTERMINAL = "empty_after_exec_nonterminal"

EMPTY_IFFC_TARGET = "[]"

# Markers from ``benchmark.iffc.format_iffc_execution_user_message``.
_EXEC_HEADER = "Function execution results"
_EXEC_FOOTER = "or [] if no further function calling is needed"


def is_empty_iffc_target(content: Any) -> bool:
    """True when the assistant target is exactly the IFFC empty list."""
    return str(content or "").strip() == EMPTY_IFFC_TARGET


def is_iffc_execution_observation(content: Any) -> bool:
    """True when a user message looks like an IFFC tool-result observation."""
    text = str(content or "")
    if _EXEC_HEADER in text:
        return True
    return _EXEC_FOOTER in text


def _previous_user_content(messages: list[dict], index: int) -> str | None:
    for j in range(index - 1, -1, -1):
        if messages[j].get("role") == "user":
            return str(messages[j].get("content") or "")
    return None


def has_nonempty_iffc_tool_call(content: Any) -> bool:
    """True when assistant text parses to at least one IFFC function call."""
    text = str(content or "")
    if is_empty_iffc_target(text):
        return False
    return bool(parse_iffc_response(text))


def has_later_nonempty_tool_call(messages: list[dict], assistant_index: int) -> bool:
    """True if a later assistant turn in the parent conversation calls a tool."""
    for msg in messages[assistant_index + 1 :]:
        if msg.get("role") != "assistant":
            continue
        if has_nonempty_iffc_tool_call(msg.get("content")):
            return True
    return False


def classify_assistant_turn(messages: list[dict], index: int) -> str:
    """Classify one assistant turn against the parent conversation."""
    if index < 0 or index >= len(messages):
        return KIND_NONEMPTY
    msg = messages[index]
    if msg.get("role") != "assistant":
        return KIND_NONEMPTY
    if not is_empty_iffc_target(msg.get("content")):
        return KIND_NONEMPTY
    prev = _previous_user_content(messages, index)
    if prev is None or not is_iffc_execution_observation(prev):
        return KIND_EMPTY
    if has_later_nonempty_tool_call(messages, index):
        return KIND_EMPTY_AFTER_EXEC_NONTERMINAL
    return KIND_EMPTY_AFTER_EXEC_TERMINAL


def should_drop_assistant_turn(messages: list[dict], index: int) -> bool:
    """Drop mid-trajectory ``[]`` after a tool-result observation."""
    return classify_assistant_turn(messages, index) == KIND_EMPTY_AFTER_EXEC_NONTERMINAL


def _source_of(row: dict, parent: dict | None = None) -> str:
    raw = row.get("source")
    if raw is None and parent is not None:
        raw = parent.get("source")
    return str(raw or "unknown")


@dataclass
class EmptyAfterExecFilterStats:
    """Before/after counts for the non-terminal empty-after-exec drop."""

    total_before: int = 0
    total_after: int = 0
    empty_targets_before: int = 0
    empty_targets_after: int = 0
    empty_after_exec_before: int = 0
    empty_after_exec_after: int = 0
    nonterminal_empty_after_exec_before: int = 0
    nonterminal_empty_after_exec_after: int = 0
    dropped: int = 0
    by_source_before: Counter[str] = field(default_factory=Counter)
    by_source_after: Counter[str] = field(default_factory=Counter)
    empty_targets_by_source_before: Counter[str] = field(default_factory=Counter)
    empty_targets_by_source_after: Counter[str] = field(default_factory=Counter)
    empty_after_exec_by_source_before: Counter[str] = field(default_factory=Counter)
    empty_after_exec_by_source_after: Counter[str] = field(default_factory=Counter)
    dropped_by_source: Counter[str] = field(default_factory=Counter)

    def _note(
        self,
        *,
        kind: str,
        source: str,
        after: bool,
    ) -> None:
        if after:
            self.total_after += 1
            self.by_source_after[source] += 1
        else:
            self.total_before += 1
            self.by_source_before[source] += 1
        is_empty = kind != KIND_NONEMPTY
        is_after_exec = kind in (
            KIND_EMPTY_AFTER_EXEC_TERMINAL,
            KIND_EMPTY_AFTER_EXEC_NONTERMINAL,
        )
        is_nonterminal = kind == KIND_EMPTY_AFTER_EXEC_NONTERMINAL
        if after:
            if is_empty:
                self.empty_targets_after += 1
                self.empty_targets_by_source_after[source] += 1
            if is_after_exec:
                self.empty_after_exec_after += 1
                self.empty_after_exec_by_source_after[source] += 1
            if is_nonterminal:
                self.nonterminal_empty_after_exec_after += 1
        else:
            if is_empty:
                self.empty_targets_before += 1
                self.empty_targets_by_source_before[source] += 1
            if is_after_exec:
                self.empty_after_exec_before += 1
                self.empty_after_exec_by_source_before[source] += 1
            if is_nonterminal:
                self.nonterminal_empty_after_exec_before += 1

    def note_before(self, kind: str, source: str) -> None:
        self._note(kind=kind, source=source, after=False)

    def note_after(self, kind: str, source: str) -> None:
        self._note(kind=kind, source=source, after=True)

    def note_dropped(self, source: str) -> None:
        self.dropped += 1
        self.dropped_by_source[source] += 1

    def as_dict(self) -> dict[str, Any]:
        def _plain(counter: Counter[str]) -> dict[str, int]:
            return dict(sorted(counter.items()))

        return {
            "total_before": self.total_before,
            "total_after": self.total_after,
            "empty_targets_before": self.empty_targets_before,
            "empty_targets_after": self.empty_targets_after,
            "empty_after_exec_before": self.empty_after_exec_before,
            "empty_after_exec_after": self.empty_after_exec_after,
            "nonterminal_empty_after_exec_before": self.nonterminal_empty_after_exec_before,
            "nonterminal_empty_after_exec_after": self.nonterminal_empty_after_exec_after,
            "dropped": self.dropped,
            "by_source_before": _plain(self.by_source_before),
            "by_source_after": _plain(self.by_source_after),
            "empty_targets_by_source_before": _plain(self.empty_targets_by_source_before),
            "empty_targets_by_source_after": _plain(self.empty_targets_by_source_after),
            "empty_after_exec_by_source_before": _plain(
                self.empty_after_exec_by_source_before
            ),
            "empty_after_exec_by_source_after": _plain(
                self.empty_after_exec_by_source_after
            ),
            "dropped_by_source": _plain(self.dropped_by_source),
        }

    def format(self, label: str, *, enabled: bool) -> str:
        sources = sorted(
            set(self.by_source_before) | set(self.by_source_after) | {"toolace", "xlam"}
        )
        lines = [
            f"empty-after-exec filter ({label}): drop_nonterminal={enabled}",
            f"  rows: {self.total_before} → {self.total_after} "
            f"(dropped {self.dropped})",
            f"  empty [] targets: {self.empty_targets_before} → "
            f"{self.empty_targets_after}",
            f"  empty-after-exec: {self.empty_after_exec_before} → "
            f"{self.empty_after_exec_after} "
            f"(non-terminal {self.nonterminal_empty_after_exec_before} → "
            f"{self.nonterminal_empty_after_exec_after})",
            "  by source:",
        ]
        for source in sources:
            if (
                source not in self.by_source_before
                and source not in self.by_source_after
            ):
                continue
            lines.append(
                "    "
                f"{source}: rows {self.by_source_before[source]}→"
                f"{self.by_source_after[source]}; "
                f"empty [] {self.empty_targets_by_source_before[source]}→"
                f"{self.empty_targets_by_source_after[source]}; "
                f"empty-after-exec {self.empty_after_exec_by_source_before[source]}→"
                f"{self.empty_after_exec_by_source_after[source]}"
            )
        return "\n".join(lines)


def unroll_and_filter(
    records: Iterable[dict],
    *,
    drop_nonterminal_empty_after_exec: bool = True,
) -> tuple[list[dict], EmptyAfterExecFilterStats]:
    """Unroll conversations, optionally dropping mid-trajectory empty-after-exec."""
    stats = EmptyAfterExecFilterStats()
    out: list[dict] = []
    for rec in records:
        messages = rec.get("messages") or []
        rows = unroll_conversation(rec)
        for row in rows:
            idx = len(row.get("messages") or []) - 1
            kind = classify_assistant_turn(messages, idx)
            source = _source_of(row, rec)
            stats.note_before(kind, source)
            if drop_nonterminal_empty_after_exec and should_drop_assistant_turn(
                messages, idx
            ):
                stats.note_dropped(source)
                continue
            stats.note_after(kind, source)
            out.append(row)
    return out, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unroll conversation-level IFFC JSONL and drop non-terminal "
            "assistant [] targets after tool-execution observations."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Conversation-level JSONL (e.g. sft/data/splits/train.jsonl).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Directory with train.jsonl / valid.jsonl (prints stats for both).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write unrolled+filtered JSONL (only with --input).",
    )
    parser.add_argument(
        "--keep-nonterminal-empty-after-exec",
        action="store_true",
        help="Disable the drop (SFT v1 unroll: keep every assistant turn).",
    )
    args = parser.parse_args(argv)
    drop = not args.keep_nonterminal_empty_after_exec

    if args.input is None and args.splits_dir is None:
        parser.error("pass --input FILE or --splits-dir DIR")

    def _run(path: Path, label: str, out_path: Path | None) -> None:
        if not path.exists():
            parser.error(f"missing {path}")
        rows, stats = unroll_and_filter(
            read_jsonl(path),
            drop_nonterminal_empty_after_exec=drop,
        )
        print(stats.format(label, enabled=drop))
        if out_path is not None:
            n = write_jsonl(out_path, rows)
            print(f"  wrote {n} → {out_path}")

    if args.splits_dir is not None:
        if args.out is not None:
            parser.error("--out cannot be combined with --splits-dir")
        _run(args.splits_dir / "train.jsonl", "train", None)
        valid = args.splits_dir / "valid.jsonl"
        if valid.exists():
            _run(valid, "valid", None)
        return 0

    _run(args.input, args.input.name, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
