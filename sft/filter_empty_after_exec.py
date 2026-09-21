"""SFT v4 empty-target curriculum (extends v3 recovery, does not replace it).

SFT v1 unrolled every assistant turn, including ToolACE replies that the
formatter maps to ``[]`` immediately after a ``format_iffc_execution_user_message``
observation. On the Mac mix that was ~5,520 empty ``[]`` targets (all ToolACE;
xLAM had 0). After an execution-result user turn, 78% of assistant targets
were ``[]`` (2121 empty vs 596 continue). That taught the adapter to stop as
soon as tools returned — BFCL Run 12 hit 70% empty turns vs ~13% on base IFFC
and collapsed ``multi_turn_base`` to 0.5%.

SFT v2 (PR #24) dropped only **non-terminal** empty-after-exec (334→0) and
still kept 552 terminal ``[]``. Run 13 scored the same 0.5%. The remaining
terminal stop-after-tools targets still taught call→results→``[]``, and BFCL
then issues a **new user turn** that the adapter never recovered from
(Run 14 stripping ``[]`` from live history only reached 2.0%).

SFT v3 dropped **all** empty-after-exec targets (terminal included) and kept /
synthesized ``recovery_after_empty``. That zeroed empty-after-exec targets
(~0) with ``recovery_after_empty`` ≈ 397, but leftover empties remained:
~2471 assistant ``[]`` after ordinary (non-exec) user messages (ToolACE
clarifying / chitchat), and ~1859 unrolled rows whose **first** assistant in
the prefix is ``[]`` (~5.7%). v3 smoke still never-started GFS
``multi_turn_base_1`` turn-0 (``[]`` vs base IFFC ``ls(a=True)``). Recovery
after ``[]`` already PASSes — do **not** weaken it.

SFT v4 (this module's default) keeps the v3 empty-after-exec drop and
recovery curriculum, and additionally:

* **Drop first-assistant-empty targets:** when the unrolled row's target is
  ``[]`` and that assistant is the first assistant message in the messages
  prefix, drop the row. This is the turn-0 never-start teacher.
* **Drop clarifying empty-after-user:** when the target is ``[]``, the
  immediately previous message is ``user``, and it is **not** an IFFC
  tool-result observation, drop the row. Keep recovery rows whose gold is a
  nonempty call after a prior ``[]`` in history.

mlx-lm chat JSONL has no sample-weight field, so these rows are **dropped**
not downweighted. Conversation-level ``split_data`` files are **not**
rewritten (split unit stays conversation id). Ablations:

* ``--keep-first-assistant-empty`` and ``--keep-clarifying-empty-after-user``
  together restore v3 (v3 empty-after-exec + recovery unchanged).
* ``--keep-terminal-empty-after-exec`` restores v2 (drop non-terminal only).
* ``--keep-nonterminal-empty-after-exec`` restores v1 (keep every assistant turn).
* ``--no-synthetic-recovery`` disables history-only ``[]`` insertion.

Do **not** copy BFCL ``multi_turn_base`` prompts into train; synthetic
structure is bookkeeping only.
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
from sft.split_data import assert_no_bfcl

KIND_NONEMPTY = "nonempty"
KIND_EMPTY = "empty"
KIND_EMPTY_AFTER_EXEC_TERMINAL = "empty_after_exec_terminal"
KIND_EMPTY_AFTER_EXEC_NONTERMINAL = "empty_after_exec_nonterminal"

RECIPE_V1 = "v1"
RECIPE_V2 = "v2"
RECIPE_V3 = "v3"
RECIPE_V4 = "v4"

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


def _previous_user_index(messages: list[dict], index: int) -> int | None:
    for j in range(index - 1, -1, -1):
        if messages[j].get("role") == "user":
            return j
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


def is_first_assistant_in_prefix(messages: list[dict], index: int) -> bool:
    """True when ``messages[index]`` is the first assistant turn in the prefix."""
    if index < 0 or index >= len(messages):
        return False
    if messages[index].get("role") != "assistant":
        return False
    return all(messages[j].get("role") != "assistant" for j in range(index))


def is_first_assistant_empty_target(messages: list[dict], index: int) -> bool:
    """True when the target is ``[]`` and it is the first assistant in the prefix.

    This is the turn-0 never-start teacher: the unrolled row's gold is empty
    and no earlier assistant turn exists in that messages prefix.
    """
    if not is_first_assistant_in_prefix(messages, index):
        return False
    return is_empty_iffc_target(messages[index].get("content"))


def is_clarifying_empty_after_user(messages: list[dict], index: int) -> bool:
    """True when gold is ``[]`` immediately after a non-exec user message.

    Distinct from empty-after-exec (previous user is an IFFC tool-result
    observation). ToolACE clarifying / chitchat ``[]`` after a normal user
    turn matches this; recovery rows with a nonempty call gold do not.
    """
    if index <= 0 or index >= len(messages):
        return False
    msg = messages[index]
    if msg.get("role") != "assistant":
        return False
    if not is_empty_iffc_target(msg.get("content")):
        return False
    prev = messages[index - 1]
    if prev.get("role") != "user":
        return False
    return not is_iffc_execution_observation(prev.get("content"))


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


def resolve_recipe(
    *,
    keep_nonterminal_empty_after_exec: bool = False,
    keep_terminal_empty_after_exec: bool = False,
    keep_first_assistant_empty: bool = False,
    keep_clarifying_empty_after_user: bool = False,
) -> str:
    """Map ablation flags to a recipe id.

    v4 (default): v3 plus drop first-assistant-empty and clarifying
    empty-after-user ``[]`` targets.
    v3: both ``--keep-first-assistant-empty`` and
    ``--keep-clarifying-empty-after-user`` (empty-after-exec + recovery only).
    v2: ``--keep-terminal-empty-after-exec`` (drop non-terminal only).
    v1: ``--keep-nonterminal-empty-after-exec`` (keep every assistant turn).
    v1 wins if both keep-flags are set; v2 wins over v3/v4 keep-flags.
    """
    if keep_nonterminal_empty_after_exec:
        return RECIPE_V1
    if keep_terminal_empty_after_exec:
        return RECIPE_V2
    if keep_first_assistant_empty and keep_clarifying_empty_after_user:
        return RECIPE_V3
    return RECIPE_V4


def recipe_drop_flags(recipe: str) -> tuple[bool, bool]:
    """Return ``(drop_nonterminal, drop_terminal)`` for a recipe id."""
    if recipe == RECIPE_V1:
        return False, False
    if recipe == RECIPE_V2:
        return True, False
    return True, True


def recipe_v4_drop_flags(recipe: str) -> tuple[bool, bool]:
    """Return ``(drop_first_assistant_empty, drop_clarifying_empty_after_user)``.

    Only v4 turns these on. v1/v2/v3 keep first-turn and clarifying ``[]``
    so those ablations stay comparable to scored Run 12 / 13 / v3 prepare.
    """
    if recipe == RECIPE_V4:
        return True, True
    return False, False


def infer_recipe(
    *,
    drop_nonterminal_empty_after_exec: bool,
    drop_terminal_empty_after_exec: bool,
    drop_first_assistant_empty: bool | None = None,
    drop_clarifying_empty_after_user: bool | None = None,
) -> str:
    """Infer a recipe id from drop kwargs (``None`` v4 flags mean default-on)."""
    if not drop_nonterminal_empty_after_exec:
        return RECIPE_V1
    if not drop_terminal_empty_after_exec:
        return RECIPE_V2
    if (
        drop_first_assistant_empty is False
        and drop_clarifying_empty_after_user is False
    ):
        return RECIPE_V3
    return RECIPE_V4


def should_drop_assistant_turn(
    messages: list[dict],
    index: int,
    *,
    drop_nonterminal_empty_after_exec: bool = True,
    drop_terminal_empty_after_exec: bool = True,
    drop_first_assistant_empty: bool = True,
    drop_clarifying_empty_after_user: bool = True,
) -> bool:
    """Drop empty ``[]`` targets under the active recipe.

    v4 default: empty-after-exec (terminal + non-terminal), first-assistant
    empty, and clarifying empty-after-user. Pass the v4 drop flags ``False``
    for the v3 policy; ``drop_terminal_empty_after_exec=False`` for v2.
    """
    kind = classify_assistant_turn(messages, index)
    if kind == KIND_EMPTY_AFTER_EXEC_NONTERMINAL and drop_nonterminal_empty_after_exec:
        return True
    if kind == KIND_EMPTY_AFTER_EXEC_TERMINAL and drop_terminal_empty_after_exec:
        return True
    if drop_first_assistant_empty and is_first_assistant_empty_target(messages, index):
        return True
    if drop_clarifying_empty_after_user and is_clarifying_empty_after_user(
        messages, index
    ):
        return True
    return False


def is_recovery_after_empty(messages: list[dict]) -> bool:
    """True when gold is a nonempty call after a prior ``[]`` then a new user.

    The last message must be a nonempty IFFC tool call. Somewhere in the
    prefix there is an assistant ``[]`` (formatter NL→``[]``, clarifying stop,
    or a synthetic history-only stop), then a later user message that is **not**
    an execution observation, then the gold call.

    This is the BFCL multi-turn recovery shape: stop the current turn, then
    start again on the next user question. Empty-after-exec ``[]`` *targets*
    are not recovery rows.
    """
    if len(messages) < 3:
        return False
    if messages[-1].get("role") != "assistant":
        return False
    if not has_nonempty_iffc_tool_call(messages[-1].get("content")):
        return False
    for i, msg in enumerate(messages[:-1]):
        if msg.get("role") != "assistant":
            continue
        if not is_empty_iffc_target(msg.get("content")):
            continue
        for later in messages[i + 1 : -1]:
            if later.get("role") != "user":
                continue
            if not is_iffc_execution_observation(later.get("content")):
                return True
    return False


def inject_recovery_stop(messages: list[dict]) -> tuple[list[dict], bool]:
    """Insert assistant ``[]`` after tool results in a user→call unroll prefix.

    Only fires when:

    * the gold (last) message is a nonempty IFFC tool call
    * the previous user message is a **new question**, not an execution
      observation (continue-after-exec is left alone)
    * history contains an IFFC execution observation **immediately** before
      that new user question
    * there is not already an assistant ``[]`` between those two user turns

    The inserted ``[]`` is history-only: this function never changes the gold
    target and never emits a new unroll row. Callers must not train on the
    inserted stop (v3 already drops empty-after-exec targets from the parent
    conversation; the synthetic token is not added to the parent).
    """
    if len(messages) < 3:
        return messages, False
    if messages[-1].get("role") != "assistant":
        return messages, False
    if not has_nonempty_iffc_tool_call(messages[-1].get("content")):
        return messages, False

    prev_user_idx = _previous_user_index(messages, len(messages) - 1)
    if prev_user_idx is None:
        return messages, False
    if is_iffc_execution_observation(messages[prev_user_idx].get("content")):
        return messages, False

    last_exec_idx = None
    for j in range(prev_user_idx - 1, -1, -1):
        if messages[j].get("role") != "user":
            continue
        if is_iffc_execution_observation(messages[j].get("content")):
            last_exec_idx = j
            break
    if last_exec_idx is None:
        return messages, False

    for j in range(last_exec_idx + 1, prev_user_idx):
        if messages[j].get("role") == "assistant" and is_empty_iffc_target(
            messages[j].get("content")
        ):
            return messages, False

    # Only the BFCL bookkeeping hole: tool results, then a new user question
    # with nothing in between. Do not splice ``[]`` into a continue-after-exec
    # subsequence (exec → [call] → new user).
    if last_exec_idx + 1 != prev_user_idx:
        return messages, False

    inserted = {"role": "assistant", "content": EMPTY_IFFC_TARGET}
    out = messages[:prev_user_idx] + [inserted] + messages[prev_user_idx:]
    return out, True


def _source_of(row: dict, parent: dict | None = None) -> str:
    raw = row.get("source")
    if raw is None and parent is not None:
        raw = parent.get("source")
    return str(raw or "unknown")


@dataclass
class EmptyAfterExecFilterStats:
    """Before/after counts for the empty-after-exec drop + recovery curriculum."""

    recipe: str = RECIPE_V4
    total_before: int = 0
    total_after: int = 0
    empty_targets_before: int = 0
    empty_targets_after: int = 0
    empty_after_exec_before: int = 0
    empty_after_exec_after: int = 0
    nonterminal_empty_after_exec_before: int = 0
    nonterminal_empty_after_exec_after: int = 0
    terminal_empty_after_exec_before: int = 0
    terminal_empty_after_exec_after: int = 0
    first_turn_empty_before: int = 0
    first_turn_empty_after: int = 0
    clarifying_empty_before: int = 0
    clarifying_empty_after: int = 0
    recovery_after_empty_before: int = 0
    recovery_after_empty_after: int = 0
    synthetic_recovery_inserted: int = 0
    dropped: int = 0
    dropped_first_turn_empty: int = 0
    dropped_clarifying_empty: int = 0
    by_source_before: Counter[str] = field(default_factory=Counter)
    by_source_after: Counter[str] = field(default_factory=Counter)
    empty_targets_by_source_before: Counter[str] = field(default_factory=Counter)
    empty_targets_by_source_after: Counter[str] = field(default_factory=Counter)
    empty_after_exec_by_source_before: Counter[str] = field(default_factory=Counter)
    empty_after_exec_by_source_after: Counter[str] = field(default_factory=Counter)
    recovery_after_empty_by_source_before: Counter[str] = field(default_factory=Counter)
    recovery_after_empty_by_source_after: Counter[str] = field(default_factory=Counter)
    dropped_by_source: Counter[str] = field(default_factory=Counter)

    def _note(
        self,
        *,
        kind: str,
        source: str,
        after: bool,
        recovery: bool = False,
        first_turn_empty: bool = False,
        clarifying_empty: bool = False,
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
        is_terminal = kind == KIND_EMPTY_AFTER_EXEC_TERMINAL
        if after:
            if is_empty:
                self.empty_targets_after += 1
                self.empty_targets_by_source_after[source] += 1
            if is_after_exec:
                self.empty_after_exec_after += 1
                self.empty_after_exec_by_source_after[source] += 1
            if is_nonterminal:
                self.nonterminal_empty_after_exec_after += 1
            if is_terminal:
                self.terminal_empty_after_exec_after += 1
            if first_turn_empty:
                self.first_turn_empty_after += 1
            if clarifying_empty:
                self.clarifying_empty_after += 1
            if recovery:
                self.recovery_after_empty_after += 1
                self.recovery_after_empty_by_source_after[source] += 1
        else:
            if is_empty:
                self.empty_targets_before += 1
                self.empty_targets_by_source_before[source] += 1
            if is_after_exec:
                self.empty_after_exec_before += 1
                self.empty_after_exec_by_source_before[source] += 1
            if is_nonterminal:
                self.nonterminal_empty_after_exec_before += 1
            if is_terminal:
                self.terminal_empty_after_exec_before += 1
            if first_turn_empty:
                self.first_turn_empty_before += 1
            if clarifying_empty:
                self.clarifying_empty_before += 1
            if recovery:
                self.recovery_after_empty_before += 1
                self.recovery_after_empty_by_source_before[source] += 1

    def note_before(
        self,
        kind: str,
        source: str,
        *,
        recovery: bool = False,
        first_turn_empty: bool = False,
        clarifying_empty: bool = False,
    ) -> None:
        self._note(
            kind=kind,
            source=source,
            after=False,
            recovery=recovery,
            first_turn_empty=first_turn_empty,
            clarifying_empty=clarifying_empty,
        )

    def note_after(
        self,
        kind: str,
        source: str,
        *,
        recovery: bool = False,
        first_turn_empty: bool = False,
        clarifying_empty: bool = False,
    ) -> None:
        self._note(
            kind=kind,
            source=source,
            after=True,
            recovery=recovery,
            first_turn_empty=first_turn_empty,
            clarifying_empty=clarifying_empty,
        )

    def note_dropped(
        self,
        source: str,
        *,
        first_turn_empty: bool = False,
        clarifying_empty: bool = False,
    ) -> None:
        self.dropped += 1
        self.dropped_by_source[source] += 1
        if first_turn_empty:
            self.dropped_first_turn_empty += 1
        if clarifying_empty:
            self.dropped_clarifying_empty += 1

    def note_synthetic_recovery(self) -> None:
        self.synthetic_recovery_inserted += 1

    def as_dict(self) -> dict[str, Any]:
        def _plain(counter: Counter[str]) -> dict[str, int]:
            return dict(sorted(counter.items()))

        return {
            "recipe": self.recipe,
            "total_before": self.total_before,
            "total_after": self.total_after,
            "empty_targets_before": self.empty_targets_before,
            "empty_targets_after": self.empty_targets_after,
            "empty_after_exec_before": self.empty_after_exec_before,
            "empty_after_exec_after": self.empty_after_exec_after,
            "nonterminal_empty_after_exec_before": self.nonterminal_empty_after_exec_before,
            "nonterminal_empty_after_exec_after": self.nonterminal_empty_after_exec_after,
            "terminal_empty_after_exec_before": self.terminal_empty_after_exec_before,
            "terminal_empty_after_exec_after": self.terminal_empty_after_exec_after,
            "first_turn_empty_before": self.first_turn_empty_before,
            "first_turn_empty_after": self.first_turn_empty_after,
            "clarifying_empty_before": self.clarifying_empty_before,
            "clarifying_empty_after": self.clarifying_empty_after,
            "recovery_after_empty_before": self.recovery_after_empty_before,
            "recovery_after_empty_after": self.recovery_after_empty_after,
            "synthetic_recovery_inserted": self.synthetic_recovery_inserted,
            "dropped": self.dropped,
            "dropped_first_turn_empty": self.dropped_first_turn_empty,
            "dropped_clarifying_empty": self.dropped_clarifying_empty,
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
            "recovery_after_empty_by_source_before": _plain(
                self.recovery_after_empty_by_source_before
            ),
            "recovery_after_empty_by_source_after": _plain(
                self.recovery_after_empty_by_source_after
            ),
            "dropped_by_source": _plain(self.dropped_by_source),
        }

    def format(
        self,
        label: str,
        *,
        enabled: bool | None = None,
        recipe: str | None = None,
        drop_nonterminal_empty_after_exec: bool | None = None,
        drop_terminal_empty_after_exec: bool | None = None,
        drop_first_assistant_empty: bool | None = None,
        drop_clarifying_empty_after_user: bool | None = None,
        synthesize_recovery_stop: bool | None = None,
    ) -> str:
        recipe_id = recipe or self.recipe
        drop_nt, drop_t = recipe_drop_flags(recipe_id)
        drop_first, drop_clarify = recipe_v4_drop_flags(recipe_id)
        if drop_nonterminal_empty_after_exec is not None:
            drop_nt = drop_nonterminal_empty_after_exec
        if drop_terminal_empty_after_exec is not None:
            drop_t = drop_terminal_empty_after_exec
        if drop_first_assistant_empty is not None:
            drop_first = drop_first_assistant_empty
        if drop_clarifying_empty_after_user is not None:
            drop_clarify = drop_clarifying_empty_after_user
        if enabled is None:
            enabled = drop_nt or drop_t or drop_first or drop_clarify
        synth = True if synthesize_recovery_stop is None else synthesize_recovery_stop
        sources = sorted(
            set(self.by_source_before) | set(self.by_source_after) | {"toolace", "xlam"}
        )
        lines = [
            f"empty-after-exec filter ({label}): recipe={recipe_id} "
            f"drop_nonterminal={drop_nt} drop_terminal={drop_t} "
            f"drop_first_assistant_empty={drop_first} "
            f"drop_clarifying_empty_after_user={drop_clarify} "
            f"synthetic_recovery={synth}",
            f"  rows: {self.total_before} → {self.total_after} "
            f"(dropped {self.dropped})",
            f"  empty [] targets: {self.empty_targets_before} → "
            f"{self.empty_targets_after}",
            f"  empty-after-exec: {self.empty_after_exec_before} → "
            f"{self.empty_after_exec_after} "
            f"(non-terminal {self.nonterminal_empty_after_exec_before} → "
            f"{self.nonterminal_empty_after_exec_after}; "
            f"terminal {self.terminal_empty_after_exec_before} → "
            f"{self.terminal_empty_after_exec_after})",
            f"  first_turn_empty: {self.first_turn_empty_before} → "
            f"{self.first_turn_empty_after} "
            f"(dropped {self.dropped_first_turn_empty})",
            f"  clarifying_empty: {self.clarifying_empty_before} → "
            f"{self.clarifying_empty_after} "
            f"(dropped {self.dropped_clarifying_empty})",
            f"  recovery_after_empty: {self.recovery_after_empty_before} → "
            f"{self.recovery_after_empty_after} "
            f"(synthetic inserted {self.synthetic_recovery_inserted})",
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
                f"{self.empty_after_exec_by_source_after[source]}; "
                f"recovery_after_empty "
                f"{self.recovery_after_empty_by_source_before[source]}→"
                f"{self.recovery_after_empty_by_source_after[source]}"
            )
        return "\n".join(lines)


def unroll_and_filter(
    records: Iterable[dict],
    *,
    drop_nonterminal_empty_after_exec: bool = True,
    drop_terminal_empty_after_exec: bool = True,
    drop_first_assistant_empty: bool | None = None,
    drop_clarifying_empty_after_user: bool | None = None,
    synthesize_recovery_stop: bool = True,
    recipe: str | None = None,
    check_bfcl_leak: bool = True,
) -> tuple[list[dict], EmptyAfterExecFilterStats]:
    """Unroll conversations, drop empty targets, inject recovery history.

    Defaults are SFT v4: v3 empty-after-exec + recovery, plus drop first-
    assistant ``[]`` targets and clarifying empty-after-user ``[]``. Pass
    both v4 drop flags ``False`` to ablate back to v3. ``None`` v4 flags
    follow the inferred recipe so v1/v2 kwargs do not silently gain v4 drops.
    """
    recs = list(records)
    if check_bfcl_leak:
        assert_no_bfcl(recs)
    if recipe is None:
        recipe = infer_recipe(
            drop_nonterminal_empty_after_exec=drop_nonterminal_empty_after_exec,
            drop_terminal_empty_after_exec=drop_terminal_empty_after_exec,
            drop_first_assistant_empty=drop_first_assistant_empty,
            drop_clarifying_empty_after_user=drop_clarifying_empty_after_user,
        )
    drop_first_v4, drop_clarify_v4 = recipe_v4_drop_flags(recipe)
    if drop_first_assistant_empty is None:
        drop_first_assistant_empty = drop_first_v4
    if drop_clarifying_empty_after_user is None:
        drop_clarifying_empty_after_user = drop_clarify_v4
    stats = EmptyAfterExecFilterStats(recipe=recipe)
    out: list[dict] = []
    for rec in recs:
        messages = rec.get("messages") or []
        rows = unroll_conversation(rec)
        for row in rows:
            idx = len(row.get("messages") or []) - 1
            kind = classify_assistant_turn(messages, idx)
            source = _source_of(row, rec)
            prefix = list(row.get("messages") or [])
            recovery_before = is_recovery_after_empty(prefix)
            first_turn = is_first_assistant_empty_target(messages, idx)
            clarifying = is_clarifying_empty_after_user(messages, idx)
            stats.note_before(
                kind,
                source,
                recovery=recovery_before,
                first_turn_empty=first_turn,
                clarifying_empty=clarifying,
            )
            if should_drop_assistant_turn(
                messages,
                idx,
                drop_nonterminal_empty_after_exec=drop_nonterminal_empty_after_exec,
                drop_terminal_empty_after_exec=drop_terminal_empty_after_exec,
                drop_first_assistant_empty=drop_first_assistant_empty,
                drop_clarifying_empty_after_user=drop_clarifying_empty_after_user,
            ):
                stats.note_dropped(
                    source,
                    first_turn_empty=first_turn,
                    clarifying_empty=clarifying,
                )
                continue
            injected = False
            if synthesize_recovery_stop:
                prefix, injected = inject_recovery_stop(prefix)
                if injected:
                    row = {**row, "messages": prefix}
                    stats.note_synthetic_recovery()
            recovery_after = is_recovery_after_empty(prefix)
            if recovery_after:
                row = {**row, "recovery_after_empty": True}
            if injected:
                row = {**row, "synthetic_recovery_stop": True}
            stats.note_after(
                kind,
                source,
                recovery=recovery_after,
                first_turn_empty=first_turn,
                clarifying_empty=clarifying,
            )
            out.append(row)
    return out, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unroll conversation-level IFFC JSONL. Default SFT v4 keeps the "
            "v3 empty-after-exec drop + recovery curriculum, and also drops "
            "first-assistant [] targets and clarifying [] after ordinary user "
            "messages. v3/v2/v1 remain behind keep-flags."
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
    parser.add_argument(
        "--keep-terminal-empty-after-exec",
        action="store_true",
        help=(
            "SFT v2 unroll: drop mid-trajectory [] after tool results, but "
            "keep true terminal empty-after-exec."
        ),
    )
    parser.add_argument(
        "--keep-first-assistant-empty",
        action="store_true",
        help=(
            "Keep unrolled rows whose gold is [] and that assistant is the "
            "first in the prefix (disable the v4 turn-0 empty drop)."
        ),
    )
    parser.add_argument(
        "--keep-clarifying-empty-after-user",
        action="store_true",
        help=(
            "Keep user→[] clarifying / chitchat targets when the previous "
            "message is not an IFFC tool-result observation. Pass together "
            "with --keep-first-assistant-empty to ablate back to v3."
        ),
    )
    parser.add_argument(
        "--no-synthetic-recovery",
        action="store_true",
        help=(
            "Do not insert a history-only [] after tool results on the next "
            "user→call unroll row (natural recovery rows are still kept)."
        ),
    )
    args = parser.parse_args(argv)
    recipe = resolve_recipe(
        keep_nonterminal_empty_after_exec=args.keep_nonterminal_empty_after_exec,
        keep_terminal_empty_after_exec=args.keep_terminal_empty_after_exec,
        keep_first_assistant_empty=args.keep_first_assistant_empty,
        keep_clarifying_empty_after_user=args.keep_clarifying_empty_after_user,
    )
    drop_nt, drop_t = recipe_drop_flags(recipe)
    drop_first, drop_clarify = recipe_v4_drop_flags(recipe)
    if args.keep_first_assistant_empty:
        drop_first = False
    if args.keep_clarifying_empty_after_user:
        drop_clarify = False
    synthesize = recipe in (RECIPE_V3, RECIPE_V4) and not args.no_synthetic_recovery

    if args.input is None and args.splits_dir is None:
        parser.error("pass --input FILE or --splits-dir DIR")

    def _run(path: Path, label: str, out_path: Path | None) -> None:
        if not path.exists():
            parser.error(f"missing {path}")
        rows, stats = unroll_and_filter(
            read_jsonl(path),
            drop_nonterminal_empty_after_exec=drop_nt,
            drop_terminal_empty_after_exec=drop_t,
            drop_first_assistant_empty=drop_first,
            drop_clarifying_empty_after_user=drop_clarify,
            synthesize_recovery_stop=synthesize,
            recipe=recipe,
        )
        print(
            stats.format(
                label,
                recipe=recipe,
                drop_nonterminal_empty_after_exec=drop_nt,
                drop_terminal_empty_after_exec=drop_t,
                drop_first_assistant_empty=drop_first,
                drop_clarifying_empty_after_user=drop_clarify,
                synthesize_recovery_stop=synthesize,
            )
        )
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
