"""SFT unroll curriculum (v5 default; v1–v4 and v6 remain as ablations).

SFT v1 unrolled every assistant turn, including ToolACE ``[]`` after exec
(Run 12 0.5%). v2 dropped only non-terminal empty-after-exec (Run 13 still
0.5%). v3 dropped all empty-after-exec and kept / synthesized recovery-
after-``[]``. v4 (PR #31, Run 15 15.0%) dropped first-assistant-empty and
clarifying ``[]`` as well — empty self-conditioning largely fixed, still
below IFFC 30.5%. Residual: state_mismatch ~64%, under_calling ~35%
(mostly turn 0). Do **not** present 15% as beating IFFC; do **not** add
epochs on the v4 mix; do **not** treat BoN as the win for this ckpt.

SFT v5 (this module's default) keeps every v4 empty-target drop and the
recovery-after-``[]`` curriculum, then adds turn-0 call-set completeness
and env-state consistency (``sft.curriculum_v5``). Conversation-level
``split_data`` files are not rewritten. Do **not** copy BFCL
``multi_turn_base`` prompts into train. **Run 17 scored v5 IFFC 23/200 =
11.5%** (regression vs Run 15 15.0%): state_mismatch 109→133, under_call
59→44, empty 2→0, avg api_calls/entry 17.3→40.3. Hypothesis: synthetic
complete turn-0 taught parallel call dumps that desync observe→act APIs.

SFT v6 is **v4 empty/recovery + only** ``drop_state_inconsistent``. Turn-0
rewrites stay off (``drop_underspecified_turn0=False``,
``synthesize_complete_turn0=False``). Distinct recipe id — not a v4
relabel. Select with ``--recipe v6``. Do **not** BoN this ckpt; do **not**
add epochs on v4/v5. v6 does **not** claim a BFCL score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from benchmark.iffc import parse_iffc_response
from sft.curriculum_v5 import (
    is_absorbed_independent_followup,
    is_incomplete_turn0,
    is_state_consistent_continue,
    is_state_inconsistent_continue,
    is_underspecified_turn0,
    plan_turn0,
)
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
RECIPE_V5 = "v5"
RECIPE_V6 = "v6"
RECIPE_IDS = (RECIPE_V1, RECIPE_V2, RECIPE_V3, RECIPE_V4, RECIPE_V5, RECIPE_V6)
RECIPES_WITH_RECOVERY = (RECIPE_V3, RECIPE_V4, RECIPE_V5, RECIPE_V6)
RECIPES_WITH_V4_EMPTY = (RECIPE_V4, RECIPE_V5, RECIPE_V6)

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
    keep_underspecified_turn0: bool = False,
    keep_state_inconsistent: bool = False,
    no_synthetic_complete_turn0: bool = False,
    recipe: str | None = None,
) -> str:
    """Map ablation flags to a recipe id.

    v5 (default): v4 plus turn-0 completeness and env-state consistency.
    v6: v4 empty/recovery plus only ``drop_state_inconsistent`` (no
    underspecified drop, no complete-turn0 synth).
    v4: all three v5 keep/no-synth flags (empty-target curriculum only).
    v3: both ``--keep-first-assistant-empty`` and
    ``--keep-clarifying-empty-after-user`` (empty-after-exec + recovery only).
    v2: ``--keep-terminal-empty-after-exec`` (drop non-terminal only).
    v1: ``--keep-nonterminal-empty-after-exec`` (keep every assistant turn).
    Explicit ``recipe=`` wins. v1 wins if both v1/v2 keep-flags are set;
    v2 wins over v3/v4/v5/v6 keep-flags; v3 wins over v4/v5/v6 keep-flags;
    v4 wins over v6 when all three v5 keep/no-synth flags are set.
    """
    if recipe:
        return recipe
    if keep_nonterminal_empty_after_exec:
        return RECIPE_V1
    if keep_terminal_empty_after_exec:
        return RECIPE_V2
    if keep_first_assistant_empty and keep_clarifying_empty_after_user:
        return RECIPE_V3
    if (
        keep_underspecified_turn0
        and keep_state_inconsistent
        and no_synthetic_complete_turn0
    ):
        return RECIPE_V4
    if keep_underspecified_turn0 and no_synthetic_complete_turn0:
        return RECIPE_V6
    return RECIPE_V5


def recipe_drop_flags(recipe: str) -> tuple[bool, bool]:
    """Return ``(drop_nonterminal, drop_terminal)`` for a recipe id."""
    if recipe == RECIPE_V1:
        return False, False
    if recipe == RECIPE_V2:
        return True, False
    return True, True


def recipe_v4_drop_flags(recipe: str) -> tuple[bool, bool]:
    """Return ``(drop_first_assistant_empty, drop_clarifying_empty_after_user)``.

    v4, v5, and v6 turn these on. v1/v2/v3 keep first-turn and clarifying
    ``[]`` so those ablations stay comparable to scored Run 12 / 13 / v3
    prepare.
    """
    if recipe in RECIPES_WITH_V4_EMPTY:
        return True, True
    return False, False


def recipe_v5_flags(recipe: str) -> tuple[bool, bool, bool, bool]:
    """Curriculum extras: underspecified, state-inconsistent, complete-turn0, absorbed.

    v5: all four on.
    v6: only state-inconsistent (v4 empty/recovery + ``drop_state_inconsistent``).
    v4 and earlier: all off so Run 15's mix stays reproducible.
    """
    if recipe == RECIPE_V5:
        return True, True, True, True
    if recipe == RECIPE_V6:
        return False, True, False, False
    return False, False, False, False


def infer_recipe(
    *,
    drop_nonterminal_empty_after_exec: bool,
    drop_terminal_empty_after_exec: bool,
    drop_first_assistant_empty: bool | None = None,
    drop_clarifying_empty_after_user: bool | None = None,
    drop_underspecified_turn0: bool | None = None,
    drop_state_inconsistent: bool | None = None,
    synthesize_complete_turn0: bool | None = None,
) -> str:
    """Infer a recipe id from drop kwargs (``None`` v4/v5 flags mean default-on)."""
    if not drop_nonterminal_empty_after_exec:
        return RECIPE_V1
    if not drop_terminal_empty_after_exec:
        return RECIPE_V2
    if (
        drop_first_assistant_empty is False
        and drop_clarifying_empty_after_user is False
    ):
        return RECIPE_V3
    v5_off = (
        drop_underspecified_turn0 is False
        and drop_state_inconsistent is False
        and synthesize_complete_turn0 is False
    )
    if v5_off:
        return RECIPE_V4
    v6_only_state = (
        drop_underspecified_turn0 is False
        and drop_state_inconsistent is True
        and synthesize_complete_turn0 is False
    )
    if v6_only_state:
        return RECIPE_V6
    return RECIPE_V5


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

    recipe: str = RECIPE_V5
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
    underspecified_turn0_before: int = 0
    underspecified_turn0_after: int = 0
    incomplete_turn0_before: int = 0
    incomplete_turn0_after: int = 0
    state_inconsistent_before: int = 0
    state_inconsistent_after: int = 0
    state_consistent_after: int = 0
    synthetic_complete_turn0: int = 0
    dropped: int = 0
    dropped_first_turn_empty: int = 0
    dropped_clarifying_empty: int = 0
    dropped_underspecified_turn0: int = 0
    dropped_incomplete_turn0: int = 0
    dropped_state_inconsistent: int = 0
    dropped_absorbed_followup: int = 0
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
        underspecified_turn0: bool = False,
        incomplete_turn0: bool = False,
        state_inconsistent: bool = False,
        state_consistent: bool = False,
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
            if underspecified_turn0:
                self.underspecified_turn0_after += 1
            if incomplete_turn0:
                self.incomplete_turn0_after += 1
            if state_inconsistent:
                self.state_inconsistent_after += 1
            if state_consistent:
                self.state_consistent_after += 1
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
            if underspecified_turn0:
                self.underspecified_turn0_before += 1
            if incomplete_turn0:
                self.incomplete_turn0_before += 1
            if state_inconsistent:
                self.state_inconsistent_before += 1
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
        underspecified_turn0: bool = False,
        incomplete_turn0: bool = False,
        state_inconsistent: bool = False,
    ) -> None:
        self._note(
            kind=kind,
            source=source,
            after=False,
            recovery=recovery,
            first_turn_empty=first_turn_empty,
            clarifying_empty=clarifying_empty,
            underspecified_turn0=underspecified_turn0,
            incomplete_turn0=incomplete_turn0,
            state_inconsistent=state_inconsistent,
        )

    def note_after(
        self,
        kind: str,
        source: str,
        *,
        recovery: bool = False,
        first_turn_empty: bool = False,
        clarifying_empty: bool = False,
        underspecified_turn0: bool = False,
        incomplete_turn0: bool = False,
        state_inconsistent: bool = False,
        state_consistent: bool = False,
    ) -> None:
        self._note(
            kind=kind,
            source=source,
            after=True,
            recovery=recovery,
            first_turn_empty=first_turn_empty,
            clarifying_empty=clarifying_empty,
            underspecified_turn0=underspecified_turn0,
            incomplete_turn0=incomplete_turn0,
            state_inconsistent=state_inconsistent,
            state_consistent=state_consistent,
        )

    def note_dropped(
        self,
        source: str,
        *,
        first_turn_empty: bool = False,
        clarifying_empty: bool = False,
        underspecified_turn0: bool = False,
        incomplete_turn0: bool = False,
        state_inconsistent: bool = False,
        absorbed_followup: bool = False,
    ) -> None:
        self.dropped += 1
        self.dropped_by_source[source] += 1
        if first_turn_empty:
            self.dropped_first_turn_empty += 1
        if clarifying_empty:
            self.dropped_clarifying_empty += 1
        if underspecified_turn0:
            self.dropped_underspecified_turn0 += 1
        if incomplete_turn0:
            self.dropped_incomplete_turn0 += 1
        if state_inconsistent:
            self.dropped_state_inconsistent += 1
        if absorbed_followup:
            self.dropped_absorbed_followup += 1

    def note_synthetic_complete(self) -> None:
        self.synthetic_complete_turn0 += 1

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
            "underspecified_turn0_before": self.underspecified_turn0_before,
            "underspecified_turn0_after": self.underspecified_turn0_after,
            "incomplete_turn0_before": self.incomplete_turn0_before,
            "incomplete_turn0_after": self.incomplete_turn0_after,
            "state_inconsistent_before": self.state_inconsistent_before,
            "state_inconsistent_after": self.state_inconsistent_after,
            "state_consistent_after": self.state_consistent_after,
            "synthetic_complete_turn0": self.synthetic_complete_turn0,
            "dropped": self.dropped,
            "dropped_first_turn_empty": self.dropped_first_turn_empty,
            "dropped_clarifying_empty": self.dropped_clarifying_empty,
            "dropped_underspecified_turn0": self.dropped_underspecified_turn0,
            "dropped_incomplete_turn0": self.dropped_incomplete_turn0,
            "dropped_state_inconsistent": self.dropped_state_inconsistent,
            "dropped_absorbed_followup": self.dropped_absorbed_followup,
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
        drop_underspecified_turn0: bool | None = None,
        drop_state_inconsistent: bool | None = None,
        synthesize_complete_turn0: bool | None = None,
    ) -> str:
        recipe_id = recipe or self.recipe
        drop_nt, drop_t = recipe_drop_flags(recipe_id)
        drop_first, drop_clarify = recipe_v4_drop_flags(recipe_id)
        drop_u, drop_s, synth_c, _drop_abs = recipe_v5_flags(recipe_id)
        if drop_nonterminal_empty_after_exec is not None:
            drop_nt = drop_nonterminal_empty_after_exec
        if drop_terminal_empty_after_exec is not None:
            drop_t = drop_terminal_empty_after_exec
        if drop_first_assistant_empty is not None:
            drop_first = drop_first_assistant_empty
        if drop_clarifying_empty_after_user is not None:
            drop_clarify = drop_clarifying_empty_after_user
        if drop_underspecified_turn0 is not None:
            drop_u = drop_underspecified_turn0
        if drop_state_inconsistent is not None:
            drop_s = drop_state_inconsistent
        if synthesize_complete_turn0 is not None:
            synth_c = synthesize_complete_turn0
        if enabled is None:
            enabled = (
                drop_nt
                or drop_t
                or drop_first
                or drop_clarify
                or drop_u
                or drop_s
                or synth_c
            )
        synth = True if synthesize_recovery_stop is None else synthesize_recovery_stop
        sources = sorted(
            set(self.by_source_before) | set(self.by_source_after) | {"toolace", "xlam"}
        )
        lines = [
            f"empty-after-exec filter ({label}): recipe={recipe_id} "
            f"drop_nonterminal={drop_nt} drop_terminal={drop_t} "
            f"drop_first_assistant_empty={drop_first} "
            f"drop_clarifying_empty_after_user={drop_clarify} "
            f"synthetic_recovery={synth} "
            f"drop_underspecified_turn0={drop_u} "
            f"drop_state_inconsistent={drop_s} "
            f"synthetic_complete_turn0={synth_c}",
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
            f"  underspecified_turn0: {self.underspecified_turn0_before} → "
            f"{self.underspecified_turn0_after} "
            f"(dropped {self.dropped_underspecified_turn0})",
            f"  incomplete_turn0: {self.incomplete_turn0_before} → "
            f"{self.incomplete_turn0_after} "
            f"(dropped {self.dropped_incomplete_turn0}; "
            f"synthetic complete {self.synthetic_complete_turn0})",
            f"  state_inconsistent: {self.state_inconsistent_before} → "
            f"{self.state_inconsistent_after} "
            f"(dropped {self.dropped_state_inconsistent}; "
            f"state_consistent kept {self.state_consistent_after}; "
            f"absorbed follow-up dropped {self.dropped_absorbed_followup})",
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
    drop_underspecified_turn0: bool | None = None,
    drop_state_inconsistent: bool | None = None,
    synthesize_complete_turn0: bool | None = None,
    drop_absorbed_independent_followup: bool | None = None,
    synthesize_recovery_stop: bool = True,
    recipe: str | None = None,
    check_bfcl_leak: bool = True,
) -> tuple[list[dict], EmptyAfterExecFilterStats]:
    """Unroll conversations, apply the active recipe, inject recovery history.

    Defaults are SFT v5: v4 empty-after-exec + recovery + first-assistant /
    clarifying ``[]`` drops, plus turn-0 completeness and env-state
    consistency. Pass ``recipe="v6"`` for v4 empty/recovery plus only
    ``drop_state_inconsistent`` (no underspecified drop, no complete-turn0
    synth). Pass ``recipe="v4"`` (or all three v5 flags off) to ablate back
    to v4. ``None`` v4/v5 flags follow the inferred recipe so v1/v2 kwargs
    do not silently gain later drops.
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
            drop_underspecified_turn0=drop_underspecified_turn0,
            drop_state_inconsistent=drop_state_inconsistent,
            synthesize_complete_turn0=synthesize_complete_turn0,
        )
    drop_first_v4, drop_clarify_v4 = recipe_v4_drop_flags(recipe)
    drop_u_v5, drop_s_v5, synth_c_v5, drop_abs_v5 = recipe_v5_flags(recipe)
    if drop_first_assistant_empty is None:
        drop_first_assistant_empty = drop_first_v4
    if drop_clarifying_empty_after_user is None:
        drop_clarifying_empty_after_user = drop_clarify_v4
    if drop_underspecified_turn0 is None:
        drop_underspecified_turn0 = drop_u_v5
    if drop_state_inconsistent is None:
        drop_state_inconsistent = drop_s_v5
    if synthesize_complete_turn0 is None:
        synthesize_complete_turn0 = synth_c_v5
    if drop_absorbed_independent_followup is None:
        drop_absorbed_independent_followup = drop_abs_v5 and synthesize_complete_turn0
    stats = EmptyAfterExecFilterStats(recipe=recipe)
    out: list[dict] = []
    for rec in recs:
        messages = rec.get("messages") or []
        plan = plan_turn0(messages)
        rows = unroll_conversation(rec)
        for row in rows:
            idx = len(row.get("messages") or []) - 1
            kind = classify_assistant_turn(messages, idx)
            source = _source_of(row, rec)
            prefix = list(row.get("messages") or [])
            recovery_before = is_recovery_after_empty(prefix)
            first_turn = is_first_assistant_empty_target(messages, idx)
            clarifying = is_clarifying_empty_after_user(messages, idx)
            underspec = is_underspecified_turn0(messages, idx)
            incomplete = is_incomplete_turn0(messages, idx)
            state_bad = is_state_inconsistent_continue(messages, idx)
            stats.note_before(
                kind,
                source,
                recovery=recovery_before,
                first_turn_empty=first_turn,
                clarifying_empty=clarifying,
                underspecified_turn0=underspec,
                incomplete_turn0=incomplete,
                state_inconsistent=state_bad,
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
            if drop_underspecified_turn0 and underspec:
                stats.note_dropped(source, underspecified_turn0=True)
                continue
            if (
                incomplete
                and not synthesize_complete_turn0
                and drop_underspecified_turn0
            ):
                stats.note_dropped(source, incomplete_turn0=True)
                continue
            if drop_state_inconsistent and state_bad:
                stats.note_dropped(source, state_inconsistent=True)
                continue
            if drop_absorbed_independent_followup and is_absorbed_independent_followup(
                messages, idx, plan.absorbed_keys
            ):
                stats.note_dropped(source, absorbed_followup=True)
                continue
            completed = False
            incomplete_after = incomplete
            if (
                synthesize_complete_turn0
                and idx == plan.first_nonempty_idx
                and plan.merged_gold
            ):
                last = dict(prefix[-1])
                last["content"] = plan.merged_gold
                prefix = prefix[:-1] + [last]
                row = {**row, "messages": prefix, "synthetic_complete_turn0": True}
                stats.note_synthetic_complete()
                completed = True
                incomplete_after = False
                underspec = False
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
            if completed:
                row = {**row, "synthetic_complete_turn0": True}
            state_ok = is_state_consistent_continue(prefix, len(prefix) - 1)
            stats.note_after(
                kind,
                source,
                recovery=recovery_after,
                first_turn_empty=first_turn,
                clarifying_empty=clarifying,
                underspecified_turn0=False if completed else underspec,
                incomplete_turn0=incomplete_after,
                state_inconsistent=False,
                state_consistent=state_ok,
            )
            out.append(row)
    return out, stats


def flags_for_recipe(
    recipe: str,
    *,
    synthesize_recovery_stop: bool | None = None,
) -> dict[str, Any]:
    """Kwargs for ``unroll_and_filter`` that realize ``recipe``."""
    drop_nt, drop_t = recipe_drop_flags(recipe)
    drop_first, drop_clarify = recipe_v4_drop_flags(recipe)
    drop_u, drop_s, synth_c, drop_abs = recipe_v5_flags(recipe)
    if synthesize_recovery_stop is None:
        synthesize_recovery_stop = recipe in RECIPES_WITH_RECOVERY
    return {
        "drop_nonterminal_empty_after_exec": drop_nt,
        "drop_terminal_empty_after_exec": drop_t,
        "drop_first_assistant_empty": drop_first,
        "drop_clarifying_empty_after_user": drop_clarify,
        "drop_underspecified_turn0": drop_u,
        "drop_state_inconsistent": drop_s,
        "synthesize_complete_turn0": synth_c,
        "drop_absorbed_independent_followup": drop_abs and synth_c,
        "synthesize_recovery_stop": synthesize_recovery_stop,
        "recipe": recipe,
    }


def compare_recipe_unroll(
    records: Iterable[dict],
    *,
    recipe: str = RECIPE_V5,
    baseline: str = RECIPE_V4,
    check_bfcl_leak: bool = True,
) -> dict[str, Any]:
    """Unroll the same conversations under two recipes and return row deltas.

    Used by ``--compare-recipe`` so a Mac prepare can print v6 vs v4 (or
    v5 vs v4) kept / dropped counts without a second CLI invocation. Does
    not write files.
    """
    recs = list(records)
    _kept_a, stats_a = unroll_and_filter(
        recs, check_bfcl_leak=check_bfcl_leak, **flags_for_recipe(recipe)
    )
    _kept_b, stats_b = unroll_and_filter(
        recs, check_bfcl_leak=False, **flags_for_recipe(baseline)
    )
    da, db = stats_a.as_dict(), stats_b.as_dict()
    return {
        "recipe": recipe,
        "baseline": baseline,
        "total_after": da["total_after"],
        "baseline_total_after": db["total_after"],
        "delta_rows": da["total_after"] - db["total_after"],
        "dropped": da["dropped"],
        "baseline_dropped": db["dropped"],
        "delta_dropped": da["dropped"] - db["dropped"],
        "recovery_after_empty_after": da["recovery_after_empty_after"],
        "baseline_recovery_after_empty_after": db["recovery_after_empty_after"],
        "synthetic_recovery_inserted": da["synthetic_recovery_inserted"],
        "synthetic_complete_turn0": da["synthetic_complete_turn0"],
        "dropped_underspecified_turn0": da["dropped_underspecified_turn0"],
        "dropped_incomplete_turn0": da["dropped_incomplete_turn0"],
        "dropped_state_inconsistent": da["dropped_state_inconsistent"],
        "dropped_absorbed_followup": da["dropped_absorbed_followup"],
        "empty_after_exec_after": da["empty_after_exec_after"],
        "baseline_empty_after_exec_after": db["empty_after_exec_after"],
        "recipe_stats": da,
        "baseline_stats": db,
    }


def format_recipe_comparison(delta: dict[str, Any], label: str) -> str:
    """Pretty-print ``compare_recipe_unroll`` for the prepare log."""
    return "\n".join(
        [
            f"recipe compare ({label}): {delta['recipe']} vs {delta['baseline']}",
            f"  rows after: {delta['baseline_total_after']} → "
            f"{delta['total_after']} (delta {delta['delta_rows']:+d})",
            f"  dropped: {delta['baseline_dropped']} → {delta['dropped']} "
            f"(delta {delta['delta_dropped']:+d})",
            f"  recovery_after_empty: "
            f"{delta['baseline_recovery_after_empty_after']} → "
            f"{delta['recovery_after_empty_after']}",
            f"  empty-after-exec after: "
            f"{delta['baseline_empty_after_exec_after']} → "
            f"{delta['empty_after_exec_after']}",
            f"  curriculum extra: underspecified dropped "
            f"{delta['dropped_underspecified_turn0']}; "
            f"incomplete dropped {delta['dropped_incomplete_turn0']}; "
            f"synthetic complete {delta['synthetic_complete_turn0']}; "
            f"state_inconsistent dropped {delta['dropped_state_inconsistent']}; "
            f"absorbed follow-up dropped {delta['dropped_absorbed_followup']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unroll conversation-level IFFC JSONL. Default SFT v5 keeps the "
            "v4 empty-after-exec + recovery curriculum and first-assistant / "
            "clarifying [] drops, then adds turn-0 call-set completeness and "
            "env-state consistency. v6 is v4 empty/recovery + only "
            "drop_state_inconsistent (no turn-0 rewrites). v4/v3/v2/v1 remain "
            "behind --recipe / keep-flags."
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
        "--recipe",
        choices=list(RECIPE_IDS),
        default=None,
        help="Unroll recipe id. Default v5. v6/v4/v3/v2/v1 are ablations.",
    )
    parser.add_argument(
        "--compare-recipe",
        choices=list(RECIPE_IDS),
        default=None,
        help="Also unroll under this baseline and print row deltas (e.g. v4).",
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
        "--keep-underspecified-turn0",
        action="store_true",
        help=(
            "Keep first-call rows whose opening user names ≥2 catalog tools "
            "but gold has one call (disable the v5 underspecified drop)."
        ),
    )
    parser.add_argument(
        "--keep-state-inconsistent",
        action="store_true",
        help=(
            "Keep continue-after-exec gold that uses a user-named entity "
            "the exec observation never confirmed (disable the v5 state drop)."
        ),
    )
    parser.add_argument(
        "--no-synthetic-complete-turn0",
        action="store_true",
        help=(
            "Do not merge independently-knowable later first-episode calls "
            "into the first nonempty gold. Incomplete first-turns are dropped "
            "instead (v5 minus synth). Pass with --keep-underspecified-turn0 "
            "for v6 (state-inconsistent drop only); pass with both v5 "
            "keep-flags for v4."
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
        keep_underspecified_turn0=args.keep_underspecified_turn0,
        keep_state_inconsistent=args.keep_state_inconsistent,
        no_synthetic_complete_turn0=args.no_synthetic_complete_turn0,
        recipe=args.recipe,
    )
    drop_nt, drop_t = recipe_drop_flags(recipe)
    drop_first, drop_clarify = recipe_v4_drop_flags(recipe)
    drop_u, drop_s, synth_c, drop_abs = recipe_v5_flags(recipe)
    if args.keep_first_assistant_empty:
        drop_first = False
    if args.keep_clarifying_empty_after_user:
        drop_clarify = False
    if args.keep_underspecified_turn0:
        drop_u = False
    if args.keep_state_inconsistent:
        drop_s = False
    if args.no_synthetic_complete_turn0:
        synth_c = False
        drop_abs = False
    synthesize = recipe in RECIPES_WITH_RECOVERY and not args.no_synthetic_recovery

    if args.input is None and args.splits_dir is None:
        parser.error("pass --input FILE or --splits-dir DIR")

    def _run(path: Path, label: str, out_path: Path | None) -> None:
        if not path.exists():
            parser.error(f"missing {path}")
        recs = read_jsonl(path)
        rows, stats = unroll_and_filter(
            recs,
            drop_nonterminal_empty_after_exec=drop_nt,
            drop_terminal_empty_after_exec=drop_t,
            drop_first_assistant_empty=drop_first,
            drop_clarifying_empty_after_user=drop_clarify,
            drop_underspecified_turn0=drop_u,
            drop_state_inconsistent=drop_s,
            synthesize_complete_turn0=synth_c,
            drop_absorbed_independent_followup=drop_abs and synth_c,
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
                drop_underspecified_turn0=drop_u,
                drop_state_inconsistent=drop_s,
                synthesize_complete_turn0=synth_c,
            )
        )
        if args.compare_recipe:
            delta = compare_recipe_unroll(
                recs,
                recipe=recipe,
                baseline=args.compare_recipe,
                check_bfcl_leak=False,
            )
            print(format_recipe_comparison(delta, label))
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

