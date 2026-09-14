"""Best-of-N majority voting overlay on IFFC+STATE (METHODS §4 / PLAN #2).

``--fc-mode iffc-state-bon`` keeps the IFFC+STATE channel (no native
``tools`` / ``tool_choice``; JSON-in-text parse; harness-written ``[STATE]``
from executed tool results; same BFCL ``multi_turn_checker``) and replaces
the single greedy completion with **MV@N**:

1. Sample *N* candidates at ``temperature=0.7`` (default ``N=5``).
2. Parse each with the IFFC JSON-in-text parser.
3. Majority-vote on the step's ``(function_name, arguments)`` tuples.
4. Execute the **winner once**, then continue the agent loop (STATE updates
   as in ``iffc-state``).

This is **not** NFC, **not** GST, **not** CD/GBNF, and **not** a verifier /
reward model. Empty parseable ``[]`` is ABSTAIN (a real ballot). Unparseable
candidates are **excluded** from the vote.

Tie-break (deterministic):

1. Highest ballot count wins.
2. If still tied, prefer any non-empty tool-call key over ABSTAIN.
3. If still tied, pick the first-seen key among remaining modes (sample order).

The winning *content* is the first candidate that produced that key, so the
transcript stores real model text. Losing samples are **not** written to
``chat_history``.

Latency is ~N× vs greedy IFFC+STATE.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import config
from .iffc import assistant_text, try_parse_iffc_response

SyncPost = Callable[[dict], dict]
AsyncPost = Callable[[dict], Awaitable[dict]]

# Empty-tuple key: parseable ``[]`` (end turn / no further calls this step).
ABSTAIN_KEY: tuple[tuple[str, str], ...] = ()

BON_TEMPERATURE = config.BON_TEMPERATURE

VoteKey = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BonCandidate:
    """One sampled completion after IFFC parse."""

    content: str
    parsed: list[dict] | None
    key: VoteKey | None
    index: int
    raw: dict = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class VoteResult:
    """Majority-vote outcome over a list of :class:`BonCandidate`."""

    winner: BonCandidate
    counts: dict[VoteKey, int]
    n_valid: int
    n_excluded: int
    tied_keys: tuple[VoteKey, ...]


def canonicalize_params(params: Any) -> dict:
    """Coerce a parameters blob to a dict for a stable vote key."""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            return {"__arg_parse_error__": params}
    if not isinstance(params, dict):
        return {"__arg_parse_error__": params}
    return params


def _params_json(params: dict) -> str:
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def calls_vote_key(parsed: list[dict] | None) -> VoteKey | None:
    """Return a hashable vote key, or ``None`` to exclude (unparseable).

    * ``None`` parsed → exclude from the ballot.
    * ``[]`` → :data:`ABSTAIN_KEY` (counts as a vote for ABSTAIN).
    * otherwise → ordered ``((name, canonical_json_args), ...)``.
    """
    if parsed is None:
        return None
    items: list[tuple[str, str]] = []
    for call in parsed:
        name = str(call.get("function_name") or call.get("name") or "")
        params = canonicalize_params(
            call.get("parameters", call.get("arguments", {}))
        )
        items.append((name, _params_json(params)))
    return tuple(items)


def candidate_from_content(content: str, index: int, raw: dict | None = None) -> BonCandidate:
    parsed = try_parse_iffc_response(content)
    return BonCandidate(
        content=content,
        parsed=parsed,
        key=calls_vote_key(parsed),
        index=index,
        raw=raw or {},
    )


def candidate_from_response(data: dict, index: int) -> BonCandidate:
    choices = data.get("choices") or []
    if not choices:
        return BonCandidate(
            content="", parsed=None, key=None, index=index, raw=data
        )
    content = assistant_text(choices[0].get("message") or {})
    return candidate_from_content(content, index, raw=data)


def majority_vote(candidates: list[BonCandidate]) -> VoteResult:
    """Pick a winner from sampled candidates.

    Unparseable candidates (``key is None``) are excluded. If every candidate
    is excluded, the first sample is returned so the IFFC retry path can fire.
    """
    if not candidates:
        empty = BonCandidate(content="", parsed=None, key=None, index=0)
        return VoteResult(
            winner=empty, counts={}, n_valid=0, n_excluded=0, tied_keys=()
        )

    ballots = [c for c in candidates if c.key is not None]
    n_excluded = len(candidates) - len(ballots)
    if not ballots:
        return VoteResult(
            winner=candidates[0],
            counts={},
            n_valid=0,
            n_excluded=n_excluded,
            tied_keys=(),
        )

    counts: dict[VoteKey, int] = {}
    first_index: dict[VoteKey, int] = {}
    for cand in ballots:
        key = cand.key
        assert key is not None
        counts[key] = counts.get(key, 0) + 1
        first_index.setdefault(key, cand.index)

    max_count = max(counts.values())
    tied = [key for key, n in counts.items() if n == max_count]

    if len(tied) > 1:
        non_empty = [key for key in tied if key != ABSTAIN_KEY]
        if non_empty and ABSTAIN_KEY in tied:
            tied = non_empty
        tied.sort(key=lambda key: first_index[key])

    winner_key = tied[0]
    winner = next(c for c in ballots if c.key == winner_key)
    return VoteResult(
        winner=winner,
        counts=counts,
        n_valid=len(ballots),
        n_excluded=n_excluded,
        tied_keys=tuple(tied),
    )


def resolve_bon_n(n: int | None = None) -> int:
    """Return a positive sample count (``n`` or ``config.BON_N``)."""
    value = config.BON_N if n is None else n
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"bon-n must be an integer >= 1, got {value!r}")
    return value


def build_bon_payload(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = BON_TEMPERATURE,
) -> dict:
    """IFFC+STATE chat payload for one BoN sample (no native tools)."""
    return {
        "model": model or config.MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }


def bon_payload_omits_tools(payload: dict) -> bool:
    return "tools" not in payload and "tool_choice" not in payload


def _merge_timings(responses: list[dict]) -> dict:
    totals = {
        "prompt_n": 0,
        "prompt_ms": 0.0,
        "predicted_n": 0,
        "predicted_ms": 0.0,
    }
    for data in responses:
        src = data.get("timings") or {}
        for key in totals:
            totals[key] += src.get(key, 0) or 0
    return totals


def _assistant_response(content: str, timings: dict) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "timings": timings,
    }


def select_winner_response(responses: list[dict]) -> tuple[dict, int]:
    """Vote over raw chat-completions JSON; return a synthetic single choice."""
    candidates = [candidate_from_response(data, i) for i, data in enumerate(responses)]
    result = majority_vote(candidates)
    timings = _merge_timings(responses)
    return _assistant_response(result.winner.content, timings), len(responses)


def complete_step(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    post: SyncPost,
    model: str,
    n: int | None = None,
    temperature: float = BON_TEMPERATURE,
) -> tuple[dict, int]:
    """Sample *N* IFFC completions, vote, return a synthetic winner response.

    *tools* is ignored (IFFC+STATE never sends ``tools`` / ``tool_choice``).
    Losing samples are not appended to *messages*.
    """
    del tools
    n = resolve_bon_n(n)
    payload = build_bon_payload(messages, model=model, temperature=temperature)
    responses = [post(payload) for _ in range(n)]
    return select_winner_response(responses)


async def complete_step_async(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    post: AsyncPost,
    model: str,
    n: int | None = None,
    temperature: float = BON_TEMPERATURE,
) -> tuple[dict, int]:
    """Async counterpart of :func:`complete_step` (N samples may run concurrently)."""
    del tools
    n = resolve_bon_n(n)
    payload = build_bon_payload(messages, model=model, temperature=temperature)
    responses = list(await asyncio.gather(*[post(payload) for _ in range(n)]))
    return select_winner_response(responses)
