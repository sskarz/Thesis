"""Multi-turn loop: emit command → parse → execute → observe → repeat."""

from __future__ import annotations

from .backend import Backend
from .dialect import SYSTEM_PROMPT, repair_message
from .parser import parse_command
from .tasks import ToyTask
from .types import EpisodeResult, Observation, ParseError, ToolCall, Turn


def run_episode(
    task: ToyTask,
    model,
    backend: Backend,
    max_turns: int | None = None,
    repair: bool = True,
) -> EpisodeResult:
    """Run consecutive command calling until `done` or turn cap.

    Scoring is always on final backend state (outcome), not on the command AST.
    """
    limit = max_turns if max_turns is not None else task.max_turns
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.instruction},
    ]
    turns: list[Turn] = []
    stopped = "max_turns"

    for _ in range(limit):
        raw = model.generate(messages)
        parsed = parse_command(raw)

        if isinstance(parsed, ParseError):
            obs = Observation(ok=False, stderr=parsed.message, exit_code=2)
            turns.append(
                Turn(
                    raw=raw,
                    call=None,
                    parse_error=parsed,
                    observation=obs,
                    repaired=repair,
                )
            )
            messages.append({"role": "assistant", "content": raw})
            if repair:
                messages.append({"role": "user", "content": repair_message(parsed.message)})
                continue
            stopped = "abort"
            break

        assert isinstance(parsed, ToolCall)
        messages.append({"role": "assistant", "content": parsed.raw or raw})

        if parsed.name == "done":
            turns.append(
                Turn(raw=raw, call=parsed, parse_error=None, observation=Observation(ok=True))
            )
            stopped = "done"
            break

        obs = backend.execute(parsed)
        turns.append(Turn(raw=raw, call=parsed, parse_error=None, observation=obs))
        feedback = obs.render()
        if not feedback:
            feedback = "(ok, no output)"
        messages.append({"role": "user", "content": feedback})

    checks = task.evaluate(backend)
    success = bool(checks) and all(c.passed for c in checks)
    return EpisodeResult(
        task_id=task.id,
        success=success,
        turns=turns,
        checks=checks,
        stopped=stopped,
    )
