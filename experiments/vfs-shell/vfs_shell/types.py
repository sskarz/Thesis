"""Shared types for the constrained shell dialect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RedirectOp = Literal[">", ">>"]


@dataclass(frozen=True)
class Redirect:
    op: RedirectOp
    path: str


@dataclass(frozen=True)
class ToolCall:
    """Structured action produced by the allowlisted parser."""

    name: str
    flags: frozenset[str] = frozenset()
    options: tuple[tuple[str, str], ...] = ()
    positional: tuple[str, ...] = ()
    redirect: Redirect | None = None
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "flags": sorted(self.flags),
            "options": [list(pair) for pair in self.options],
            "positional": list(self.positional),
            "redirect": (
                {"op": self.redirect.op, "path": self.redirect.path}
                if self.redirect
                else None
            ),
        }


@dataclass(frozen=True)
class ParseError:
    message: str
    raw: str
    kind: str = "parse"  # parse | deny | unsafe


@dataclass(frozen=True)
class Observation:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    def render(self) -> str:
        if self.ok:
            return self.stdout if self.stdout else ""
        text = self.stderr or "command failed"
        return f"(exit {self.exit_code}) {text}"


@dataclass
class Turn:
    raw: str
    call: ToolCall | None
    parse_error: ParseError | None
    observation: Observation
    repaired: bool = False


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    turns: list[Turn] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    stopped: str = "max_turns"  # done | max_turns | abort

    @property
    def parse_errors(self) -> int:
        return sum(1 for t in self.turns if t.parse_error is not None)

    @property
    def executed_calls(self) -> int:
        return sum(1 for t in self.turns if t.call is not None and t.call.name != "done")
