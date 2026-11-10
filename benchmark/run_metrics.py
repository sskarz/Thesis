"""Per-task harness counters (R81). Logging only: never changes prompt text.

Enabled when ``RUN_METRICS_JSONL`` is set; one JSON line per entry with
format errors (parse retries + final unparseable), invalid tool names,
invalid args (unknown key, missing required, or non-dict parameters),
prefill tokens and per-request latency (prompt_ms ~ TTFT, wall ms).
"""

from __future__ import annotations

import json
import os
import threading
import time

_LOCK = threading.Lock()


def enabled() -> bool:
    return bool(os.environ.get("RUN_METRICS_JSONL"))


def _schemas(active_tools: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in active_tools or []:
        f = t.get("function", t)
        out[f.get("name", "")] = f.get("parameters") or {}
    return out


class TaskMetrics:
    def __init__(self, entry_id: str, fc_mode: str) -> None:
        self.entry_id = entry_id
        self.fc_mode = fc_mode
        self.requests: list[dict] = []
        self.format_retries = 0
        self.format_unparseable = 0
        self.calls = 0
        self.invalid_name = 0
        self.invalid_args = 0
        self.invalid_examples: list[str] = []
        self._t0: float | None = None

    def start_request(self) -> None:
        self._t0 = time.perf_counter()

    def end_request(self, data: dict, turn: int, step: int) -> None:
        wall = (time.perf_counter() - self._t0) * 1000 if self._t0 else None
        t = data.get("timings", {}) or {}
        self.requests.append({
            "turn": turn,
            "step": step,
            "prompt_n": t.get("prompt_n"),
            "cache_n": t.get("cache_n"),
            "prompt_ms": t.get("prompt_ms"),
            "predicted_n": t.get("predicted_n"),
            "predicted_ms": t.get("predicted_ms"),
            "wall_ms": round(wall, 2) if wall is not None else None,
        })
        self._t0 = None

    def on_step(self, action: str, flag: str | None) -> None:
        if action == "retry":
            self.format_retries += 1
        if flag and flag.endswith("_unparseable"):
            self.format_unparseable += 1

    def check_calls(self, parsed: list | None, active_tools: list[dict]) -> None:
        schemas = _schemas(active_tools)
        for call in parsed or []:
            self.calls += 1
            if not isinstance(call, dict):
                self.invalid_name += 1
                continue
            name = str(call.get("function_name") or call.get("name") or "")
            if name not in schemas:
                self.invalid_name += 1
                if len(self.invalid_examples) < 5:
                    self.invalid_examples.append(f"name:{name}")
                continue
            args = call.get("parameters", call.get("arguments", {}))
            sch = schemas[name]
            props = set((sch.get("properties") or {}).keys())
            req = set(sch.get("required") or [])
            if not isinstance(args, dict):
                bad = True
            else:
                bad = bool(set(args) - props) or bool(req - set(args))
            if bad:
                self.invalid_args += 1
                if len(self.invalid_examples) < 5:
                    self.invalid_examples.append(f"args:{name}")

    def write(self) -> None:
        path = os.environ.get("RUN_METRICS_JSONL")
        if not path:
            return
        rec = {
            "id": self.entry_id,
            "fc_mode": self.fc_mode,
            "n_requests": len(self.requests),
            "format_retries": self.format_retries,
            "format_unparseable": self.format_unparseable,
            "format_errors": self.format_retries + self.format_unparseable,
            "calls": self.calls,
            "invalid_name": self.invalid_name,
            "invalid_args": self.invalid_args,
            "invalid_examples": self.invalid_examples,
            "first_prompt_n": self.requests[0]["prompt_n"] if self.requests else None,
            "first_prompt_ms": self.requests[0]["prompt_ms"] if self.requests else None,
            "requests": self.requests,
        }
        with _LOCK:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
