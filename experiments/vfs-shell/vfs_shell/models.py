"""Model adapters: scripted oracle (tests) and llama-server stub (later)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class Model(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class ScriptedModel:
    """Deterministic stand-in: emit a fixed command list, then `done`."""

    def __init__(self, commands: list[str]) -> None:
        self.commands = list(commands)
        self.index = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.index >= len(self.commands):
            return "done"
        cmd = self.commands[self.index]
        self.index += 1
        return cmd


class LlamaServerModel:
    """OpenAI-compatible llama-server client.

    Intentionally does **not** send a `tools` array. The action space is the
    constrained shell dialect in the system prompt (see dialect.SYSTEM_PROMPT).
    Plug this in once a local server is up:

        ./llama.cpp/build/bin/llama-server --jinja -fa on ... --port 8080 --temp 0.0

    Default base URL matches `benchmark.config.SERVER_URL` + `/v1`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "local",
        temperature: float = 0.0,
        timeout_s: float = 120.0,
    ) -> None:
        env_url = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
        raw = (base_url or env_url).rstrip("/")
        if raw.endswith("/v1"):
            self.base_url = raw
        else:
            self.base_url = raw + "/v1"
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"llama-server not reachable at {self.base_url} ({exc}). "
                "Start llama-server or use ScriptedModel for toy tasks."
            ) from exc

        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        # Some templates put extra reasoning alongside content; ignore it.
        return content or ""
