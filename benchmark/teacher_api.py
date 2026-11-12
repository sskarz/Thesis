"""Teacher (DeepSeek) chat adapter for the multi-turn loop. Distillation only.

``generate_multi_turn`` posts to ``config.SERVER_URL`` with no auth. For
teacher-student data generation the same loop must talk to a hosted
OpenAI-compatible API instead. This module is a drop-in replacement for the
HTTP post: install a :class:`TeacherClient` and ``_post_chat_sync`` /
``_post_chat_async`` delegate to it. Nothing changes when none is installed,
so every existing benchmark run is unaffected.

What it does
------------
* Base URL, model name and key come from config / the environment. The key is
  read only from ``DEEPSEEK_API_KEY`` (name overridable), kept in a private
  attribute, sent only in the ``Authorization`` header, and never logged,
  printed, repr'd or written to disk.
* Reasoning is switched off with ``{"thinking": {"type": "disabled"}}``
  on the DeepSeek API (thinking is ON by default), or ``{"reasoning":
  {"effort": "none"}}`` via OpenRouter, where the request is also pinned to
  DeepSeek's own endpoint with no fallbacks. Defence in depth:
  if ``reasoning_content`` comes back anyway it is removed from the message,
  because ``iffc.assistant_text`` would otherwise prepend it to the reply and
  it would leak into history and training targets. A counter records it.
* Retries with exponential backoff + jitter on 429 / 5xx / timeouts /
  connection errors (honours ``Retry-After``). Other 4xx raise
  ``requests.HTTPError`` unchanged, so the loop's 400 handling still works.
* Client-side rate limit: max concurrent requests plus a minimum interval
  between request starts. Defaults are conservative, far under the account
  limit in DeepSeek's rate-limit page.
* Logs every call (messages sent, text returned, token usage, finish reason,
  latency, attempts, the exact ``model`` string the API reported) to
  ``<log_dir>/calls.jsonl`` and one summary per episode to
  ``<log_dir>/episodes.jsonl``. ``verified`` is left ``None``; the verifier
  fills it. Keep ``log_dir`` under ``sft/data/`` (gitignored).

No network call happens at import time or when constructing a client.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Pinned dated version (the un-dated id is an alias that can move).
OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash-1017"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
# DeepSeek's own endpoint ("deepseek") trains on prompts and is excluded by
# Sans's OpenRouter privacy setting (404 "paid model training violation"), so
# the pilot pins one non-training third-party host (fp8 weights). One provider
# for the whole run keeps rows consistent; it is logged on every call.
OPENROUTER_PROVIDER = "deepinfra"
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


class TeacherConfigError(RuntimeError):
    """Missing or unusable teacher configuration (e.g. no API key)."""


class BudgetExceeded(RuntimeError):
    """The client's hard dollar cap was reached; no further calls are sent."""


@dataclass
class TeacherConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    key_env: str = DEFAULT_KEY_ENV
    backend: str = "deepseek"  # "deepseek" (direct) or "openrouter"
    # OpenRouter only: provider slugs allowed (no fallbacks), and data policy.
    # "allow" is the default because DeepSeek's endpoint may train on prompts
    # and "deny" can leave no endpoint; tasks are synthetic, nothing private.
    openrouter_only: tuple[str, ...] = (OPENROUTER_PROVIDER,)
    data_collection: str = "allow"
    thinking: str = "disabled"  # "disabled" is the only value the student format allows
    temperature: float | None = 0.0
    max_tokens: int = 1024
    timeout_s: float = 180.0
    max_retries: int = 6
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    max_concurrency: int = 4
    min_interval_s: float = 0.0
    log_dir: Path | None = None
    # Hard stop. Spend is read from the API's own ``usage.cost`` (OpenRouter);
    # checked before every request. None = no cap (do not use for paid runs).
    max_cost_usd: float | None = None

    @classmethod
    def local(cls, **kw) -> "TeacherConfig":
        """A local llama-server (OpenAI-compatible) used as the STUDENT, with full recording.

        Requests are sent exactly as the benchmark harness would send them (no auth, no
        extra fields, reasoning_content left in the message so history matches a normal run).
        """
        base = dict(backend="local", base_url="http://127.0.0.1:8080/v1", model="local",
                    key_env="", temperature=None, max_retries=2, timeout_s=900.0)
        base.update(kw)
        return cls(**base)

    @classmethod
    def openrouter(cls, **kw) -> "TeacherConfig":
        """DeepSeek V4.1 Flash via OpenRouter, pinned to one provider."""
        base = dict(
            backend="openrouter",
            base_url=OPENROUTER_BASE_URL,
            model=OPENROUTER_MODEL,
            key_env=OPENROUTER_KEY_ENV,
        )
        base.update(kw)
        return cls(**base)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_EPISODE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "teacher_episode", default=None
)
_ACTIVE: "TeacherClient | None" = None


def get_active() -> "TeacherClient | None":
    return _ACTIVE


def install(client: "TeacherClient | None") -> None:
    """Route ``generate_multi_turn`` chat posts through *client* (None = off)."""
    global _ACTIVE
    _ACTIVE = client


class TeacherClient:
    def __init__(
        self,
        cfg: TeacherConfig | None = None,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg or TeacherConfig()
        key = os.environ.get(self.cfg.key_env, "").strip() if self.cfg.key_env else ""
        if not key and self.cfg.backend != "local":
            raise TeacherConfigError(
                f"{self.cfg.key_env} is not set. Export it in your shell; the "
                "adapter never reads a key from a file or argument."
            )
        self.__key = key  # name-mangled; never logged
        self._session = session or requests.Session()
        self._sleep = sleep
        self._gate = threading.BoundedSemaphore(max(1, self.cfg.max_concurrency))
        self._interval_lock = threading.Lock()
        self._next_start = 0.0
        self._log_lock = threading.Lock()
        self._episodes: dict[str, dict] = {}
        self.stats = {
            "calls": 0,
            "retries": 0,
            "reasoning_stripped": 0,
            "failures": 0,
            "cost_usd": 0.0,
        }
        if self.cfg.thinking != "disabled":
            raise TeacherConfigError(
                "thinking must be 'disabled': the student format has no "
                "reasoning text (handoff section 5)."
            )

    def __repr__(self) -> str:  # never include the key
        return (
            f"TeacherClient(base_url={self.cfg.base_url!r}, "
            f"model={self.cfg.model!r}, key=<redacted>)"
        )

    # ------------------------------------------------------------------ request
    def _body(self, payload: dict) -> dict:
        body = copy.deepcopy(payload)
        if self.cfg.backend == "local":
            return body  # identical to what the harness posts to llama-server
        body["model"] = self.cfg.model
        if self.cfg.backend == "openrouter":
            # OpenRouter docs: effort "none" disables reasoning entirely
            # (exclude=true would still bill reasoning tokens).
            body["reasoning"] = {"effort": "none"}
            body["provider"] = {
                "only": list(self.cfg.openrouter_only),
                "allow_fallbacks": False,
                "data_collection": self.cfg.data_collection,
            }
        else:
            body["thinking"] = {"type": "disabled"}
        body.setdefault("max_tokens", self.cfg.max_tokens)
        if self.cfg.temperature is not None:
            body["temperature"] = body.get("temperature", self.cfg.temperature)
        ep = _EPISODE.get()
        if ep is not None and ep.get("temperature") is not None:
            body["temperature"] = ep["temperature"]  # sampling episodes (pass@k)
        body.pop("tools", None)  # JSON-text loop only; never native tools
        body.pop("tool_choice", None)
        body["stream"] = False
        return body

    def _throttle(self) -> None:
        if self.cfg.min_interval_s <= 0:
            return
        with self._interval_lock:
            now = time.monotonic()
            wait = self._next_start - now
            self._next_start = max(now, self._next_start) + self.cfg.min_interval_s
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.cfg.backoff_max_s)
            except ValueError:
                pass
        base = min(self.cfg.backoff_base_s * (2**attempt), self.cfg.backoff_max_s)
        return base * (0.5 + random.random() / 2)

    def post_chat(self, payload: dict) -> dict:
        """POST one chat completion. Returns a llama-server-shaped dict."""
        cap = self.cfg.max_cost_usd
        if cap is not None and self.stats["cost_usd"] >= cap:
            raise BudgetExceeded(
                f"cost cap ${cap:.2f} reached (spent ${self.stats['cost_usd']:.4f})"
            )
        body = self._body(payload)
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.__key:
            headers["Authorization"] = f"Bearer {self.__key}"
        t0 = time.monotonic()
        last_err: Exception | None = None
        with self._gate:
            for attempt in range(self.cfg.max_retries + 1):
                self._throttle()
                retry_after = None
                try:
                    resp = self._session.post(
                        url, json=body, headers=headers, timeout=self.cfg.timeout_s
                    )
                    if resp.status_code in RETRY_STATUS:
                        retry_after = resp.headers.get("Retry-After")
                        last_err = requests.HTTPError(
                            f"{resp.status_code} from teacher API", response=resp
                        )
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                        return self._finish(payload, data, attempt, t0)
                except (requests.Timeout, requests.ConnectionError) as e:
                    last_err = e
                if attempt < self.cfg.max_retries:
                    self.stats["retries"] += 1
                    self._sleep(self._backoff(attempt, retry_after))
        self.stats["failures"] += 1
        assert last_err is not None
        raise last_err

    async def post_chat_async(self, payload: dict) -> dict:
        import asyncio

        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(None, ctx.run, self.post_chat, payload)

    # ------------------------------------------------------------------ results
    def _finish(self, payload: dict, data: dict, attempt: int, t0: float) -> dict:
        choices = data.get("choices") or []
        stripped = 0
        for ch in choices:
            msg = ch.get("message") or {}
            if self.cfg.backend == "local":
                rc = msg.get("reasoning_content")  # student: keep it, the harness persists it
                reasoning_text = rc
            else:
                rc = msg.pop("reasoning_content", None)
                reasoning_text = None
            if rc and self.cfg.backend != "local":
                stripped += len(rc)
        if stripped:
            self.stats["reasoning_stripped"] += 1
        usage = {k: (data.get("usage") or {}).get(k) for k in _USAGE_KEYS}
        usage["cost"] = (data.get("usage") or {}).get("cost")
        text = ""
        finish = None
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
            finish = choices[0].get("finish_reason")
        ep = _EPISODE.get()
        record = {
            "ts": _utcnow(),
            "episode_id": ep["id"] if ep else None,
            "requested_model": self.cfg.model,
            "api_model": data.get("model"),  # exact version string the API reports
            "api_provider": data.get("provider"),  # OpenRouter: who served it
            "cost_usd": (data.get("usage") or {}).get("cost"),
            "system_fingerprint": data.get("system_fingerprint"),
            "usage": usage,
            "finish_reason": finish,
            "latency_s": round(time.monotonic() - t0, 3),
            "attempts": attempt + 1,
            "reasoning_chars_stripped": stripped,
            "reasoning_content": (choices[0].get("message") or {}).get("reasoning_content") if (choices and self.cfg.backend == "local") else None,
            "messages": payload.get("messages"),
            "response_text": text,
        }
        self.stats["calls"] += 1
        with self._log_lock:
            self.stats["cost_usd"] += usage.get("cost") or 0.0
        self._log("calls.jsonl", record)
        if ep is not None:
            with self._log_lock:
                s = ep["summary"]
                s["calls"] += 1
                s["api_models"].add(data.get("model") or "?")
                s["api_providers"].add(str(data.get("provider") or "?"))
                s["cost_usd"] += usage.get("cost") or 0.0
                s["reasoning_chars_stripped"] += stripped
                for k in ("prompt_tokens", "completion_tokens"):
                    s[k] += usage.get(k) or 0
                for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                    s[k] += usage.get(k) or 0
        # Same shape the llama-server loop reads (no ``timings`` is fine: .get).
        return data

    def _log(self, name: str, record: dict) -> None:
        if self.cfg.log_dir is None:
            return
        line = json.dumps(record, ensure_ascii=False)
        with self._log_lock:
            Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(self.cfg.log_dir) / name, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ----------------------------------------------------------------- episodes
    @contextlib.contextmanager
    def episode(self, episode_id: str, **meta: Any):
        """Tag every call made inside the block (this thread/task) as one episode.

        On exit writes one summary row to ``episodes.jsonl``: requested model,
        every API model string seen, date, token counts, ``verified=None``.
        """
        summary = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "reasoning_chars_stripped": 0,
            "cost_usd": 0.0,
            "api_models": set(),
            "api_providers": set(),
        }
        ep = {"id": episode_id, "summary": summary, "temperature": meta.get("temperature")}
        token = _EPISODE.set(ep)
        t0 = time.monotonic()
        try:
            yield ep
        finally:
            _EPISODE.reset(token)
            row = {
                "episode_id": episode_id,
                "date": _utcnow(),
                "requested_model": self.cfg.model,
                **{k: v for k, v in summary.items()
                   if k not in ("api_models", "api_providers")},
                "api_models": sorted(summary["api_models"]),
                "api_providers": sorted(summary["api_providers"]),
                "wall_s": round(time.monotonic() - t0, 3),
                "verified": None,
                **meta,
            }
            with self._log_lock:
                self._episodes[episode_id] = row
            self._log("episodes.jsonl", row)
