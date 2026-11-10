"""Offline smoke for the DeepSeek teacher adapter (no network, no real key).

    uv run python -m benchmark.smoke_teacher_api

Uses a local scripted HTTP server and a fake key. Checks auth header, the
thinking-off flag, reasoning stripping, retry/backoff, no-retry on 400, rate
limiting, per-call/per-episode logs, key never written anywhere, and a full
``generate_for_entry`` run routed through the adapter.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from . import config, generate_multi_turn, teacher_api
from .smoke_iffc import WEATHER_TOOL

FAKE_KEY = "sk-smoke-NOT-A-REAL-KEY-123456"
VERSION = "DeepSeek-V4.1-Flash-SMOKE"


def _ok(content: str, reasoning: str | None = None) -> tuple[int, dict, dict]:
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return 200, {
        "model": VERSION,
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7,
                  "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60},
    }, {}


class _H(BaseHTTPRequestHandler):
    script: list = []
    seen: list = []
    idx = 0

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                "body": json.loads(raw or b"{}")})
        status, body, hdrs = (type(self).script[type(self).idx]
                              if type(self).idx < len(type(self).script)
                              else (500, {"error": "script exhausted"}, {}))
        type(self).idx += 1
        out = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a) -> None:
        return


def _serve(script: list):
    _H.script, _H.seen, _H.idx = script, [], 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _client(base: str, log_dir: Path | None = None, **kw) -> teacher_api.TeacherClient:
    cfg = teacher_api.TeacherConfig(base_url=base, log_dir=log_dir, **kw)
    return teacher_api.TeacherClient(cfg, sleep=lambda s: None)


def check_config_errors() -> None:
    os.environ.pop(teacher_api.DEFAULT_KEY_ENV, None)
    try:
        teacher_api.TeacherClient()
        raise AssertionError("expected TeacherConfigError without key")
    except teacher_api.TeacherConfigError:
        pass
    os.environ[teacher_api.DEFAULT_KEY_ENV] = FAKE_KEY
    try:
        teacher_api.TeacherClient(teacher_api.TeacherConfig(thinking="enabled"))
        raise AssertionError("thinking must be refused")
    except teacher_api.TeacherConfigError:
        pass
    c = teacher_api.TeacherClient()
    assert FAKE_KEY not in repr(c) and FAKE_KEY not in str(c.__dict__.keys())
    assert teacher_api.get_active() is None


def check_request_shape_retry_and_strip(tmp: Path) -> None:
    srv, base = _serve([
        (429, {"error": "busy"}, {"Retry-After": "0"}),
        (503, {"error": "down"}, {}),
        _ok('[{"function_name": "get_weather", "parameters": {"city": "SF"}}]',
            reasoning="secret chain of thought"),
    ])
    try:
        c = _client(base, tmp)
        payload = {"model": "ignored", "temperature": 0.0, "tools": [WEATHER_TOOL],
                   "messages": [{"role": "user", "content": "hi"}]}
        with c.episode("ep-shape", task="x"):
            data = c.post_chat(payload)
        req = _H.seen[-1]
        assert len(_H.seen) == 3 and c.stats["retries"] == 2, (len(_H.seen), c.stats)
        assert req["path"] == "/chat/completions"
        assert req["auth"] == f"Bearer {FAKE_KEY}"
        b = req["body"]
        assert b["model"] == "deepseek-flash" and b["thinking"] == {"type": "disabled"}
        assert "tools" not in b and "tool_choice" not in b and b["stream"] is False
        assert b["temperature"] == 0.0 and b["max_tokens"] == 1024
        assert "reasoning_content" not in data["choices"][0]["message"]
        assert c.stats["reasoning_stripped"] == 1
        calls = [json.loads(l) for l in (tmp / "calls.jsonl").read_text().splitlines()]
        assert calls[0]["api_model"] == VERSION and calls[0]["episode_id"] == "ep-shape"
        assert calls[0]["attempts"] == 3 and calls[0]["usage"]["prompt_tokens"] == 100
        assert "secret chain" not in json.dumps(calls)
        ep = [json.loads(l) for l in (tmp / "episodes.jsonl").read_text().splitlines()][0]
        assert ep["api_models"] == [VERSION] and ep["completion_tokens"] == 7
        assert ep["verified"] is None and ep["requested_model"] == "deepseek-flash"
        for p in tmp.iterdir():
            assert FAKE_KEY not in p.read_text(), f"key leaked into {p.name}"
    finally:
        srv.shutdown(); srv.server_close()


def check_openrouter_shape() -> None:
    os.environ[teacher_api.OPENROUTER_KEY_ENV] = FAKE_KEY
    srv, base = _serve([_ok("[]")])
    try:
        cfg = teacher_api.TeacherConfig.openrouter(base_url=base)
        c = teacher_api.TeacherClient(cfg, sleep=lambda s: None)
        c.post_chat({"messages": [{"role": "user", "content": "hi"}]})
        r = _H.seen[0]
        b = r["body"]
        assert r["auth"] == f"Bearer {FAKE_KEY}"
        assert b["model"] == "deepseek/deepseek-v4.1-flash-1017"
        assert b["reasoning"] == {"effort": "none"} and "thinking" not in b
        assert b["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False,
                                 "data_collection": "allow"}
    finally:
        os.environ.pop(teacher_api.OPENROUTER_KEY_ENV, None)
        srv.shutdown(); srv.server_close()


def check_no_retry_on_400_and_exhaustion() -> None:
    srv, base = _serve([(400, {"error": "bad"}, {})])
    try:
        c = _client(base)
        try:
            c.post_chat({"messages": []})
            raise AssertionError("expected HTTPError")
        except requests.HTTPError as e:
            assert e.response.status_code == 400
        assert len(_H.seen) == 1 and c.stats["retries"] == 0
    finally:
        srv.shutdown(); srv.server_close()
    srv, base = _serve([(500, {"error": "x"}, {})] * 5)
    try:
        c = _client(base, max_retries=2)
        try:
            c.post_chat({"messages": []})
            raise AssertionError("expected failure after retries")
        except requests.HTTPError:
            pass
        assert len(_H.seen) == 3 and c.stats["failures"] == 1
    finally:
        srv.shutdown(); srv.server_close()


def check_local_backend(tmp: Path) -> None:
    srv, base = _serve([_ok("[]", reasoning="student thoughts")])
    try:
        cfg = teacher_api.TeacherConfig.local(base_url=base, log_dir=tmp)
        c = teacher_api.TeacherClient(cfg, sleep=lambda s: None)
        payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
        with c.episode("loc-1"):
            data = c.post_chat(payload)
        req = _H.seen[0]
        assert req["auth"] is None and req["body"] == payload, req
        assert data["choices"][0]["message"]["reasoning_content"] == "student thoughts"
        rec = json.loads((tmp / "calls.jsonl").read_text().splitlines()[0])
        assert rec["reasoning_content"] == "student thoughts" and rec["response_text"] == "[]"
    finally:
        srv.shutdown(); srv.server_close()


def check_budget_cap() -> None:
    ok = _ok("[]")
    ok[1]["usage"]["cost"] = 0.4
    srv, base = _serve([ok, ok, ok])
    try:
        c = _client(base, max_cost_usd=0.5)
        c.post_chat({"messages": []})
        c.post_chat({"messages": []})  # spent 0.8 >= 0.5 after this one
        try:
            c.post_chat({"messages": []})
            raise AssertionError("expected BudgetExceeded")
        except teacher_api.BudgetExceeded:
            pass
        assert len(_H.seen) == 2 and abs(c.stats["cost_usd"] - 0.8) < 1e-9
    finally:
        srv.shutdown(); srv.server_close()


def check_rate_limit() -> None:
    srv, base = _serve([_ok("[]")] * 3)
    sleeps: list[float] = []
    try:
        cfg = teacher_api.TeacherConfig(base_url=base, min_interval_s=5.0)
        c = teacher_api.TeacherClient(cfg, sleep=sleeps.append)
        for _ in range(3):
            c.post_chat({"messages": []})
        # sleep is stubbed so the clock does not advance: waits stack at ~5 s, ~10 s
        assert len(sleeps) == 2 and 4.5 < sleeps[0] <= 5.0 and 9.5 < sleeps[1] <= 10.0, sleeps
    finally:
        srv.shutdown(); srv.server_close()


def check_full_loop(tmp: Path) -> None:
    """generate_for_entry end to end through the adapter; reasoning never in history."""
    srv, base = _serve([
        _ok('[{"function_name": "get_weather", "parameters": {"city": "San Francisco"}}]',
            reasoning="private thoughts"),
        _ok("[]"),
    ])
    prev_exec = generate_multi_turn.execute_multi_turn_func_call
    try:
        generate_multi_turn.execute_multi_turn_func_call = lambda **kw: (["72 F"], None)
        teacher_api.install(_client(base, tmp / "loop"))
        client = teacher_api.get_active()
        entry = {"id": "smoke-teacher-1", "initial_config": {}, "involved_classes": [],
                 "question": [[{"role": "user", "content": "Weather in San Francisco?"}]]}
        with client.episode("smoke-teacher-1"):
            turns, timing = generate_multi_turn.generate_for_entry(
                entry, "multi_turn_base", [WEATHER_TOOL], fc_mode=config.FC_IFFC)
        assert turns == [[["get_weather(city='San Francisco')"]]], turns
        assert timing["api_calls"] == 2
        second = _H.seen[1]["body"]["messages"]
        assistant = [m for m in second if m["role"] == "assistant"]
        assert assistant and "private thoughts" not in json.dumps(second)
        assert all(r["body"]["thinking"] == {"type": "disabled"} for r in _H.seen)
        ep = json.loads((tmp / "loop" / "episodes.jsonl").read_text().splitlines()[0])
        assert ep["calls"] == 2 and ep["prompt_tokens"] == 200
    finally:
        teacher_api.install(None)
        generate_multi_turn.execute_multi_turn_func_call = prev_exec
        srv.shutdown(); srv.server_close()
    assert teacher_api.get_active() is None


def main() -> None:
    prev = os.environ.get(teacher_api.DEFAULT_KEY_ENV)
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            check_config_errors()
            check_request_shape_retry_and_strip(root / "a")
            check_openrouter_shape()
            check_no_retry_on_400_and_exhaustion()
            check_budget_cap()
            check_local_backend(root / 'loc')
            check_rate_limit()
            check_full_loop(root / "b")
    finally:
        if prev is None:
            os.environ.pop(teacher_api.DEFAULT_KEY_ENV, None)
        else:
            os.environ[teacher_api.DEFAULT_KEY_ENV] = prev
    print("smoke_teacher_api: OK")


if __name__ == "__main__":
    main()
