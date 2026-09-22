"""Smoke IFFC parser + a 2-step multi-turn loop against a mock (or live) server.

Offline (no llama-server; used in CI):

    uv run python -m benchmark.smoke_iffc

Live llama-server at ``config.SERVER_URL``:

    uv run python -m benchmark.smoke_iffc --live

IFFC = arXiv:2608.22472 Appendix B JSON-in-text; NFC = native tools.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from . import generate_multi_turn
from .iffc import (
    IFFC_SYSTEM_PROMPT,
    build_iffc_system_prompt,
    iffc_calls_to_func_strings,
    parse_iffc_response,
    try_parse_iffc_response,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def run_fixture_checks() -> None:
    """Assert parser + func-string conversion on known fixtures (no network)."""
    assert "{available_functions}" in IFFC_SYSTEM_PROMPT
    prompt = build_iffc_system_prompt([WEATHER_TOOL])
    assert "{available_functions}" not in prompt
    assert "get_weather" in prompt

    empty = parse_iffc_response("[]")
    assert empty == []

    one = parse_iffc_response(
        '[{"function_name": "mv", "parameters": {"source": "a", "destination": "b"}}]'
    )
    assert one == [
        {"function_name": "mv", "parameters": {"source": "a", "destination": "b"}}
    ]
    assert iffc_calls_to_func_strings(one) == ["mv(source='a', destination='b')"]

    multi = parse_iffc_response(
        '[{"function_name": "cd", "parameters": {"folder": "docs"}},'
        ' {"function_name": "ls", "parameters": {}}]'
    )
    assert [c["function_name"] for c in multi] == ["cd", "ls"]
    assert iffc_calls_to_func_strings(multi) == ["cd(folder='docs')", "ls()"]

    fenced = parse_iffc_response(
        'Sure:\n```json\n[{"function_name": "ping", "parameters": {}}]\n```\n'
    )
    assert fenced == [{"function_name": "ping", "parameters": {}}]

    noisy = parse_iffc_response(
        'I will call the tool now.\n[{"function_name": "ls", "parameters": {}}]\nDone.'
    )
    assert noisy == [{"function_name": "ls", "parameters": {}}]

    assert try_parse_iffc_response("not json at all") is None
    assert parse_iffc_response("not json at all") == []

    xlam_wrap = parse_iffc_response(
        '{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"location": "New York", "unit": "fahrenheit"}}]}'
    )
    assert xlam_wrap == [
        {
            "function_name": "get_weather",
            "parameters": {"location": "New York", "unit": "fahrenheit"},
        }
    ]
    assert parse_iffc_response('{"tool_calls": []}') == []
    assert parse_iffc_response(
        '{"function_calls": [{"function_name": "ls", "parameters": {}}]}'
    ) == [{"function_name": "ls", "parameters": {}}]


class _ScriptedChatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat completions endpoint with scripted replies."""

    contents: list[str] = []
    payloads: list[dict] = []
    _idx = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        if raw:
            try:
                type(self).payloads.append(json.loads(raw.decode()))
            except json.JSONDecodeError:
                type(self).payloads.append({})
        if self._idx >= len(self.contents):
            body = json.dumps({"error": "no more scripted responses"}).encode()
            self.send_response(500)
        else:
            content = self.contents[self._idx]
            type(self)._idx += 1
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "timings": {
                        "prompt_n": 10,
                        "prompt_ms": 5,
                        "predicted_n": 4,
                        "predicted_ms": 8,
                    },
                }
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _start_scripted_server(contents: list[str]) -> tuple[ThreadingHTTPServer, str]:
    _ScriptedChatHandler.contents = contents
    _ScriptedChatHandler.payloads = []
    _ScriptedChatHandler._idx = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def run_mock_multi_step_loop() -> None:
    """Two IFFC steps against a scripted chat-completions server."""
    contents = [
        '[{"function_name": "get_weather", "parameters": {"city": "San Francisco"}}]',
        "[]",
    ]
    server, base_url = _start_scripted_server(contents)
    prev_url = config.SERVER_URL
    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    try:
        config.SERVER_URL = base_url
        generate_multi_turn.execute_multi_turn_func_call = (
            lambda **kwargs: (["72 F"], None)
        )
        entry = {
            "id": "smoke-iffc-1",
            "initial_config": {},
            "involved_classes": [],
            "question": [
                [
                    {
                        "role": "user",
                        "content": "What is the weather in San Francisco?",
                    }
                ]
            ],
        }
        turn_results, timing = generate_multi_turn.generate_for_entry(
            entry,
            "multi_turn_base",
            [WEATHER_TOOL],
            fc_mode=config.FC_IFFC,
        )
        assert turn_results == [[["get_weather(city='San Francisco')"]]], turn_results
        assert timing["id"] == "smoke-iffc-1"
        assert timing["api_calls"] == 2
        assert timing["category"] == "multi_turn_base"
    finally:
        config.SERVER_URL = prev_url
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        server.shutdown()
        server.server_close()


def run_mock_strip_empty_assistant_history() -> None:
    """Two BFCL user turns: flag on drops assistant ``[]``; flag off keeps it."""
    contents = [
        '[{"function_name": "get_weather", "parameters": {"city": "San Francisco"}}]',
        "[]",
        "[]",
    ]
    entry = {
        "id": "smoke-iffc-strip",
        "initial_config": {},
        "involved_classes": [],
        "question": [
            [{"role": "user", "content": "What is the weather in San Francisco?"}],
            [{"role": "user", "content": "What about tomorrow?"}],
        ],
    }

    def _assistant_empties(messages: list[dict]) -> list[dict]:
        return [
            m
            for m in messages
            if m.get("role") == "assistant"
            and try_parse_iffc_response(m.get("content")) == []
        ]

    server, base_url = _start_scripted_server(contents)
    prev_url = config.SERVER_URL
    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    prev_strip = config.IFFC_STRIP_EMPTY_ASSISTANT
    try:
        config.SERVER_URL = base_url
        generate_multi_turn.execute_multi_turn_func_call = (
            lambda **kwargs: (["72 F"], None)
        )

        generate_multi_turn.generate_for_entry(
            entry,
            "multi_turn_base",
            [WEATHER_TOOL],
            fc_mode=config.FC_IFFC,
            strip_empty_assistant=False,
        )
        off_payloads = list(_ScriptedChatHandler.payloads)
        assert len(off_payloads) == 3
        assert _assistant_empties(off_payloads[2]["messages"]), (
            "flag off must keep assistant [] in history for the next user turn"
        )

        _ScriptedChatHandler.payloads = []
        _ScriptedChatHandler._idx = 0
        generate_multi_turn.generate_for_entry(
            entry,
            "multi_turn_base",
            [WEATHER_TOOL],
            fc_mode=config.FC_IFFC,
            strip_empty_assistant=True,
        )
        on_payloads = list(_ScriptedChatHandler.payloads)
        assert len(on_payloads) == 3
        assert _assistant_empties(on_payloads[2]["messages"]) == []
    finally:
        config.SERVER_URL = prev_url
        config.IFFC_STRIP_EMPTY_ASSISTANT = prev_strip
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        server.shutdown()
        server.server_close()


def run_offline_smoke() -> None:
    run_fixture_checks()
    run_mock_multi_step_loop()
    run_mock_strip_empty_assistant_history()


def run_live_smoke() -> None:
    """One synthetic IFFC turn against a running llama-server (structural checks)."""
    import requests

    try:
        resp = requests.get(f"{config.SERVER_URL}/v1/models", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"Live smoke requires llama-server at {config.SERVER_URL}: {exc}")

    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    generate_multi_turn.execute_multi_turn_func_call = lambda **kwargs: (["72 F"], None)
    try:
        entry = {
            "id": "smoke-iffc-live",
            "initial_config": {},
            "involved_classes": [],
            "question": [
                [
                    {
                        "role": "user",
                        "content": "Call get_weather for San Francisco, then stop.",
                    }
                ]
            ],
        }
        turn_results, timing = generate_multi_turn.generate_for_entry(
            entry,
            "multi_turn_base",
            [WEATHER_TOOL],
            fc_mode=config.FC_IFFC,
        )
    finally:
        generate_multi_turn.execute_multi_turn_func_call = prev_execute

    assert isinstance(turn_results, list)
    assert timing["api_calls"] >= 1
    print(f"live turn_results={turn_results!r}")
    print(f"live api_calls={timing['api_calls']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also hit llama-server at SERVER_URL with a synthetic IFFC turn",
    )
    args = parser.parse_args(argv)
    run_offline_smoke()
    print("offline IFFC smoke: OK")
    if args.live:
        run_live_smoke()
        print("live IFFC smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
