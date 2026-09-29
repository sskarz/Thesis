"""Smoke PTC-A: live tool-name GBNF (args free) + a mock IFFC loop.

Offline (no llama-server; used in CI):

    uv run python -m benchmark.smoke_ptc_a

Live llama-server at ``config.SERVER_URL`` (structural checks only):

    uv run python -m benchmark.smoke_ptc_a --live

PTC-A = IFFC JSON-in-text + name-only DFA. It is **not** CD (no ``json_schema``
pass-2), **not** Plan-to-Act, **not** ``iffc-ooda-card``, **not**
``iffc-stop-repeat``. Default ``nfc`` is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from . import generate_multi_turn
from .cd import build_name_dispatch_grammar, grammar_allowed_names, tool_names
from .ptc_a import (
    apply_name_dfa_constraints,
    assert_ptc_a_payload,
    build_name_dfa_grammar,
    grammar_constrains_argument_schemas,
    instance_matches_name_dfa,
    text_matches_name_dfa,
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

OTHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the time for a timezone",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
        },
    },
}


def run_constraint_checks() -> None:
    """Prove name DFA accept/reject and that args stay unconstrained (no GPU)."""
    tools = [WEATHER_TOOL, OTHER_TOOL]
    names = tool_names(tools)
    assert names == ["get_weather", "get_time"]

    grammar = build_name_dfa_grammar(tools)
    allowed = grammar_allowed_names(grammar)
    assert allowed == names
    assert "not_a_tool" not in allowed
    assert not grammar_constrains_argument_schemas(grammar)
    assert "parameters" in grammar
    assert "json-object" in grammar
    # CD pass-1 forbids parameters — PTC-A must not reuse that grammar.
    cd_grammar = build_name_dispatch_grammar(tools)
    assert grammar != cd_grammar
    assert "parameters" not in cd_grammar

    assert instance_matches_name_dfa(tools, [])
    assert instance_matches_name_dfa(
        tools,
        [{"function_name": "get_weather", "parameters": {"city": "SF"}}],
    )
    # Args unconstrained: empty object, extra keys, non-schema types all OK.
    assert instance_matches_name_dfa(
        tools, [{"function_name": "get_weather", "parameters": {}}]
    )
    assert instance_matches_name_dfa(
        tools,
        [
            {
                "function_name": "get_weather",
                "parameters": {"city": "SF", "extra": True, "n": 1},
            }
        ],
    )
    assert not instance_matches_name_dfa(
        tools, [{"function_name": "not_a_tool", "parameters": {"city": "SF"}}]
    )
    assert not instance_matches_name_dfa(
        tools, [{"function_name": "get_weather"}]
    )
    assert text_matches_name_dfa(tools, "[]")
    assert not text_matches_name_dfa(
        tools, '[{"function_name": "hack", "parameters": {}}]'
    )

    payload = {
        "model": "m",
        "messages": [],
        "temperature": 0.0,
        "tools": tools,
        "tool_choice": "auto",
        "json_schema": {"type": "object"},
    }
    apply_name_dfa_constraints(payload, tools)
    assert_ptc_a_payload(payload, tools)
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "json_schema" not in payload
    assert "city" not in payload["grammar"]
    assert "tz" not in payload["grammar"]


class _ScriptedChatHandler(BaseHTTPRequestHandler):
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


def run_mock_single_pass_loop() -> None:
    """PTC-A: one constrained IFFC completion (with args) → execute → []."""
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
            "id": "smoke-ptc-a-1",
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
            [WEATHER_TOOL, OTHER_TOOL],
            fc_mode=config.FC_IFFC_PTC_A,
        )
        assert turn_results == [
            [["get_weather(city='San Francisco')"]]
        ], turn_results
        assert timing["id"] == "smoke-ptc-a-1"
        # Single pass per decision (tool call + end-turn []) — not CD's 3.
        assert timing["api_calls"] == 2
        assert timing["category"] == "multi_turn_base"

        payloads = _ScriptedChatHandler.payloads
        assert len(payloads) == 2
        for payload in payloads:
            assert_ptc_a_payload(payload, [WEATHER_TOOL, OTHER_TOOL])
            assert "json_schema" not in payload
            assert "tools" not in payload
            allowed = grammar_allowed_names(payload["grammar"])
            assert allowed == ["get_weather", "get_time"]
            assert "city" not in payload["grammar"]
            assert "tz" not in payload["grammar"]
    finally:
        config.SERVER_URL = prev_url
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        server.shutdown()
        server.server_close()


def run_offline_smoke() -> None:
    run_constraint_checks()
    run_mock_single_pass_loop()


def run_live_smoke() -> None:
    """One synthetic PTC-A turn against a running llama-server (structural)."""
    import requests

    try:
        resp = requests.get(f"{config.SERVER_URL}/v1/models", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"Live smoke requires llama-server at {config.SERVER_URL}: {exc}"
        )

    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    generate_multi_turn.execute_multi_turn_func_call = lambda **kwargs: (["72 F"], None)
    try:
        entry = {
            "id": "smoke-ptc-a-live",
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
            fc_mode=config.FC_IFFC_PTC_A,
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
        help="Also hit llama-server at SERVER_URL with a synthetic PTC-A turn",
    )
    args = parser.parse_args(argv)
    run_offline_smoke()
    print("offline PTC-A (name DFA, args free) smoke: OK")
    if args.live:
        run_live_smoke()
        print("live PTC-A smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
