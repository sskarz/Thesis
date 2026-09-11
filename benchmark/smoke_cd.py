"""Smoke TagDispatch-style CD: grammar/schema constraints + a mock two-pass loop.

Offline (no llama-server; used in CI):

    uv run python -m benchmark.smoke_cd

Live llama-server at ``config.SERVER_URL`` (structural checks only):

    uv run python -m benchmark.smoke_cd --live

CD = two-pass constrained decoding (name GBNF, then per-tool ``json_schema``).
It is **not** NFC native ``tools`` GBNF. llama.cpp has no XGrammar; this is a
TagDispatch *approximation* on stock llama-server.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from . import generate_multi_turn
from .cd import (
    apply_args_constraints,
    apply_dispatch_constraints,
    assert_cd_payload,
    build_args_schema,
    build_name_dispatch_grammar,
    build_name_dispatch_schema,
    grammar_allowed_names,
    instance_matches_schema,
    tool_names,
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
    """Prove name/arg constraints reject malformed output (no GPU)."""
    tools = [WEATHER_TOOL, OTHER_TOOL]
    names = tool_names(tools)
    assert names == ["get_weather", "get_time"]

    grammar = build_name_dispatch_grammar(tools)
    assert "tools" not in grammar
    allowed = grammar_allowed_names(grammar)
    assert allowed == names
    assert "not_a_tool" not in allowed
    assert 'hack' not in grammar_allowed_names(grammar)

    name_schema = build_name_dispatch_schema(tools)
    assert instance_matches_schema(name_schema, [])
    assert instance_matches_schema(
        name_schema, [{"function_name": "get_weather"}]
    )
    assert instance_matches_schema(
        name_schema,
        [{"function_name": "get_weather"}, {"function_name": "get_time"}],
    )
    # Malformed names rejected
    assert not instance_matches_schema(
        name_schema, [{"function_name": "not_a_tool"}]
    )
    assert not instance_matches_schema(
        name_schema, [{"function_name": "get_weather_please"}]
    )
    # Pass 1 forbids parameters (NFC-style extra keys)
    assert not instance_matches_schema(
        name_schema,
        [{"function_name": "get_weather", "parameters": {"city": "SF"}}],
    )

    args_schema = build_args_schema(tools, ["get_weather"])
    assert instance_matches_schema(
        args_schema,
        [{"function_name": "get_weather", "parameters": {"city": "San Francisco"}}],
    )
    # Wrong name for this dispatched schema
    assert not instance_matches_schema(
        args_schema,
        [{"function_name": "get_time", "parameters": {"tz": "UTC"}}],
    )
    # Missing required arg is still schema-valid only if the nested
    # parameters schema does not require city… WEATHER_TOOL requires city.
    assert not instance_matches_schema(
        args_schema,
        [{"function_name": "get_weather", "parameters": {}}],
    )
    # Extra wrapper keys rejected
    assert not instance_matches_schema(
        args_schema,
        [
            {
                "function_name": "get_weather",
                "parameters": {"city": "SF"},
                "extra": True,
            }
        ],
    )

    payload: dict = {"model": "test", "messages": [], "temperature": 0.0}
    apply_dispatch_constraints(payload, tools)
    assert_cd_payload(payload, pass_kind="dispatch")
    payload2: dict = {"model": "test", "messages": [], "temperature": 0.0}
    apply_args_constraints(payload2, tools, ["get_weather"])
    assert_cd_payload(payload2, pass_kind="args")


class _ScriptedChatHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible chat completions; records payloads and scripts replies."""

    contents: list[str] = []
    payloads: list[dict] = []
    _idx = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            req = {}
        type(self).payloads.append(req)
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


def run_mock_two_pass_loop() -> None:
    """CD step: name dispatch → args json_schema → execute → [] end turn."""
    contents = [
        '[{"function_name": "get_weather"}]',
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
            "id": "smoke-cd-1",
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
            fc_mode=config.FC_CD,
        )
        assert turn_results == [
            [["get_weather(city='San Francisco')"]]
        ], turn_results
        assert timing["id"] == "smoke-cd-1"
        assert timing["api_calls"] == 3  # dispatch + args + end-turn []
        assert timing["category"] == "multi_turn_base"

        payloads = _ScriptedChatHandler.payloads
        assert len(payloads) == 3
        assert_cd_payload(payloads[0], pass_kind="dispatch")
        assert "get_weather" in grammar_allowed_names(payloads[0]["grammar"])
        assert "get_time" in grammar_allowed_names(payloads[0]["grammar"])
        assert_cd_payload(payloads[1], pass_kind="args")
        schema = payloads[1]["json_schema"]
        assert instance_matches_schema(
            schema,
            [
                {
                    "function_name": "get_weather",
                    "parameters": {"city": "San Francisco"},
                }
            ],
        )
        assert not instance_matches_schema(
            schema,
            [{"function_name": "get_time", "parameters": {"tz": "UTC"}}],
        )
        # End-turn is another dispatch pass (empty list), still no native tools
        assert_cd_payload(payloads[2], pass_kind="dispatch")
    finally:
        config.SERVER_URL = prev_url
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        server.shutdown()
        server.server_close()


def run_offline_smoke() -> None:
    run_constraint_checks()
    run_mock_two_pass_loop()


def run_live_smoke() -> None:
    """One synthetic CD turn against a running llama-server (structural checks)."""
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
            "id": "smoke-cd-live",
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
            fc_mode=config.FC_CD,
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
        help="Also hit llama-server at SERVER_URL with a synthetic CD turn",
    )
    args = parser.parse_args(argv)
    run_offline_smoke()
    print("offline CD (TagDispatch-style) smoke: OK")
    if args.live:
        run_live_smoke()
        print("live CD smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
