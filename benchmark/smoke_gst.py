"""Smoke GST: template-in-payload + NFC tools + a mock native-tool loop.

Offline (no llama-server; used in CI):

    uv run python -m benchmark.smoke_gst

Live llama-server at ``config.SERVER_URL`` (structural checks only):

    uv run python -m benchmark.smoke_gst --live

GST = Guided-Structured Templates (arXiv:2509.18076) on the NFC channel.
Requests still send ``tools`` / ``tool_choice``; GST text is not executed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from . import generate_multi_turn
from .gst import (
    GST_CLOSE,
    GST_OPEN,
    GST_STEPS,
    build_gst_system_prompt,
    extract_gst_block,
    gst_payload_has_curriculum,
    gst_payload_has_nfc_tools,
    has_gst_scaffold,
    parse_gst_steps,
    strip_gst_block,
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

_GST_WEATHER = (
    f"{GST_OPEN}\n"
    "1. get_weather matches the city weather request\n"
    "2. pick get_weather\n"
    "3. city: string, required\n"
    "4. required city; no optional args needed\n"
    "5. city = San Francisco\n"
    "6. get_weather(city='San Francisco')\n"
    "7. revalidate: docs + query ok\n"
    f"{GST_CLOSE}"
)

_GST_ABSTAIN = (
    f"{GST_OPEN}\n"
    "1. weather already fetched\n"
    "2. no further function needed\n"
    "3. n/a\n"
    "4. n/a\n"
    "5. prior observation: 72 F\n"
    "6. none\n"
    "7. ABSTAIN\n"
    f"{GST_CLOSE}"
)


def run_fixture_checks() -> None:
    """Assert curriculum + parse/ignore GST text (no network)."""
    prompt = build_gst_system_prompt()
    assert GST_OPEN in prompt and GST_CLOSE in prompt
    for step in GST_STEPS:
        assert step in prompt
    assert "ABSTAIN" in prompt
    assert "prior tool" in prompt.lower() or "observations" in prompt.lower()

    block = extract_gst_block(_GST_WEATHER)
    assert block is not None
    steps = parse_gst_steps(_GST_WEATHER)
    assert set(steps) == set(range(1, 8))
    assert has_gst_scaffold(_GST_WEATHER)

    stripped = strip_gst_block(_GST_WEATHER + " leftover")
    assert GST_OPEN not in stripped
    assert "leftover" in stripped

    unlabeled = (
        "1. identify get_weather\n2. pick it\n3. docs\n"
        "4. city required\n5. SF\n6. draft\n7. ABSTAIN"
    )
    assert has_gst_scaffold(unlabeled)
    assert extract_gst_block(unlabeled) is not None

    assert extract_gst_block("Sure, I can help.") is None
    assert not has_gst_scaffold("Sure, I can help.")


class _ScriptedChatHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoint with scripted assistant messages."""

    replies: list[dict] = []
    payloads: list[dict] = []
    _idx = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            type(self).payloads.append(json.loads(raw.decode() or "{}"))
        except json.JSONDecodeError:
            type(self).payloads.append({})
        if self._idx >= len(self.replies):
            body = json.dumps({"error": "no more scripted responses"}).encode()
            self.send_response(500)
        else:
            message = self.replies[self._idx]
            type(self)._idx += 1
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": message,
                            "finish_reason": (
                                "tool_calls" if message.get("tool_calls") else "stop"
                            ),
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


def _start_scripted_server(
    replies: list[dict],
) -> tuple[ThreadingHTTPServer, str]:
    _ScriptedChatHandler.replies = replies
    _ScriptedChatHandler.payloads = []
    _ScriptedChatHandler._idx = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def run_mock_multi_step_loop() -> None:
    """GST+NFC: GST text in content, executable action from tool_calls."""
    replies = [
        {
            "role": "assistant",
            "content": _GST_WEATHER,
            "tool_calls": [
                {
                    "id": "call-gst-1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "San Francisco"}),
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": _GST_ABSTAIN,
        },
    ]
    server, base_url = _start_scripted_server(replies)
    prev_url = config.SERVER_URL
    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    try:
        config.SERVER_URL = base_url
        generate_multi_turn.execute_multi_turn_func_call = (
            lambda **kwargs: (["72 F"], None)
        )
        entry = {
            "id": "smoke-gst-1",
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
            fc_mode=config.FC_GST,
        )
        assert turn_results == [[["get_weather(city='San Francisco')"]]], turn_results
        assert timing["id"] == "smoke-gst-1"
        assert timing["api_calls"] == 2
        assert timing["category"] == "multi_turn_base"
        assert timing["flags"] == ""

        payloads = _ScriptedChatHandler.payloads
        assert payloads, "expected chat-completions payloads"
        first = payloads[0]
        assert gst_payload_has_nfc_tools(first), first.keys()
        assert gst_payload_has_curriculum(first)
        assert "grammar" not in first
        assert "json_schema" not in first
        assert first["messages"][0]["role"] == "system"
        sys_text = first["messages"][0]["content"]
        for step in GST_STEPS:
            assert step in sys_text

        followup = payloads[1]["messages"]
        assert any(m.get("role") == "tool" for m in followup)
        assert not any(
            m.get("role") == "user" and "Function execution results" in (m.get("content") or "")
            for m in followup
        )
        assistant_with_tools = next(
            m for m in followup if m.get("role") == "assistant" and m.get("tool_calls")
        )
        assert has_gst_scaffold(assistant_with_tools.get("content") or "")
    finally:
        config.SERVER_URL = prev_url
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        server.shutdown()
        server.server_close()


def run_offline_smoke() -> None:
    run_fixture_checks()
    run_mock_multi_step_loop()


def run_live_smoke() -> None:
    """One synthetic GST turn against a running llama-server (structural)."""
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
            "id": "smoke-gst-live",
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
            fc_mode=config.FC_GST,
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
        help="Also hit llama-server at SERVER_URL with a synthetic GST turn",
    )
    args = parser.parse_args(argv)
    run_offline_smoke()
    print("offline GST smoke: OK")
    if args.live:
        run_live_smoke()
        print("live GST smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
