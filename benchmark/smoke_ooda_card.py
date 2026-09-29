"""Smoke one-pass IFFC-OODA-card: JSON-in-text, no native tools, compact card.

Offline (no llama-server; used in CI):

    uv run python -m benchmark.smoke_ooda_card

Live llama-server at ``config.SERVER_URL`` (structural checks only):

    uv run python -m benchmark.smoke_ooda_card --live

iffc-ooda-card = IFFC Appendix B JSON-in-text plus a harness-written
``[OODA]`` card (goal / slots / last / next) stacked on stop-repeat MIN=6.
Not iffc-state, not GST, not a Specialized 4-pass OODA loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from . import generate_multi_turn
from .iffc import IFFC_SYSTEM_PROMPT, try_parse_iffc_response
from .ooda_card import (
    CARD_CLOSE,
    CARD_OPEN,
    OODA_ADDENDUM,
    OodaCardTracker,
    inject_card,
    payload_has_ooda_card,
    payload_has_state_block,
    payload_omits_tools,
    record_execution,
    render_card,
)
from .stop_on_repeat import OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS

CD_TOOL = {
    "type": "function",
    "function": {
        "name": "cd",
        "description": "change directory",
        "parameters": {
            "type": "object",
            "properties": {"folder": {"type": "string"}},
            "required": ["folder"],
        },
    },
}


def run_offline_smoke() -> None:
    """Assert card format, injection, IFFC parse compatibility (no network)."""
    assert OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS == 6
    assert config.FC_IFFC_OODA_CARD in config.FC_MODES
    assert config.FC_IFFC_OODA_CARD != config.FC_NFC

    tracker = OodaCardTracker()
    history = [{"role": "user", "content": "Go to docs"}]
    block = render_card(tracker, history, has_tools=True)
    assert CARD_OPEN in block and CARD_CLOSE in block
    assert "goal: Go to docs" in block
    assert "next: call" in block

    record_execution(
        tracker,
        ["cd(folder='docs')"],
        [{"current_working_directory": "docs"}],
    )
    after = render_card(tracker, history, has_tools=True)
    assert "cwd=docs" in after or "folder=docs" in after

    record_execution(
        tracker,
        ["cd(folder='docs')"],
        [{"current_working_directory": "docs"}],
    )
    history.append(
        {
            "role": "user",
            "content": (
                "Function execution results:\n- cd(folder='docs') -> ok\n"
                'Respond with a JSON list of function calls (each with '
                '"function_name" and "parameters"), or [] if no further '
                "function calling is needed."
            ),
        }
    )
    injected = inject_card(history, tracker, has_tools=True)
    assert injected[-1]["content"].startswith(CARD_OPEN)
    assert history[-1]["content"].startswith("Function execution results:")
    assert "[STATE]" not in injected[-1]["content"]

    echoed = f"{block}\n[]"
    assert try_parse_iffc_response(echoed) == []
    call_json = '[{"function_name": "cd", "parameters": {"folder": "docs"}}]'
    assert try_parse_iffc_response(f"{block}\n{call_json}")[0]["function_name"] == "cd"
    assert OODA_ADDENDUM
    assert "{available_functions}" in IFFC_SYSTEM_PROMPT
    print("ooda-card offline smoke: ok")


def _mock_handler(replies: list[str]):
    remaining = iter(replies)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            assert "tools" not in body and "tool_choice" not in body
            assert payload_omits_tools(body)
            assert payload_has_ooda_card(body)
            assert not payload_has_state_block(body)
            content = next(remaining)
            payload = {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "timings": {
                    "prompt_n": 8,
                    "prompt_ms": 10,
                    "predicted_n": 4,
                    "predicted_ms": 8,
                },
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def run_mock_loop() -> None:
    replies = [
        json.dumps([{"function_name": "cd", "parameters": {"folder": "docs"}}]),
        "[]",
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _mock_handler(replies))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    previous = config.SERVER_URL
    try:
        config.SERVER_URL = f"http://{host}:{port}"
        entry = {
            "id": "ooda-smoke-1",
            "initial_config": {},
            "involved_classes": ["fs"],
            "question": [[{"role": "user", "content": "Go to docs"}]],
        }

        def fake_execute(**kwargs):
            return [{"current_working_directory": "docs"}], None

        previous_exec = generate_multi_turn.execute_multi_turn_func_call
        generate_multi_turn.execute_multi_turn_func_call = fake_execute  # type: ignore[method-assign]
        try:
            turn_results, timing = generate_multi_turn.generate_for_entry(
                entry,
                "multi_turn_base",
                [CD_TOOL],
                fc_mode=config.FC_IFFC_OODA_CARD,
            )
            assert turn_results == [[["cd(folder='docs')"]]]
            assert timing["api_calls"] == 2
            print("ooda-card mock loop: ok")
        finally:
            generate_multi_turn.execute_multi_turn_func_call = previous_exec
    finally:
        config.SERVER_URL = previous
        server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke iffc-ooda-card harness.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Hit config.SERVER_URL (structural checks only; not a BFCL score).",
    )
    args = parser.parse_args()
    run_offline_smoke()
    if args.live:
        import requests

        try:
            requests.get(f"{config.SERVER_URL}/v1/models", timeout=5).raise_for_status()
        except Exception as exc:
            print(f"live smoke skipped: server not reachable ({exc})", file=sys.stderr)
            return 0
        print(
            "live smoke: server reachable. Run generate_for_entry against "
            "a real model on Mac; this script does not score multi_turn_base."
        )
        return 0
    run_mock_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
