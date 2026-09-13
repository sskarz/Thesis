"""Smoke IFFC+STATE: JSON-in-text, no native tools, harness-written STATE.

Offline (no llama-server; used in CI):

    uv run python -m benchmark.smoke_iffc_state

Live llama-server at ``config.SERVER_URL`` (structural checks only):

    uv run python -m benchmark.smoke_iffc_state --live

iffc-state = IFFC Appendix B JSON-in-text plus a verified ``[STATE]`` snapshot
from actual tool execution results. Not GST, not CD/GBNF.
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
from .iffc_state import (
    KEEP_RECENT_TURNS,
    STATE_ADDENDUM,
    STATE_CLOSE,
    STATE_OPEN,
    StateTracker,
    compress_older_dumps,
    iffc_state_payload_has_state,
    iffc_state_payload_omits_tools,
)

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
ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "file_name": {"type": "string"},
            },
            "required": ["content"],
        },
    },
}
PWD_TOOL = {
    "type": "function",
    "function": {
        "name": "pwd",
        "description": "print working directory",
        "parameters": {"type": "object", "properties": {}},
    },
}


def run_fixture_checks() -> None:
    """Assert STATE construction + compression (no network)."""
    tracker = StateTracker()
    tracker.update(
        ["cd(folder='docs')"],
        [{"current_working_directory": "docs"}],
    )
    tracker.update(
        ["echo(content='hello', file_name='notes.txt')"],
        [None],
    )
    tracker.update(
        ["ls()"],
        [{"current_directory_content": ["notes.txt", "todo.md"]}],
    )
    block = tracker.render()
    assert STATE_OPEN in block and STATE_CLOSE in block
    assert "cwd:" in block and "docs" in block
    assert "notes.txt" in block
    assert "[GST]" not in block
    assert "Curriculum:" not in block

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn0"},
        {"role": "assistant", "content": "[]"},
        {
            "role": "user",
            "content": "Function execution results:\n- old0 -> dump0\n\n[STATE]\ncwd: /old\n[/STATE]",
        },
        {"role": "user", "content": "turn1"},
        {
            "role": "user",
            "content": "Function execution results:\n- old1 -> dump1",
        },
        {"role": "user", "content": "turn2"},
        {
            "role": "user",
            "content": "Function execution results:\n- old2 -> dump2",
        },
    ]
    compress_older_dumps(history, keep_recent=KEEP_RECENT_TURNS)
    blob = "\n".join(m["content"] for m in history)
    assert "turn0" in blob and "dump0" not in blob
    assert "dump1" in blob and "dump2" in blob
    assert history[0]["role"] == "system"


class _ScriptedChatHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoint with scripted assistant JSON replies."""

    contents: list[str] = []
    payloads: list[dict] = []
    _idx = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            type(self).payloads.append(json.loads(raw.decode() or "{}"))
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


def _fake_execute(**kwargs):
    results = []
    for call in kwargs.get("func_call_list") or []:
        if call.startswith("cd"):
            results.append({"current_working_directory": "docs"})
        elif call.startswith("echo"):
            results.append(None)
        elif call.startswith("pwd"):
            results.append({"current_working_directory": "/docs"})
        else:
            results.append("ok")
    return results, None


def run_mock_multi_step_loop() -> None:
    """Three IFFC turns: STATE injected, last two dumps kept, no native tools."""
    contents = [
        '[{"function_name": "cd", "parameters": {"folder": "docs"}}]',
        "[]",
        '[{"function_name": "echo", "parameters": {"content": "hello", "file_name": "notes.txt"}}]',
        "[]",
        '[{"function_name": "pwd", "parameters": {}}]',
        "[]",
    ]
    server, base_url = _start_scripted_server(contents)
    prev_url = config.SERVER_URL
    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    try:
        config.SERVER_URL = base_url
        generate_multi_turn.execute_multi_turn_func_call = _fake_execute
        entry = {
            "id": "smoke-iffc-state-1",
            "initial_config": {},
            "involved_classes": [],
            "question": [
                [{"role": "user", "content": "Go to docs"}],
                [{"role": "user", "content": "Write hello to notes.txt"}],
                [{"role": "user", "content": "Where am I?"}],
            ],
        }
        turn_results, timing = generate_multi_turn.generate_for_entry(
            entry,
            "multi_turn_base",
            [CD_TOOL, ECHO_TOOL, PWD_TOOL],
            fc_mode=config.FC_IFFC_STATE,
        )
        assert turn_results == [
            [["cd(folder='docs')"]],
            [["echo(content='hello', file_name='notes.txt')"]],
            [["pwd()"]],
        ], turn_results
        assert timing["id"] == "smoke-iffc-state-1"
        assert timing["api_calls"] == 6
        assert timing["flags"] == ""

        payloads = _ScriptedChatHandler.payloads
        assert len(payloads) == 6
        first = payloads[0]
        assert iffc_state_payload_omits_tools(first), first.keys()
        assert "grammar" not in first
        assert "json_schema" not in first
        sys_text = first["messages"][0]["content"]
        assert first["messages"][0]["role"] == "system"
        assert "expert in function calling" in sys_text
        assert STATE_ADDENDUM.split("Trust STATE", 1)[0][:20] in sys_text or "verified" in sys_text
        assert STATE_OPEN in sys_text or "STATE" in sys_text
        assert "Curriculum:" not in sys_text
        assert "[GST]" not in sys_text

        follow_cd = payloads[1]["messages"]
        assert iffc_state_payload_has_state(payloads[1])
        assert any(
            m.get("role") == "user" and "Function execution results" in (m.get("content") or "")
            for m in follow_cd
        )
        assert not any(m.get("role") == "tool" for m in follow_cd)
        cd_dump = next(
            m["content"]
            for m in follow_cd
            if m.get("role") == "user" and "Function execution results" in (m.get("content") or "")
        )
        assert STATE_OPEN in cd_dump and "docs" in cd_dump
        parsed = try_parse_iffc_response(
            next(m["content"] for m in follow_cd if m.get("role") == "assistant")
        )
        assert parsed and parsed[0]["function_name"] == "cd"

        last = payloads[-1]["messages"]
        last_blob = "\n".join(str(m.get("content") or "") for m in last)
        assert "Go to docs" in last_blob
        assert "Write hello to notes.txt" in last_blob
        assert "Where am I?" in last_blob
        dumps = [
            m.get("content") or ""
            for m in last
            if m.get("role") == "user" and "Function execution results" in (m.get("content") or "")
        ]
        assert len(dumps) == KEEP_RECENT_TURNS, dumps
        assert any("echo" in d for d in dumps)
        assert any("pwd" in d for d in dumps)
        assert not any("cd(folder='docs')" in d for d in dumps)
        assert STATE_OPEN in last_blob and "/docs" in last_blob
        assert IFFC_SYSTEM_PROMPT.split("\n", 1)[0][:20] in last[0]["content"]
        for payload in payloads:
            assert iffc_state_payload_omits_tools(payload)
            assert "tools" not in payload
    finally:
        config.SERVER_URL = prev_url
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        server.shutdown()
        server.server_close()


def run_offline_smoke() -> None:
    run_fixture_checks()
    run_mock_multi_step_loop()


def run_live_smoke() -> None:
    """One synthetic IFFC+STATE turn against a running llama-server (structural)."""
    import requests

    try:
        resp = requests.get(f"{config.SERVER_URL}/v1/models", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"Live smoke requires llama-server at {config.SERVER_URL}: {exc}"
        )

    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    generate_multi_turn.execute_multi_turn_func_call = _fake_execute
    try:
        entry = {
            "id": "smoke-iffc-state-live",
            "initial_config": {},
            "involved_classes": [],
            "question": [
                [
                    {
                        "role": "user",
                        "content": "Call cd for docs, then stop.",
                    }
                ]
            ],
        }
        turn_results, timing = generate_multi_turn.generate_for_entry(
            entry,
            "multi_turn_base",
            [CD_TOOL],
            fc_mode=config.FC_IFFC_STATE,
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
        help="Also hit llama-server at SERVER_URL with a synthetic IFFC+STATE turn",
    )
    args = parser.parse_args(argv)
    run_offline_smoke()
    print("offline IFFC+STATE smoke: OK")
    if args.live:
        run_live_smoke()
        print("live IFFC+STATE smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
