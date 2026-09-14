"""Smoke Best-of-N: majority vote on IFFC+STATE, no native tools.

Offline (no llama-server; used in CI):

    uv run python -m benchmark.smoke_bon

Live llama-server at ``config.SERVER_URL`` (structural checks only):

    uv run python -m benchmark.smoke_bon --live

iffc-state-bon = IFFC+STATE channel plus MV@N (default N=5, temperature=0.7).
Not NFC, not GST, not CD/GBNF. Does not start a live BFCL generate.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from . import generate_multi_turn
from .bon import (
    ABSTAIN_KEY,
    BON_TEMPERATURE,
    candidate_from_content,
    majority_vote,
)
from .iffc import try_parse_iffc_response
from .iffc_state import (
    KEEP_RECENT_TURNS,
    STATE_ADDENDUM,
    STATE_CLOSE,
    STATE_OPEN,
    StateTracker,
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

SMOKE_BON_N = 3


def run_fixture_checks() -> None:
    """Assert vote rules + STATE construction (no network)."""
    # Majority: two cd(docs) beat one ls().
    cands = [
        candidate_from_content("Sure I can help", 0),
        candidate_from_content(
            '[{"function_name": "cd", "parameters": {"folder": "docs"}}]', 1
        ),
        candidate_from_content(
            '[{"function_name": "ls", "parameters": {}}]', 2
        ),
        candidate_from_content(
            '[{"function_name": "cd", "parameters": {"folder": "docs"}}]', 3
        ),
    ]
    result = majority_vote(cands)
    assert result.n_excluded == 1
    assert result.winner.parsed[0]["function_name"] == "cd"
    assert result.winner.parsed[0]["parameters"]["folder"] == "docs"

    # Empty [] is ABSTAIN (a ballot); unparseable is excluded.
    mixed = [
        candidate_from_content("not json", 0),
        candidate_from_content("[]", 1),
        candidate_from_content("[]", 2),
    ]
    abstain = majority_vote(mixed)
    assert abstain.winner.key == ABSTAIN_KEY
    assert abstain.n_excluded == 1
    assert abstain.n_valid == 2

    # Tie ABSTAIN vs non-empty: prefer the tool call.
    tied = [
        candidate_from_content("[]", 0),
        candidate_from_content(
            '[{"function_name": "pwd", "parameters": {}}]', 1
        ),
    ]
    prefer = majority_vote(tied)
    assert prefer.winner.parsed[0]["function_name"] == "pwd"

    tracker = StateTracker()
    tracker.update(
        ["cd(folder='docs')"],
        [{"current_working_directory": "docs"}],
    )
    block = tracker.render()
    assert STATE_OPEN in block and STATE_CLOSE in block
    assert "docs" in block
    assert "[GST]" not in block


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
    """Two IFFC+STATE turns with N=3: majority wins; STATE injected; no tools."""
    # Per step: 3 samples. Execute then ABSTAIN, twice.
    contents = [
        # turn 0 step 0: minority ls / unparseable lose to cd(docs)
        "Sure, I can help with that.",
        '[{"function_name": "cd", "parameters": {"folder": "docs"}}]',
        '[{"function_name": "cd", "parameters": {"folder": "docs"}}]',
        # turn 0 step 1: ABSTAIN
        "[]",
        "[]",
        "not json at all",
        # turn 1 step 0: majority echo
        '[{"function_name": "pwd", "parameters": {}}]',
        '[{"function_name": "echo", "parameters": {"content": "hello", "file_name": "notes.txt"}}]',
        '[{"function_name": "echo", "parameters": {"content": "hello", "file_name": "notes.txt"}}]',
        # turn 1 step 1: ABSTAIN
        "[]",
        "[]",
        "[]",
    ]
    server, base_url = _start_scripted_server(contents)
    prev_url = config.SERVER_URL
    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    prev_n = config.BON_N
    try:
        config.SERVER_URL = base_url
        config.set_bon_n(SMOKE_BON_N)
        generate_multi_turn.execute_multi_turn_func_call = _fake_execute
        entry = {
            "id": "smoke-iffc-state-bon-1",
            "initial_config": {},
            "involved_classes": [],
            "question": [
                [{"role": "user", "content": "Go to docs"}],
                [{"role": "user", "content": "Write hello to notes.txt"}],
            ],
        }
        turn_results, timing = generate_multi_turn.generate_for_entry(
            entry,
            "multi_turn_base",
            [CD_TOOL, ECHO_TOOL, PWD_TOOL],
            fc_mode=config.FC_IFFC_STATE_BON,
        )
        assert turn_results == [
            [["cd(folder='docs')"]],
            [["echo(content='hello', file_name='notes.txt')"]],
        ], turn_results
        assert timing["id"] == "smoke-iffc-state-bon-1"
        assert timing["api_calls"] == SMOKE_BON_N * 4
        assert timing["flags"] == ""

        payloads = _ScriptedChatHandler.payloads
        assert len(payloads) == SMOKE_BON_N * 4
        first = payloads[0]
        assert "tools" not in first and "tool_choice" not in first
        assert "grammar" not in first
        assert "json_schema" not in first
        assert first["temperature"] == BON_TEMPERATURE
        sys_text = first["messages"][0]["content"]
        assert first["messages"][0]["role"] == "system"
        assert "expert in function calling" in sys_text
        assert STATE_ADDENDUM.split("Trust STATE", 1)[0][:20] in sys_text or "verified" in sys_text
        assert STATE_OPEN in sys_text or "STATE" in sys_text
        assert "Curriculum:" not in sys_text
        assert "[GST]" not in sys_text

        # After first execute, STATE is in the next step's samples.
        after_cd = payloads[SMOKE_BON_N]
        assert after_cd["temperature"] == BON_TEMPERATURE
        follow = after_cd["messages"]
        dumps = [
            m.get("content") or ""
            for m in follow
            if m.get("role") == "user"
            and "Function execution results" in (m.get("content") or "")
        ]
        assert dumps, follow
        assert STATE_OPEN in dumps[-1] and "docs" in dumps[-1]
        assert not any(m.get("role") == "tool" for m in follow)
        parsed = try_parse_iffc_response(
            next(m["content"] for m in follow if m.get("role") == "assistant")
        )
        assert parsed and parsed[0]["function_name"] == "cd"

        last = payloads[-1]["messages"]
        last_blob = "\n".join(str(m.get("content") or "") for m in last)
        assert "Go to docs" in last_blob
        assert "Write hello to notes.txt" in last_blob
        dumps = [
            m.get("content") or ""
            for m in last
            if m.get("role") == "user"
            and "Function execution results" in (m.get("content") or "")
        ]
        assert len(dumps) <= KEEP_RECENT_TURNS
        assert any("echo" in d for d in dumps)
        assert STATE_OPEN in last_blob
        for payload in payloads:
            assert "tools" not in payload
            assert payload["temperature"] == BON_TEMPERATURE
    finally:
        config.SERVER_URL = prev_url
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        config.set_bon_n(prev_n)
        server.shutdown()
        server.server_close()


def run_offline_smoke() -> None:
    run_fixture_checks()
    run_mock_multi_step_loop()


def run_live_smoke() -> None:
    """One synthetic BoN turn against a running llama-server (structural)."""
    import requests

    try:
        resp = requests.get(f"{config.SERVER_URL}/v1/models", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"Live smoke requires llama-server at {config.SERVER_URL}: {exc}"
        )

    prev_execute = generate_multi_turn.execute_multi_turn_func_call
    prev_n = config.BON_N
    generate_multi_turn.execute_multi_turn_func_call = _fake_execute
    try:
        config.set_bon_n(2)
        entry = {
            "id": "smoke-iffc-state-bon-live",
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
            fc_mode=config.FC_IFFC_STATE_BON,
        )
    finally:
        generate_multi_turn.execute_multi_turn_func_call = prev_execute
        config.set_bon_n(prev_n)

    assert isinstance(turn_results, list)
    assert timing["api_calls"] >= 2
    print(f"live turn_results={turn_results!r}")
    print(f"live api_calls={timing['api_calls']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also hit llama-server at SERVER_URL with a synthetic BoN turn",
    )
    args = parser.parse_args(argv)
    run_offline_smoke()
    print("offline IFFC+STATE+BoN smoke: OK")
    if args.live:
        run_live_smoke()
        print("live IFFC+STATE+BoN smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
