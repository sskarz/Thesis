"""Smoke full-stack ``iffc-state-ooda`` harness. Fully OFFLINE.

    uv run python -m benchmark.smoke_state_ooda

No llama-server and no model: a local mock HTTP server on 127.0.0.1:0 and a
fake BFCL executor. Asserts:

- mode is registered in FC_MODES / JSON-text / STATE / OODA / stop-repeat sets;
- merged system prompt = Appendix B + STATE addendum + OODA addendum (both present);
- every request omits native tools, and every post-execution request carries
  the ``[STATE]`` block *before* the ``[OODA]`` card on the latest user message;
- gated stop-on-repeat (MIN=6) still fires: an executed-call re-emit after 6
  assistant tool-turns is force-stopped with a harness ``[]``, not executed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from . import generate_multi_turn
from .iffc_state import STATE_ADDENDUM, STATE_CLOSE, STATE_OPEN
from .ooda_card import (
    CARD_CLOSE,
    CARD_OPEN,
    OODA_ADDENDUM_AFTER_STATE,
    OodaCardTracker,
    build_state_ooda_system_prompt,
    inject_card_after_state,
    payload_has_ooda_card,
    payload_has_state_block,
    payload_omits_tools,
)
from .stop_on_repeat import OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS

MODE = config.FC_IFFC_STATE_OODA
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


def _assert_state_then_card(body: dict) -> None:
    last_user = [m for m in body["messages"] if m.get("role") == "user"][-1]
    text = last_user["content"]
    s, c = text.find(STATE_OPEN), text.find(CARD_OPEN)
    assert s >= 0 and c >= 0, text
    assert text.find(STATE_CLOSE) < c < text.find(CARD_CLOSE), text
    assert text.count(CARD_OPEN) == 1


def run_offline_smoke() -> None:
    assert OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS == 6
    assert MODE == "iffc-state-ooda"
    for group in (
        config.FC_MODES,
        config.FC_JSON_TEXT_MODES,
        config.FC_IFFC_STATE_MODES,
        config.FC_IFFC_OODA_MODES,
        config.FC_IFFC_STOP_REPEAT_MODES,
    ):
        assert MODE in group
    assert MODE not in config.FC_IFFC_DECISION_MODES

    sys_prompt = build_state_ooda_system_prompt([CD_TOOL])
    assert STATE_ADDENDUM in sys_prompt and OODA_ADDENDUM_AFTER_STATE in sys_prompt
    assert sys_prompt.index(STATE_ADDENDUM) < sys_prompt.index(OODA_ADDENDUM_AFTER_STATE)

    history = [
        {"role": "user", "content": "Go to docs"},
        {
            "role": "user",
            "content": (
                "Function execution results:\n- cd(folder='docs') -> ok\n\n"
                f"{STATE_OPEN}\ncwd: docs\n{STATE_CLOSE}\n\n"
                'Respond with a JSON list of function calls (each with '
                '"function_name" and "parameters"), or [] if no further '
                "function calling is needed."
            ),
        },
    ]
    out = inject_card_after_state(history, OodaCardTracker(), has_tools=True)
    _assert_state_then_card({"messages": out})
    assert out[-1]["content"].rstrip().endswith("function calling is needed.")
    assert CARD_OPEN not in history[-1]["content"]  # transcript untouched
    # No STATE yet (step 0): card is prefixed, like iffc-ooda-card.
    out0 = inject_card_after_state(history[:1], OodaCardTracker(), has_tools=True)
    assert out0[0]["content"].startswith(CARD_OPEN)
    # Idempotent: re-injecting on an already-carded copy keeps one card.
    again = inject_card_after_state(out, OodaCardTracker(), has_tools=True)
    _assert_state_then_card({"messages": again})
    print("state-ooda offline smoke: ok")


def run_mock_loop() -> None:
    """Mocked server: 7 distinct cd calls, then a re-emit of an executed call."""
    folders = [f"d{i}" for i in range(7)]
    replies = [
        json.dumps([{"function_name": "cd", "parameters": {"folder": f}}])
        for f in folders
    ]
    # 8th step: re-emit an already-executed call after >=6 tool-turns -> gated stop.
    replies.append(json.dumps([{"function_name": "cd", "parameters": {"folder": "d0"}}]))
    bodies: list[dict] = []
    remaining = iter(replies)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            bodies.append(body)
            raw = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": next(remaining)}}],
                "timings": {"prompt_n": 8, "prompt_ms": 10, "predicted_n": 4, "predicted_ms": 8},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    prev_url, prev_exec = config.SERVER_URL, generate_multi_turn.execute_multi_turn_func_call

    def fake_execute(**kwargs):
        calls = kwargs.get("func_call_list") or []
        return [{"current_working_directory": "x"} for _ in calls] or [{}], None

    try:
        config.SERVER_URL = f"http://{host}:{port}"
        generate_multi_turn.execute_multi_turn_func_call = fake_execute  # type: ignore[method-assign]
        entry = {
            "id": "state-ooda-smoke-1",
            "initial_config": {},
            "involved_classes": ["fs"],
            "question": [[{"role": "user", "content": "Walk the dirs"}]],
        }
        turn_results, timing = generate_multi_turn.generate_for_entry(
            entry, "multi_turn_base", [CD_TOOL], fc_mode=MODE
        )
    finally:
        config.SERVER_URL = prev_url
        generate_multi_turn.execute_multi_turn_func_call = prev_exec
        server.shutdown()

    assert len(bodies) == 8, len(bodies)
    for i, body in enumerate(bodies):
        assert payload_omits_tools(body)
        assert body["temperature"] == 0.0
        sysmsg = body["messages"][0]
        assert sysmsg["role"] == "system"
        assert STATE_ADDENDUM in sysmsg["content"]
        assert OODA_ADDENDUM_AFTER_STATE in sysmsg["content"]
        assert payload_has_ooda_card(body)
        if i >= 1:
            assert payload_has_state_block(body)
            _assert_state_then_card(body)
    # 7 executed steps; the 8th (re-emit of executed d0 after 7 tool-turns) is
    # force-stopped by the MIN=6 gate and NOT executed.
    executed = [s for s in turn_results[0] if s]
    assert executed == [[f"cd(folder='{f}')"] for f in folders], turn_results
    flags = json.dumps(timing)
    assert "stop_on_repeat" in flags, timing
    print("state-ooda mock loop (STATE->card order + stop gate): ok")


def main() -> int:
    run_offline_smoke()
    run_mock_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
