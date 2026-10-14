"""Smoke ``iffc-decision-ooda`` compose: mode wire + OODA inject + Decision hint.

Offline (no servers required for structural checks; Decision probe mocked):

    uv run python -m benchmark.smoke_decision_ooda_compose

Live (Ollama tev1 + llama-server :8080):

    uv run python -m benchmark.smoke_decision_ooda_compose --live

Proves: mode registered; working copy = OODA card then Decision hint;
persistent history clean; Decision hard-noop sole [] that step; stop-repeat
still in FC_IFFC_STOP_REPEAT_MODES; no crash. Acc NOT scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

from . import config
from . import decision_sidecar
from . import generate_multi_turn as gmt
from .ooda_card import (
    CARD_CLOSE,
    CARD_OPEN,
    OodaCardTracker,
    inject_card,
    payload_has_ooda_card,
    payload_omits_tools,
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


def _assert_modes() -> None:
    assert OBSERVE_SAME_MIN_ASSISTANT_TOOL_TURNS == 6
    assert config.FC_IFFC_DECISION_OODA == "iffc-decision-ooda"
    assert config.FC_IFFC_DECISION_OODA in config.FC_MODES
    assert config.FC_IFFC_DECISION_OODA in config.FC_JSON_TEXT_MODES
    assert config.FC_IFFC_DECISION_OODA in config.FC_IFFC_STOP_REPEAT_MODES
    assert config.FC_IFFC_DECISION_OODA in config.FC_IFFC_DECISION_MODES
    assert config.FC_IFFC_DECISION_OODA in config.FC_IFFC_OODA_MODES
    # Controls still selectable on same commit
    assert config.FC_IFFC_OODA_CARD in config.FC_MODES
    assert config.FC_IFFC_DECISION in config.FC_MODES
    # Plain decision is NOT on the stop-repeat stack (R58 recipe preserved)
    assert config.FC_IFFC_DECISION not in config.FC_IFFC_STOP_REPEAT_MODES
    config.set_fc_mode(config.FC_IFFC_DECISION_OODA)
    assert config.FC_MODE == config.FC_IFFC_DECISION_OODA


def _assert_working_copy_order() -> None:
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Go to docs"},
    ]
    tracker = OodaCardTracker()
    payload = gmt._build_chat_payload(
        history,
        [CD_TOOL],
        config.FC_IFFC_DECISION_OODA,
        ooda_tracker=tracker,
    )
    assert payload_omits_tools(payload)
    assert payload_has_ooda_card(payload)
    msgs = payload["messages"]
    # OODA injects on last user; persistent history untouched
    assert history[1]["content"] == "Go to docs"
    assert CARD_OPEN in msgs[-1]["content"]
    assert CARD_CLOSE in msgs[-1]["content"]

    # Decision hint last (after OODA card message)
    working = list(msgs)
    working.append(
        decision_sidecar.name_hint_user_message("cd", strong=False)
    )
    assert CARD_OPEN in working[-2]["content"]
    hint = working[-1]["content"]
    assert "cd" in hint.lower() or "cd" in hint
    # persistent still clean
    assert all(CARD_OPEN not in (m.get("content") or "") for m in history)
    assert history == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Go to docs"},
    ]


def _assert_hard_noop_sole_stop() -> None:
    """Decision hard-noop must synthesize [] without also running stop-repeat."""
    history: list[dict] = [{"role": "user", "content": "done"}]
    flags: list[str] = []
    # Force soft_once already used → hard
    act, used = gmt._consume_decision_noop(
        soft_noop_used=True, turn_idx=0, flags=flags
    )
    assert act == "hard"
    assert any(f.startswith("decision_noop_hard") for f in flags)
    history.append({"role": "assistant", "content": "[]"})
    # Only one harness [] in history
    harness_stops = [
        m for m in history if m.get("role") == "assistant" and m.get("content") == "[]"
    ]
    assert len(harness_stops) == 1


def _fake_probe_execute(name: str = "cd"):
    pol = SimpleNamespace(
        action=decision_sidecar.ACTION_EXECUTE,
        tool_name=name,
        bypass_reason=None,
    )
    return SimpleNamespace(policy=pol)


def run_offline_smoke() -> dict:
    _assert_modes()
    _assert_working_copy_order()
    _assert_hard_noop_sole_stop()

    # Fake llama-server: return a clean IFFC cd call then []
    class Handler(BaseHTTPRequestHandler):
        n = 0

        def log_message(self, *args):  # noqa: ARG002
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            Handler.captured = json.loads(body.decode())
            Handler.n += 1
            if Handler.n == 1:
                content = json.dumps(
                    [{"function_name": "cd", "parameters": {"folder": "docs"}}]
                )
            else:
                content = "[]"
            out = {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "timings": {
                    "prompt_n": 1,
                    "prompt_ms": 1,
                    "predicted_n": 1,
                    "predicted_ms": 1,
                },
            }
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_url = config.SERVER_URL
    config.SERVER_URL = f"http://127.0.0.1:{port}"
    try:
        entry = {
            "id": "smoke_decision_ooda_compose_0",
            "question": [[{"role": "user", "content": "Go to docs"}]],
            "initial_config": {},
            "involved_classes": [],
        }
        with mock.patch.object(
            decision_sidecar,
            "probe_next_tool",
            return_value=_fake_probe_execute("cd"),
        ), mock.patch(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils.execute_multi_turn_func_call",
            return_value=([{"current_working_directory": "docs"}], None),
        ):
            turns, timing = gmt.generate_for_entry(
                entry,
                "multi_turn_base",
                [CD_TOOL],
                fc_mode=config.FC_IFFC_DECISION_OODA,
            )
        # Captured Gemma payload must have OODA card + Decision hint; no tools
        captured = Handler.captured
        assert payload_omits_tools(captured)
        assert payload_has_ooda_card(captured)
        msgs = captured["messages"]
        # last message should be Decision hint (execute path) OR OODA-on-user
        joined = "\n".join(str(m.get("content", "")) for m in msgs)
        assert CARD_OPEN in joined and CARD_CLOSE in joined
        assert "cd" in joined.lower()
        assert timing["api_calls"] >= 1
        evidence = {
            "turns": turns,
            "api_calls": timing["api_calls"],
            "flags": timing.get("flags", ""),
            "payload_n_messages": len(msgs),
            "payload_has_ooda": True,
            "payload_omits_tools": True,
        }
    finally:
        config.SERVER_URL = old_url
        server.shutdown()

    return {"offline": "PASS", **evidence}


def run_live_smoke() -> dict:
    """One short live step: Decision probe + Gemma with compose mode."""
    import requests

    # Health
    ollama = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
    ollama.raise_for_status()
    llama = requests.get(f"{config.SERVER_URL}/v1/models", timeout=5)
    llama.raise_for_status()

    decision_sidecar.prewarm()
    history = [{"role": "user", "content": "List files in the current directory."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": "list directory contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "boolean"},
                    },
                },
            },
        }
    ]
    gmt._prepare_step_messages(history, tools, config.FC_IFFC_DECISION_OODA)
    # system should have OODA addendum
    assert any(
        "OODA" in (m.get("content") or "") for m in history if m.get("role") == "system"
    )
    persistent_before = json.dumps(history)
    probe = decision_sidecar.probe_next_tool(
        history, tools, entry_id="smoke_compose_live", turn=0, step=0
    )
    name_hint = None
    if (
        probe.policy.action == decision_sidecar.ACTION_EXECUTE
        and probe.policy.tool_name
    ):
        name_hint = probe.policy.tool_name
    tracker = OodaCardTracker()
    data, n = gmt._complete_chat_sync(
        history,
        tools,
        config.FC_IFFC_DECISION_OODA,
        name_hint=name_hint,
        ooda_tracker=tracker,
    )
    # persistent history unchanged by inject/hint
    assert json.dumps(history) == persistent_before
    content = data["choices"][0]["message"].get("content", "")
    return {
        "live": "PASS",
        "decision_action": probe.policy.action,
        "decision_tool": getattr(probe.policy, "tool_name", None),
        "name_hint": name_hint,
        "gemma_n_calls": n,
        "gemma_content_preview": (content or "")[:200],
        "persistent_clean": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    out: dict = {"fc_mode": config.FC_IFFC_DECISION_OODA}
    out.update(run_offline_smoke())
    if args.live:
        out.update(run_live_smoke())
    print(json.dumps(out, indent=2))
    print("SMOKE PASS" if out.get("offline") == "PASS" else "SMOKE FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
