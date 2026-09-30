"""Unit tests for iffc-decision Decision sidecar (mocked client; no live Ollama)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmark import config
from benchmark import decision_sidecar
from benchmark import stop_on_repeat

# Import side package policy/client directly for focused unit tests.
_PROBE = Path.home() / "Documents" / "decision-probe"
if str(_PROBE) not in sys.path:
    sys.path.insert(0, str(_PROBE))

from sidecar.client import SystemOneResult, post_systemone  # noqa: E402
from sidecar.policy import (  # noqa: E402
    ACTION_BYPASS,
    ACTION_EXECUTE,
    ACTION_NOOP,
    decide,
)


_EXEC = (
    "Function execution results:\n"
    "cd → ok\n"
    "or [] if no further function calling is needed"
)


def _ok_result(choice: str, probs: dict[str, float], **kw) -> SystemOneResult:
    return SystemOneResult(
        ok=True,
        choice=choice,
        probabilities=probs,
        confidence=kw.get("confidence", 0.5),
        noul=kw.get("noul", 0.2),
        latency_ms=kw.get("latency_ms", 10.0),
        raw=None,
        error=None,
        http_status=200,
        input_tokens=100,
    )


def test_fc_modes_include_decision_three_way():
    """Mirror test_stop_on_repeat.py:321-323 — all three memberships."""
    assert config.FC_IFFC_DECISION in config.FC_MODES
    assert config.FC_IFFC_DECISION in config.FC_JSON_TEXT_MODES
    assert config.FC_IFFC != config.FC_IFFC_DECISION
    # Family helper used by --iffc-strip-empty-assistant
    from benchmark.generate_multi_turn import _iffc_family_mode

    assert _iffc_family_mode(config.FC_IFFC_DECISION) is True


def test_position_guard_blocks_noop_at_step0():
    """noop never fires on a fresh BFCL user question (not exec observation)."""
    history = [{"role": "user", "content": "Please cd into docs"}]
    result = _ok_result(
        "noop",
        {"cd": 0.1, "noop": 0.9},
    )
    decision = decide(
        result,
        active_tool_names=["cd", "ls"],
        chat_history=history,
        noop_min=0.30,
        choice_min=0.35,
        is_exec_observation_fn=stop_on_repeat.is_exec_observation,
    )
    assert decision.action == ACTION_BYPASS
    assert decision.is_exec_observation is False
    assert decision.bypass_reason == "noop_position_guard"


def test_position_guard_allows_noop_after_exec_obs():
    """When choice==noop and mass fires with exec obs → noop_end_turn."""
    history = [
        {"role": "assistant", "content": '[{"function_name":"cd","parameters":{}}]'},
        {"role": "user", "content": _EXEC},
    ]
    result = _ok_result("noop", {"get_weather": 0.28, "noop": 0.61, "cd": 0.11})
    decision = decide(
        result,
        active_tool_names=["cd", "get_weather"],
        chat_history=history,
        noop_min=0.30,
        choice_min=0.35,
        is_exec_observation_fn=stop_on_repeat.is_exec_observation,
        argmax_respecting=True,
    )
    assert decision.action == ACTION_NOOP
    assert decision.is_exec_observation is True


def test_argmax_respecting_rescues_book_flight_163():
    """_163 shape: choice=book_flight p=0.519 > p_noop=0.465 → execute, not noop."""
    history = [{"role": "user", "content": _EXEC}]
    result = _ok_result(
        "book_flight",
        {"book_flight": 0.519, "noop": 0.465, "search": 0.016},
    )
    decision = decide(
        result,
        active_tool_names=["book_flight", "search"],
        chat_history=history,
        noop_min=0.30,
        choice_min=0.35,
        is_exec_observation_fn=stop_on_repeat.is_exec_observation,
        argmax_respecting=True,
    )
    assert decision.action == ACTION_EXECUTE
    assert decision.tool_name == "book_flight"


def test_argmax_respecting_rescues_84_shape():
    """_84 t1s2 shape: choice tool p=0.514 > p_noop=0.366 → execute, not noop."""
    history = [{"role": "user", "content": _EXEC}]
    result = _ok_result(
        "cd",
        {"cd": 0.514, "noop": 0.366, "ls": 0.120},
    )
    decision = decide(
        result,
        active_tool_names=["cd", "ls"],
        chat_history=history,
        noop_min=0.30,
        choice_min=0.35,
        is_exec_observation_fn=stop_on_repeat.is_exec_observation,
        argmax_respecting=True,
    )
    assert decision.action == ACTION_EXECUTE
    assert decision.tool_name == "cd"


def test_noop_position_guard_requires_confident_noop():
    """Low p_noop at ask step → low_confidence, not noop_position_guard."""
    history = [{"role": "user", "content": "please list files"}]
    result = _ok_result("noop", {"noop": 0.05, "ls": 0.40})
    decision = decide(
        result,
        active_tool_names=["ls"],
        chat_history=history,
        noop_min=0.30,
        choice_min=0.35,
        is_exec_observation_fn=stop_on_repeat.is_exec_observation,
        argmax_respecting=True,
    )
    assert decision.action == ACTION_BYPASS
    assert decision.bypass_reason != "noop_position_guard"


def test_argmax_off_restores_mass_noop():
    """DECISION_NOOP_ARGMAX=0: mass noop ignores argmax (status quo)."""
    history = [{"role": "user", "content": _EXEC}]
    result = _ok_result(
        "get_weather",
        {"get_weather": 0.5826, "noop": 0.3043, "search": 0.1131},
    )
    decision = decide(
        result,
        active_tool_names=["get_weather", "search"],
        chat_history=history,
        noop_min=0.30,
        choice_min=0.35,
        is_exec_observation_fn=stop_on_repeat.is_exec_observation,
        argmax_respecting=False,
    )
    assert decision.action == ACTION_NOOP


def test_execute_when_choice_confident():
    history = [{"role": "user", "content": "go"}]
    result = _ok_result("cd", {"cd": 0.70, "noop": 0.05, "ls": 0.25})
    decision = decide(
        result,
        active_tool_names=["cd", "ls"],
        chat_history=history,
        noop_min=0.30,
        choice_min=0.35,
        is_exec_observation_fn=stop_on_repeat.is_exec_observation,
    )
    assert decision.action == ACTION_EXECUTE
    assert decision.tool_name == "cd"


def test_client_exception_returns_ok_false(monkeypatch):
    """Client must never raise into harness — Exception → ok=False."""

    def _boom(*_a, **_k):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(
        "sidecar.client.urllib.request.urlopen",
        _boom,
    )
    out = post_systemone(state="x", questions={"next_tool": {"type": "choice", "criteria": {"noop": "x"}}})
    assert out.ok is False
    assert out.choice is None
    assert out.error is not None
    assert out.error.startswith("client_exception:")


def test_probe_fail_open_on_client_exception(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("down")

    monkeypatch.setattr(
        "sidecar.client.urllib.request.urlopen",
        _boom,
    )
    history = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "cd", "description": ""}}]
    probe = decision_sidecar.probe_next_tool(history, tools, log=False)
    assert probe.policy.action == ACTION_BYPASS
    assert probe.policy.bypass_reason == "client_exception"


def test_is_noop_allowed_matches_stop_on_repeat():
    assert decision_sidecar.is_noop_allowed([]) is False
    assert (
        decision_sidecar.is_noop_allowed(
            [{"role": "user", "content": "fresh question"}]
        )
        is False
    )
    assert (
        decision_sidecar.is_noop_allowed([{"role": "user", "content": _EXEC}])
        is True
    )


# === client_fix extensions (criteria cap / truncate / error log / prewarm) ===

from sidecar.state import (  # noqa: E402
    MAX_ACTIVE_TOOLS,
    MAX_CHOICE_CRITERIA,
    MAX_STATE_PROMPT_TOKENS,
    build_questions,
    chat_history_to_state,
    clamp_tool_names,
    clamp_tool_names_relevant,
    clamp_tools,
    estimate_tokens,
    short_tool_desc,
)


def test_criteria_cap_clamps_to_26_with_noop():
    """Multi-API menus (32 tools) must clamp so active ∪ {noop} ≤ 26."""
    names = [f"tool_{i:02d}" for i in range(32)]
    kept, dropped = clamp_tool_names(names)
    assert len(kept) == MAX_ACTIVE_TOOLS == 25
    assert len(dropped) == 7
    # Stable lexicographic: first 25 of sorted names
    assert kept == sorted(names)[:25]
    assert dropped == sorted(names)[25:]
    q = build_questions(names, desc_mode="off")
    criteria = q["next_tool"]["criteria"]
    assert "noop" in criteria
    assert len(criteria) == MAX_CHOICE_CRITERIA == 26
    assert set(criteria) - {"noop"} == set(kept)


def test_criteria_cap_under_limit_unchanged_order():
    names = ["cd", "ls", "find"]
    kept, dropped = clamp_tool_names(names)
    assert kept == names
    assert dropped == []
    q = build_questions(names, desc_mode="off")
    assert len(q["next_tool"]["criteria"]) == 4  # 3 + noop


def test_state_truncate_under_token_budget():
    """Long transcripts must yield state with estimate_tokens ≤ budget."""
    tools = [
        {
            "type": "function",
            "function": {"name": f"tool_{i}", "description": "x" * 200},
        }
        for i in range(30)
    ]
    history = []
    files = ", ".join(f'"f{j}"' for j in range(40))
    blob = (
        "Function execution results:\n"
        + (f'cd → {{"ok": true, "files": [{files}]}}\n') * 5
        + "or [] if no further function calling is needed"
    )
    for i in range(40):
        history.append(
            {
                "role": "assistant",
                "content": f'[{{"function_name":"tool_{i % 30}","parameters":{{}}}}]',
            }
        )
        history.append({"role": "user", "content": blob})
    state = chat_history_to_state(history, tools)
    est = estimate_tokens(state)
    assert est <= MAX_STATE_PROMPT_TOKENS, (est, len(state), state[:200])
    assert len(state) <= MAX_STATE_PROMPT_TOKENS * 1 + 8


def test_http_error_preserves_status_and_body(monkeypatch):
    """HTTPError must surface http_status + body, not opaque client_exception."""
    import io
    import urllib.error

    def _boom(*_a, **_k):
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:11434/v1/systemone",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error":"question \\"next_tool\\": criteria must contain 2-26 candidates"}'
            ),
        )

    monkeypatch.setattr("sidecar.client.urllib.request.urlopen", _boom)
    out = post_systemone(
        state="x",
        questions={
            "next_tool": {"type": "choice", "criteria": {"a": "A", "noop": "n"}}
        },
    )
    assert out.ok is False
    assert out.http_status == 400
    assert out.error is not None
    assert out.error.startswith("http_400:")
    assert "criteria" in out.error or "2-26" in out.error or "2" in out.error


def test_probe_log_record_includes_client_error_and_http_status(monkeypatch):
    import io
    import urllib.error

    def _boom(*_a, **_k):
        raise urllib.error.HTTPError(
            url="http://x",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error":"prompt 0 has 2377 tokens; expected 1-2050"}'
            ),
        )

    monkeypatch.setattr("sidecar.client.urllib.request.urlopen", _boom)
    history = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "cd", "description": ""}}]
    probe = decision_sidecar.probe_next_tool(history, tools, log=False)
    assert probe.policy.action == ACTION_BYPASS
    assert probe.policy.bypass_reason == "http_400"
    assert probe.log_record.get("http_status") == 400
    assert probe.log_record.get("client_error", "").startswith("http_400:")
    assert (
        "2377" in probe.log_record["client_error"]
        or "prompt" in probe.log_record["client_error"]
    )


def test_prewarm_questions_are_legal_2_to_26():
    """Prewarm must not be noop-only (1 cand); need dummy tool + noop."""
    q = build_questions(["__prewarm__"], include_noul=False, desc_mode="off")
    criteria = q["next_tool"]["criteria"]
    assert 2 <= len(criteria) <= MAX_CHOICE_CRITERIA
    assert "noop" in criteria
    assert "__prewarm__" in criteria
    # Empty tool list alone is illegal for live prewarm — document shape.
    q_bad = build_questions([], include_noul=False, desc_mode="off")
    assert len(q_bad["next_tool"]["criteria"]) == 1  # noop only — must not be posted


def test_prewarm_uses_legal_criteria(monkeypatch):
    captured = {}

    def _fake_post(**kwargs):
        captured["questions"] = kwargs["questions"]
        captured["criteria_names"] = list(kwargs.get("criteria_names") or [])
        return SystemOneResult(
            ok=True,
            choice="__prewarm__",
            probabilities={"__prewarm__": 0.6, "noop": 0.4},
            confidence=0.5,
            noul=None,
            latency_ms=100.0,
            raw={},
            error=None,
            http_status=200,
            input_tokens=50,
        )

    monkeypatch.setattr(decision_sidecar, "post_systemone", _fake_post)
    decision_sidecar._prewarmed = False
    result = decision_sidecar.prewarm(force=True)
    assert result is not None and result.ok
    crit = captured["questions"]["next_tool"]["criteria"]
    assert 2 <= len(crit) <= 26
    assert "noop" in crit
    assert decision_sidecar._prewarmed is True


# === Lever 1 — real tool descriptions ===

def test_short_tool_desc_extracts_bfcl_suffix():
    raw = (
        "This tool belongs to the Gorilla file system. "
        "Tool description: List the contents of the current directory."
    )
    assert short_tool_desc(raw, max_chars=80) == (
        "List the contents of the current directory."
    )
    tool = {
        "type": "function",
        "function": {"name": "ls", "description": raw},
    }
    assert "List the contents" in short_tool_desc(tool, max_chars=80)


def test_build_questions_embeds_descriptions():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": (
                    "This tool belongs to FS. "
                    "Tool description: List the contents of the current directory."
                ),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cd",
                "description": (
                    "This tool belongs to FS. "
                    "Tool description: Change the current working directory."
                ),
            },
        },
    ]
    q = build_questions(
        ["ls", "cd"],
        active_tools=tools,
        desc_mode="short",
        desc_max_chars=80,
    )
    crit = q["next_tool"]["criteria"]
    assert "List the contents" in crit["ls"]
    assert "Change the current" in crit["cd"]
    assert crit["noop"] == "Do not call a tool; reply to the user or stop."
    assert q["_meta"]["desc_mode"] in ("short", "shrunk")
    assert "as the next action." not in crit["ls"]


def test_build_questions_template_fallback_when_missing_desc():
    q = build_questions(["mystery"], desc_mode="short", desc_by_name={})
    crit = q["next_tool"]["criteria"]
    assert crit["mystery"] == "Call tool `mystery` as the next action."
    assert q["_meta"]["per_tool_desc_mode"]["mystery"] == "template_fallback"


def test_build_questions_desc_mode_off_bare_template():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": "Tool description: List files.",
            },
        }
    ]
    q = build_questions(["ls"], active_tools=tools, desc_mode="off")
    assert q["next_tool"]["criteria"]["ls"] == "Call tool `ls` as the next action."
    assert q["_meta"]["desc_mode"] == "template_fallback"


def test_desc_shrink_ladder_logs_mode():
    """Synthetic many long descs → shrunk or template_fallback, never crash."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i:02d}",
                "description": (
                    "Tool description: " + ("word " * 40) + f"action{i}"
                ),
            },
        }
        for i in range(25)
    ]
    names = [f"tool_{i:02d}" for i in range(25)]
    q = build_questions(
        names,
        active_tools=tools,
        desc_mode="short",
        desc_max_chars=80,
        state_tokens_est=1750,
        criteria_budget_tokens=50,
    )
    assert 2 <= len(q["next_tool"]["criteria"]) <= 26
    assert q["_meta"]["desc_mode"] in ("short", "shrunk", "template_fallback")


# === Lever 2 — soft_once consumer ===

def test_soft_once_consumer_first_then_hard():
    from benchmark.generate_multi_turn import _consume_decision_noop

    flags: list[str] = []
    # Force soft_once via env already default; monkeypatch mode helper
    act1, used = _consume_decision_noop(
        soft_noop_used=False, turn_idx=0, flags=flags
    )
    assert act1 == "soft"
    assert used is True
    assert any("decision_noop_soft" in f for f in flags)
    flags2: list[str] = []
    act2, used2 = _consume_decision_noop(
        soft_noop_used=True, turn_idx=0, flags=flags2
    )
    assert act2 == "hard"
    assert any("decision_noop_hard" in f for f in flags2)


def test_noop_consumer_hard_break_env(monkeypatch):
    monkeypatch.setattr(decision_sidecar, "DECISION_NOOP_HARD_BREAK", 1)
    # Re-read via noop_consumer_mode which checks the module attr
    assert decision_sidecar.noop_consumer_mode() == "hard"
    from benchmark.generate_multi_turn import _consume_decision_noop

    flags: list[str] = []
    act, used = _consume_decision_noop(
        soft_noop_used=False, turn_idx=1, flags=flags
    )
    assert act == "hard"
    monkeypatch.setattr(decision_sidecar, "DECISION_NOOP_HARD_BREAK", 0)


# === Lever 3 — relevance / API-class clamp ===

MESSAGE = [
    "add_contact", "delete_message", "get_message_stats", "get_user_id",
    "list_users", "message_get_login_status", "message_login",
    "search_messages", "send_message", "view_messages_sent",
]
VEHICLE = [
    "activateParkingBrake", "adjustClimateControl", "check_tire_pressure",
    "displayCarStatus", "display_log", "estimate_distance",
    "estimate_drive_feasibility_by_mileage", "fillFuelTank",
    "find_nearest_tire_shop", "gallon_to_liter", "get_current_speed",
    "get_outside_temperature_from_google",
    "get_outside_temperature_from_weather_com", "get_zipcode_based_on_city",
    "liter_to_gallon", "lockDoors", "pressBrakePedal", "releaseBrakePedal",
    "setCruiseControl", "setHeadlights", "set_navigation", "startEngine",
]
FS = [
    "cat", "cd", "cp", "diff", "du", "echo", "find", "grep", "ls", "mkdir",
    "mv", "pwd", "rm", "rmdir", "sort", "tail", "touch", "wc",
]
TWITTER = [
    "authenticate_twitter", "comment", "follow_user", "get_tweet",
    "get_tweet_comments", "get_user_stats", "get_user_tweets",
    "list_all_following", "mention", "post_tweet", "posting_get_login_status",
    "retweet", "search_tweets", "unfollow_user",
]


def test_relevant_clamp_keeps_startEngine_on_95_shaped():
    """_95 ask about starting engine must keep startEngine in kept (≤25)."""
    names = MESSAGE + VEHICLE  # 32
    assert len(names) == 32
    n2c = {n: "MessageAPI" for n in MESSAGE}
    n2c.update({n: "VehicleControlAPI" for n in VEHICLE})
    descs = {
        "startEngine": "Starts the car engine.",
        "send_message": "Send a message to a user.",
        "view_messages_sent": "View messages previously sent.",
        "pressBrakePedal": "Press the brake pedal.",
    }
    kept, dropped = clamp_tool_names_relevant(
        names,
        user_ask="Please start the engine and check messages",
        name_to_class=n2c,
        desc_by_name=descs,
    )
    assert len(kept) == 25
    assert "startEngine" in kept
    assert "noop" not in kept
    # Message class not entirely guillotined
    assert any(n in kept for n in MESSAGE)
    assert any(n in kept for n in ("send_message", "view_messages_sent", "search_messages"))


def test_relevant_clamp_keeps_sort_on_4_shaped():
    names = TWITTER + FS  # 32
    n2c = {n: "TwitterAPI" for n in TWITTER}
    n2c.update({n: "GorillaFileSystem" for n in FS})
    descs = {"sort": "Sort lines of a text file.", "ls": "List directory contents."}
    # Lex clamp drops sort (ASCII tail). Relevant + ask must keep it.
    lex_kept, lex_dropped = clamp_tool_names(names)
    assert "sort" in lex_dropped or "sort" not in lex_kept
    kept, dropped = clamp_tool_names_relevant(
        names,
        user_ask="Sort the report file and list the directory",
        name_to_class=n2c,
        desc_by_name=descs,
    )
    assert len(kept) <= 25
    assert "sort" in kept


def test_relevant_displacement_preserves_lex_kept_gold_shaped():
    """Real-bank gate: 0 gold-missing and 0 lex→relevant gold regressions.

    Covers _4/_30/_46/_95 (esp. rmdir on _46). Uses BFCL asks + ground_truth
    via offline scorer helpers — no gold in harness runtime.
    """
    import sys
    from pathlib import Path

    probe = Path.home() / "Documents" / "decision-probe"
    if str(probe) not in sys.path:
        sys.path.insert(0, str(probe))
    from offline.score_clamp_gold import displacement_on_bank, load_bfcl_bank

    bank = load_bfcl_bank(
        [
            "multi_turn_base_4",
            "multi_turn_base_30",
            "multi_turn_base_46",
            "multi_turn_base_95",
        ]
    )
    assert bank, "BFCL bank failed to load"
    result = displacement_on_bank(bank=bank)
    assert result["n_gold_missing"] == 0, result
    assert result["n_lex_regressions"] == 0, result
    # Explicit _46 rmdir
    c46 = next(c for c in result["cases"] if c["id"].startswith("multi_turn_base_46"))
    assert "rmdir" in c46["gold"]
    assert "rmdir" not in c46["gold_missing_relevant"]
    assert "rmdir" not in c46["lex_to_relevant_gold_regressions"]


def test_clamp_strategy_dispatch():
    names = [f"z_tool_{i:02d}" for i in range(30)] + ["aaa"]
    lex_k, _ = clamp_tools(names, strategy="lex")
    assert lex_k[0] == "aaa" or "aaa" in lex_k
    rel_k, _ = clamp_tools(
        names, strategy="relevant", user_ask="please use z_tool_29 now"
    )
    assert "z_tool_29" in rel_k
    assert len(rel_k) == 25


def test_kill_switch_envs_exported():
    assert hasattr(decision_sidecar, "DECISION_DESC_MODE")
    assert hasattr(decision_sidecar, "DECISION_NOOP_ARGMAX")
    assert hasattr(decision_sidecar, "DECISION_NOOP_SOFT")
    assert hasattr(decision_sidecar, "DECISION_NOOP_HARD_BREAK")
    assert hasattr(decision_sidecar, "DECISION_CLAMP_STRATEGY")
    assert hasattr(decision_sidecar, "DECISION_DESC_MAX_CHARS")

def test_soft_once_integration_bypass_obs_reprobe(monkeypatch):
    """soft_once workhorse: first noop → soft; after obs, second noop → hard.

    Simulates the generate_multi_turn contract without launching Ollama:
    consume soft → pretend Gemma appended an exec observation → next Decision
    noop hard-ends. Does not require a live model.
    """
    from benchmark.generate_multi_turn import _consume_decision_noop

    flags: list[str] = []
    act1, used = _consume_decision_noop(
        soft_noop_used=False, turn_idx=0, flags=flags
    )
    assert act1 == "soft" and used is True
    # After soft, harness falls through to Gemma; exec obs is appended.
    history = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "call tool"},
        {"role": "user", "content": _EXEC},
    ]
    assert stop_on_repeat.is_exec_observation(history[-1]["content"])
    # Re-probe path: second Decision noop after real obs → hard
    flags2: list[str] = []
    act2, used2 = _consume_decision_noop(
        soft_noop_used=True, turn_idx=0, flags=flags2
    )
    assert act2 == "hard"
    assert any("decision_noop_hard" in f for f in flags2)
    # soft_noact flag helper: when soft step ends with no tool call
    soft_noop_this_step = True
    action = "end_turn"
    out_flags: list[str] = []
    if action == "end_turn" and soft_noop_this_step:
        out_flags.append("decision_noop_soft_noact:turn_0")
    assert out_flags == ["decision_noop_soft_noact:turn_0"]


def test_short_survives_at_state_cap_1800():
    """Lever 1: template_fallback rate <5% at state_tokens_est=1800."""
    tools = []
    for i, name in enumerate((FS + MESSAGE)[:25]):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        f"Tool description: Useful operation {name} on files "
                        f"or messages in the workspace."
                    ),
                },
            }
        )
    names = [t["function"]["name"] for t in tools]
    q = build_questions(
        kept_names=names,
        active_tools=tools,
        desc_mode="short",
        state_tokens_est=1800,
    )
    assert q["_meta"]["desc_mode"] != "template_fallback"
    assert q["_meta"]["desc_mode_requested"] == "short"
    per = q["_meta"]["per_tool_desc_mode"]
    tf = sum(1 for m in per.values() if m == "template_fallback")
    assert tf / max(1, len(per)) < 0.05

