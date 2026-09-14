import json

import pytest

from benchmark import config
from benchmark.bon import (
    ABSTAIN_KEY,
    BON_TEMPERATURE,
    candidate_from_content,
    candidate_from_response,
    calls_vote_key,
    complete_step,
    majority_vote,
    resolve_bon_n,
    select_winner_response,
)
from benchmark.iffc import IFFC_SYSTEM_PROMPT
from benchmark.iffc_state import STATE_ADDENDUM, STATE_OPEN
from benchmark.smoke_bon import run_offline_smoke


def _call(name: str, **params) -> str:
    return json.dumps([{"function_name": name, "parameters": params}])


def test_majority_picks_most_frequent_call():
    cands = [
        candidate_from_content(_call("ls"), 0),
        candidate_from_content(_call("cd", folder="docs"), 1),
        candidate_from_content(_call("cd", folder="docs"), 2),
        candidate_from_content(_call("cd", folder="other"), 3),
        candidate_from_content(_call("cd", folder="docs"), 4),
    ]
    result = majority_vote(cands)
    assert result.winner.parsed[0]["function_name"] == "cd"
    assert result.winner.parsed[0]["parameters"] == {"folder": "docs"}
    assert result.winner.index == 1
    key = calls_vote_key(result.winner.parsed)
    assert result.counts[key] == 3
    assert result.n_valid == 5
    assert result.n_excluded == 0


def test_unparseable_excluded_empty_counts_as_abstain():
    cands = [
        candidate_from_content("Sure, I can help.", 0),
        candidate_from_content("[]", 1),
        candidate_from_content("", 2),
        candidate_from_content("[]", 3),
        candidate_from_content(_call("ls"), 4),
    ]
    result = majority_vote(cands)
    assert result.n_excluded == 2
    assert result.n_valid == 3
    assert result.winner.key == ABSTAIN_KEY
    assert result.counts[ABSTAIN_KEY] == 2
    assert result.winner.content == "[]"


def test_tie_prefers_non_empty_over_abstain():
    cands = [
        candidate_from_content("[]", 0),
        candidate_from_content(_call("pwd"), 1),
        candidate_from_content("[]", 2),
        candidate_from_content(_call("pwd"), 3),
    ]
    result = majority_vote(cands)
    assert result.winner.parsed[0]["function_name"] == "pwd"
    assert result.winner.index == 1
    assert ABSTAIN_KEY in result.counts
    assert result.counts[ABSTAIN_KEY] == result.counts[result.winner.key]


def test_tie_among_non_empty_is_first_seen():
    cands = [
        candidate_from_content(_call("cd", folder="a"), 0),
        candidate_from_content(_call("cd", folder="b"), 1),
        candidate_from_content(_call("cd", folder="a"), 2),
        candidate_from_content(_call("cd", folder="b"), 3),
    ]
    result = majority_vote(cands)
    assert result.winner.parsed[0]["parameters"]["folder"] == "a"
    assert result.winner.index == 0


def test_all_unparseable_returns_first_for_retry_path():
    cands = [
        candidate_from_content("nope", 0),
        candidate_from_content("still nope", 1),
    ]
    result = majority_vote(cands)
    assert result.n_valid == 0
    assert result.n_excluded == 2
    assert result.winner.index == 0
    assert result.winner.parsed is None
    assert result.winner.content == "nope"


def test_argument_key_order_does_not_split_votes():
    a = '[{"function_name": "echo", "parameters": {"file_name": "a.txt", "content": "hi"}}]'
    b = '[{"function_name": "echo", "parameters": {"content": "hi", "file_name": "a.txt"}}]'
    cands = [
        candidate_from_content(a, 0),
        candidate_from_content(b, 1),
        candidate_from_content(_call("ls"), 2),
    ]
    result = majority_vote(cands)
    assert result.winner.parsed[0]["function_name"] == "echo"
    assert result.counts[result.winner.key] == 2


def test_parallel_calls_vote_as_the_whole_step():
    both = json.dumps(
        [
            {"function_name": "cd", "parameters": {"folder": "docs"}},
            {"function_name": "ls", "parameters": {}},
        ]
    )
    only_cd = _call("cd", folder="docs")
    cands = [
        candidate_from_content(both, 0),
        candidate_from_content(only_cd, 1),
        candidate_from_content(both, 2),
    ]
    result = majority_vote(cands)
    assert [c["function_name"] for c in result.winner.parsed] == ["cd", "ls"]


def test_empty_choices_are_excluded():
    raw = {"choices": []}
    cand = candidate_from_response(raw, 0)
    assert cand.key is None
    result = majority_vote(
        [cand, candidate_from_content(_call("ls"), 1)]
    )
    assert result.winner.parsed[0]["function_name"] == "ls"
    assert result.n_excluded == 1


def test_bon_n_configurable_and_rejects_invalid():
    assert config.BON_N_DEFAULT == 5
    assert resolve_bon_n(None) == config.BON_N
    config.set_bon_n(3)
    assert config.BON_N == 3
    assert resolve_bon_n(None) == 3
    assert resolve_bon_n(7) == 7
    with pytest.raises(ValueError, match="bon-n"):
        config.set_bon_n(0)
    with pytest.raises(ValueError, match="bon-n"):
        config.set_bon_n(-1)
    with pytest.raises(ValueError):
        resolve_bon_n(0)
    config.set_bon_n(config.BON_N_DEFAULT)


def test_set_fc_mode_accepts_iffc_state_bon():
    config.set_fc_mode("iffc-state-bon")
    assert config.FC_MODE == config.FC_IFFC_STATE_BON
    assert config.FC_IFFC_STATE_BON in config.FC_MODES
    assert config.FC_IFFC_STATE_BON in config.FC_JSON_TEXT_MODES
    assert config.FC_IFFC_STATE_BON in config.FC_IFFC_STATE_MODES
    assert config.FC_IFFC_STATE in config.FC_IFFC_STATE_MODES
    assert config.FC_GST not in config.FC_JSON_TEXT_MODES
    assert config.FC_NFC not in config.FC_IFFC_STATE_MODES
    config.set_fc_mode("nfc")
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("best-of-n")


def test_complete_step_samples_n_votes_and_omits_tools():
    payloads: list[dict] = []
    replies = [
        "garbage",
        _call("cd", folder="docs"),
        _call("cd", folder="docs"),
    ]

    def post(payload):
        payloads.append(json.loads(json.dumps(payload)))
        content = replies[len(payloads) - 1]
        return {
            "choices": [{"message": {"content": content}}],
            "timings": {
                "prompt_n": 1,
                "prompt_ms": 2,
                "predicted_n": 3,
                "predicted_ms": 4,
            },
        }

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Open docs"},
    ]
    data, n_calls = complete_step(
        history,
        [{"type": "function", "function": {"name": "cd"}}],
        post=post,
        model="test-model",
        n=3,
    )
    assert n_calls == 3
    assert len(payloads) == 3
    for payload in payloads:
        assert "tools" not in payload
        assert "tool_choice" not in payload
        assert "grammar" not in payload
        assert payload["temperature"] == BON_TEMPERATURE
        assert payload["messages"] == history
    parsed = json.loads(data["choices"][0]["message"]["content"])
    assert parsed[0]["function_name"] == "cd"
    assert data["timings"]["prompt_n"] == 3
    assert data["timings"]["predicted_ms"] == 12
    assert history[-1]["content"] == "Open docs"


def test_select_winner_does_not_invent_scores():
    responses = [
        {"choices": [{"message": {"content": _call("ls")}}], "timings": {}},
        {"choices": [{"message": {"content": _call("ls")}}], "timings": {}},
        {"choices": [{"message": {"content": _call("pwd")}}], "timings": {}},
    ]
    data, n = select_winner_response(responses)
    assert n == 3
    assert "ls" in data["choices"][0]["message"]["content"]


def test_iffc_state_prompt_pieces_still_present_for_bon_mode():
    from benchmark.iffc_state import build_iffc_state_system_prompt

    prompt = build_iffc_state_system_prompt(
        [{"name": "cd", "description": "change directory", "parameters": {"type": "object"}}]
    )
    assert "expert in function calling" in prompt
    assert STATE_ADDENDUM in prompt
    assert STATE_OPEN in prompt
    assert "[GST]" not in prompt
    assert IFFC_SYSTEM_PROMPT.split("{available_functions}")[0] in prompt


def test_offline_smoke_script_passes():
    run_offline_smoke()
