import json

import pytest

from benchmark import config
from benchmark.cd import (
    apply_args_constraints,
    apply_dispatch_constraints,
    assert_cd_payload,
    build_args_schema,
    build_name_dispatch_grammar,
    build_name_dispatch_schema,
    build_tagged_union_schema,
    complete_step,
    grammar_allowed_names,
    instance_matches_schema,
    tool_names,
)
from benchmark.smoke_cd import run_offline_smoke


WEATHER = {
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
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
        },
    },
}


def test_tool_names_unwraps_openai_and_bfcl():
    mixed = [WEATHER, {"name": "ls"}]
    assert tool_names(mixed) == ["get_weather", "ls"]


def test_name_dispatch_schema_rejects_unknown_and_extra_keys():
    schema = build_name_dispatch_schema([WEATHER, TIME])
    assert instance_matches_schema(schema, [])
    assert instance_matches_schema(schema, [{"function_name": "get_weather"}])
    assert not instance_matches_schema(
        schema, [{"function_name": "not_a_tool"}]
    )
    assert not instance_matches_schema(
        schema,
        [{"function_name": "get_weather", "parameters": {"city": "x"}}],
    )


def test_name_dispatch_grammar_enumerates_only_live_tools():
    grammar = build_name_dispatch_grammar([WEATHER, TIME])
    assert grammar_allowed_names(grammar) == ["get_weather", "get_time"]
    assert "not_a_tool" not in grammar_allowed_names(grammar)
    empty = build_name_dispatch_grammar([])
    assert grammar_allowed_names(empty) == []
    assert "empty ::=" in empty


def test_args_schema_is_per_selected_tool_not_union_of_all():
    schema = build_args_schema([WEATHER, TIME], ["get_weather"])
    good = [
        {"function_name": "get_weather", "parameters": {"city": "SF"}},
    ]
    assert instance_matches_schema(schema, good)
    # Dispatched to weather: time is not a legal tag on this pass
    assert not instance_matches_schema(
        schema,
        [{"function_name": "get_time", "parameters": {"tz": "UTC"}}],
    )
    assert not instance_matches_schema(
        schema,
        [{"function_name": "get_weather", "parameters": {}}],
    )


def test_tagged_union_schema_allows_any_live_tool_with_its_args():
    schema = build_tagged_union_schema([WEATHER, TIME])
    assert instance_matches_schema(schema, [])
    assert instance_matches_schema(
        schema,
        [{"function_name": "get_time", "parameters": {"tz": "UTC"}}],
    )
    assert not instance_matches_schema(
        schema, [{"function_name": "hack", "parameters": {}}]
    )


def test_payloads_never_include_native_tools():
    p1 = {"model": "m", "messages": [], "temperature": 0.0, "tools": [WEATHER]}
    apply_dispatch_constraints(p1, [WEATHER, TIME])
    assert_cd_payload(p1, pass_kind="dispatch")
    p2 = {"model": "m", "messages": [], "temperature": 0.0, "tool_choice": "auto"}
    apply_args_constraints(p2, [WEATHER, TIME], ["get_time"])
    assert_cd_payload(p2, pass_kind="args")
    assert p2["json_schema"]["minItems"] == 1


def test_assert_cd_payload_rejects_nfc_fields():
    with pytest.raises(AssertionError, match="tools/tool_choice"):
        assert_cd_payload(
            {"tools": [WEATHER], "grammar": "root ::= ws"},
            pass_kind="dispatch",
        )


def test_complete_step_two_pass_and_empty_dispatch():
    posts: list[dict] = []

    def post(payload):
        posts.append(payload)
        if "grammar" in payload:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"function_name": "get_weather"}]'
                        }
                    }
                ],
                "timings": {"prompt_n": 3, "prompt_ms": 1, "predicted_n": 2, "predicted_ms": 4},
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '[{"function_name": "get_weather",'
                            ' "parameters": {"city": "SF"}}]'
                        )
                    }
                }
            ],
            "timings": {"prompt_n": 5, "prompt_ms": 2, "predicted_n": 6, "predicted_ms": 8},
        }

    data, n = complete_step(
        [{"role": "user", "content": "weather?"}],
        [WEATHER, TIME],
        post=post,
        model="test-model",
    )
    assert n == 2
    assert json.loads(data["choices"][0]["message"]["content"]) == [
        {"function_name": "get_weather", "parameters": {"city": "SF"}}
    ]
    assert data["timings"]["prompt_n"] == 8
    assert_cd_payload(posts[0], pass_kind="dispatch")
    assert_cd_payload(posts[1], pass_kind="args")
    assert "tools" not in posts[0] and "tools" not in posts[1]

    def post_empty(payload):
        posts.append(payload)
        return {
            "choices": [{"message": {"content": "[]"}}],
            "timings": {},
        }

    posts.clear()
    data, n = complete_step(
        [{"role": "user", "content": "thanks"}],
        [WEATHER],
        post=post_empty,
        model="test-model",
    )
    assert n == 1
    assert data["choices"][0]["message"]["content"] == "[]"
    assert len(posts) == 1
    assert_cd_payload(posts[0], pass_kind="dispatch")


def test_set_fc_mode_accepts_cd_and_rejects_unknown():
    config.set_fc_mode("cd")
    assert config.FC_MODE == config.FC_CD
    config.set_fc_mode("nfc")
    with pytest.raises(ValueError, match="Unknown fc-mode"):
        config.set_fc_mode("tools-gbnf")


def test_offline_smoke_script_passes():
    run_offline_smoke()
