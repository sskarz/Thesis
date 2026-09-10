"""Unit tests for Gemma native tool-call content parsing."""

from benchmark.parse_gemma_tool_calls import parse_gemma_tool_call_content


PARALLEL_SPOTIFY = (
    '<|tool_call>call:spotify.play{artist:<|"|>Taylor Swift<|"|>,duration:20}<tool_call|>'
    '<|tool_call>call:spotify.play{artist:<|"|>Maroon 5<|"|>,duration:15}<tool_call|>'
)


def test_two_parallel_concatenated_calls():
    parsed = parse_gemma_tool_call_content(PARALLEL_SPOTIFY)
    assert parsed == [
        {"spotify.play": {"artist": "Taylor Swift", "duration": 20}},
        {"spotify.play": {"artist": "Maroon 5", "duration": 15}},
    ]


def test_dotted_function_names():
    content = (
        "<|tool_call>call:math_toolkit.sum_of_multiples"
        "{numbers:[15],multiple:3}<tool_call|>"
        "<|tool_call>call:area_rectangle.calculate"
        "{length:5.5,width:2}<tool_call|>"
    )
    parsed = parse_gemma_tool_call_content(content)
    assert parsed == [
        {"math_toolkit.sum_of_multiples": {"numbers": [15], "multiple": 3}},
        {"area_rectangle.calculate": {"length": 5.5, "width": 2}},
    ]


def test_string_wrap_ints_floats_bools_lists():
    content = (
        "<|tool_call>call:demo.run{"
        'title:<|"|>Hello world<|"|>,'
        "count:20,"
        "ratio:1.5,"
        "ok:true,"
        "skip:false,"
        "items:[15, 30]"
        "}<tool_call|>"
    )
    parsed = parse_gemma_tool_call_content(content)
    assert parsed == [
        {
            "demo.run": {
                "title": "Hello world",
                "count": 20,
                "ratio": 1.5,
                "ok": True,
                "skip": False,
                "items": [15, 30],
            }
        }
    ]


def test_colon_inside_wrapped_string_is_not_a_key_split():
    content = (
        "<|tool_call>call:logger.write"
        '{message:<|"|>Error: not found<|"|>}<tool_call|>'
    )
    parsed = parse_gemma_tool_call_content(content)
    assert parsed == [{"logger.write": {"message": "Error: not found"}}]


def test_prose_clarify_string_returns_none():
    assert parse_gemma_tool_call_content(
        "Which playlist should I play? Please clarify the artist."
    ) is None
    assert parse_gemma_tool_call_content(
        "I should call spotify.play but I need the duration first."
    ) is None


def test_functiongemma_and_qwen_markup_are_not_stolen():
    assert parse_gemma_tool_call_content(
        "<start_function_call>"
        "call:lookup_weather{city:<escape>San Francisco<escape>}"
        "<end_function_call>"
    ) is None
    assert parse_gemma_tool_call_content(
        '<tool_call>\n{"name": "math.factorial", "arguments": {"number": 5}}\n</tool_call>'
    ) is None
