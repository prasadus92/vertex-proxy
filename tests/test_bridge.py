"""OpenAI Chat Completions <-> Anthropic Messages translation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeVertex
from vertex_proxy.bridge import StreamTranslator, anthropic_to_openai, openai_to_anthropic
from vertex_proxy.bridge.response import finish_reason
from vertex_proxy.errors import ProxyError

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


# --- request ------------------------------------------------------------------


def test_system_and_developer_messages_become_the_system_prompt() -> None:
    body = openai_to_anthropic(
        {
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "developer", "content": [{"type": "text", "text": "Use metric."}]},
                {"role": "user", "content": "hi"},
            ]
        }
    )
    assert body["system"] == "Be terse.\n\nUse metric."
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_parameters_are_mapped() -> None:
    body = openai_to_anthropic(
        {
            "messages": [],
            "max_completion_tokens": 50,
            "max_tokens": 10,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": "END",
            "stream": True,
            "user": "u-1",
            "stream_options": {"include_usage": True},
        }
    )
    assert body["max_tokens"] == 50
    assert (body["temperature"], body["top_p"], body["stream"]) == (0.2, 0.9, True)
    assert body["stop_sequences"] == ["END"]
    assert body["metadata"] == {"user_id": "u-1"}
    assert "stream_options" not in body


def test_max_tokens_has_a_default() -> None:
    assert openai_to_anthropic({"messages": []})["max_tokens"] == 4096


def test_tools_and_tool_choice_are_translated() -> None:
    body = openai_to_anthropic(
        {
            "messages": [],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
    )
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Weather for a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (None, None),
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "tool", "name": "get_weather"},
        ),
    ],
)
def test_tool_choice_variants(choice: Any, expected: Any) -> None:
    request: dict[str, Any] = {"messages": [], "tools": [WEATHER_TOOL]}
    if choice is not None:
        request["tool_choice"] = choice
    assert openai_to_anthropic(request).get("tool_choice") == expected


def test_assistant_tool_calls_and_parallel_results() -> None:
    body = openai_to_anthropic(
        {
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "not json"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "4C"},
                {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "9C"}]},
                {"role": "user", "content": "thanks"},
            ]
        }
    )
    assistant, results, follow_up = body["messages"][1:]
    assert assistant["content"] == [
        {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Oslo"}},
        {"type": "tool_use", "id": "c2", "name": "get_weather", "input": {"raw": "not json"}},
    ]
    assert results == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "4C"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "9C"},
        ],
    }
    assert follow_up == {"role": "user", "content": "thanks"}


def test_empty_assistant_turns_are_dropped() -> None:
    body = openai_to_anthropic({"messages": [{"role": "assistant", "content": ""}]})
    assert body["messages"] == []


def test_images_become_image_blocks() -> None:
    body = openai_to_anthropic(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                    ],
                }
            ]
        }
    )
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.jpg"}},
    ]


@pytest.mark.parametrize(
    "request_body",
    [
        {"messages": [{"role": "narrator", "content": "x"}]},
        {"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]},
        {"messages": [], "tools": [{"type": "web_search"}]},
        {"messages": [], "tools": [WEATHER_TOOL], "tool_choice": "sometimes"},
    ],
)
def test_unsupported_inputs_are_rejected_clearly(request_body: dict[str, Any]) -> None:
    with pytest.raises(ProxyError) as caught:
        openai_to_anthropic(request_body)
    assert caught.value.status == 400


# --- response -----------------------------------------------------------------


def test_response_with_text_and_tool_use() -> None:
    result = anthropic_to_openai(
        {
            "id": "msg_1",
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "Checking. "},
                {"type": "text", "text": "One moment."},
                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Oslo"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5},
        },
        "claude-opus-5-5",
    )
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Checking. One moment."
    assert choice["message"]["tool_calls"] == [
        {
            "id": "t1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
        }
    ]
    assert result["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "total_tokens": 105,
        "prompt_tokens_details": {"cached_tokens": 90},
    }


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("unknown", "stop"),
    ],
)
def test_finish_reasons(stop_reason: str, finish: str) -> None:
    result = anthropic_to_openai(
        {"content": [{"type": "text", "text": "x"}], "stop_reason": stop_reason}, "m"
    )
    assert result["choices"][0]["finish_reason"] == finish


@pytest.mark.parametrize(
    ("stop_reason", "finish"),
    [("tool_use", "tool_calls"), ("end_turn", "tool_calls"), ("max_tokens", "length")],
)
def test_finish_reason_with_tool_calls(stop_reason: str, finish: str) -> None:
    assert finish_reason(stop_reason, has_tool_calls=True) == finish


async def test_streamed_tool_call_cut_off_by_max_tokens_finishes_with_length() -> None:
    lines = _events(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "f"},
        },
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
    )
    chunks = await _collect(StreamTranslator("m"), lines)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


async def test_streamed_tool_call_with_end_turn_finishes_with_tool_calls() -> None:
    lines = _events(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "f"},
        },
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    )
    chunks = await _collect(StreamTranslator("m"), lines)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_only_response_has_null_content() -> None:
    result = anthropic_to_openai(
        {
            "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}],
            "stop_reason": "tool_use",
        },
        "m",
    )
    assert result["choices"][0]["message"]["content"] is None


# --- streaming ------------------------------------------------------------------


def _events(*events: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for event in events:
        lines += [f"event: {event['type']}", f"data: {json.dumps(event)}", ""]
    return lines


async def _collect(translator: StreamTranslator, lines: list[str]) -> list[Any]:
    async def source() -> AsyncIterator[str]:
        for line in lines:
            yield line

    out: list[Any] = []
    async for chunk in translator.translate(source()):
        text = chunk.decode()
        assert text.startswith("data: ") and text.endswith("\n\n")
        payload = text[6:-2]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


TOOL_STREAM = _events(
    {"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Let me check."},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '"Oslo"}'},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "t2", "name": "get_time", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": "{}"},
    },
    {"type": "ping"},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}},
    {"type": "message_stop"},
)


async def test_streamed_tool_calls_are_translated() -> None:
    chunks = await _collect(StreamTranslator("claude-opus-5-5"), TOOL_STREAM)
    deltas = [c["choices"][0]["delta"] for c in chunks if c != "[DONE]"]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert deltas[1] == {"content": "Let me check."}
    assert deltas[2] == {
        "tool_calls": [
            {
                "index": 0,
                "id": "t1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            }
        ]
    }
    args = "".join(
        d["tool_calls"][0]["function"]["arguments"]
        for d in deltas
        if "tool_calls" in d and d["tool_calls"][0]["index"] == 0
    )
    assert json.loads(args) == {"city": "Oslo"}
    assert deltas[5]["tool_calls"][0]["index"] == 1
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c != "[DONE]"]
    assert finishes.count("tool_calls") == 1 and finishes[-1] == "tool_calls"
    assert chunks[-1] == "[DONE]"
    assert len({c["id"] for c in chunks if c != "[DONE]"}) == 1


async def test_usage_chunk_when_requested() -> None:
    chunks = await _collect(StreamTranslator("m", include_usage=True), TOOL_STREAM)
    usage_chunk = chunks[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 12
    assert usage_chunk["usage"]["completion_tokens"] == 30


async def test_no_usage_chunk_by_default() -> None:
    chunks = await _collect(StreamTranslator("m"), TOOL_STREAM)
    assert all(c == "[DONE]" or c["choices"] for c in chunks)


async def test_stream_error_event_is_forwarded() -> None:
    lines = _events({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
    chunks = await _collect(StreamTranslator("m"), lines)
    assert chunks == [{"error": {"type": "overloaded_error", "message": "busy"}}, "[DONE]"]


def test_openai_route_streams_tool_calls_end_to_end(
    make_client: Callable[..., TestClient], vertex: FakeVertex
) -> None:
    vertex.respond_sse(["\n".join(TOOL_STREAM[i : i + 2]) for i in range(0, len(TOOL_STREAM), 3)])
    client = make_client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5-5",
            "stream": True,
            "tools": [WEATHER_TOOL],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert vertex.last_json["tools"][0]["name"] == "get_weather"
    assert vertex.last_json["stream"] is True
    payloads = [line[6:] for line in r.text.split("\n\n") if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    assert any("tool_calls" in json.loads(p)["choices"][0]["delta"] for p in payloads[:-1])


async def test_nothing_is_emitted_after_message_stop() -> None:
    lines = _events({"type": "message_stop"}) + _events(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "late"}}
    )
    assert await _collect(StreamTranslator("m"), lines) == ["[DONE]"]
