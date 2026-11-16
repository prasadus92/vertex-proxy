"""OpenAI Responses API: translation to Chat Completions and back, and the /v1/responses route."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeVertex
from vertex_proxy.errors import ProxyError
from vertex_proxy.responses import ResponsesStream, ToolNames, chat_to_response, responses_to_chat
from vertex_proxy.responses.carrier import read_tool_extras, tool_extras_item
from vertex_proxy.responses.request import ToolName

EXEC = {
    "type": "function",
    "name": "exec_command",
    "description": "Run a command",
    "strict": False,
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}
NAMESPACE = {
    "type": "namespace",
    "name": "github",
    "description": "GitHub tools",
    "tools": [{"type": "function", "name": "open_pr", "parameters": {"type": "object"}}],
}
SIGNATURE = {"google": {"thought_signature": "sig-1"}}


# --- request ------------------------------------------------------------------


def test_codex_style_request_becomes_chat_completions() -> None:
    chat, _ = responses_to_chat(
        {
            "model": "claude-opus-5-5",
            "instructions": "You are Codex.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "sandbox"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            ],
            "tools": [EXEC, {"type": "web_search", "external_web_access": False}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"summary": "auto", "effort": "high"},
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "s1",
        }
    )
    assert chat["messages"] == [
        {"role": "system", "content": "You are Codex.\n\nsandbox"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    assert [t["function"]["name"] for t in chat["tools"]] == ["exec_command"]
    assert chat["tool_choice"] == "auto"
    assert chat["parallel_tool_calls"] is True
    assert chat["reasoning_effort"] == "high"
    assert chat["stream_options"] == {"include_usage": True}
    for dropped in ("store", "include", "prompt_cache_key", "reasoning", "instructions", "input"):
        assert dropped not in chat


def test_string_input_and_sampling_options() -> None:
    chat, _ = responses_to_chat(
        {"model": "m", "input": "hello", "max_output_tokens": 99, "temperature": 0.3, "top_p": 0.8}
    )
    assert chat["messages"] == [{"role": "user", "content": "hello"}]
    assert (chat["max_completion_tokens"], chat["temperature"], chat["top_p"]) == (99, 0.3, 0.8)
    assert "stream" not in chat


def test_namespace_tools_are_flattened_and_mapped_back() -> None:
    chat, names = responses_to_chat({"model": "m", "input": [], "tools": [NAMESPACE]})
    assert chat["tools"][0]["function"]["name"] == "github__open_pr"
    assert names.lookup("github__open_pr") == ToolName("open_pr", namespace="github")


def test_custom_tools_take_a_single_string_input() -> None:
    chat, names = responses_to_chat(
        {
            "model": "m",
            "input": [],
            "tools": [{"type": "custom", "name": "apply_patch", "description": "patch"}],
        }
    )
    parameters = chat["tools"][0]["function"]["parameters"]
    assert parameters["properties"] == {"input": {"type": "string"}}
    assert names.lookup("apply_patch").custom


def test_tool_history_becomes_assistant_tool_calls_and_tool_messages() -> None:
    chat, _ = responses_to_chat(
        {
            "model": "m",
            "tools": [EXEC, NAMESPACE],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "On it."}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                },
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "open_pr",
                    "namespace": "github",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": [{"type": "input_text", "text": "ok"}],
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "c3",
                    "name": "apply_patch",
                    "input": "*** Begin Patch",
                },
                {"type": "custom_tool_call_output", "call_id": "c3", "output": "Done"},
                {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
            ],
        }
    )
    user, assistant, tool1, tool2, patch_call, patch_result = chat["messages"]
    assert assistant["content"] == "On it."
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == [
        "exec_command",
        "github__open_pr",
    ]
    assert tool1 == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}
    assert tool2 == {"role": "tool", "tool_call_id": "c2", "content": "ok"}
    assert json.loads(patch_call["tool_calls"][0]["function"]["arguments"]) == {
        "input": "*** Begin Patch"
    }
    assert patch_result["content"] == "Done"


def test_images_in_tool_output_follow_as_a_user_message() -> None:
    chat, _ = responses_to_chat(
        {
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "view_image", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
                },
            ],
        }
    )
    assert chat["messages"][-1] == {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    }


def test_json_schema_output_becomes_response_format() -> None:
    schema = {"type": "object"}
    chat, _ = responses_to_chat(
        {
            "model": "m",
            "input": "x",
            "text": {
                "format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}
            },
        }
    )
    assert chat["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "out", "schema": schema, "strict": True},
    }


@pytest.mark.parametrize(
    "body",
    [
        {"input": [{"type": "message", "role": "user", "content": [{"type": "input_audio"}]}]},
        {"input": [], "tools": [{"type": "brand_new_tool"}]},
        {"input": [], "tools": [EXEC], "tool_choice": {"type": "allowed_tools"}},
    ],
)
def test_unsupported_inputs_are_named_400s(body: dict[str, Any]) -> None:
    with pytest.raises(ProxyError) as caught:
        responses_to_chat({"model": "m", **body})
    assert caught.value.status == 400


def test_tool_extras_round_trip_through_a_reasoning_item() -> None:
    item = tool_extras_item({"c1": SIGNATURE})
    assert item["type"] == "reasoning" and item["summary"] == []
    lookalike = "x" * 28 + base64.urlsafe_b64encode(json.dumps({"c2": SIGNATURE}).encode()).decode()
    others = [
        {"type": "reasoning", "encrypted_content": "opaque"},
        {"type": "reasoning", "encrypted_content": lookalike},
    ]
    assert read_tool_extras([item, *others]) == {"c1": SIGNATURE}
    chat, _ = responses_to_chat(
        {
            "model": "m",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "exec_command",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "x"},
                item,
            ],
        }
    )
    assert chat["messages"][0]["tool_calls"][0]["extra_content"] == SIGNATURE


# --- non-streamed response ------------------------------------------------------


def test_chat_completion_becomes_a_response() -> None:
    names = ToolNames()
    names.add(ToolName("open_pr", namespace="github"))
    result = chat_to_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "Opening it.",
                        "tool_calls": [
                            {
                                "id": "c9",
                                "type": "function",
                                "function": {"name": "github__open_pr", "arguments": "{}"},
                                "extra_content": SIGNATURE,
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
        },
        "gemini-3.5-flash",
        names,
    )
    assert result["object"] == "response" and result["status"] == "completed"
    message, call, carrier = result["output"]
    assert message["content"][0] == {
        "type": "output_text",
        "text": "Opening it.",
        "annotations": [],
    }
    assert (call["type"], call["call_id"], call["name"], call["namespace"]) == (
        "function_call",
        "c9",
        "open_pr",
        "github",
    )
    assert read_tool_extras([carrier]) == {"c9": SIGNATURE}
    assert result["usage"]["input_tokens_details"]["cached_tokens"] == 6
    assert result["usage"]["total_tokens"] == 14


def test_truncated_completion_is_incomplete() -> None:
    result = chat_to_response(
        {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]}, "m", ToolNames()
    )
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}


# --- streaming --------------------------------------------------------------------


def _chat_sse(*chunks: Any) -> list[bytes]:
    frames = [
        f"data: {json.dumps(c) if not isinstance(c, str) else c}\n\n".encode() for c in chunks
    ]
    return [b"".join(frames)[i : i + 7] for i in range(0, len(b"".join(frames)), 7)]


async def _events(stream: ResponsesStream, raw: list[bytes]) -> list[dict[str, Any]]:
    async def source() -> AsyncIterator[bytes]:
        for piece in raw:
            yield piece

    out = []
    async for frame in stream.translate(source()):
        event_line, data_line = frame.decode().strip().split("\n")
        event = json.loads(data_line.removeprefix("data: "))
        assert event_line == f"event: {event['type']}"
        out.append(event)
    return out


def _delta(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _finish(reason: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}}


async def test_streamed_text_and_tool_calls_become_items() -> None:
    names = ToolNames()
    names.add(ToolName("exec_command"))
    raw = _chat_sse(
        _delta(role="assistant", content="Let me "),
        _delta(content="look."),
        _delta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": ""},
                    "extra_content": SIGNATURE,
                }
            ]
        ),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '{"cmd":'}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '"ls"}'}}]),
        _finish("tool_calls"),
        USAGE,
        "[DONE]",
    )
    events = await _events(ResponsesStream("gemini-3.5-flash", names), raw)
    types = [e["type"] for e in events]
    assert types[:2] == ["response.created", "response.in_progress"]
    assert types[-1] == "response.completed"
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    assert deltas == "Let me look."
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert done[0]["type"] == "message" and done[0]["content"][0]["text"] == "Let me look."
    assert done[1] == {
        "type": "function_call",
        "id": done[1]["id"],
        "status": "completed",
        "call_id": "c1",
        "name": "exec_command",
        "arguments": '{"cmd":"ls"}',
    }
    assert read_tool_extras([done[2]]) == {"c1": SIGNATURE}
    completed = events[-1]["response"]
    assert completed["id"].startswith("resp_")
    assert completed["usage"]["input_tokens"] == 20 and completed["usage"]["output_tokens"] == 5
    assert [item["type"] for item in completed["output"]] == [
        "message",
        "function_call",
        "reasoning",
    ]
    added = [e for e in events if e["type"] == "response.output_item.added"]
    assert added[0]["item"]["status"] == "in_progress"
    first_delta = types.index("response.output_text.delta")
    assert types.index("response.output_item.added") < first_delta
    message_done = next(i for i, e in enumerate(events) if e["type"] == "response.output_item.done")
    call_added = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call"
    )
    assert message_done < call_added


async def test_parallel_tool_calls_are_separate_items_in_order() -> None:
    raw = _chat_sse(
        _delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}]),
        _delta(tool_calls=[{"index": 1, "id": "b", "function": {"name": "g", "arguments": ""}}]),
        _finish("tool_calls"),
        "[DONE]",
    )
    events = await _events(ResponsesStream("m", ToolNames()), raw)
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert [(i["call_id"], i["name"], i["arguments"]) for i in done] == [
        ("a", "f", "{}"),
        ("b", "g", "{}"),
    ]
    assert [e["output_index"] for e in events if e["type"] == "response.output_item.done"] == [0, 1]


async def test_streamed_custom_tool_call_is_unwrapped() -> None:
    names = ToolNames()
    names.add(ToolName("apply_patch", custom=True))
    raw = _chat_sse(
        _delta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "p",
                    "function": {"name": "apply_patch", "arguments": '{"input": "*** Begin"}'},
                }
            ]
        ),
        _finish("tool_calls"),
        "[DONE]",
    )
    events = await _events(ResponsesStream("m", names), raw)
    item = [e["item"] for e in events if e["type"] == "response.output_item.done"][0]
    assert (item["type"], item["input"]) == ("custom_tool_call", "*** Begin")
    assert not any(e["type"] == "response.function_call_arguments.delta" for e in events)


async def test_length_finish_is_response_incomplete() -> None:
    events = await _events(
        ResponsesStream("m", ToolNames()),
        _chat_sse(_delta(content="x"), _finish("length"), "[DONE]"),
    )
    assert events[-1]["type"] == "response.incomplete"
    assert events[-1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        ({"type": "rate_limit_error", "message": "slow down"}, "rate_limit_exceeded"),
        (
            {"type": "invalid_request_error", "message": "prompt is too long: 300000 tokens"},
            "context_length_exceeded",
        ),
        ({"type": "overloaded_error", "message": "busy"}, "server_error"),
    ],
)
async def test_stream_errors_become_response_failed(error: dict[str, str], code: str) -> None:
    events = await _events(
        ResponsesStream("m", ToolNames()), _chat_sse(_delta(content="x"), {"error": error})
    )
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"] == {"code": code, "message": error["message"]}


async def test_stream_that_ends_early_fails() -> None:
    events = await _events(ResponsesStream("m", ToolNames()), _chat_sse(_delta(content="x")))
    assert events[-1]["type"] == "response.failed"
    assert "ended early" in events[-1]["response"]["error"]["message"]


async def test_transport_error_mid_stream_fails() -> None:
    async def broken() -> AsyncIterator[bytes]:
        yield _chat_sse(_delta(content="x"))[0]
        raise httpx2.ReadError("reset")

    stream = ResponsesStream("m", ToolNames())
    types = [
        json.loads(f.decode().split("data: ", 1)[1])["type"]
        async for f in stream.translate(broken())
    ]
    assert types[-1] == "response.failed"


# --- route ---------------------------------------------------------------------------


def _anthropic_sse(*events: dict[str, Any]) -> list[str]:
    return [f"event: {e['type']}\ndata: {json.dumps(e)}" for e in events]


@pytest.mark.parametrize(
    "path", ["/v1/responses", "/responses", "/openai/v1/responses", "/openai/responses"]
)
def test_streamed_responses_request_for_claude(
    client: TestClient, vertex: FakeVertex, path: str
) -> None:
    vertex.respond_sse(
        _anthropic_sse(
            {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 9}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "exec_command"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"cmd":"ls"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 3},
            },
            {"type": "message_stop"},
        )
    )
    r = client.post(
        path,
        json={
            "model": "claude-opus-5-5",
            "instructions": "be useful",
            "input": "list files",
            "tools": [EXEC],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert str(vertex.last.url).endswith(
        "/publishers/anthropic/models/claude-opus-5-5:streamRawPredict"
    )
    assert vertex.last_json["system"] == "be useful"
    assert vertex.last_json["tools"][0]["name"] == "exec_command"
    events = [json.loads(block.split("data: ", 1)[1]) for block in r.text.strip().split("\n\n")]
    assert events[-1]["type"] == "response.completed"
    call = [e["item"] for e in events if e["type"] == "response.output_item.done"][0]
    assert (call["call_id"], call["arguments"]) == ("toolu_1", '{"cmd":"ls"}')
    assert events[-1]["response"]["usage"]["input_tokens"] == 9


def test_non_streamed_responses_request_for_gemini(client: TestClient, vertex: FakeVertex) -> None:
    vertex.respond(
        200,
        {
            "choices": [
                {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
            ]
        },
    )
    r = client.post("/v1/responses", json={"model": "gemini-3.5-flash", "input": "hi"})
    assert r.status_code == 200
    assert str(vertex.last.url).endswith("/endpoints/openapi/chat/completions")
    assert vertex.last_json["messages"] == [{"role": "user", "content": "hi"}]
    assert r.json()["output"][0]["content"][0]["text"] == "hello"


def test_context_overflow_before_streaming_is_reported_in_stream(
    client: TestClient, vertex: FakeVertex
) -> None:
    vertex.respond(
        400,
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 999999 tokens > 1000000 maximum",
            },
        },
    )
    r = client.post(
        "/v1/responses", json={"model": "claude-opus-5-5", "input": "hi", "stream": True}
    )
    assert r.status_code == 200
    last = json.loads(r.text.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert last["type"] == "response.failed"
    assert last["response"]["error"]["code"] == "context_length_exceeded"


def test_other_errors_before_streaming_are_http_errors(
    client: TestClient, vertex: FakeVertex
) -> None:
    vertex.respond(
        429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )
    r = client.post(
        "/v1/responses", json={"model": "claude-opus-5-5", "input": "hi", "stream": True}
    )
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"


def test_responses_route_requires_the_api_key(make_client: Callable[..., TestClient]) -> None:
    client = make_client(api_key="k")
    assert client.post("/v1/responses", json={"model": "m", "input": "x"}).status_code == 401


# --- review regressions ------------------------------------------------------------


async def test_parallel_calls_without_index_in_one_chunk_stay_separate() -> None:
    raw = _chat_sse(
        _delta(
            tool_calls=[
                {"id": "a", "function": {"name": "f", "arguments": '{"x":1}'}},
                {"id": "b", "function": {"name": "f", "arguments": '{"x":2}'}},
            ]
        ),
        _finish("tool_calls"),
        "[DONE]",
    )
    done = [
        e["item"]
        for e in await _events(ResponsesStream("m", ToolNames()), raw)
        if e["type"] == "response.output_item.done"
    ]
    assert [(i["call_id"], i["arguments"]) for i in done] == [("a", '{"x":1}'), ("b", '{"x":2}')]


async def test_name_and_id_that_arrive_late_are_kept() -> None:
    names = ToolNames()
    names.add(ToolName("open_pr", namespace="github"))
    raw = _chat_sse(
        _delta(tool_calls=[{"index": 0, "function": {"arguments": "{"}}]),
        _delta(tool_calls=[{"index": 0, "id": "late-id", "function": {"name": "github__"}}]),
        _delta(tool_calls=[{"index": 0, "function": {"name": "open_pr", "arguments": "}"}}]),
        _finish("tool_calls"),
        "[DONE]",
    )
    events = await _events(ResponsesStream("m", names), raw)
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert len(done) == 1
    assert (done[0]["call_id"], done[0]["name"], done[0]["namespace"], done[0]["arguments"]) == (
        "late-id",
        "open_pr",
        "github",
        "{}",
    )
    streamed = "".join(
        e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta"
    )
    assert streamed == "{}"


async def test_text_after_a_tool_call_is_a_new_message() -> None:
    raw = _chat_sse(
        _delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}]),
        _delta(content="and also"),
        _finish("stop"),
        "[DONE]",
    )
    events = await _events(ResponsesStream("m", ToolNames()), raw)
    done = [e["item"]["type"] for e in events if e["type"] == "response.output_item.done"]
    assert sorted(done) == ["function_call", "message"]
    assert events[-1]["response"]["output"][1]["content"][0]["text"] == "and also"


def test_images_from_parallel_tool_outputs_follow_the_whole_run() -> None:
    chat, _ = responses_to_chat(
        {
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "a", "name": "view_image", "arguments": "{}"},
                {
                    "type": "function_call",
                    "call_id": "b",
                    "name": "exec_command",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "a",
                    "output": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}],
                },
                {"type": "function_call_output", "call_id": "b", "output": "done"},
                {"type": "message", "role": "user", "content": "next"},
            ],
        }
    )
    assert [m["role"] for m in chat["messages"]] == ["assistant", "tool", "tool", "user", "user"]
    assert chat["messages"][3]["content"][0]["type"] == "image_url"


def test_custom_tools_inside_a_namespace_round_trip() -> None:
    chat, names = responses_to_chat(
        {
            "model": "m",
            "input": [],
            "tools": [
                {
                    "type": "namespace",
                    "name": "ns",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "patch",
                            "description": "Apply a patch.",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": "start: /.+/",
                            },
                        }
                    ],
                }
            ],
        }
    )
    function = chat["tools"][0]["function"]
    assert function["name"] == "ns__patch"
    assert "lark grammar" in function["description"] and "start: /.+/" in function["description"]
    assert names.lookup("ns__patch") == ToolName("patch", namespace="ns", custom=True)
    result = chat_to_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c",
                                "function": {"name": "ns__patch", "arguments": '{"input": "diff"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        "m",
        names,
    )
    item = result["output"][0]
    assert (item["type"], item["name"], item["namespace"], item["input"]) == (
        "custom_tool_call",
        "patch",
        "ns",
        "diff",
    )


@pytest.mark.parametrize(
    "chunk",
    [
        [{"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}],
        {"error": {"code": 429, "message": "Resource exhausted"}},
    ],
)
async def test_google_shaped_stream_errors_are_rate_limits(chunk: Any) -> None:
    events = await _events(ResponsesStream("m", ToolNames()), _chat_sse(_delta(content="x"), chunk))
    assert events[-1]["response"]["error"]["code"] == "rate_limit_exceeded"


async def test_chunks_that_are_not_objects_are_skipped() -> None:
    raw = _chat_sse(["not", "an", "object"], 7, _delta(content="ok"), _finish("stop"), "[DONE]")
    events = await _events(ResponsesStream("m", ToolNames()), raw)
    assert events[-1]["type"] == "response.completed"


def test_agent_messages_become_user_turns_and_unknown_items_are_dropped() -> None:
    chat, _ = responses_to_chat(
        {
            "model": "m",
            "input": [
                {
                    "type": "agent_message",
                    "author": "planner",
                    "recipient": "me",
                    "content": [{"type": "input_text", "text": "do X"}],
                },
                {"type": "configuration_update", "anything": True},
                {"type": "compaction", "encrypted_content": "zz"},
            ],
        }
    )
    assert chat["messages"] == [{"role": "user", "content": "[planner] do X"}]


def test_system_messages_after_user_turns_are_hoisted_to_the_front() -> None:
    chat, _ = responses_to_chat(
        {
            "model": "m",
            "instructions": "base",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "env"}],
                },
                {"type": "message", "role": "system", "content": "more"},
            ],
        }
    )
    assert chat["messages"] == [
        {"role": "system", "content": "base\n\nenv\n\nmore"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.parametrize("content", ["", [], [{"type": "input_text", "text": ""}]])
def test_empty_user_messages_are_skipped(content: Any) -> None:
    chat, _ = responses_to_chat(
        {"model": "m", "input": [{"type": "message", "role": "user", "content": content}]}
    )
    assert chat["messages"] == []


@pytest.mark.parametrize(
    "items",
    [
        ["not-a-dict", {"type": "reasoning", "encrypted_content": None}],
        [
            {
                "type": "reasoning",
                "encrypted_content": tool_extras_item({"c": SIGNATURE})["encrypted_content"][:-4],
            }
        ],
    ],
)
def test_carrier_ignores_malformed_items(items: list[Any]) -> None:
    assert read_tool_extras(items) == {}


def test_carrier_ignores_payloads_that_are_not_maps_of_objects() -> None:
    prefix = tool_extras_item({})["encrypted_content"].split(":", 1)[0] + ":"
    for payload in ([1, 2], {"c": "not-an-object"}):
        content = prefix + base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        assert read_tool_extras([{"type": "reasoning", "encrypted_content": content}]) == {}


async def test_content_filter_finish_is_incomplete() -> None:
    events = await _events(
        ResponsesStream("m", ToolNames()),
        _chat_sse(_delta(content="x"), _finish("content_filter"), "[DONE]"),
    )
    assert events[-1]["type"] == "response.incomplete"
    assert events[-1]["response"]["incomplete_details"] == {"reason": "content_filter"}


def test_previous_response_id_is_rejected() -> None:
    with pytest.raises(ProxyError, match="previous_response_id"):
        responses_to_chat({"model": "m", "input": "x", "previous_response_id": "resp_1"})


@pytest.mark.parametrize(
    ("effort", "sent"),
    [("xhigh", "high"), ("minimal", "low"), ("medium", "medium"), ("none", None)],
)
def test_reasoning_effort_is_mapped_to_chat_values(effort: str, sent: str | None) -> None:
    chat, _ = responses_to_chat({"model": "m", "input": "x", "reasoning": {"effort": effort}})
    assert chat.get("reasoning_effort") == sent


def test_responses_stream_is_always_event_stream(client: TestClient, vertex: FakeVertex) -> None:
    body = 'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    vertex.reply = lambda _r: httpx2.Response(
        200, content=body.encode(), headers={"content-type": "application/json"}
    )
    r = client.post(
        "/v1/responses", json={"model": "gemini-3.5-flash", "input": "hi", "stream": True}
    )
    assert r.headers["content-type"].startswith("text/event-stream")


async def test_a_new_id_at_the_same_index_starts_a_new_call() -> None:
    raw = _chat_sse(
        _delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}]),
        _delta(tool_calls=[{"index": 0, "id": "b", "function": {"name": "g", "arguments": "{}"}}]),
        _finish("tool_calls"),
        "[DONE]",
    )
    events = await _events(ResponsesStream("m", ToolNames()), raw)
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert [(i["call_id"], i["name"]) for i in done] == [("a", "f"), ("b", "g")]


async def test_custom_tool_with_a_late_name_streams_no_argument_deltas() -> None:
    names = ToolNames()
    names.add(ToolName("apply_patch", custom=True))
    raw = _chat_sse(
        _delta(tool_calls=[{"index": 0, "id": "p", "function": {"arguments": '{"input": '}}]),
        _delta(tool_calls=[{"index": 0, "function": {"name": "apply_patch", "arguments": '"x"}'}}]),
        _finish("tool_calls"),
        "[DONE]",
    )
    events = await _events(ResponsesStream("m", names), raw)
    assert not any(e["type"].startswith("response.function_call_arguments") for e in events)
    item = [e["item"] for e in events if e["type"] == "response.output_item.done"][0]
    assert (item["type"], item["input"]) == ("custom_tool_call", "x")
