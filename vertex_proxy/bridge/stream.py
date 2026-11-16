"""Anthropic Messages SSE stream to OpenAI Chat Completions chunk stream."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .response import finish_reason, openai_usage

DONE = b"data: [DONE]\n\n"


class StreamTranslator:
    """Translates 1 Anthropic stream. Tool calls are numbered in the order their blocks start."""

    def __init__(self, model: str, *, include_usage: bool = False) -> None:
        self._model = model
        self._include_usage = include_usage
        self._id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self._created = int(time.time())
        self._tool_index_by_block: dict[int, int] = {}
        self._usage: dict[str, Any] = {}
        self._finished = False

    async def translate(self, lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
        async for line in lines:
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            for chunk in self.on_event(event):
                yield chunk
            if self._finished:
                return

    def on_event(self, event: dict[str, Any]) -> list[bytes]:
        kind = event.get("type")
        if kind == "message_start":
            self._usage.update(event.get("message", {}).get("usage") or {})
            return [self._chunk({"role": "assistant", "content": ""})]
        if kind == "content_block_start":
            return self._block_start(event.get("index", 0), event.get("content_block") or {})
        if kind == "content_block_delta":
            return self._block_delta(event.get("index", 0), event.get("delta") or {})
        if kind == "message_delta":
            self._usage.update(event.get("usage") or {})
            reason = finish_reason(
                event.get("delta", {}).get("stop_reason"), bool(self._tool_index_by_block)
            )
            return [self._chunk({}, finish_reason=reason)]
        if kind == "message_stop":
            self._finished = True
            usage = [self._usage_chunk()] if self._include_usage else []
            return [*usage, DONE]
        if kind == "error":
            self._finished = True
            return [_sse({"error": event.get("error") or {}}), DONE]
        return []

    def _block_start(self, index: int, block: dict[str, Any]) -> list[bytes]:
        if block.get("type") != "tool_use":
            return []
        tool_index = len(self._tool_index_by_block)
        self._tool_index_by_block[index] = tool_index
        call = {
            "index": tool_index,
            "id": block.get("id", ""),
            "type": "function",
            "function": {"name": block.get("name", ""), "arguments": ""},
        }
        return [self._chunk({"tool_calls": [call]})]

    def _block_delta(self, index: int, delta: dict[str, Any]) -> list[bytes]:
        if delta.get("type") == "text_delta":
            return [self._chunk({"content": delta.get("text", "")})]
        if delta.get("type") == "input_json_delta" and index in self._tool_index_by_block:
            call = {
                "index": self._tool_index_by_block[index],
                "function": {"arguments": delta.get("partial_json", "")},
            }
            return [self._chunk({"tool_calls": [call]})]
        return []

    def _base(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
        }

    def _chunk(self, delta: dict[str, Any], *, finish_reason: str | None = None) -> bytes:
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        return _sse({**self._base(), "choices": [choice]})

    def _usage_chunk(self) -> bytes:
        return _sse({**self._base(), "choices": [], "usage": openai_usage(self._usage)})


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()
