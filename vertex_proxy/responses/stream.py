"""Chat Completions SSE stream to Responses API SSE events."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx2

from .carrier import tool_extras_item
from .output import (
    failure_code,
    message_item,
    new_id,
    outcome,
    response_object,
    responses_usage,
    tool_call_item,
)
from .request import ToolName, ToolNames

_DONE = "[DONE]"


@dataclass
class _OpenMessage:
    item_id: str
    output_index: int
    text: str = ""


@dataclass
class _OpenCall:
    item_id: str
    output_index: int
    call_id: str
    has_upstream_id: bool = False
    name: str = ""
    arguments: str = ""
    streamed: int = 0


class ResponsesStream:
    """Translates a single Chat Completions stream into Responses events."""

    def __init__(self, model: str, names: ToolNames) -> None:
        self._model = model
        self._names = names
        self._id = new_id("resp")
        self._created = int(time.time())
        self._sequence = 0
        self._next_index = 0
        self._message: _OpenMessage | None = None
        self._calls: list[_OpenCall] = []
        self._call_by_index: dict[int, _OpenCall] = {}
        self._done_items: list[tuple[int, dict[str, Any]]] = []
        self._extras: dict[str, Any] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None

    async def translate(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        for event in self.start():
            yield event
        ended = False
        try:
            async for payload in _sse_payloads(chunks):
                if payload == _DONE:
                    ended = True
                    break
                if isinstance(payload, list) and len(payload) == 1:
                    payload = payload[0]
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("error"), dict):
                    error = payload["error"]
                    message = str(error.get("message") or "upstream error")
                    yield self.fail(failure_code(error, message), message)
                    return
                for event in self.on_chunk(payload):
                    yield event
        except httpx2.HTTPError as exc:
            yield self.fail("server_error", f"vertex-proxy: upstream stream failed: {exc}")
            return
        if not ended and self._finish_reason is None:
            yield self.fail("server_error", "vertex-proxy: upstream stream ended early")
            return
        for event in self.finish():
            yield event

    def start(self) -> list[bytes]:
        snapshot = self._snapshot("in_progress")
        return [
            self._event("response.created", response=snapshot),
            self._event("response.in_progress", response=snapshot),
        ]

    def on_chunk(self, chunk: dict[str, Any]) -> list[bytes]:
        if chunk.get("usage"):
            self._usage = chunk["usage"]
        events: list[bytes] = []
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                events += self._text(delta["content"])
            for call in delta.get("tool_calls") or []:
                events += self._tool_call(call)
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]
        return events

    def finish(self) -> list[bytes]:
        events = self._close_message()
        for call in self._calls:
            events += self._close_call(call)
        self._calls.clear()
        if self._extras:
            item = tool_extras_item(self._extras)
            output_index = self._claim_index()
            events.append(
                self._event("response.output_item.added", output_index=output_index, item=item)
            )
            events += self._item_done(output_index, item)
        status, reason = outcome(self._finish_reason)
        kind = "response.completed" if status == "completed" else "response.incomplete"
        return [*events, self._event(kind, response=self._snapshot(status, reason=reason))]

    def fail(self, code: str, message: str) -> bytes:
        snapshot = self._snapshot("failed", error={"code": code, "message": message})
        return self._event("response.failed", response=snapshot)

    def _text(self, text: str) -> list[bytes]:
        events: list[bytes] = []
        if self._message is None:
            m = self._message = _OpenMessage(new_id("msg"), self._claim_index())
            events.append(
                self._event(
                    "response.output_item.added",
                    output_index=m.output_index,
                    item=message_item(m.item_id, "", status="in_progress"),
                )
            )
            events.append(
                self._event(
                    "response.content_part.added",
                    item_id=m.item_id,
                    output_index=m.output_index,
                    content_index=0,
                    part=_text_part(""),
                )
            )
        m = self._message
        m.text += text
        events.append(
            self._event(
                "response.output_text.delta",
                item_id=m.item_id,
                output_index=m.output_index,
                content_index=0,
                delta=text,
                logprobs=[],
            )
        )
        return events

    def _tool_call(self, delta: dict[str, Any]) -> list[bytes]:
        """Matches a delta to its call by id, then by index; a new id is always a new call."""
        events: list[bytes] = []
        index = delta.get("index")
        call_id = delta.get("id")
        call = self._find_call(index, call_id)
        if call is not None and call_id and not call.has_upstream_id:
            call.call_id, call.has_upstream_id = call_id, True
        if call is None or (call_id and call_id != call.call_id):
            events += self._close_message()
            call = _OpenCall(
                new_id("fc"), self._claim_index(), call_id or new_id("call"), bool(call_id)
            )
            self._calls.append(call)
            if index is not None:
                self._call_by_index[index] = call
            item = tool_call_item(
                call.item_id, call.call_id, ToolName(""), "", status="in_progress"
            )
            events.append(
                self._event("response.output_item.added", output_index=call.output_index, item=item)
            )
        function = delta.get("function") or {}
        call.name += function.get("name") or ""
        call.arguments += function.get("arguments") or ""
        if delta.get("extra_content"):
            self._extras[call.call_id] = delta["extra_content"]
        return events + self._argument_deltas(call)

    def _find_call(self, index: int | None, call_id: str | None) -> _OpenCall | None:
        if call_id:
            for call in self._calls:
                if call.has_upstream_id and call.call_id == call_id:
                    return call
        if index is not None:
            return self._call_by_index.get(index)
        return self._calls[-1] if self._calls and not call_id else None

    def _argument_deltas(self, call: _OpenCall) -> list[bytes]:
        """Argument deltas wait until the name says whether the call is a custom tool."""
        if not call.name or self._names.lookup(call.name).custom:
            return []
        fragment = call.arguments[call.streamed :]
        if not fragment:
            return []
        call.streamed = len(call.arguments)
        return [
            self._event(
                "response.function_call_arguments.delta",
                item_id=call.item_id,
                output_index=call.output_index,
                delta=fragment,
            )
        ]

    def _close_message(self) -> list[bytes]:
        m = self._message
        if m is None:
            return []
        self._message = None
        common = {"item_id": m.item_id, "output_index": m.output_index, "content_index": 0}
        return [
            self._event("response.output_text.done", **common, text=m.text, logprobs=[]),
            self._event("response.content_part.done", **common, part=_text_part(m.text)),
            *self._item_done(m.output_index, message_item(m.item_id, m.text)),
        ]

    def _close_call(self, call: _OpenCall) -> list[bytes]:
        tool = self._names.lookup(call.name)
        arguments = call.arguments or "{}"
        events: list[bytes] = []
        if not tool.custom:
            events.append(
                self._event(
                    "response.function_call_arguments.done",
                    item_id=call.item_id,
                    output_index=call.output_index,
                    name=tool.name,
                    arguments=arguments,
                )
            )
        item = tool_call_item(call.item_id, call.call_id, tool, arguments)
        return [*events, *self._item_done(call.output_index, item)]

    def _item_done(self, output_index: int, item: dict[str, Any]) -> list[bytes]:
        self._done_items.append((output_index, item))
        return [self._event("response.output_item.done", output_index=output_index, item=item)]

    def _claim_index(self) -> int:
        self._next_index += 1
        return self._next_index - 1

    def _snapshot(
        self, status: str, *, error: dict[str, Any] | None = None, reason: str | None = None
    ) -> dict[str, Any]:
        output = [item for _, item in sorted(self._done_items, key=lambda pair: pair[0])]
        usage = responses_usage(self._usage) if status in ("completed", "incomplete") else None
        return response_object(
            self._id,
            self._model,
            status=status,
            output=output,
            usage=usage,
            error=error,
            incomplete_reason=reason,
            created_at=self._created,
        )

    def _event(self, kind: str, **fields: Any) -> bytes:
        payload = {"type": kind, "sequence_number": self._sequence, **fields}
        self._sequence += 1
        return f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


async def _sse_payloads(chunks: AsyncIterator[bytes]) -> AsyncIterator[Any]:
    """The decoded `data:` payloads of an SSE byte stream, with "[DONE]" passed through as text."""
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == _DONE:
                yield _DONE
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
