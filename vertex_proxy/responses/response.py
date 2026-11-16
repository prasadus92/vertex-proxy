"""Chat Completions response to Responses API response."""

from __future__ import annotations

from typing import Any

from .carrier import tool_extras_item
from .output import (
    message_item,
    new_id,
    outcome,
    response_object,
    responses_usage,
    tool_call_item,
)
from .request import ToolNames


def chat_to_response(chat: dict[str, Any], model: str, names: ToolNames) -> dict[str, Any]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    if message.get("content"):
        output.append(message_item(new_id("msg"), message["content"]))
    extras: dict[str, Any] = {}
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        call_id = call.get("id") or new_id("call")
        tool = names.lookup(function.get("name", ""))
        output.append(
            tool_call_item(new_id("fc"), call_id, tool, function.get("arguments") or "{}")
        )
        if call.get("extra_content"):
            extras[call_id] = call["extra_content"]
    if extras:
        output.append(tool_extras_item(extras))
    status, reason = outcome(choice.get("finish_reason"))
    return response_object(
        new_id("resp"),
        model,
        status=status,
        output=output,
        usage=responses_usage(chat.get("usage")),
        incomplete_reason=reason,
    )
