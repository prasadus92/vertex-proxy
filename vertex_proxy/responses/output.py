"""Responses API output items, usage and response objects built from Chat Completions data."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .request import ToolName

CONTEXT_OVERFLOW_PHRASES = (
    "prompt is too long",
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def message_item(item_id: str, text: str, status: str = "completed") -> dict[str, Any]:
    content = (
        [{"type": "output_text", "text": text, "annotations": []}] if status == "completed" else []
    )
    return {
        "type": "message",
        "id": item_id,
        "status": status,
        "role": "assistant",
        "content": content,
    }


def tool_call_item(
    item_id: str, call_id: str, tool: ToolName, arguments: str, status: str = "completed"
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "custom_tool_call" if tool.custom else "function_call",
        "id": item_id,
        "status": status,
        "call_id": call_id,
        "name": tool.name,
    }
    if tool.custom:
        item["input"] = _custom_input(arguments)
    else:
        item["arguments"] = arguments
    if tool.namespace:
        item["namespace"] = tool.namespace
    return item


def _custom_input(arguments: str) -> str:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    return parsed.get("input", "") if isinstance(parsed, dict) else arguments


def responses_usage(chat_usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = chat_usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": prompt,
        "input_tokens_details": {
            "cached_tokens": int(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            )
        },
        "output_tokens": completion,
        "output_tokens_details": {
            "reasoning_tokens": int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            )
        },
        "total_tokens": int(usage.get("total_tokens") or prompt + completion),
    }


def response_object(
    response_id: str,
    model: str,
    *,
    status: str,
    output: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at or int(time.time()),
        "status": status,
        "model": model,
        "output": output or [],
        "usage": usage,
        "error": error,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
    }


_INCOMPLETE_REASONS = {"length": "max_output_tokens", "content_filter": "content_filter"}


def outcome(finish_reason: str | None) -> tuple[str, str | None]:
    """The response status and incomplete reason for a Chat Completions finish_reason."""
    reason = _INCOMPLETE_REASONS.get(finish_reason or "")
    return ("incomplete", reason) if reason else ("completed", None)


def failure_code(error: dict[str, Any], message: str) -> str:
    """The Responses error code that makes a client such as Codex react the right way.

    ``error`` is an OpenAI-shaped (``type``) or Google-shaped (``code``, ``status``) error.
    """
    lowered = message.lower()
    if any(phrase in lowered for phrase in CONTEXT_OVERFLOW_PHRASES):
        return "context_length_exceeded"
    kind = error.get("type")
    if (
        kind == "rate_limit_error"
        or error.get("code") == 429
        or error.get("status") == "RESOURCE_EXHAUSTED"
    ):
        return "rate_limit_exceeded"
    if kind == "invalid_request_error":
        return "invalid_prompt"
    return "server_error"
