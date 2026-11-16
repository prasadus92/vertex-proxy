"""Anthropic Messages response to OpenAI Chat Completions response."""

from __future__ import annotations

import json
import time
from typing import Any

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def finish_reason(stop_reason: str | None, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return FINISH_REASONS.get(stop_reason or "end_turn", "stop")


def openai_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """OpenAI usage from Anthropic usage; cached input counts as prompt tokens, as in OpenAI."""
    cached = usage.get("cache_read_input_tokens") or 0
    prompt = (
        (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0) + cached
    )
    completion = usage.get("output_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def anthropic_to_openai(response: dict[str, Any], model: str) -> dict[str, Any]:
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in response.get("content") or []:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{response.get('id', '')}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason(response.get("stop_reason"), bool(tool_calls)),
            }
        ],
        "usage": openai_usage(response.get("usage") or {}),
    }
