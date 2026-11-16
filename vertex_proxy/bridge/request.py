"""OpenAI Chat Completions request body to Anthropic Messages request body."""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import ProxyError

DEFAULT_MAX_TOKENS = 4096

_DATA_URL = re.compile(r"^data:(?P<media_type>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)
_SYSTEM_ROLES = frozenset({"system", "developer"})


def openai_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    system: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        if role in _SYSTEM_ROLES:
            system.extend(_texts(message.get("content")))
        elif role == "tool":
            _append_tool_result(messages, message)
        elif role == "assistant":
            converted = _assistant(message)
            if converted["content"]:
                messages.append(converted)
        elif role == "user":
            messages.append({"role": "user", "content": _user_content(message.get("content"))})
        else:
            raise ProxyError.bad_request(f"unsupported message role {role!r}")

    result: dict[str, Any] = {"messages": messages}
    if system:
        result["system"] = "\n\n".join(system)
    result["max_tokens"] = (
        body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS
    )
    for key in ("temperature", "top_p", "stream"):
        if body.get(key) is not None:
            result[key] = body[key]
    if (stop := body.get("stop")) is not None:
        result["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if isinstance(body.get("user"), str):
        result["metadata"] = {"user_id": body["user"]}
    if tools := body.get("tools"):
        result["tools"] = [_tool(tool) for tool in tools]
        if (choice := _tool_choice(body)) is not None:
            result["tool_choice"] = choice
    return result


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content] if content else []
    return [
        part["text"]
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
    ]


def _user_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        kind = part.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif kind == "image_url":
            url = part.get("image_url", {}).get("url", "")
            blocks.append({"type": "image", "source": _image_source(url)})
        else:
            raise ProxyError.bad_request(f"unsupported content part type {kind!r}")
    return blocks


def _image_source(url: str) -> dict[str, str]:
    if match := _DATA_URL.match(url):
        return {"type": "base64", "media_type": match["media_type"], "data": match["data"]}
    return {"type": "url", "url": url}


def _assistant(message: dict[str, Any]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": t} for t in _texts(message.get("content"))
    ]
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "input": _arguments(function.get("arguments")),
            }
        )
    return {"role": "assistant", "content": blocks}


def _arguments(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw or {}
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _append_tool_result(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    """Results of parallel tool calls must share 1 user message, so consecutive ones merge."""
    content = message.get("content")
    block = {
        "type": "tool_result",
        "tool_use_id": message.get("tool_call_id", ""),
        "content": "\n".join(_texts(content)) if isinstance(content, list) else content or "",
    }
    previous = messages[-1] if messages else None
    if (
        previous is not None
        and previous["role"] == "user"
        and isinstance(previous["content"], list)
        and previous["content"]
        and all(b.get("type") == "tool_result" for b in previous["content"])
    ):
        previous["content"].append(block)
    else:
        messages.append({"role": "user", "content": [block]})


def _tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        raise ProxyError.bad_request(f"unsupported tool type {tool.get('type')!r}")
    function = tool.get("function", {})
    converted = {
        "name": function.get("name", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }
    if function.get("description"):
        converted["description"] = function["description"]
    return converted


def _tool_choice(body: dict[str, Any]) -> dict[str, Any] | None:
    choice = body.get("tool_choice")
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        converted: dict[str, Any] = {"type": "any"}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        converted = {"type": "tool", "name": choice.get("function", {}).get("name", "")}
    elif choice in (None, "auto"):
        converted = {"type": "auto"}
    else:
        raise ProxyError.bad_request(f"unsupported tool_choice {choice!r}")
    if body.get("parallel_tool_calls") is False:
        converted["disable_parallel_tool_use"] = True
    elif choice is None:
        return None
    return converted
