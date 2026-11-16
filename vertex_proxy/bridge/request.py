"""OpenAI Chat Completions request body to Anthropic Messages request body."""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import ProxyError

DEFAULT_MAX_TOKENS = 8192
MAX_TEMPERATURE = 1.0

_DATA_URL = re.compile(r"^data:(?P<media_type>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)
_SYSTEM_ROLES = frozenset({"system", "developer"})
_UNSUPPORTED_RESPONSE_FORMATS = frozenset({"json_object", "json_schema"})

Block = dict[str, Any]


def openai_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    _reject_unsupported_options(body)
    system: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        role = message.get("role")
        if role in _SYSTEM_ROLES:
            system.extend(_texts(message.get("content")))
        elif role == "tool":
            _append(messages, "user", [_tool_result(message)])
        elif role == "assistant":
            _append(messages, "assistant", _assistant_blocks(message))
        elif role == "user":
            blocks = _user_blocks(message.get("content"))
            if not blocks:
                raise ProxyError.bad_request("a user message has no content")
            _append(messages, "user", blocks)
        else:
            raise ProxyError.bad_request(f"unsupported message role {role!r}")

    result: dict[str, Any] = {"messages": [_compact(m) for m in messages]}
    if system:
        result["system"] = "\n\n".join(system)
    result["max_tokens"] = (
        body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS
    )
    _copy_sampling(body, result)
    if (stop := body.get("stop")) is not None:
        result["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if isinstance(body.get("user"), str):
        result["metadata"] = {"user_id": body["user"]}
    if tools := body.get("tools"):
        result["tools"] = [_tool(tool) for tool in tools]
        if (choice := _tool_choice(body)) is not None:
            result["tool_choice"] = choice
    return result


def _reject_unsupported_options(body: dict[str, Any]) -> None:
    if (body.get("n") or 1) > 1:
        raise ProxyError.bad_request("n > 1 is not supported for Claude models")
    response_format = (body.get("response_format") or {}).get("type")
    if response_format in _UNSUPPORTED_RESPONSE_FORMATS:
        raise ProxyError.bad_request(
            f"response_format {response_format!r} is not supported for Claude models; "
            "describe the JSON you want in the prompt, or use a tool"
        )


def _copy_sampling(body: dict[str, Any], result: dict[str, Any]) -> None:
    """Claude takes temperature in [0, 1] and, on current models, not with top_p as well."""
    temperature = body.get("temperature")
    if temperature is not None:
        if temperature > MAX_TEMPERATURE:
            raise ProxyError.bad_request(
                f"temperature {temperature} is above {MAX_TEMPERATURE}, the maximum for Claude"
            )
        result["temperature"] = temperature
    elif body.get("top_p") is not None:
        result["top_p"] = body["top_p"]
    if body.get("stream") is not None:
        result["stream"] = body["stream"]


def _append(messages: list[dict[str, Any]], role: str, blocks: list[Block]) -> None:
    """Claude needs alternating roles, so a turn with the same role as the last one is merged."""
    if not blocks:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": list(blocks)})


def _compact(message: dict[str, Any]) -> dict[str, Any]:
    content = message["content"]
    if len(content) == 1 and content[0]["type"] == "text":
        return {"role": message["role"], "content": content[0]["text"]}
    return message


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content] if content else []
    return [
        part["text"]
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
    ]


def _user_blocks(content: Any) -> list[Block]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[Block] = []
    for part in content or []:
        kind = part.get("type")
        if kind == "text":
            if part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            image = part.get("image_url")
            url = image if isinstance(image, str) else (image or {}).get("url", "")
            blocks.append({"type": "image", "source": _image_source(url)})
        else:
            raise ProxyError.bad_request(f"unsupported content part type {kind!r}")
    return blocks


def _image_source(url: str) -> dict[str, str]:
    if match := _DATA_URL.match(url):
        return {"type": "base64", "media_type": match["media_type"], "data": match["data"]}
    if url.startswith(("https://", "http://")):
        return {"type": "url", "url": url}
    raise ProxyError.bad_request("image_url must be a data URL or an http(s) URL")


def _assistant_blocks(message: dict[str, Any]) -> list[Block]:
    blocks: list[Block] = [{"type": "text", "text": t} for t in _texts(message.get("content"))]
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
    return blocks


def _arguments(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw or {}
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _tool_result(message: dict[str, Any]) -> Block:
    content = message.get("content")
    return {
        "type": "tool_result",
        "tool_use_id": message.get("tool_call_id", ""),
        "content": "\n".join(_texts(content)) if isinstance(content, list) else content or "",
    }


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
    serial = body.get("parallel_tool_calls") is False
    if choice == "none":
        return {"type": "none"}
    if choice is None and not serial:
        return None
    if choice in (None, "auto"):
        converted: dict[str, Any] = {"type": "auto"}
    elif choice == "required":
        converted = {"type": "any"}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        converted = {"type": "tool", "name": choice.get("function", {}).get("name", "")}
    else:
        raise ProxyError.bad_request(f"unsupported tool_choice {choice!r}")
    if serial:
        converted["disable_parallel_tool_use"] = True
    return converted
