"""OpenAI Responses API request body to Chat Completions request body."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..errors import ProxyError
from .carrier import read_tool_extras

logger = logging.getLogger(__name__)

# Responses tools with no Chat Completions equivalent; Codex sends web_search by default.
_DROPPED_TOOL_TYPES = frozenset(
    {
        "web_search",
        "web_search_preview",
        "file_search",
        "code_interpreter",
        "image_generation",
        "computer_use_preview",
        "local_shell",
        "mcp",
        "tool_search",
    }
)
# History items the proxy never produces and has nothing to send upstream for.
_DROPPED_ITEM_TYPES = frozenset(
    {
        "reasoning",
        "web_search_call",
        "local_shell_call",
        "image_generation_call",
        "tool_search_call",
        "tool_search_output",
        "compaction",
        "item_reference",
    }
)
_SYSTEM_ROLES = frozenset({"system", "developer"})
_CUSTOM_INPUT = "input"


@dataclass(frozen=True)
class ToolName:
    """How a flattened Chat Completions tool name maps back to the Responses tool."""

    name: str
    namespace: str | None = None
    custom: bool = False


@dataclass
class ToolNames:
    by_chat_name: dict[str, ToolName] = field(default_factory=dict)

    def add(self, tool: ToolName) -> str:
        chat_name = f"{tool.namespace}__{tool.name}" if tool.namespace else tool.name
        self.by_chat_name[chat_name] = tool
        return chat_name

    def chat_name(self, name: str, namespace: str | None) -> str:
        return f"{namespace}__{name}" if namespace else name

    def lookup(self, chat_name: str) -> ToolName:
        return self.by_chat_name.get(chat_name) or ToolName(chat_name)


def responses_to_chat(body: dict[str, Any]) -> tuple[dict[str, Any], ToolNames]:
    """The Chat Completions body for a Responses request, and the map to undo tool flattening."""
    names = ToolNames()
    tools = [chat_tool for tool in body.get("tools") or [] for chat_tool in _tools(tool, names)]
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    items = body.get("input") or []
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    extras = read_tool_extras(items)
    for item in items:
        _append_item(messages, item, names, extras)

    chat: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if tools:
        chat["tools"] = tools
        chat["tool_choice"] = _tool_choice(body.get("tool_choice"), names)
        if body.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("max_output_tokens"):
        chat["max_completion_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "user"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if effort := (body.get("reasoning") or {}).get("effort"):
        chat["reasoning_effort"] = effort
    if (text_format := (body.get("text") or {}).get("format") or {}).get("type") == "json_schema":
        chat["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                k: text_format[k] for k in ("name", "schema", "strict") if k in text_format
            },
        }
    if body.get("stream"):
        chat["stream"] = True
        chat["stream_options"] = {"include_usage": True}
    return chat, names


def _tools(tool: dict[str, Any], names: ToolNames) -> list[dict[str, Any]]:
    """Chat Completions tools for 1 Responses tool; an MCP namespace becomes 1 per inner tool."""
    kind = tool.get("type")
    if kind == "function":
        return [_function(names.add(ToolName(tool["name"])), tool)]
    if kind == "namespace":
        return [
            _function(names.add(ToolName(inner["name"], namespace=tool["name"])), inner)
            for inner in tool.get("tools") or []
        ]
    if kind == "custom":
        chat_name = names.add(ToolName(tool["name"], custom=True))
        schema = {
            "type": "object",
            "properties": {_CUSTOM_INPUT: {"type": "string"}},
            "required": [_CUSTOM_INPUT],
        }
        return [
            _function(chat_name, {"description": tool.get("description"), "parameters": schema})
        ]
    if kind in _DROPPED_TOOL_TYPES:
        logger.debug("responses: dropping %s tool, which has no Chat Completions form", kind)
        return []
    raise ProxyError.bad_request(f"unsupported tool type {kind!r}")


def _function(chat_name: str, tool: dict[str, Any]) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": chat_name,
        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
    }
    if tool.get("description"):
        function["description"] = tool["description"]
    return {"type": "function", "function": function}


def _tool_choice(choice: Any, names: ToolNames) -> Any:
    if choice in (None, "auto", "none", "required"):
        return choice or "auto"
    if isinstance(choice, dict) and choice.get("type") == "function":
        chat_name = names.chat_name(choice["name"], choice.get("namespace"))
        return {"type": "function", "function": {"name": chat_name}}
    raise ProxyError.bad_request(f"unsupported tool_choice {choice!r}")


def _append_item(
    messages: list[dict[str, Any]],
    item: dict[str, Any],
    names: ToolNames,
    extras: dict[str, Any],
) -> None:
    kind = item.get("type", "message")
    if kind == "message":
        messages.append(_message(item))
    elif kind in ("function_call", "custom_tool_call"):
        _append_tool_call(messages, item, names, extras)
    elif kind in ("function_call_output", "custom_tool_call_output"):
        messages.extend(_tool_output(item))
    elif kind in _DROPPED_ITEM_TYPES:
        return
    else:
        raise ProxyError.bad_request(f"unsupported input item type {kind!r}")


def _message(item: dict[str, Any]) -> dict[str, Any]:
    role = item.get("role", "user")
    content = item.get("content")
    if isinstance(content, str):
        return {"role": "system" if role in _SYSTEM_ROLES else role, "content": content}
    if role in _SYSTEM_ROLES or role == "assistant":
        text = "".join(part.get("text", "") for part in content or [] if "text" in part)
        return {"role": "system" if role in _SYSTEM_ROLES else "assistant", "content": text}
    return {"role": role, "content": [_user_part(part) for part in content or []]}


def _user_part(part: dict[str, Any]) -> dict[str, Any]:
    kind = part.get("type")
    if kind in ("input_text", "output_text", "text"):
        return {"type": "text", "text": part.get("text", "")}
    if kind == "input_image":
        return {"type": "image_url", "image_url": {"url": part.get("image_url", "")}}
    raise ProxyError.bad_request(f"unsupported content part type {kind!r}")


def _append_tool_call(
    messages: list[dict[str, Any]],
    item: dict[str, Any],
    names: ToolNames,
    extras: dict[str, Any],
) -> None:
    custom = item["type"] == "custom_tool_call"
    arguments = (
        json.dumps({_CUSTOM_INPUT: item.get("input", "")})
        if custom
        else item.get("arguments", "{}")
    )
    call: dict[str, Any] = {
        "id": item.get("call_id", ""),
        "type": "function",
        "function": {
            "name": names.chat_name(item.get("name", ""), item.get("namespace")),
            "arguments": arguments,
        },
    }
    if extra := extras.get(call["id"]):
        call["extra_content"] = extra
    previous = messages[-1] if messages else None
    if previous is not None and previous["role"] == "assistant" and "tool_calls" in previous:
        previous["tool_calls"].append(call)
    elif previous is not None and previous["role"] == "assistant":
        previous["tool_calls"] = [call]
    else:
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})


def _tool_output(item: dict[str, Any]) -> list[dict[str, Any]]:
    """A tool message, plus a user message for any images, which tool messages cannot carry."""
    output = item.get("output", "")
    if isinstance(output, str):
        return [{"role": "tool", "tool_call_id": item.get("call_id", ""), "content": output}]
    texts = [
        part.get("text", "") for part in output if part.get("type") in ("input_text", "output_text")
    ]
    images = [part for part in output if part.get("type") == "input_image"]
    result = [
        {"role": "tool", "tool_call_id": item.get("call_id", ""), "content": "\n".join(texts)}
    ]
    if images:
        result.append({"role": "user", "content": [_user_part(image) for image in images]})
    return result
