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
_TOOL_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})
_SYSTEM_ROLES = frozenset({"system", "developer"})
_TEXT_PARTS = frozenset({"input_text", "output_text", "text"})
_CUSTOM_INPUT = "input"
# Codex sends minimal and xhigh; Chat Completions providers accept low, medium and high.
_REASONING_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}

Message = dict[str, Any]


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
        chat_name = self.chat_name(tool.name, tool.namespace)
        self.by_chat_name[chat_name] = tool
        return chat_name

    @staticmethod
    def chat_name(name: str, namespace: str | None) -> str:
        return f"{namespace}__{name}" if namespace else name

    def lookup(self, chat_name: str) -> ToolName:
        return self.by_chat_name.get(chat_name) or ToolName(chat_name)


def responses_to_chat(body: dict[str, Any]) -> tuple[dict[str, Any], ToolNames]:
    """The Chat Completions body for a Responses request, and the map to undo tool flattening."""
    if body.get("previous_response_id"):
        raise ProxyError.bad_request(
            "previous_response_id is not supported: vertex-proxy stores nothing, so send the "
            "whole conversation in input"
        )
    names = ToolNames()
    tools = [chat_tool for tool in body.get("tools") or [] for chat_tool in _tools(tool, names)]
    items = body.get("input") or []
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    conversation = _Conversation(names, read_tool_extras(items))
    if body.get("instructions"):
        conversation.system.append(body["instructions"])
    for item in items:
        conversation.add(item)

    chat: dict[str, Any] = {"model": body.get("model"), "messages": conversation.messages()}
    if tools:
        chat["tools"] = tools
        chat["tool_choice"] = _tool_choice(body.get("tool_choice"))
        if body.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("max_output_tokens"):
        chat["max_completion_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "user"):
        if body.get(key) is not None:
            chat[key] = body[key]
    effort = _REASONING_EFFORT.get((body.get("reasoning") or {}).get("effort") or "")
    if effort:
        chat["reasoning_effort"] = effort
    text_format = (body.get("text") or {}).get("format") or {}
    if text_format.get("type") == "json_schema":
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


class _Conversation:
    """Builds Chat Completions messages from Responses input items, in order."""

    def __init__(self, names: ToolNames, extras: dict[str, Any]) -> None:
        self._names = names
        self._extras = extras
        self.system: list[str] = []
        self._turns: list[Message] = []
        # Tool messages must sit together, so images from tool outputs wait for the run to end.
        self._pending_images: list[dict[str, Any]] = []

    def messages(self) -> list[Message]:
        self._flush_images()
        # Some open-model chat templates accept a system message only in first position.
        head = [{"role": "system", "content": "\n\n".join(self.system)}] if self.system else []
        return head + self._turns

    def add(self, item: dict[str, Any]) -> None:
        kind = item.get("type", "message")
        if kind not in _TOOL_OUTPUT_TYPES:
            self._flush_images()
        if kind == "message":
            self._message(item)
        elif kind == "agent_message":
            text = "\n".join(_texts(item.get("content")))
            if text:
                self._turns.append(
                    {"role": "user", "content": f"[{item.get('author', 'agent')}] {text}"}
                )
        elif kind in ("function_call", "custom_tool_call"):
            self._tool_call(item)
        elif kind in _TOOL_OUTPUT_TYPES:
            self._tool_output(item)
        else:
            logger.debug("responses: dropping %s input item", kind)

    def _message(self, item: dict[str, Any]) -> None:
        role = item.get("role", "user")
        content = item.get("content")
        if role in _SYSTEM_ROLES:
            self.system.extend(_texts(content))
            return
        if role == "assistant":
            text = "".join(_texts(content))
            if text:
                self._turns.append({"role": "assistant", "content": text})
            return
        if isinstance(content, str):
            if content:
                self._turns.append({"role": role, "content": content})
            return
        blocks = [_user_part(part) for part in content or [] if _has_content(part)]
        if blocks:
            self._turns.append({"role": role, "content": blocks})

    def _tool_call(self, item: dict[str, Any]) -> None:
        custom = item["type"] == "custom_tool_call"
        arguments = (
            json.dumps({_CUSTOM_INPUT: item.get("input", "")})
            if custom
            else item.get("arguments") or "{}"
        )
        call: dict[str, Any] = {
            "id": item.get("call_id", ""),
            "type": "function",
            "function": {
                "name": self._names.chat_name(item.get("name", ""), item.get("namespace")),
                "arguments": arguments,
            },
        }
        if extra := self._extras.get(call["id"]):
            call["extra_content"] = extra
        previous = self._turns[-1] if self._turns else None
        if previous is not None and previous["role"] == "assistant":
            previous.setdefault("tool_calls", []).append(call)
        else:
            self._turns.append({"role": "assistant", "content": None, "tool_calls": [call]})

    def _tool_output(self, item: dict[str, Any]) -> None:
        output = item.get("output", "")
        parts = [{"type": "input_text", "text": output}] if isinstance(output, str) else output
        self._turns.append(
            {
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": "\n".join(_texts(parts)),
            }
        )
        self._pending_images += [
            _user_part(part) for part in parts if part.get("type") == "input_image"
        ]

    def _flush_images(self) -> None:
        if self._pending_images:
            self._turns.append({"role": "user", "content": self._pending_images})
            self._pending_images = []


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content] if content else []
    return [
        part.get("text", "")
        for part in content or []
        if isinstance(part, dict) and part.get("type") in _TEXT_PARTS and part.get("text")
    ]


def _has_content(part: dict[str, Any]) -> bool:
    return part.get("type") not in _TEXT_PARTS or bool(part.get("text"))


def _user_part(part: dict[str, Any]) -> dict[str, Any]:
    kind = part.get("type")
    if kind in _TEXT_PARTS:
        return {"type": "text", "text": part.get("text", "")}
    if kind == "input_image":
        return {"type": "image_url", "image_url": {"url": part.get("image_url", "")}}
    raise ProxyError.bad_request(f"unsupported content part type {kind!r}")


def _tools(
    tool: dict[str, Any], names: ToolNames, namespace: str | None = None
) -> list[dict[str, Any]]:
    """Chat Completions tools for 1 Responses tool; an MCP namespace becomes 1 per inner tool."""
    kind = tool.get("type")
    if kind == "function":
        return [_function(names.add(ToolName(tool["name"], namespace)), tool)]
    if kind == "namespace":
        return [
            chat for inner in tool.get("tools") or [] for chat in _tools(inner, names, tool["name"])
        ]
    if kind == "custom":
        chat_name = names.add(ToolName(tool["name"], namespace, custom=True))
        schema = {
            "type": "object",
            "properties": {_CUSTOM_INPUT: {"type": "string"}},
            "required": [_CUSTOM_INPUT],
        }
        return [
            _function(chat_name, {"description": _custom_description(tool), "parameters": schema})
        ]
    if kind in _DROPPED_TOOL_TYPES:
        logger.debug("responses: dropping %s tool, which has no Chat Completions form", kind)
        return []
    raise ProxyError.bad_request(f"unsupported tool type {kind!r}")


def _custom_description(tool: dict[str, Any]) -> str:
    """A custom tool's grammar is its only input spec, so it goes into the description."""
    description = tool.get("description") or ""
    grammar = tool.get("format") or {}
    if grammar.get("definition"):
        syntax = grammar.get("syntax", "grammar")
        description += f"\n\nThe input must follow this {syntax} grammar:\n{grammar['definition']}"
    return description.strip()


def _function(chat_name: str, tool: dict[str, Any]) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": chat_name,
        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
    }
    if tool.get("description"):
        function["description"] = tool["description"]
    return {"type": "function", "function": function}


def _tool_choice(choice: Any) -> Any:
    if choice in (None, "auto", "none", "required"):
        return choice or "auto"
    if isinstance(choice, dict) and choice.get("type") == "function":
        chat_name = ToolNames.chat_name(choice["name"], choice.get("namespace"))
        return {"type": "function", "function": {"name": chat_name}}
    raise ProxyError.bad_request(f"unsupported tool_choice {choice!r}")
