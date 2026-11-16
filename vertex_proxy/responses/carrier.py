"""Carries provider extensions on tool calls through a client that would drop them.

Gemini 3 returns `extra_content` (a thought signature) on each tool call and rejects the next turn
without it. Responses clients such as Codex keep only standard fields on a function_call, but they
send reasoning items back unchanged, so the extras travel in a reasoning item's encrypted_content.
The content is base64-encoded JSON, readable by the client; it is not encrypted.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

_PREFIX = "vertex-proxy.tool-extras.v1:"


def tool_extras_item(extras: dict[str, Any]) -> dict[str, Any]:
    """A reasoning output item holding ``extras``, a map of call_id to extra_content."""
    payload = base64.urlsafe_b64encode(json.dumps(extras).encode()).decode()
    return {
        "type": "reasoning",
        "id": f"rs_{uuid.uuid4().hex}",
        "summary": [],
        "encrypted_content": _PREFIX + payload,
    }


def read_tool_extras(items: list[Any]) -> dict[str, Any]:
    """Every call_id to extra_content map found in reasoning items this module wrote."""
    extras: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        content = item.get("encrypted_content")
        if not isinstance(content, str) or not content.startswith(_PREFIX):
            continue
        try:
            decoded = json.loads(base64.urlsafe_b64decode(content[len(_PREFIX) :]))
        except ValueError:
            continue
        if isinstance(decoded, dict):
            extras.update(
                {k: v for k, v in decoded.items() if isinstance(k, str) and isinstance(v, dict)}
            )
    return extras
