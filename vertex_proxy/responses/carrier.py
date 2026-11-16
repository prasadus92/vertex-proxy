"""Carries provider extensions on tool calls through a client that would drop them.

Gemini 3 returns `extra_content` (a thought signature) on each tool call and rejects the next turn
without it. Responses clients such as Codex keep only standard fields on a function_call, but they
send reasoning items back unchanged, so the extras travel in a reasoning item's encrypted_content.
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
        content = item.get("encrypted_content") if isinstance(item, dict) else None
        if item.get("type") != "reasoning" or not isinstance(content, str):
            continue
        if not content.startswith(_PREFIX):
            continue
        try:
            extras.update(json.loads(base64.urlsafe_b64decode(content[len(_PREFIX) :])))
        except ValueError:
            continue
    return extras
