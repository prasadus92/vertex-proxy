"""Translation between the OpenAI Responses API and the Chat Completions API."""

from .request import ToolNames, responses_to_chat
from .response import chat_to_response
from .stream import ResponsesStream

__all__ = ["ResponsesStream", "ToolNames", "chat_to_response", "responses_to_chat"]
