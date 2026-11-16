"""OpenAI Chat Completions API."""

from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response

from ..chat import ChatService
from ..errors import Dialect
from ..upstream import relay_stream
from .common import Services, dialect_router, get_services, read_json, require_model

# Clients build the URL by appending /chat/completions to whatever base URL the user set.
CHAT_PATHS = (
    "/v1/chat/completions",
    "/chat/completions",
    "/openai/v1/chat/completions",
    "/openai/chat/completions",
    "/gemini/v1/chat/completions",
    "/gemini/chat/completions",
)

router = dialect_router(Dialect.OPENAI)


async def chat_completions(
    request: Request, services: Services = Depends(get_services)
) -> Response:
    body = await read_json(request)
    model = require_model(body)
    chat = ChatService(services)
    if body.get("stream"):
        stream = await chat.stream(body, model)
        return relay_stream(stream.response, Dialect.OPENAI, stream.chunks)
    return JSONResponse(await chat.complete(body, model))


for _path in CHAT_PATHS:
    router.add_api_route(_path, chat_completions, methods=["POST"])
