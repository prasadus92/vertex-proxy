"""OpenAI Responses API, served through the Chat Completions routing; this is what Codex speaks."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..chat import ChatService
from ..errors import Dialect, ProxyError
from ..responses import ResponsesStream, chat_to_response, responses_to_chat
from ..responses.output import failure_code
from ..upstream import relay_stream
from .common import Services, dialect_router, get_services, read_json, require_model

RESPONSES_PATHS = ("/v1/responses", "/responses", "/openai/v1/responses", "/openai/responses")

router = dialect_router(Dialect.OPENAI)


async def create_response(request: Request, services: Services = Depends(get_services)) -> Response:
    body = await read_json(request)
    model = require_model(body)
    chat_body, names = responses_to_chat(body)
    chat = ChatService(services)
    if not body.get("stream"):
        return JSONResponse(chat_to_response(await chat.complete(chat_body, model), model, names))
    translator = ResponsesStream(model, names)
    try:
        stream = await chat.stream(chat_body, model)
    except ProxyError as exc:
        # In-stream, so clients such as Codex compact the conversation and retry.
        if failure_code(exc.kind.api_type, exc.message) != "context_length_exceeded":
            raise
        events = [*translator.start(), translator.fail("context_length_exceeded", exc.message)]
        return StreamingResponse(_replay(events), media_type="text/event-stream")
    return relay_stream(stream.response, Dialect.OPENAI, translator.translate(stream.chunks))


async def _replay(events: list[bytes]) -> AsyncIterator[bytes]:
    for event in events:
        yield event


for _path in RESPONSES_PATHS:
    router.add_api_route(_path, create_response, methods=["POST"])
