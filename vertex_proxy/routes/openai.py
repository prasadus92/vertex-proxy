"""OpenAI Chat Completions API, routed to Claude, Gemini, Vertex MaaS models or Ollama."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response

from ..bridge import StreamTranslator, anthropic_to_openai, openai_to_anthropic
from ..catalog import ChatRoute, Provider
from ..errors import Dialect, ProxyError, UpstreamContext
from ..upstream import relay_stream
from .anthropic import ANTHROPIC_VERSION
from .common import (
    Services,
    dialect_router,
    forward_json,
    get_services,
    json_body,
    read_json,
    require_model,
)

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
    requested = require_model(body)
    route = services.catalog.chat_route(requested)
    if route is None:
        raise ProxyError.bad_request(
            f"unknown model '{requested}'. GET /v1/models lists the models this proxy serves; "
            "Vertex MaaS models can also be named as publisher/model."
        )
    stream = bool(body.get("stream"))
    if route.provider is Provider.ANTHROPIC:
        return await _via_anthropic(services, body, requested, route.upstream_model, stream)
    url, context, authenticated = _openai_target(services, route)
    return await forward_json(
        services,
        url,
        {**body, "model": route.upstream_model},
        context=context,
        dialect=Dialect.OPENAI,
        metric_route=route.provider.value,
        model=requested,
        stream=stream,
        authenticated=authenticated,
    )


for _path in CHAT_PATHS:
    router.add_api_route(_path, chat_completions, methods=["POST"])


def _openai_target(services: Services, route: ChatRoute) -> tuple[str, UpstreamContext, bool]:
    if route.provider is Provider.OLLAMA:
        return (
            f"{route.ollama_url}/v1/chat/completions",
            UpstreamContext(model=route.upstream_model),
            False,
        )
    s = services.settings
    location = s.gemini_region if route.provider is Provider.GEMINI else s.maas_region
    context = UpstreamContext(services.endpoints.project_id, location, route.upstream_model)
    return services.endpoints.openai_chat(location), context, True


async def _via_anthropic(
    services: Services, body: dict[str, Any], requested: str, vertex_model: str, stream: bool
) -> Response:
    payload = openai_to_anthropic(body)
    payload["anthropic_version"] = ANTHROPIC_VERSION
    location = services.settings.anthropic_region
    url = services.endpoints.anthropic(location, vertex_model, stream=stream)
    context = UpstreamContext(services.endpoints.project_id, location, vertex_model)
    metrics = services.metrics
    try:
        if stream:
            response = await services.upstream.open_stream(url, json=payload, context=context)
            metrics.record("openai-anthropic", requested, response.status_code)
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            translator = StreamTranslator(requested, include_usage=include_usage)
            return relay_stream(
                response, Dialect.OPENAI, translator.translate(response.aiter_lines())
            )
        response = await services.upstream.post(url, json=payload, context=context)
    except ProxyError as exc:
        metrics.record("openai-anthropic", requested, exc.status)
        raise
    result = anthropic_to_openai(json_body(response.content), requested)
    usage = result["usage"]
    metrics.record(
        "openai-anthropic", requested, 200, usage["prompt_tokens"], usage["completion_tokens"]
    )
    return JSONResponse(result)
