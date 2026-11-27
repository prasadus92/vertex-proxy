"""OpenAI Chat Completions requests, routed to Claude, Gemini, Vertex MaaS models or Ollama.

Both the Chat Completions and the Responses routes use this, so every provider is reached the
same way whichever OpenAI API the client speaks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx2

from .bridge import StreamTranslator, anthropic_to_openai, openai_to_anthropic
from .catalog import ChatRoute, Provider
from .errors import ProxyError, UpstreamContext
from .metrics import usage_tokens
from .services import Services
from .upstream import json_payload

ANTHROPIC_VERSION = "vertex-2023-10-16"


@dataclass(frozen=True)
class ChatStream:
    """An open upstream response and its body as OpenAI Chat Completions SSE."""

    response: httpx2.Response
    chunks: AsyncIterator[bytes]


class ChatService:
    def __init__(self, services: Services) -> None:
        self._services = services

    def resolve(self, model: str) -> ChatRoute:
        route = self._services.catalog.chat_route(model)
        if route is None:
            raise ProxyError.bad_request(
                f"unknown model '{model}'. GET /v1/models lists the models this proxy serves; "
                "Vertex MaaS models can also be named as publisher/model."
            )
        return route

    async def complete(self, body: dict[str, Any], model: str) -> dict[str, Any]:
        route = self.resolve(model)
        metrics = self._services.metrics
        with metrics.counting_failures(route.provider.value, model):
            if route.provider is Provider.ANTHROPIC:
                url, payload, context = self._anthropic_call(body, route, stream=False)
                response = await self._services.upstream.post(url, json=payload, context=context)
                result = anthropic_to_openai(json_payload(response), model)
            elif route.provider is Provider.CUSTOM:
                result, response = await self._custom_complete(body, route)
            else:
                url, context, authenticated = self._openai_target(route)
                response = await self._services.upstream.post(
                    url,
                    json={**body, "model": route.upstream_model},
                    context=context,
                    authenticated=authenticated,
                )
                result = json_payload(response)
        metrics.record(route.provider.value, model, response.status_code, *usage_tokens(result))
        return result

    async def stream(self, body: dict[str, Any], model: str) -> ChatStream:
        route = self.resolve(model)
        metrics = self._services.metrics
        upstream = self._services.upstream
        with metrics.counting_failures(route.provider.value, model):
            if route.provider is Provider.ANTHROPIC:
                url, payload, context = self._anthropic_call(body, route, stream=True)
                response = await upstream.open_stream(url, json=payload, context=context)
                include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                translator = StreamTranslator(model, include_usage=include_usage)
                chunks = translator.translate(response.aiter_lines())
            elif route.provider is Provider.CUSTOM:
                response, chunks = await self._custom_stream(body, route)
            else:
                url, context, authenticated = self._openai_target(route)
                response = await upstream.open_stream(
                    url,
                    json={**body, "model": route.upstream_model, "stream": True},
                    context=context,
                    authenticated=authenticated,
                )
                chunks = response.aiter_bytes()
        metrics.record(route.provider.value, model, response.status_code)
        return ChatStream(response, chunks)

    def _anthropic_call(
        self, body: dict[str, Any], route: ChatRoute, *, stream: bool
    ) -> tuple[str, dict[str, Any], UpstreamContext]:
        payload = openai_to_anthropic({**body, "stream": stream})
        payload["anthropic_version"] = ANTHROPIC_VERSION
        location = self._services.settings.anthropic_region
        endpoints = self._services.endpoints
        url = endpoints.anthropic(location, route.upstream_model, stream=stream)
        return url, payload, UpstreamContext(endpoints.project_id, location, route.upstream_model)

    async def _custom_complete(
        self, body: dict[str, Any], route: ChatRoute
    ) -> tuple[dict[str, Any], httpx2.Response]:
        """Forward a non-streaming request to a custom OpenAI-compatible provider."""
        assert route.custom is not None
        info = route.custom
        url = f"{info.base_url}/chat/completions"
        context = UpstreamContext(model=info.upstream_model)
        response = await self._services.upstream.post(
            url,
            json={**body, "model": info.upstream_model},
            headers={"Authorization": f"Bearer {info.api_key}"},
            context=context,
            authenticated=False,
        )
        return json_payload(response), response

    async def _custom_stream(
        self, body: dict[str, Any], route: ChatRoute
    ) -> tuple[httpx2.Response, AsyncIterator[bytes]]:
        """Forward a streaming request to a custom OpenAI-compatible provider."""
        assert route.custom is not None
        info = route.custom
        url = f"{info.base_url}/chat/completions"
        context = UpstreamContext(model=info.upstream_model)
        response = await self._services.upstream.open_stream(
            url,
            json={**body, "model": info.upstream_model, "stream": True},
            headers={"Authorization": f"Bearer {info.api_key}"},
            context=context,
            authenticated=False,
        )
        return response, response.aiter_bytes()

    def _openai_target(self, route: ChatRoute) -> tuple[str, UpstreamContext, bool]:
        if route.provider is Provider.OLLAMA:
            context = UpstreamContext(model=route.upstream_model)
            return f"{route.ollama_url}/v1/chat/completions", context, False
        settings, endpoints = self._services.settings, self._services.endpoints
        location = (
            settings.gemini_region if route.provider is Provider.GEMINI else settings.maas_region
        )
        context = UpstreamContext(endpoints.project_id, location, route.upstream_model)
        return endpoints.openai_chat(location), context, True
