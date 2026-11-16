"""Anthropic Messages API, served by Claude on Vertex AI."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Request
from fastapi.responses import Response

from ..chat import ANTHROPIC_VERSION
from ..errors import Dialect, ProxyError, UpstreamContext
from .common import Services, dialect_router, forward_json, get_services, read_json, require_model

# Vertex rejects these. Claude Code sends context_management alongside a beta header the
# proxy does not forward.
_UNSUPPORTED_FIELDS = frozenset({"context_management"})

router = dialect_router(Dialect.ANTHROPIC)


def _resolve(services: Services, body: dict[str, Any]) -> tuple[str, str]:
    requested = require_model(body)
    vertex_model = services.catalog.anthropic_model(requested)
    if vertex_model is None:
        raise ProxyError.bad_request(
            f"'{requested}' is not a Claude model. Known models: "
            f"{', '.join(sorted(services.catalog.anthropic))}"
        )
    return requested, vertex_model


def _context(services: Services, vertex_model: str) -> UpstreamContext:
    return UpstreamContext(
        services.endpoints.project_id, services.settings.anthropic_region, vertex_model
    )


@router.post("/anthropic/v1/messages")
@router.post("/v1/messages")
async def messages(request: Request, services: Services = Depends(get_services)) -> Response:
    body = await read_json(request)
    requested, vertex_model = _resolve(services, body)
    payload = {k: v for k, v in body.items() if k != "model" and k not in _UNSUPPORTED_FIELDS}
    payload.setdefault("anthropic_version", ANTHROPIC_VERSION)
    stream = bool(body.get("stream"))
    url = services.endpoints.anthropic(
        services.settings.anthropic_region, vertex_model, stream=stream
    )
    return await forward_json(
        services,
        url,
        payload,
        context=_context(services, vertex_model),
        dialect=Dialect.ANTHROPIC,
        metric_route="anthropic",
        model=requested,
        stream=stream,
    )


@router.post("/anthropic/v1/messages/count_tokens")
@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request, services: Services = Depends(get_services)) -> Response:
    body = await read_json(request)
    requested, vertex_model = _resolve(services, body)
    payload = {k: v for k, v in body.items() if k not in _UNSUPPORTED_FIELDS}
    payload["model"] = vertex_model
    url = services.endpoints.anthropic_count_tokens(services.settings.anthropic_region)
    return await forward_json(
        services,
        url,
        payload,
        context=_context(services, vertex_model),
        dialect=Dialect.ANTHROPIC,
        metric_route="anthropic-count-tokens",
        model=requested,
        stream=False,
    )
