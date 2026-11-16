"""Gemini generateContent API, served by Gemini on Vertex AI."""

from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import Response

from ..errors import Dialect, ProxyError, UpstreamContext
from .common import (
    Services,
    dialect_router,
    forward_json,
    forwarded_query,
    get_services,
    path_segment,
    read_json,
    with_query,
)

router = dialect_router(Dialect.GOOGLE)


# google-genai defaults to v1 and older Gemini SDKs to v1beta; both map to the same Vertex call.
@router.post("/gemini/v1beta/models/{target:path}")
@router.post("/gemini/v1/models/{target:path}")
@router.post("/v1beta/models/{target:path}")
async def generate(
    target: str, request: Request, services: Services = Depends(get_services)
) -> Response:
    model, separator, action = target.rpartition(":")
    if not separator or not model:
        raise ProxyError.bad_request(
            "the path must name a model and an action, as in models/gemini-3.5-flash:generateContent"
        )
    requested = path_segment(model, "model")
    action = path_segment(action, "action")
    vertex_model = services.catalog.gemini_model(requested)
    location = services.settings.gemini_region
    url = with_query(
        services.endpoints.gemini(location, vertex_model, action), forwarded_query(request)
    )
    return await forward_json(
        services,
        url,
        await read_json(request),
        context=UpstreamContext(services.endpoints.project_id, location, vertex_model),
        dialect=Dialect.GOOGLE,
        metric_route="gemini",
        model=requested,
        stream=action.startswith("stream"),
    )
