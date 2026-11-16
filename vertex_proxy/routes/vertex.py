"""Vertex AI pass-through: clients that speak Vertex's own API get the proxy's credentials.

Claude Code uses this with CLAUDE_CODE_USE_VERTEX=1, ANTHROPIC_VERTEX_BASE_URL pointing here and
CLAUDE_CODE_SKIP_VERTEX_AUTH=1, which makes it send only the betas Vertex accepts.
"""

from __future__ import annotations

import re

from fastapi import Depends, Request
from fastapi.responses import Response

from ..errors import Dialect, ErrorKind, ProxyError, UpstreamContext
from ..upstream import relay_stream
from .common import (
    Services,
    dialect_router,
    forwarded_query,
    get_services,
    with_query,
)

_PUBLISHER_MODEL_PATH = re.compile(
    r"^(?:(?P<version>v1|v1beta1)/)?projects/[^/]+/locations/(?P<location>[a-z0-9-]+)"
    r"/publishers/(?P<publisher>[a-z0-9-]+)/models/(?P<model>[A-Za-z0-9._@-]+)"
    r":(?P<method>[A-Za-z]+)$"
)
_FORWARDED_HEADERS = ("anthropic-beta", "anthropic-version", "content-type")

router = dialect_router(Dialect.RAW)


@router.post("/vertex/{path:path}")
async def passthrough(
    path: str, request: Request, services: Services = Depends(get_services)
) -> Response:
    match = _PUBLISHER_MODEL_PATH.match(path)
    if match is None:
        raise ProxyError(
            404,
            ErrorKind.NOT_FOUND,
            "vertex-proxy forwards only publisher model calls, as in "
            "/vertex/v1/projects/P/locations/L/publishers/anthropic/models/M:rawPredict",
        )
    location, model, method = match["location"], match["model"], match["method"]
    query = forwarded_query(request)
    url = with_query(
        services.endpoints.publisher_model(
            location, match["publisher"], model, method, match["version"] or "v1"
        ),
        query,
    )
    headers = {h: request.headers[h] for h in _FORWARDED_HEADERS if h in request.headers}
    context = UpstreamContext(services.endpoints.project_id, location, model)
    content = await request.body()
    upstream, metrics = services.upstream, services.metrics
    stream = method.startswith("stream") or "alt=sse" in query
    dialect = Dialect.ANTHROPIC if match["publisher"] == "anthropic" else Dialect.GOOGLE
    with metrics.counting_failures("vertex", model):
        if stream:
            response = await upstream.open_stream(
                url, content=content, headers=headers, context=context
            )
            metrics.record("vertex", model, response.status_code)
            return relay_stream(response, dialect)
        response = await upstream.post(url, content=content, headers=headers, context=context)
    metrics.record("vertex", model, response.status_code)
    return Response(
        response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )
