"""Liveness and Prometheus metrics; neither requires the proxy API key."""

from __future__ import annotations

from fastapi import Depends
from fastapi.responses import JSONResponse, PlainTextResponse

from .. import __version__
from ..errors import Dialect, ErrorKind, ProxyError
from .common import Services, dialect_router, get_services

router = dialect_router(Dialect.OPENAI, authenticated=False)


@router.get("/health")
async def health(services: Services = Depends(get_services)) -> JSONResponse:
    try:
        await services.upstream.access_token()
    except ProxyError as exc:
        return JSONResponse({"status": "unhealthy", "error": exc.message}, status_code=503)
    body = {"status": "ok", "version": __version__}
    if not services.settings.api_key:
        body["project"] = services.endpoints.project_id
    return JSONResponse(body)


@router.get("/metrics")
async def metrics(services: Services = Depends(get_services)) -> PlainTextResponse:
    if not services.settings.metrics_enabled:
        raise ProxyError(
            404,
            ErrorKind.NOT_FOUND,
            "metrics are disabled; set VERTEX_PROXY_METRICS_ENABLED=true to enable them",
        )
    return PlainTextResponse(services.metrics.render(), media_type="text/plain; version=0.0.4")
