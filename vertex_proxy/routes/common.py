"""Shared route plumbing: services, error dialect, the proxy API key and JSON forwarding."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from ..catalog import ModelCatalog
from ..config import Settings
from ..errors import Dialect, ErrorKind, ProxyError, UpstreamContext
from ..metrics import Metrics, usage_tokens
from ..upstream import Upstream, relay_stream
from ..vertex import VertexEndpoints

_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._@-]+$")


@dataclass(frozen=True)
class Services:
    settings: Settings
    catalog: ModelCatalog
    endpoints: VertexEndpoints
    upstream: Upstream
    metrics: Metrics


def get_services(request: Request) -> Services:
    return request.app.state.services


def _set_dialect(dialect: Dialect) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        request.state.dialect = dialect

    return dependency


def _supplied_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return (
        request.headers.get("x-api-key")
        or request.headers.get("x-goog-api-key")
        or request.query_params.get("key")
    )


def require_api_key(request: Request, services: Services = Depends(get_services)) -> None:
    expected = services.settings.api_key
    if not expected:
        return
    supplied = _supplied_key(request)
    if supplied is None or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise ProxyError(
            401,
            ErrorKind.AUTHENTICATION,
            "vertex-proxy: missing or invalid API key. Send VERTEX_PROXY_API_KEY as "
            "`Authorization: Bearer <key>`, `x-api-key` or `x-goog-api-key`.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def dialect_router(dialect: Dialect, *, authenticated: bool = True) -> APIRouter:
    """A router whose errors use ``dialect`` and, optionally, that requires the proxy API key."""
    dependencies = [Depends(_set_dialect(dialect))]
    if authenticated:
        dependencies.append(Depends(require_api_key))
    return APIRouter(dependencies=dependencies)


async def read_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        body = await request.json()
    except ValueError as exc:
        raise ProxyError.bad_request("request body must be JSON") from exc
    if not isinstance(body, dict):
        raise ProxyError.bad_request("request body must be a JSON object")
    return body


def require_model(body: dict[str, Any]) -> str:
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ProxyError.bad_request("missing 'model' in request body")
    return model.strip()


def path_segment(value: str, what: str) -> str:
    """Reject values that would change the shape of the upstream URL."""
    if not _PATH_SEGMENT.match(value):
        raise ProxyError.bad_request(f"invalid {what}: {value!r}")
    return value


def forwarded_query(request: Request) -> str:
    """The client's query string without an API key, which must not travel upstream."""
    return urlencode([(k, v) for k, v in request.query_params.multi_items() if k != "key"])


def with_query(url: str, query: str) -> str:
    return f"{url}?{query}" if query else url


async def forward_json(
    services: Services,
    url: str,
    payload: Any,
    *,
    context: UpstreamContext,
    dialect: Dialect,
    metric_route: str,
    model: str,
    stream: bool,
    authenticated: bool = True,
) -> Response:
    """POST ``payload`` upstream and return the reply, streamed or as JSON."""
    upstream, metrics = services.upstream, services.metrics
    try:
        if stream:
            response = await upstream.open_stream(
                url, json=payload, context=context, authenticated=authenticated
            )
            metrics.record(metric_route, model, response.status_code)
            return relay_stream(response, dialect)
        response = await upstream.post(
            url, json=payload, context=context, authenticated=authenticated
        )
    except ProxyError as exc:
        metrics.record(metric_route, model, exc.status)
        raise
    data = json_body(response.content)
    metrics.record(metric_route, model, response.status_code, *usage_tokens(data))
    return JSONResponse(data, status_code=response.status_code)


def json_body(content: bytes) -> Any:
    try:
        return json.loads(content)
    except ValueError as exc:
        raise ProxyError(502, ErrorKind.UPSTREAM, "upstream returned a non-JSON response") from exc
