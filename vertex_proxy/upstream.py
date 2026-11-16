"""Authenticated calls to Vertex AI (and Ollama), with upstream failures raised as ProxyError."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx2
from fastapi.responses import StreamingResponse

from .auth import CredentialsError, TokenSource
from .errors import Dialect, ErrorKind, ProxyError, UpstreamContext

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx2.Timeout(120.0, connect=10.0)
# A thinking model can go quiet mid-stream for longer than any fixed read window.
STREAM_TIMEOUT = httpx2.Timeout(connect=10.0, read=None, write=120.0, pool=120.0)


class Upstream:
    def __init__(self, http: httpx2.AsyncClient, tokens: TokenSource) -> None:
        self.http = http
        self._tokens = tokens

    async def post(
        self,
        url: str,
        *,
        context: UpstreamContext,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        authenticated: bool = True,
    ) -> httpx2.Response:
        request = await self._build(url, json, content, headers, authenticated, DEFAULT_TIMEOUT)
        return await self._send(request, context, stream=False)

    async def open_stream(
        self,
        url: str,
        *,
        context: UpstreamContext,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        authenticated: bool = True,
    ) -> httpx2.Response:
        """Send a request and return the response unread, after checking its status."""
        request = await self._build(url, json, content, headers, authenticated, STREAM_TIMEOUT)
        return await self._send(request, context, stream=True)

    async def _build(
        self,
        url: str,
        json: Any,
        content: bytes | None,
        headers: dict[str, str] | None,
        authenticated: bool,
        timeout: httpx2.Timeout,
    ) -> httpx2.Request:
        merged = httpx2.Headers({"content-type": "application/json"})
        merged.update(headers or {})
        if authenticated:
            merged["Authorization"] = f"Bearer {await self.access_token()}"
        return self.http.build_request(
            "POST", url, json=json, content=content, headers=merged, timeout=timeout
        )

    async def access_token(self) -> str:
        try:
            return await self._tokens.get_token()
        except CredentialsError as exc:
            raise ProxyError(503, ErrorKind.AUTHENTICATION, f"vertex-proxy: {exc}") from exc

    async def _send(
        self, request: httpx2.Request, context: UpstreamContext, *, stream: bool
    ) -> httpx2.Response:
        try:
            response = await self.http.send(request, stream=stream)
        except httpx2.HTTPError as exc:
            raise ProxyError(
                502, ErrorKind.UPSTREAM, f"vertex-proxy: could not reach {request.url.host}: {exc}"
            ) from exc
        if response.status_code >= 400:
            body = await response.aread()
            await response.aclose()
            raise ProxyError.from_upstream(
                response.status_code, body, dict(response.headers), context
            )
        return response


def json_payload(response: httpx2.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ProxyError(502, ErrorKind.UPSTREAM, "upstream returned a non-JSON response") from exc


def relay_stream(
    response: httpx2.Response,
    dialect: Dialect,
    chunks: AsyncIterator[bytes] | None = None,
    media_type: str | None = None,
) -> StreamingResponse:
    """Stream ``chunks`` (default: the raw upstream body) to the client, then close the upstream."""
    media_type = media_type or response.headers.get("content-type", "text/event-stream")
    is_sse = media_type.startswith("text/event-stream")

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in chunks if chunks is not None else response.aiter_bytes():
                yield chunk
        except httpx2.HTTPError as exc:
            logger.warning("upstream stream from %s failed: %s", response.url.host, exc)
            if is_sse:
                yield ProxyError(
                    502, ErrorKind.UPSTREAM, f"vertex-proxy: upstream stream failed: {exc}"
                ).to_sse(dialect)
        finally:
            await response.aclose()

    return StreamingResponse(body(), media_type=media_type)
