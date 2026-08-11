"""Unit tests for the chat web UI endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from vertex_proxy.config import Settings
from vertex_proxy.main import build_app


@pytest.mark.anyio
async def test_chat_endpoint_returns_html() -> None:
    """The /chat endpoint returns an HTML page."""
    settings = Settings(enable_chat_ui=True)
    app = build_app(settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/chat")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "vertex-proxy" in resp.text
        assert "<script>" in resp.text


@pytest.mark.anyio
async def test_chat_endpoint_disabled() -> None:
    """The /chat endpoint returns 404 when disabled."""
    settings = Settings(enable_chat_ui=False)
    app = build_app(settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/chat")
        assert resp.status_code == 404
