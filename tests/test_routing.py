"""Tests for the FastAPI routing layer.

These tests use a fake TokenManager + mocked httpx so they don't need GCP.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from vertex_proxy.config import Settings
from vertex_proxy.main import build_app


def _install_mock_http(client: TestClient) -> AsyncMock:
    """Install an AsyncMock httpx client that can be both awaited (for aclose)
    and used for .post() calls. Returns the mock for assertion."""
    mock = AsyncMock()
    mock.aclose = AsyncMock()
    mock.post = AsyncMock()
    mock.post.return_value = httpx.Response(
        200,
        json={"id": "ok", "content": [{"type": "text", "text": "ok"}]},
        request=httpx.Request("POST", "http://x"),
    )
    client.app.state.http = mock
    return mock


class _FakeTokenManager:
    """Stand-in for auth.TokenManager that never touches GCP."""

    project_id = "test-project"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def get_token(self) -> str:
        return "fake-token"

    @property
    def token(self) -> str:
        return "fake-token"


def _build_test_app(capture: dict[str, Any]) -> Any:
    """Build the app with a fake TokenManager + mocked httpx client.

    `capture` gets populated with the last upstream URL + request body so
    tests can assert on what we forwarded.
    """
    cfg = Settings(project_id="test-project")

    # Patch TokenManager used inside build_app. We do this by monkey-patching
    # the auth module attribute, since build_app instantiates it directly.
    import vertex_proxy.main as main_mod

    original_tm_cls = main_mod.TokenManager
    main_mod.TokenManager = lambda **kwargs: _FakeTokenManager()  # type: ignore[assignment, misc]

    try:
        app = build_app(cfg)
    finally:
        main_mod.TokenManager = original_tm_cls

    # Replace the http client with a mock after startup
    async def fake_post(url, headers=None, json=None, **kwargs):  # type: ignore[no-untyped-def]
        capture["url"] = url
        capture["headers"] = headers or {}
        capture["body"] = json
        # Shape a minimal successful response per path
        if "publishers/anthropic" in url:
            payload = {"id": "msg_x", "content": [{"type": "text", "text": "ok"}]}
        elif "publishers/google" in url:
            payload = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        else:
            payload = {"choices": [{"message": {"content": "ok"}}]}
        response = httpx.Response(200, json=payload, request=httpx.Request("POST", url))
        return response

    # The mock client is installed via lifespan; we patch after TestClient starts.
    return app, capture


def test_health_endpoint() -> None:
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        # Install mock http client after startup
        _install_mock_http(client)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["project"] == "test-project"


def test_list_models_endpoint() -> None:
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        ids = {m["id"] for m in data}
        # Spot-check one from each provider family
        assert "claude-sonnet-4-5-20250929" in ids
        assert "gemini-2.5-pro" in ids
        assert "kimi-k2.5" in ids


def test_anthropic_unknown_model_rejected() -> None:
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        _install_mock_http(client)
        r = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "unknown-claude-99",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 400
        assert "unknown anthropic model" in r.json()["detail"]


def test_anthropic_model_alias_resolution() -> None:
    captured: dict[str, Any] = {}
    app, _ = _build_test_app(captured)
    with TestClient(app) as client:
        mock_http = _install_mock_http(client)
        mock_http.post.return_value = httpx.Response(
            200,
            json={"id": "msg_test", "content": [{"type": "text", "text": "ok"}]},
            request=httpx.Request("POST", "http://x"),
        )

        r = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        # The upstream URL should contain the '@' model ID and rawPredict
        call_args = mock_http.post.await_args
        url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        assert "claude-sonnet-4-5@20250929:rawPredict" in url
        # Bearer token
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer fake-token"
        # 'model' stripped, 'anthropic_version' injected
        body = call_args.kwargs["json"]
        assert "model" not in body
        assert body["anthropic_version"] == "vertex-2023-10-16"


def test_gemini_path_forwarding() -> None:
    captured: dict[str, Any] = {}
    app, _ = _build_test_app(captured)
    with TestClient(app) as client:
        mock_http = _install_mock_http(client)
        mock_http.post.return_value = httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
            request=httpx.Request("POST", "http://x"),
        )

        r = client.post(
            "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
        assert r.status_code == 200
        call_args = mock_http.post.await_args
        url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        # We translate /v1beta/... → /v1/projects/.../publishers/google/models/
        assert "publishers/google/models/gemini-2.5-flash:generateContent" in url
        assert "/v1/projects/test-project" in url


def test_maas_unknown_model_rejected() -> None:
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        _install_mock_http(client)
        r = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "not-a-real-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 400
        assert "unknown MaaS model" in r.json()["detail"]


def test_openai_route_variants_all_accepted() -> None:
    """Hermes + various OpenAI clients use different URL shapes; all must work."""
    captured: dict[str, Any] = {}
    app, _ = _build_test_app(captured)
    with TestClient(app) as client:
        mock_http = _install_mock_http(client)
        mock_http.post.return_value = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", "http://x"),
        )
        for path in ["/v1/chat/completions", "/chat/completions", "/openai/v1/chat/completions"]:
            r = client.post(
                path,
                json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, f"path {path} failed"


def test_openai_gemini_routes_via_openai_compat_endpoint() -> None:
    """Gemini models via /openai/v1/chat/completions should hit Vertex's
    OpenAI-compat endpoint (not the MaaS publishers path)."""
    captured: dict[str, Any] = {}
    app, _ = _build_test_app(captured)
    with TestClient(app) as client:
        mock_http = _install_mock_http(client)
        mock_http.post.return_value = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", "http://x"),
        )
        r = client.post(
            "/openai/v1/chat/completions",
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        call_args = mock_http.post.await_args
        url = call_args.args[0] if call_args.args else call_args.kwargs["url"]
        assert "endpoints/openapi/chat/completions" in url, url
        body = call_args.kwargs["json"]
        assert body["model"] == "google/gemini-2.5-flash"


def test_v1_models_specific_lookup() -> None:
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        r = client.get("/v1/models/gemini-2.5-flash")
        assert r.status_code == 200
        assert r.json()["id"] == "gemini-2.5-flash"
        r = client.get("/v1/models/unknown-model")
        assert r.status_code == 404


def test_metrics_endpoint_disabled_by_default() -> None:
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 404
        assert "metrics disabled" in r.json()["detail"]


def test_metrics_endpoint_enabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VERTEX_PROXY_METRICS_ENABLED", "true")
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "vertex_proxy_uptime_seconds" in r.text
        assert "vertex_proxy_requests_total" in r.text


def test_api_key_required_when_configured(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VERTEX_PROXY_API_KEY", "s3cret")
    app, _ = _build_test_app({})
    with TestClient(app) as client:
        _install_mock_http(client)
        r = client.post("/v1/chat/completions", json={"model": "gemini-2.5-flash", "messages": []})
        assert r.status_code == 401
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "gemini-2.5-flash", "messages": []},
        )
        assert r.status_code == 401
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer s3cret"},
            json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code != 401
