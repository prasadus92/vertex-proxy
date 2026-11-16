"""Unit tests for Ollama backend integration."""

from __future__ import annotations

import pytest
from httpx2 import ASGITransport, AsyncClient

from vertex_proxy.config import Settings
from vertex_proxy.main import build_app


def test_ollama_backends_default_empty() -> None:
    """Ollama backends default to an empty dict."""
    s = Settings()
    assert s.ollama_backends == {}


def test_ollama_backends_explicit_model() -> None:
    """Explicit model-to-URL mapping is parsed correctly."""
    s = Settings(ollama_backends={"qwen3:30b": "http://localhost:11434"})
    assert s.ollama_backends["qwen3:30b"] == "http://localhost:11434"


def test_ollama_backends_wildcard() -> None:
    """Wildcard key '*' is accepted in config."""
    s = Settings(ollama_backends={"*": "http://gpu:11434"})
    assert s.ollama_backends["*"] == "http://gpu:11434"


@pytest.mark.anyio
async def test_ollama_models_in_v1_models() -> None:
    """Ollama models registered via explicit config appear in /v1/models."""
    settings = Settings(
        ollama_backends={"test-model:7b": "http://fake:11434"},
    )
    app = build_app(settings)

    # Manually populate the Ollama registry (skip async discovery)
    from vertex_proxy import main as main_mod

    main_mod._OLLAMA_MODELS["test-model:7b"] = "http://fake:11434"

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            model_ids = [m["id"] for m in data["data"]]
            assert "test-model:7b" in model_ids
            # Check owned_by and provider
            ollama_model = [m for m in data["data"] if m["id"] == "test-model:7b"][0]
            assert ollama_model["owned_by"] == "ollama"
            assert ollama_model["provider"] == "ollama"
    finally:
        main_mod._OLLAMA_MODELS.clear()
