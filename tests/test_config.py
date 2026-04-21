"""Unit tests for configuration + model alias resolution."""

from __future__ import annotations

from vertex_proxy.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8787
    assert s.anthropic_region == "us-east5"
    assert s.gemini_region == "us-central1"
    assert s.token_refresh_seconds == 3000


def test_anthropic_aliases_include_sonnet() -> None:
    s = Settings()
    assert "claude-sonnet-4-5-20250929" in s.anthropic_model_aliases
    assert s.anthropic_model_aliases["claude-sonnet-4-5-20250929"] == "claude-sonnet-4-5@20250929"


def test_anthropic_aliases_have_at_sign_format() -> None:
    """Vertex expects the '@' format for all model IDs."""
    s = Settings()
    for alias, vertex_id in s.anthropic_model_aliases.items():
        assert "@" in vertex_id, f"anthropic alias {alias!r} → {vertex_id!r} missing '@'"


def test_gemini_aliases_include_pro_and_flash() -> None:
    s = Settings()
    assert "gemini-2.5-pro" in s.gemini_model_aliases
    assert "gemini-2.5-flash" in s.gemini_model_aliases


def test_maas_aliases_have_publisher_path_shape() -> None:
    """MaaS aliases must follow 'publishers/{vendor}/models/{id}' shape."""
    s = Settings()
    for alias, path in s.maas_model_aliases.items():
        parts = path.split("/")
        assert len(parts) == 4, f"maas alias {alias!r} → {path!r} not 4 segments"
        assert parts[0] == "publishers", f"maas alias {alias!r} must start with 'publishers/'"
        assert parts[2] == "models", f"maas alias {alias!r} must have 'models/' segment"


def test_known_maas_vendors_present() -> None:
    """Spot-check that each known vendor has at least one model."""
    s = Settings()
    vendors_seen = {path.split("/")[1] for path in s.maas_model_aliases.values()}
    expected = {"moonshotai", "zhipu", "minimax", "qwen", "xai"}
    missing = expected - vendors_seen
    assert not missing, f"missing MaaS vendor(s): {missing}"


def test_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VERTEX_PROXY_PORT", "9999")
    monkeypatch.setenv("VERTEX_PROXY_ANTHROPIC_REGION", "europe-west4")
    s = Settings()
    assert s.port == 9999
    assert s.anthropic_region == "europe-west4"
