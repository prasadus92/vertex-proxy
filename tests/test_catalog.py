from __future__ import annotations

import httpx2
import pytest

from vertex_proxy.catalog import ModelCatalog, Provider
from vertex_proxy.config import Settings


def _catalog(**settings: object) -> ModelCatalog:
    return ModelCatalog(Settings(_env_file=None, **settings))


@pytest.mark.parametrize(
    ("requested", "vertex_id"),
    [
        ("claude-opus-5-5", "claude-opus-5-5"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-opus-4-5", "claude-opus-4-5@20251101"),
        ("claude-haiku-4-5", "claude-haiku-4-5@20251001"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5@20250929"),
        ("claude-haiku-4-5@20251001", "claude-haiku-4-5@20251001"),
        ("claude-opus-9-1", "claude-opus-9-1"),
        ("claude-sonnet-9-1-20271231", "claude-sonnet-9-1@20271231"),
    ],
)
def test_anthropic_model_resolution(requested: str, vertex_id: str) -> None:
    assert _catalog().anthropic_model(requested) == vertex_id


@pytest.mark.parametrize("name", ["gpt-5", "gemini-3.5-flash", "claude", "claude-opus/../x"])
def test_non_claude_names_are_rejected(name: str) -> None:
    assert _catalog().anthropic_model(name) is None


def test_user_aliases_extend_and_override_the_builtin_catalog() -> None:
    catalog = _catalog(
        anthropic_model_aliases={"fast": "claude-haiku-5-5", "claude-opus-4-5": "pinned@1"}
    )
    assert catalog.anthropic_model("fast") == "claude-haiku-5-5"
    assert catalog.anthropic_model("claude-opus-4-5") == "pinned@1"
    assert catalog.anthropic_model("claude-opus-5-5") == "claude-opus-5-5"


def test_retired_gemini_models_are_not_listed() -> None:
    gemini = {e["id"] for e in _catalog().entries(Provider.GEMINI)}
    assert "gemini-3.5-flash" in gemini
    assert "gemini-2.0-flash" not in gemini


@pytest.mark.parametrize(
    ("requested", "provider", "upstream"),
    [
        ("claude-opus-5-5", Provider.ANTHROPIC, "claude-opus-5-5"),
        ("gemini-3.5-flash", Provider.GEMINI, "google/gemini-3.5-flash"),
        ("google/gemini-3.5-flash", Provider.GEMINI, "google/gemini-3.5-flash"),
        ("gemini-9-ultra", Provider.GEMINI, "google/gemini-9-ultra"),
        ("glm-5.2", Provider.MAAS, "zai-org/glm-5.2-maas"),
        (
            "deepseek-ai/deepseek-v4.1-flash-maas",
            Provider.MAAS,
            "deepseek-ai/deepseek-v4.1-flash-maas",
        ),
    ],
)
def test_chat_routes(requested: str, provider: Provider, upstream: str) -> None:
    route = _catalog().chat_route(requested)
    assert route is not None
    assert (route.provider, route.upstream_model) == (provider, upstream)


def test_unknown_chat_model_has_no_route() -> None:
    assert _catalog().chat_route("mystery-model") is None


def test_ollama_cannot_shadow_a_vertex_model() -> None:
    catalog = _catalog()
    catalog.ollama["gemini-3.5-flash"] = "http://ollama:11434"
    route = catalog.chat_route("gemini-3.5-flash")
    assert route is not None and route.provider is Provider.GEMINI


async def test_discover_ollama_wildcard_and_explicit_models() -> None:
    def tags(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "http://gpu:11434/api/tags"
        return httpx2.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": ""}]})

    catalog = _catalog(
        ollama_backends={"*": "http://gpu:11434/", "llama3.3:70b": "http://other:11434"}
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(tags)) as http:
        await catalog.discover_ollama(http)
    assert catalog.ollama == {"qwen3:8b": "http://gpu:11434", "llama3.3:70b": "http://other:11434"}


async def test_discover_ollama_survives_an_unreachable_server() -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    catalog = _catalog(ollama_backends={"*": "http://gpu:11434"})
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(down)) as http:
        await catalog.discover_ollama(http)
    assert catalog.ollama == {}


def test_custom_provider_routing() -> None:
    catalog = _catalog(
        custom_providers={
            "deepseek": {
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-test",
                "models": {"deepseek-chat": "deepseek-chat"},
            }
        }
    )
    # Custom providers need async discover; set up manually.
    from vertex_proxy.catalog import CustomProviderInfo

    catalog.custom["deepseek-chat"] = CustomProviderInfo(
        provider_name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        upstream_model="deepseek-chat",
    )
    route = catalog.chat_route("deepseek-chat")
    assert route is not None
    assert route.provider is Provider.CUSTOM
    assert route.custom is not None
    assert route.custom.provider_name == "deepseek"
    assert route.upstream_model == "deepseek-chat"


def test_custom_provider_does_not_shadow_anthropic_or_gemini() -> None:
    from vertex_proxy.catalog import CustomProviderInfo

    catalog = _catalog()
    catalog.custom["claude-opus-5-5"] = CustomProviderInfo(
        provider_name="fake", base_url="http://x", api_key="k", upstream_model="x",
    )
    catalog.custom["gemini-3.5-flash"] = CustomProviderInfo(
        provider_name="fake", base_url="http://x", api_key="k", upstream_model="x",
    )
    assert catalog.chat_route("claude-opus-5-5").provider is Provider.ANTHROPIC
    assert catalog.chat_route("gemini-3.5-flash").provider is Provider.GEMINI


def test_custom_provider_shadows_maas() -> None:
    from vertex_proxy.catalog import CustomProviderInfo

    catalog = _catalog()
    catalog.custom["glm-5.2"] = CustomProviderInfo(
        provider_name="direct-glm", base_url="http://x", api_key="k", upstream_model="glm-5.2",
    )
    route = catalog.chat_route("glm-5.2")
    assert route.provider is Provider.CUSTOM


async def test_discover_custom_providers_explicit_models() -> None:
    catalog = _catalog(
        custom_providers={
            "minimax": {
                "base_url": "https://api.minimax.chat/v1",
                "api_key": "sk-test",
                "models": {"minimax-m3": "minimax-m3", "minimax-m1": "MiniMax-Text-01"},
            }
        }
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda r: httpx2.Response(200))) as http:
        await catalog.discover_custom_providers(http)
    assert "minimax-m3" in catalog.custom
    assert "minimax-m1" in catalog.custom
    assert catalog.custom["minimax-m1"].upstream_model == "MiniMax-Text-01"


async def test_discover_custom_providers_auto_discovery() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={
            "data": [{"id": "model-a"}, {"id": "model-b"}, {"id": ""}]
        })

    catalog = _catalog(
        custom_providers={
            "test": {"base_url": "https://api.test.com/v1", "api_key": "sk-x", "models": {}}
        }
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        await catalog.discover_custom_providers(http)
    assert "model-a" in catalog.custom
    assert "model-b" in catalog.custom
    assert "" not in catalog.custom


def test_custom_provider_entries_listed() -> None:
    from vertex_proxy.catalog import CustomProviderInfo

    catalog = _catalog()
    catalog.custom["my-model"] = CustomProviderInfo(
        provider_name="acme", base_url="http://x", api_key="k", upstream_model="upstream-m",
    )
    entries = catalog.entries(Provider.CUSTOM)
    ids = [e["id"] for e in entries]
    assert "my-model" in ids
    custom_entry = next(e for e in entries if e["id"] == "my-model")
    assert custom_entry["provider"] == "custom"
    assert "vertex_model_id" not in custom_entry
