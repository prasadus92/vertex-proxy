"""Which models the proxy knows, and how a requested model name maps to a Vertex AI model."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx2

from .config import Settings

logger = logging.getLogger(__name__)

# Source: platform.claude.com, "Claude on Vertex AI", checked 2026-11-16. Models from 4.6 on use
# dateless IDs; earlier ones carry an @date snapshot, which a -YYYYMMDD suffix is rewritten to.
ANTHROPIC_MODELS: dict[str, str] = {
    "claude-fable-5-2": "claude-fable-5-2",
    "claude-fable-5-1": "claude-fable-5-1",
    "claude-fable-5": "claude-fable-5",
    "claude-opus-5-5": "claude-opus-5-5",
    "claude-opus-5": "claude-opus-5",
    "claude-opus-4-8": "claude-opus-4-8",
    "claude-opus-4-7": "claude-opus-4-7",
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-sonnet-5-5": "claude-sonnet-5-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-haiku-5-5": "claude-haiku-5-5",
    "claude-opus-4-5": "claude-opus-4-5@20251101",
    "claude-sonnet-4-5": "claude-sonnet-4-5@20250929",
    "claude-haiku-4-5": "claude-haiku-4-5@20251001",
}

# Source: Vertex AI "Model versions and lifecycle", checked 2026-11-16 (GA and current previews).
GEMINI_MODELS: dict[str, str] = {
    name: name
    for name in (
        "gemini-4-hydrogen",
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
    )
}

# Source: Vertex AI "Call open model APIs", checked 2026-11-16. MaaS models share the
# OpenAI-compatible endpoint and are addressed by a publisher-prefixed model string.
MAAS_MODELS: dict[str, str] = {
    "gpt-oss-120b": "openai/gpt-oss-120b-maas",
    "glm-5.2": "zai-org/glm-5.2-maas",
    "gemma-4-26b-a4b-it": "google/gemma-4-26b-a4b-it-maas",
}

_DATED_CLAUDE = re.compile(r"^(claude-[a-z0-9-]+?)-(\d{8})$")
_CLAUDE = re.compile(r"^claude-[a-z0-9.-]+(@\d{8})?$")


class Provider(StrEnum):
    ANTHROPIC = "anthropic-vertex"
    GEMINI = "gemini-vertex"
    MAAS = "maas-vertex"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass(frozen=True)
class CustomProviderInfo:
    """Connection details for a custom OpenAI-compatible provider."""

    provider_name: str
    base_url: str
    api_key: str
    upstream_model: str


@dataclass(frozen=True)
class ChatRoute:
    """Where an OpenAI Chat Completions request for a given model goes."""

    provider: Provider
    upstream_model: str
    ollama_url: str | None = None
    custom: CustomProviderInfo | None = None


class ModelCatalog:
    """Built-in models merged with the user's aliases, plus any Ollama models found at startup."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.anthropic = {**ANTHROPIC_MODELS, **settings.anthropic_model_aliases}
        self.gemini = {**GEMINI_MODELS, **settings.gemini_model_aliases}
        self.maas = {**MAAS_MODELS, **settings.maas_model_aliases}
        self.ollama: dict[str, str] = {}
        self.custom: dict[str, CustomProviderInfo] = {}

    def anthropic_model(self, name: str) -> str | None:
        """Vertex model ID for a Claude model name, or None when the name is not a Claude model."""
        if name in self.anthropic:
            return self.anthropic[name]
        dated = _DATED_CLAUDE.match(name)
        if dated:
            return f"{dated.group(1)}@{dated.group(2)}"
        return name if _CLAUDE.match(name) else None

    def gemini_model(self, name: str) -> str:
        return self.gemini.get(name, name)

    def chat_route(self, name: str) -> ChatRoute | None:
        """Route for an OpenAI-format request. Vertex models are matched before Ollama ones.

        Custom providers sit between Ollama and MaaS: they override MaaS aliases
        but not Anthropic, Gemini or Ollama."""
        if (claude := self.anthropic_model(name)) is not None:
            return ChatRoute(Provider.ANTHROPIC, claude)
        bare = name.removeprefix("google/")
        if bare in self.gemini or bare.startswith("gemini-"):
            return ChatRoute(Provider.GEMINI, f"google/{self.gemini_model(bare)}")
        if name in self.ollama:
            return ChatRoute(Provider.OLLAMA, name, ollama_url=self.ollama[name])
        if name in self.custom:
            info = self.custom[name]
            return ChatRoute(Provider.CUSTOM, info.upstream_model, custom=info)
        if name in self.maas:
            return ChatRoute(Provider.MAAS, self.maas[name])
        if "/" in name:
            return ChatRoute(Provider.MAAS, name)
        return None

    def knows(self, name: str) -> bool:
        return self.chat_route(name) is not None

    def entries(self, *providers: Provider) -> list[dict[str, Any]]:
        """Model listing in OpenAI shape, with Anthropic-friendly fields, for the given providers."""
        wanted = set(providers) or set(Provider)
        s = self._settings
        groups: list[tuple[Provider, dict[str, str], str | None]] = [
            (Provider.ANTHROPIC, self.anthropic, s.anthropic_region),
            (Provider.GEMINI, self.gemini, s.gemini_region),
            (Provider.MAAS, self.maas, s.maas_region),
            (Provider.OLLAMA, {name: name for name in self.ollama}, None),
            (Provider.CUSTOM, {alias: info.upstream_model for alias, info in self.custom.items()}, None),
        ]
        return [
            _entry(alias, target, provider, region)
            for provider, models, region in groups
            if provider in wanted
            for alias, target in models.items()
        ]

    async def discover_custom_providers(self, http: httpx2.AsyncClient) -> None:
        """Register models from custom OpenAI-compatible providers.

        Each provider entry must have 'base_url' and 'api_key'. 'models' is a dict
        mapping alias → upstream model ID. If 'models' is empty, we attempt to
        auto-discover via GET {base_url}/models.
        """
        self.custom.clear()
        for provider_name, provider_cfg in self._settings.custom_providers.items():
            base_url = provider_cfg.get("base_url", "").rstrip("/")
            api_key = provider_cfg.get("api_key", "")
            models: dict[str, str] = dict(provider_cfg.get("models", {}))

            if not base_url:
                logger.warning("custom_provider %s: missing base_url, skipping", provider_name)
                continue
            if not api_key:
                logger.warning("custom_provider %s: missing api_key, skipping", provider_name)
                continue

            if not models:
                try:
                    resp = await http.get(
                        f"{base_url}/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for m in data.get("data", []):
                        model_id = m.get("id", "")
                        if model_id:
                            models[model_id] = model_id
                    logger.info(
                        "custom_provider %s: discovered %d models from %s",
                        provider_name, len(models), base_url,
                    )
                except Exception as exc:
                    logger.warning(
                        "custom_provider %s: failed to discover models from %s: %s",
                        provider_name, base_url, exc,
                    )
                    continue

            for alias, upstream_id in models.items():
                self.custom[alias] = CustomProviderInfo(
                    provider_name=provider_name,
                    base_url=base_url,
                    api_key=api_key,
                    upstream_model=upstream_id,
                )
            logger.info(
                "custom_provider %s: %d models registered: %s",
                provider_name, len(models), sorted(models.keys()),
            )

    async def discover_ollama(self, http: httpx2.AsyncClient) -> None:
        """Register configured Ollama models; a "*" key lists every model on that server."""
        self.ollama.clear()
        for name, base_url in self._settings.ollama_backends.items():
            base_url = base_url.rstrip("/")
            if name != "*":
                self.ollama[name] = base_url
                continue
            try:
                resp = await http.get(f"{base_url}/api/tags", timeout=10.0)
                resp.raise_for_status()
                found = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
            except (httpx2.HTTPError, ValueError) as exc:
                logger.warning("ollama: could not list models at %s: %s", base_url, exc)
                continue
            self.ollama.update(dict.fromkeys(found, base_url))
            logger.info("ollama: found %d models at %s", len(found), base_url)


_OWNERS = {
    Provider.ANTHROPIC: "anthropic",
    Provider.GEMINI: "google",
    Provider.OLLAMA: "ollama",
    Provider.CUSTOM: "custom",
}


def _entry(alias: str, target: str, provider: Provider, region: str | None) -> dict[str, Any]:
    owner = _OWNERS.get(provider) or target.split("/", 1)[0]
    entry: dict[str, Any] = {
        "id": alias,
        "object": "model",
        "type": "model",
        "display_name": alias,
        "created": 0,
        "owned_by": owner,
        "provider": provider.value,
    }
    if provider not in (Provider.OLLAMA, Provider.CUSTOM):
        entry["vertex_model_id"] = target
        entry["region"] = region
    return entry
