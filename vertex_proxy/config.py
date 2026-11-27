"""Runtime settings, read from CLI flags, environment, a .env file and an optional TOML file."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

CONFIG_ENV_VAR = "VERTEX_PROXY_CONFIG"
_LOCATION = re.compile(r"^[a-z0-9-]+$")


class ConfigError(Exception):
    """The configuration cannot be loaded or is incomplete."""


class Settings(BaseSettings):
    """Runtime configuration. Precedence: CLI flags, environment, .env, TOML file, defaults."""

    model_config = SettingsConfigDict(
        env_prefix="VERTEX_PROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    credentials_path: Path | None = None
    project_id: str | None = None

    # Claude 4.7 and later are served only from the global and multi-region (us, eu) endpoints.
    anthropic_region: str = "global"
    gemini_region: str = "global"
    maas_region: str = "global"

    host: str = "127.0.0.1"
    port: int = 8787
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    api_key: str | None = None
    metrics_enabled: bool = False

    # Access tokens live 60 minutes.
    token_refresh_seconds: int = 3000

    # Merged over the built-in catalog in vertex_proxy.catalog, so these add or override entries.
    anthropic_model_aliases: dict[str, str] = {}
    gemini_model_aliases: dict[str, str] = {}
    maas_model_aliases: dict[str, str] = {}

    # Model name to Ollama base URL; the key "*" discovers every model on that server.
    ollama_backends: dict[str, str] = {}

    # --- Custom OpenAI-compatible providers ---
    # Route models directly to third-party OpenAI-compatible APIs (MiniMax,
    # Mistral, DeepSeek, etc.) instead of going through Vertex.
    #
    # Each key is a provider name (used in model listing and logs). Value is a
    # dict with:
    #   base_url  : API base URL (must support /chat/completions)
    #   api_key   : Bearer token for the provider
    #   models    : dict mapping alias → upstream model ID.
    #              If empty ({}), the proxy will auto-discover via GET /models.
    #
    # Environment variable shape (JSON):
    #   VERTEX_PROXY_CUSTOM_PROVIDERS='{"minimax": {"base_url": "https://api.minimax.chat/v1", "api_key": "sk-...", "models": {"minimax-m3": "minimax-m3"}}}'
    #
    # Models from custom providers take priority over MaaS aliases but NOT
    # over Anthropic/Gemini/Ollama. This lets you override a Vertex MaaS
    # model with a direct-API version while keeping the same model name.
    custom_providers: dict[str, dict[str, Any]] = {}

    @field_validator("anthropic_region", "gemini_region", "maas_region")
    @classmethod
    def _location_name(cls, value: str) -> str:
        if not _LOCATION.match(value):
            raise ValueError(f"{value!r} is not a Vertex AI location such as global or us-east5")
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _lowercase(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @field_validator("credentials_path")
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(config_home) / "vertex-proxy" / "config.toml"


def resolve_config_path(explicit: Path | None = None) -> Path | None:
    """Return the TOML file to read: the explicit path, then $VERTEX_PROXY_CONFIG, then the default.

    An explicit or environment path must exist. The default path is optional.
    """
    requested = explicit or (
        Path(os.environ[CONFIG_ENV_VAR]) if os.environ.get(CONFIG_ENV_VAR) else None
    )
    if requested is not None:
        requested = requested.expanduser()
        if not requested.is_file():
            raise ConfigError(f"config file not found: {requested}")
        return requested
    fallback = default_config_path()
    return fallback if fallback.is_file() else None


def load_settings(config_path: Path | None = None, **overrides: Any) -> Settings:
    """Build settings; ``overrides`` (CLI flags) win over every other source and skip None values."""
    path = resolve_config_path(config_path)
    settings_cls: type[Settings] = Settings
    if path is not None:
        settings_cls = type(
            "FileSettings", (Settings,), {"model_config": SettingsConfigDict(toml_file=path)}
        )
    try:
        return settings_cls(**{key: value for key, value in overrides.items() if value is not None})
    except ValueError as exc:
        source = f" in {path}" if path else ""
        raise ConfigError(f"invalid configuration{source}: {exc}") from exc
