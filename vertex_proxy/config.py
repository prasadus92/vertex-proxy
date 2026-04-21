"""Configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for vertex-proxy."""

    model_config = SettingsConfigDict(
        env_prefix="VERTEX_PROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- GCP ---
    # Path to service-account JSON. Uses GOOGLE_APPLICATION_CREDENTIALS if unset.
    credentials_path: Path | None = None
    project_id: str | None = None
    # Region for Claude (Anthropic) models. us-east5 is the primary serving region.
    anthropic_region: str = "us-east5"
    # Region for Gemini models. us-central1 has the widest coverage.
    gemini_region: str = "us-central1"

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8787
    log_level: str = "info"

    # Optional bearer-token auth on the proxy itself. When set, every request
    # must include `Authorization: Bearer <this value>`. Leave unset for
    # localhost-only deploys (the default). Set it if you expose the proxy on
    # a LAN or reverse-proxy it to the internet.
    api_key: str | None = None

    # Prometheus-format metrics endpoint. Adds request counters + token
    # counters by model and provider. Off by default to keep the footprint
    # minimal; enable by setting VERTEX_PROXY_METRICS_ENABLED=true.
    metrics_enabled: bool = False

    # --- Auth refresh ---
    # Access tokens live 60 minutes. Refresh at this interval to stay ahead.
    token_refresh_seconds: int = 3000  # 50 minutes

    # --- Model aliases ---
    # Map canonical Anthropic model names → Vertex publisher model IDs.
    # Keep this list explicit; we want to know exactly what we're routing.
    # Hermes/Claude-Code typically request `claude-sonnet-4-5-20250929`; Vertex
    # uses `claude-sonnet-4-5@20250929`. The proxy translates.
    anthropic_model_aliases: dict[str, str] = {
        # Sonnet 4.5
        "claude-sonnet-4-5": "claude-sonnet-4-5@20250929",
        "claude-sonnet-4-5-20250929": "claude-sonnet-4-5@20250929",
        # Opus 4.5
        "claude-opus-4-5": "claude-opus-4-5@20250929",
        "claude-opus-4-5-20250929": "claude-opus-4-5@20250929",
        # Haiku 4.5
        "claude-haiku-4-5": "claude-haiku-4-5@20250929",
        "claude-haiku-4-5-20250929": "claude-haiku-4-5@20250929",
    }

    # Map canonical Gemini model names → Vertex publisher model IDs.
    gemini_model_aliases: dict[str, str] = {
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.0-flash": "gemini-2.0-flash-001",
    }

    # Region for Vertex MaaS (Model as a Service) open-source partner models:
    # Kimi K2.5, GLM 5, MiniMax-M2.5, Qwen 3.5, Grok 4.20, etc.
    # Vertex typically serves these via the global endpoint or us-central1.
    maas_region: str = "us-central1"

    # Map canonical MaaS model names → Vertex publisher/model path fragments.
    # Path shape on Vertex MaaS is:
    #   publishers/{PUBLISHER}/models/{MODEL_ID}
    # We store the full path fragment so different publishers can coexist.
    # Check each model's "How to use" tab in Model Garden for the exact shape.
    maas_model_aliases: dict[str, str] = {
        # Moonshot (Kimi)
        "kimi-k2.5": "publishers/moonshotai/models/kimi-k2.5",
        "kimi-k2": "publishers/moonshotai/models/kimi-k2",
        # Zhipu (GLM)
        "glm-5": "publishers/zhipu/models/glm-5",
        "glm-5.1": "publishers/zhipu/models/glm-5.1",
        "glm-4.6": "publishers/zhipu/models/glm-4.6",
        # MiniMax
        "minimax-m2.5": "publishers/minimax/models/minimax-m2.5",
        "minimax-m1": "publishers/minimax/models/minimax-m1",
        # Alibaba (Qwen)
        "qwen3.5": "publishers/qwen/models/qwen3.5",
        "qwen-3": "publishers/qwen/models/qwen-3",
        # xAI (Grok on Vertex)
        "grok-4.20": "publishers/xai/models/grok-4.20",
        "grok-4.1-fast": "publishers/xai/models/grok-4.1-fast",
    }


def load_settings() -> Settings:
    return Settings()
