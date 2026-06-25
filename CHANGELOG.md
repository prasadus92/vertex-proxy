# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-06-25

### Added
- Ollama backend support: route OpenAI-compatible requests to local Ollama servers via `VERTEX_PROXY_OLLAMA_BACKENDS` config. Supports wildcard (`"*"`) for auto-discovery of all models from an Ollama server's `/api/tags` endpoint. Ollama is matched after the Vertex routes so a discovered local model cannot shadow a managed model.
- Ollama models appear in `/v1/models` with `owned_by=ollama` and `provider=ollama`.
- docker-compose updated with Ollama env config and `host.docker.internal` networking.

### Fixed
- Strip `context_management` field from Anthropic requests before forwarding to Vertex AI. Claude Code v2.1+ sends this field, and Vertex rejects it with 400 "Extra inputs are not permitted".

## [0.3.0] - 2026-06-06

### Added
- OpenAI -> Anthropic bridge: Claude models are reachable through the OpenAI Chat Completions route (`/v1/chat/completions`); system/tool/assistant messages, max_tokens/temperature/stop, streaming, and tool calls are translated, and response-side Claude `tool_use` is mapped back to OpenAI `tool_calls`.
- Recent Anthropic models on Vertex: Opus 4.8/4.7/4.6 and Sonnet 4.6 (dateless 4.6-generation IDs).
- Per-provider model-listing endpoints: `/anthropic/v1/models`, `/gemini/v1/models`, `/gemini/v1beta/models`; plus OpenAI-compatible `created`/`owned_by` fields on `/v1/models`.

### Fixed
- Corrected the Vertex IDs for the pre-4.6 Claude family: Opus 4.5 -> `@20251101` and Haiku 4.5 -> `@20251001` (both previously 404'd at the Vertex call).
- The OpenAI-route Anthropic model validation now accepts the dateless 4.6-generation IDs (the old guard required an `@` and rejected `claude-opus-4-8` etc.).

## [0.2.0] - 2026-06-02

### Added
- Accept the OpenAI Chat Completions request shape under the `/gemini` and `/openai` prefixes (`/gemini/chat/completions`, `/gemini/v1/chat/completions`, `/openai/chat/completions`) in addition to the bare root, so OpenAI-chat clients work regardless of which `base_url` prefix they are pointed at.
- Mirror model-discovery endpoints (`/v1/models`, `/models`) under the `/gemini` and `/openai` prefixes so clients that probe for a model catalog before dispatching don't get 404s.
- Publish to PyPI via a tag-triggered OIDC Trusted Publishing workflow, enabling `pipx install vertex-proxy` (and `uv tool install` / `uvx`).

### Changed
- Single-source the package version from `vertex_proxy/__init__.py` (hatchling dynamic version), so a release only needs one version bump.

### Fixed
- Fix `404 {"detail":"Not Found"}` for OpenAI-chat clients (e.g. Hermes) configured with a `/gemini` base URL. The client appends `/chat/completions` to `base_url`, which previously had no handler under the native Gemini route prefix ([#1](https://github.com/prasadus92/vertex-proxy/issues/1)).
- Prevent long Vertex streaming responses from failing with incomplete chunked reads by removing the fixed upstream read timeout for stream requests.
- Return structured SSE error events when upstream streaming fails instead of letting the response terminate abruptly.

## [0.1.0] - 2026-04-21

Initial release.

### Added
- Anthropic Messages API-compatible route (`POST /anthropic/v1/messages`) forwarding to Vertex AI's Claude models via `:rawPredict` / `:streamRawPredict`.
- Gemini generateContent API-compatible route (`POST /gemini/v1beta/models/{model}:{action}`) forwarding to Vertex AI Gemini.
- OpenAI Chat Completions API-compatible route (`POST /openai/v1/chat/completions`) for Vertex MaaS partner models (Kimi, GLM, MiniMax, Qwen, Grok).
- Automatic GCP access-token refresh (50-min cadence).
- Streaming support on Anthropic and Gemini routes.
- Model alias mapping (e.g., `claude-sonnet-4-5-20250929` → `claude-sonnet-4-5@20250929`).
- `/health` endpoint for liveness + auth check.
- `/v1/models` endpoint listing all routable models.
- launchd plist template + install script for macOS.
