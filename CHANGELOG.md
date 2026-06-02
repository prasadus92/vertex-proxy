# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
