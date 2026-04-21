# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
