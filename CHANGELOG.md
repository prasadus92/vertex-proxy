# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-11-16

### Added
- OpenAI Responses API at `POST /v1/responses` (also `/responses` and under `/openai`), so Codex can use any model the proxy serves. Requests are translated to Chat Completions and go through the same routing as `/v1/chat/completions`, streaming included.
  - Function tools, MCP namespace tools and custom tools are translated, and tool calls come back as `function_call` or `custom_tool_call` items with their namespace.
  - Images in tool output, such as from Codex's `view_image`, are passed to the model.
  - Gemini 3 thought signatures on tool calls are carried through the client in a `reasoning` item, so multi-turn tool use keeps working.
  - A context-length rejection arrives in the stream as `context_length_exceeded`, so Codex compacts the conversation and retries.
  - Tools with no Chat Completions form, such as web search, are left out.
  - History items with no Chat Completions form are left out. Messages from other agents (`agent_message`) become user turns, and system and developer messages are merged into 1 leading system message, which some open-model chat templates require.
  - `previous_response_id` is rejected with a 400, because the proxy stores nothing.
  - A `content_filter` stop is reported as `incomplete`.
  - Tested end to end with the Codex CLI 0.155.1 against a scripted fake Vertex AI upstream, on Claude and Gemini models. Not yet run against live Vertex AI.
- `examples/codex-config.toml`.

### Changed
- The OpenAI-to-Claude bridge defaults `max_tokens` to 32000, up from 8192, when the client sends none. Claude 4.5 and later allow at least 64000, and agent turns often write whole files.
- Requests to Claude through the OpenAI routes are counted in metrics under `anthropic-vertex` instead of `openai-anthropic`. Update any dashboards or alerts that filter on the old label.

## [0.5.0] - 2026-11-16

### Added
- Claude Code pass-through: `POST /vertex/...` accepts Vertex AI's own `rawPredict` and `streamRawPredict` calls and adds the proxy's credentials. With `CLAUDE_CODE_USE_VERTEX=1`, `ANTHROPIC_VERTEX_BASE_URL` and `CLAUDE_CODE_SKIP_VERTEX_AUTH=1`, Claude Code sends only the betas Vertex accepts, so beta features that Vertex supports keep working.
- `POST /v1/messages/count_tokens` (and under `/anthropic`), backed by Vertex's Claude count-tokens endpoint, so Claude Code's `/context` shows exact counts.
- TOML config file: `--config`, `$VERTEX_PROXY_CONFIG`, or `~/.config/vertex-proxy/config.toml`. CLI flags beat environment variables, which beat the file. See `examples/config.toml`.
- Global and multi-region (`us`, `eu`) Vertex endpoints.
- Errors come back in the shape of the API the client called (Anthropic, OpenAI or Google), with a hint for the common failures: rejected credentials, a disabled Vertex AI API, a missing IAM role, a model not enabled in the location, and exhausted quota. `retry-after` is passed through.
- Startup fails with a one-line message, and exit code 2, when credentials or the project are missing or rejected.
- The proxy API key is accepted as `Authorization: Bearer`, `x-api-key` or `x-goog-api-key`, so Anthropic and Gemini SDKs can send it.
- Any `claude-*` model name now works without a catalog entry, and `-YYYYMMDD` suffixes are rewritten to Vertex's `@YYYYMMDD` form. Any `publisher/model` name is sent to the Vertex MaaS endpoint.
- `--version`, and the version in `/health`.
- `POST /gemini/v1/models/{model}:{action}`, the path current google-genai releases call, alongside `v1beta`.
- Tagged releases also publish a multi-architecture Docker image to `ghcr.io/prasadus92/vertex-proxy`.
- `CLAUDE.md` (with an `AGENTS.md` link) and `.claude/settings.json` for coding agents working on the repo.
- OpenAI-to-Claude bridge parity:
  - `tools` and `tool_choice` (including `required`, a named function and `parallel_tool_calls: false`) are translated. Before, tools were dropped, so Claude never saw them.
  - Streamed tool calls arrive as OpenAI `tool_calls` deltas.
  - `stream_options.include_usage` adds a final usage chunk.
  - The first chunk carries `role: assistant`.
  - Stream `error` events are forwarded.
- OpenAI image inputs (`image_url` with a data URL or an https URL) become Claude image blocks. The `developer` role is treated as a system message.
- Consecutive turns with the same role, such as tool results followed by a user message, are merged into 1 Claude turn.
- A tool called with no arguments streams `"{}"` as its arguments.
- Cached input tokens count toward `prompt_tokens`, and are also reported as `prompt_tokens_details.cached_tokens`.

### Changed
- The README is rewritten around setup for Claude Code, the Anthropic, OpenAI and Gemini SDKs, and other clients.
- The macOS LaunchAgent label is now `io.github.prasadus92.vertex-proxy`. `launchd/install.sh` removes the agent an earlier release installed, and it uses the `vertex-proxy` found on `PATH` (or `--bin`) instead of a virtualenv inside the repo.
- Supports Python 3.11 to 3.15, and the Docker image runs Python 3.15. Dependencies are raised to their current releases and pinned in a committed `uv.lock`.
- The HTTP client is now `httpx2`, the maintained continuation of `httpx`. It verifies TLS against the operating system's trust store instead of the `certifi` bundle. On a minimal Linux host, install `ca-certificates`, or set `SSL_CERT_FILE` to a CA bundle.
- `uvicorn[standard]` is replaced by `uvicorn` with `uvloop` and `httptools`, dropping unused file-watching and YAML dependencies.
- All 3 locations default to `global`. Claude 4.7 and later are not served from regional endpoints, and global is the cheapest option.
- The model catalog matches Vertex AI as of November 2026: Claude Fable 5.x, Opus 5.5 and 5, Sonnet 5.5 and 5, Haiku 5.5, and the 4.x models still served; Gemini 4, 3.8, 3.7, 3.5 and 3.1; GLM 5.2, gpt-oss-120b and Gemma 4 on MaaS. Models that Vertex has retired or deprecated, or no longer lists for this endpoint (Gemini 2.0 and 2.5, Opus 4.1 and 4, Sonnet 4, Kimi, MiniMax, Qwen, Grok), were removed from the listing. They can still be requested by name.
- Gemini and MaaS chat requests use Vertex's `v1` OpenAI-compatible endpoint with publisher-prefixed model IDs.
- `*_model_aliases` settings now add to the built-in catalog instead of replacing it.
- A credentials file can be a service-account key, user credentials, a workload identity federation config (file, URL, executable or AWS sourced), workforce identity user credentials, or an impersonated service account.
- The bridge reports `finish_reason: length`, not `tool_calls`, when output stops at `max_tokens` in the middle of a tool call. Unsupported roles, content parts, tool types and `tool_choice` values, `n` above 1, `response_format` `json_object` or `json_schema`, and a `temperature` above 1 return a 400 that names them. Before, they were passed on or dropped silently. `top_p` is dropped when `temperature` is set, because current Claude models reject both. The default `max_tokens` is 8192.
- Upstream errors on streaming requests are returned as real HTTP errors before the stream starts, so clients retry them.
- Split the single 880-line module into routes, catalog, errors, upstream and config modules; tests run against a fake Vertex transport.

### Security
- The proxy API key is compared in constant time, model probes now require it, and a key sent in the query string is not forwarded upstream.
- Model and action names in URLs, and the configured locations, are validated before they are placed in an upstream URL.
- The OpenAPI schema and docs pages are no longer served, and `/health` omits the project when an API key is required.

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
