# vertex-proxy

[![CI](https://github.com/prasadus92/vertex-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/prasadus92/vertex-proxy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A small, local-only proxy that bridges **any tool speaking the Anthropic Messages API, Gemini API, or OpenAI Chat Completions API** to **Google Cloud Vertex AI**, so you can point existing clients at Vertex without changing their code.

## What this is for

You have a tool (Claude Code, Hermes Agent, opencode, Cline, Continue.dev, a custom SDK integration, etc.) that already knows how to talk to:

- `api.anthropic.com`
- `generativelanguage.googleapis.com`
- any OpenAI-compatible endpoint

You want that same tool to hit Vertex AI instead, maybe because you want to burn GCP credits, unify billing, or get higher quotas than the public APIs offer.

The problem: Vertex uses **short-lived OAuth access tokens** from a service-account key. Most tools expect a static `Authorization: Bearer xxx` header. Nobody wants to rebuild auth in every client.

vertex-proxy runs on `127.0.0.1:8787`, handles the auth refresh loop, and translates between the public API shapes and Vertex's publisher-model endpoints.

```
┌──────────────┐   Anthropic/Gemini/OpenAI   ┌──────────────┐   GCP auth   ┌────────────┐
│  your tool   │ ──────────────────────────► │ vertex-proxy │ ──────────►  │ Vertex AI  │
└──────────────┘   localhost:8787            └──────────────┘   SA JWT     └────────────┘
```

No client changes. ~400 lines of Python. MIT licensed.

## Install

Python 3.11+, a GCP project with Vertex AI API enabled, and a service-account JSON key with `roles/aiplatform.user`.

```bash
git clone https://github.com/prasadus92/vertex-proxy.git
cd vertex-proxy
python -m venv .venv
.venv/bin/pip install -e .
```

## Run

```bash
export VERTEX_PROXY_CREDENTIALS_PATH=/path/to/service-account.json
export VERTEX_PROXY_PROJECT_ID=your-gcp-project
.venv/bin/vertex-proxy
# → listening on http://127.0.0.1:8787
```

Or inline:

```bash
.venv/bin/vertex-proxy \
  --credentials ~/.vertex/key.json \
  --project-id my-project \
  --port 8787
```

Verify:

```bash
curl http://127.0.0.1:8787/health
# {"status":"ok","project":"my-project"}

curl -X POST http://127.0.0.1:8787/gemini/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}'
```

## Endpoints

| Path | API compat | Vertex backend |
|---|---|---|
| `POST /anthropic/v1/messages` | Anthropic Messages API | `publishers/anthropic/models/{model}:rawPredict` |
| `POST /gemini/v1beta/models/{m}:{action}` | Gemini generateContent API | `publishers/google/models/{m}:{action}` |
| `POST /openai/v1/chat/completions` | OpenAI Chat Completions | Vertex MaaS partner models (Kimi, GLM, MiniMax, Qwen, Grok) |
| `GET /v1/models` | - | Lists routable models |
| `GET /health` | - | Liveness + auth check |

Streaming is supported on Anthropic and Gemini routes.

## Pre-configured models

All aliases live in [`vertex_proxy/config.py`](vertex_proxy/config.py); extend as needed.

**Anthropic** (on Vertex, `us-east5` by default)
- `claude-sonnet-4-5-20250929` → `claude-sonnet-4-5@20250929`
- `claude-opus-4-5-20250929` → `claude-opus-4-5@20250929`
- `claude-haiku-4-5-20250929` → `claude-haiku-4-5@20250929`

**Gemini** (on Vertex, `us-central1` by default)
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`

**MaaS partner models** (OpenAI-compatible route)
- `kimi-k2.5`, `kimi-k2` (Moonshot)
- `glm-5`, `glm-5.1`, `glm-4.6` (Zhipu)
- `minimax-m2.5`, `minimax-m1` (MiniMax)
- `qwen3.5`, `qwen-3` (Alibaba)
- `grok-4.20`, `grok-4.1-fast` (xAI)

## Recipes

### Claude Code CLI

Point Claude Code at the proxy via `ANTHROPIC_BASE_URL`:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
export ANTHROPIC_AUTH_TOKEN=bypass   # proxy ignores this; Vertex auth is server-side
claude
```

Your local Claude Code session now bills against your GCP project instead of api.anthropic.com.

### Hermes Agent

Add to `~/.hermes/config.yaml`:

```yaml
custom_providers:
  - name: vertex-gemini
    base_url: http://127.0.0.1:8787/gemini
    transport: openai_chat

  - name: vertex-anthropic
    base_url: http://127.0.0.1:8787/anthropic
    transport: anthropic_messages

fallback_model:
  provider: vertex-gemini
  model: gemini-2.5-pro
```

Zero Hermes source changes required. Picks up the existing `custom_providers` mechanism.

### opencode / Cline / any Anthropic-SDK client

Set the base URL environment variable the client supports (usually one of `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_URL`, or the equivalent in your client's config):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
```

## Run as a service (macOS launchd)

```bash
cd launchd
./install.sh --credentials /path/to/key.json --project my-gcp-project
```

This renders the plist template, copies it to `~/Library/LaunchAgents/`, loads it, and does a health check. Logs go to `~/Library/Logs/vertex-proxy.{log,err}`.

Stop:
```bash
launchctl unload ~/Library/LaunchAgents/ai.hermes.vertex-proxy.plist
```

For Linux, the same pattern works with systemd; see [`examples/systemd.service`](examples/systemd.service).

## Configuration reference

All settings accept `VERTEX_PROXY_` env var prefix or CLI flags.

| Env var | Default | Purpose |
|---|---|---|
| `VERTEX_PROXY_CREDENTIALS_PATH` | - | Service-account JSON path (falls back to ADC) |
| `VERTEX_PROXY_PROJECT_ID` | inferred from key | GCP project ID |
| `VERTEX_PROXY_ANTHROPIC_REGION` | `us-east5` | Region for Claude |
| `VERTEX_PROXY_GEMINI_REGION` | `us-central1` | Region for Gemini |
| `VERTEX_PROXY_MAAS_REGION` | `us-central1` | Region for Kimi / GLM / MiniMax / Qwen / Grok |
| `VERTEX_PROXY_HOST` | `127.0.0.1` | Bind host |
| `VERTEX_PROXY_PORT` | `8787` | Bind port |
| `VERTEX_PROXY_TOKEN_REFRESH_SECONDS` | `3000` | Token refresh interval (50 min) |
| `VERTEX_PROXY_LOG_LEVEL` | `info` | uvicorn log level |

## A word on GCP credits

**GCP promotional credits (startup, free trial, partner) typically do NOT cover Google Cloud Marketplace purchases.** On Vertex AI, this matters because:

- **First-party Google models** (Gemini 2.5 Pro / Flash, Gemma) are billed as "Vertex AI API" usage → **credits cover ✅**
- **Partner models** (Claude, Kimi, GLM, MiniMax, Grok) are typically billed via GCP Marketplace → **credits usually don't cover ❌**

The "Promotional credits" section of your model's agreement page in Google Cloud Console will tell you explicitly. Quote from a typical Claude-on-Vertex agreement:

> *Most Google Cloud promotional credits don't apply to Google Cloud Marketplace purchases.*

If credit-burn is your goal, point vertex-proxy at Gemini. If billing unification is your goal, vertex-proxy works for everything.

## Security

vertex-proxy binds to `127.0.0.1` by default and **ships with no authentication**. It's designed as a local-loopback shim; anyone who can reach it can spend your GCP credits via your service account.

Do not expose it to a public interface. If you need remote access, put it behind a reverse proxy with proper auth (nginx + basic auth, Tailscale, Cloud Run with IAP, etc.).

## Status

- [x] Anthropic Messages API → Vertex Claude (with streaming)
- [x] Gemini generateContent API → Vertex Gemini (with streaming)
- [x] OpenAI Chat Completions → Vertex Gemini via Vertex's OpenAI-compat layer
- [x] OpenAI Chat Completions → Vertex MaaS partner models (Kimi, GLM, MiniMax, Qwen, Grok)
- [x] Multiple URL shapes accepted for OpenAI client compatibility (`/v1/chat/completions`, `/chat/completions`, `/openai/v1/chat/completions`)
- [x] Automatic GCP service-account token refresh
- [x] launchd (macOS) + systemd (Linux) service recipes
- [x] Dockerfile + docker-compose for containerized deploy
- [x] Optional bearer-token auth on the proxy itself (for remote deploys)
- [x] Prometheus metrics endpoint at `/metrics`
- [x] 13 unit tests, GitHub Actions CI on Python 3.11 + 3.12

### Tested with
- [x] Hermes Agent — verified end-to-end with live Gemini 2.5 Flash dispatch
- [x] Claude Code CLI — via `ANTHROPIC_BASE_URL` env
- [x] Direct `curl` against all routes

## Troubleshooting

### 404 "model not found" on Claude routes

Most Vertex AI Claude model endpoints require one-time enablement in Model Garden. Go to https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden and click ENABLE on the specific model (Sonnet, Opus, Haiku). Accept the Marketplace T&Cs. Your service account can then call them.

Note: GCP promotional credits typically don't cover Marketplace models. See "A word on GCP credits" above.

### 404 "model not found" on MaaS routes (Kimi, GLM, MiniMax, Qwen, Grok)

Same as Claude: Vertex partner models require Model Garden enablement per model. Additionally, the MaaS path in `config.py` is a best-effort guess at Vertex's URL shape for these partners. If you hit 404s after enablement, check the "How to use" tab on the model's page in Model Garden and update the `maas_model_aliases` entry with the exact path fragment Google shows.

### 401 / 403 on all routes

Your service account lacks `roles/aiplatform.user`. Grant it:
```
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Gemini 2.5 returns empty content with `reasoning_tokens` populated

Gemini 2.5 models use an internal "thinking" budget that counts against `max_tokens`. If `max_tokens` is too low, the model may use all its budget on thinking and return no visible output. Raise `max_tokens` to at least 100 for anything beyond trivial replies.

### Request works with `curl` but fails from my OpenAI client

Your client is probably sending requests to a URL shape the shim didn't expect. The shim accepts `/v1/chat/completions`, `/chat/completions`, and `/openai/v1/chat/completions` for OpenAI-compatible traffic. If your client sends something else, file an issue with the exact URL shape and we'll add it.

### Token refresh errors in logs

The background refresh task logs errors but doesn't crash the process. If you see repeated refresh failures, check:
1. Service account JSON path is correct (`VERTEX_PROXY_CREDENTIALS_PATH`)
2. Machine clock is in sync (GCP JWT exchange is clock-sensitive)
3. Service account isn't disabled or rotated in GCP IAM

## Comparison with alternatives

| Tool | What it does | Fit |
|---|---|---|
| **vertex-proxy** (this) | Bridge existing Anthropic/Gemini/OpenAI clients to Vertex AI with auto-auth | You already use a tool with configurable base URL and want to point it at Vertex without rewriting auth |
| **LiteLLM** | Full-featured multi-provider router with caching, budgets, observability | Managing many providers centrally with policies; heavier dependency |
| **openai-compat-server** (various) | OpenAI shape over arbitrary backend | Similar to one route of vertex-proxy; doesn't handle GCP SA auth natively |
| **Vertex AI Python SDK** | Direct first-party Google SDK | You're writing new code and want to talk Vertex directly |
| **Anthropic Python SDK with Vertex backend** | First-party SDK with Vertex mode flag | You're writing new Anthropic code and control the client |

Use vertex-proxy when you have an **existing** tool you can't modify and need to redirect its traffic to Vertex.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome.

## License

MIT. See [LICENSE](LICENSE).

## Credits

Built by Prasad Subrahmanya ([prasad.tech](https://prasad.tech) · [@prasadus92](https://github.com/prasadus92)) as part of solving the "Hermes fallback model" problem for [Luminik](https://luminik.io), then extracted into a standalone tool because the shim turned out to be useful beyond Hermes.
