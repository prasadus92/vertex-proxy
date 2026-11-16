# vertex-proxy

[![CI](https://github.com/prasadus92/vertex-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/prasadus92/vertex-proxy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vertex-proxy.svg)](https://pypi.org/project/vertex-proxy/)
[![Python 3.11 to 3.15](https://img.shields.io/badge/python-3.11%20to%203.15-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

vertex-proxy is a local proxy that lets tools built for the Anthropic, Gemini or OpenAI APIs run on Google Cloud Vertex AI. Claude Code, Codex, the official SDKs and OpenAI-compatible agent frameworks work against it without code changes.

Vertex AI authenticates with short-lived OAuth tokens from a Google Cloud credential. Most clients expect a static API key and a public API's request shape. The proxy holds the credential, refreshes the token in the background, and translates requests to Vertex's endpoints.

```
your tool  --Anthropic / Gemini / OpenAI API-->  vertex-proxy  --Google auth-->  Vertex AI
           http://127.0.0.1:8787
```

Use it to keep model spend on a GCP bill, to draw on GCP credits or quota, or to keep traffic inside a Google Cloud project.

## Quickstart

You need Python 3.11 or later, a GCP project with the Vertex AI API enabled, and credentials with the `roles/aiplatform.user` role. Partner models such as Claude must also be enabled once in [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden).

```bash
uv tool install vertex-proxy        # or: pipx install vertex-proxy
# the latest main branch: uv tool install git+https://github.com/prasadus92/vertex-proxy
```

Run it with a service-account key:

```bash
vertex-proxy --credentials ~/keys/vertex.json
```

Or with your own Google login:

```bash
gcloud auth application-default login
vertex-proxy --project-id my-gcp-project
```

Check that it is up and that Google accepted the credentials:

```bash
curl http://127.0.0.1:8787/health
```

If the credentials are missing or rejected, the proxy exits with a one-line message that says what to fix.

## Use it with your tools

### Claude Code

Run Claude Code in its Vertex AI mode and send it through the proxy. In this mode Claude Code sends only the features Vertex supports, and the proxy adds the credentials:

```bash
export CLAUDE_CODE_USE_VERTEX=1
export CLAUDE_CODE_SKIP_VERTEX_AUTH=1
export ANTHROPIC_VERTEX_BASE_URL=http://127.0.0.1:8787/vertex
export ANTHROPIC_VERTEX_PROJECT_ID=vertex-proxy   # any value; the proxy uses its own project
export CLOUD_ML_REGION=global
claude
```

In this mode the location comes from `CLOUD_ML_REGION`, model names are Vertex IDs, and the proxy replaces only the project and the credentials, so its own location and alias settings do not apply. The same variables are in [`examples/claude-code-env.sh`](examples/claude-code-env.sh), together with optional model pins and the header to send when the proxy requires an API key. Run `/status` in Claude Code to confirm the provider shows as Google Vertex AI.

Claude Code can also use the proxy as a plain Anthropic endpoint. That works for older Claude Code versions and other Anthropic clients, but Claude Code then sends beta features that Vertex rejects, so set `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
export ANTHROPIC_AUTH_TOKEN=unused
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
```

### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8787/anthropic", api_key="unused")
reply = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### OpenAI SDK

The OpenAI-compatible route serves Claude, Gemini and Vertex's open models from 1 base URL. For Claude, the proxy translates to the Anthropic Messages API, including tools and streamed tool calls.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="unused")
reply = client.chat.completions.create(
    model="gemini-3.5-flash",  # or claude-opus-5-5, glm-5.2, gpt-oss-120b
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Gemini SDK

```python
from google import genai

client = genai.Client(api_key="unused", http_options={"base_url": "http://127.0.0.1:8787/gemini"})
reply = client.models.generate_content(model="gemini-3.5-flash", contents="Hello")
```

### Other OpenAI-compatible clients

Agent frameworks and coding tools such as opencode, Cline, Continue and Hermes Agent take a base URL and a model name. Use `http://127.0.0.1:8787/v1` and any model from `GET /v1/models`. The chat route is also mounted at `/chat/completions`, `/openai/...` and `/gemini/...`, so the base URL works whichever form the client expects.

### Codex

Codex talks to custom providers through the OpenAI Responses API, which the proxy serves at `/v1/responses` for every model it knows. Add a provider to `~/.codex/config.toml`:

```toml
model = "claude-opus-5-5"
model_provider = "vertex"

[model_providers.vertex]
name = "Vertex AI"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
```

Or try it without editing the file:

```bash
codex -c model_provider=vertex \
  -c 'model_providers.vertex.name="Vertex AI"' \
  -c 'model_providers.vertex.base_url="http://127.0.0.1:8787/v1"' \
  -c 'model_providers.vertex.wire_api="responses"' \
  -m gemini-3.5-flash
```

Tool calls, MCP tools and streaming work with Claude, Gemini and the open models. Codex warns that it has no metadata for the model name; that is expected for models outside OpenAI's list. Codex's built-in web search has no Vertex equivalent, so the proxy leaves it out of the request. [`examples/codex-config.toml`](examples/codex-config.toml) has the same settings.

### Local models with Ollama

Ollama models can sit next to the Vertex models on the OpenAI route:

```bash
export VERTEX_PROXY_OLLAMA_BACKENDS='{"*": "http://localhost:11434"}'
```

`"*"` lists every model on that server at startup. You can also map single model names to servers. A Vertex model always wins over an Ollama model with the same name.

## Endpoints

| Path | Speaks | Goes to |
|---|---|---|
| `POST /vertex/...` | Vertex AI `rawPredict`, `streamRawPredict` | The same call on Vertex, with the proxy's credentials and project |
| `POST /anthropic/v1/messages`, `/v1/messages` | Anthropic Messages | Claude on Vertex |
| `POST /anthropic/v1/messages/count_tokens`, `/v1/messages/count_tokens` | Anthropic token counting | Vertex Claude count-tokens |
| `POST /gemini/v1beta/models/{model}:{action}`, `/v1beta/models/...` | Gemini `generateContent`, `streamGenerateContent`, `countTokens` | Gemini on Vertex |
| `POST /v1/chat/completions` (and mirrors) | OpenAI Chat Completions | Claude (translated), Gemini, Vertex open models, Ollama |
| `POST /v1/responses` (and mirrors) | OpenAI Responses | The same models, through the Chat Completions routing |
| `GET /v1/models`, `/anthropic/v1/models`, `/gemini/v1beta/models` | Model listings | |
| `GET /health` | Liveness and a credential check | |
| `GET /metrics` | Prometheus counters, when enabled | |

The messages, generate, chat and responses routes support streaming.

## Models

The built-in catalog in [`vertex_proxy/catalog.py`](vertex_proxy/catalog.py) follows Google's model list as of November 2026:

- **Claude**: Fable 5.2, 5.1 and 5; Opus 5.5, 5, 4.8, 4.7, 4.6 and 4.5; Sonnet 5.5, 5, 4.6 and 4.5; Haiku 5.5 and 4.5.
- **Gemini**: 4 (Hydrogen, Carbon, Argon), 3.8 Flash, 3.7 Flash, 3.5 Flash and Flash-Lite, 3.1 Pro preview and 3.1 Flash-Lite.
- **Open models on Vertex**: `glm-5.2`, `gpt-oss-120b`, `gemma-4-26b-a4b-it`.

You don't need a catalog entry to use a newer model:

- Any `claude-*` name is sent to Vertex. A dated name such as `claude-sonnet-4-5-20250929` becomes Vertex's `claude-sonnet-4-5@20250929`.
- Any `gemini-*` name is sent to Gemini on Vertex.
- Any `publisher/model` name, such as `deepseek-ai/deepseek-v4.1-flash-maas`, is sent to Vertex's open-model endpoint.

To add short names, put them in the config file. They are merged over the built-in catalog.

## Configuration

CLI flags override environment variables, which override a `.env` file, which overrides the TOML config file. The config file is read from `--config`, `$VERTEX_PROXY_CONFIG`, or `$XDG_CONFIG_HOME/vertex-proxy/config.toml` (by default `~/.config/vertex-proxy/config.toml`). [`examples/config.toml`](examples/config.toml) shows every option.

| Setting | Environment variable | Default |
|---|---|---|
| Credentials file | `VERTEX_PROXY_CREDENTIALS_PATH` | Application Default Credentials |
| Project | `VERTEX_PROXY_PROJECT_ID` | From the credentials |
| Claude location | `VERTEX_PROXY_ANTHROPIC_REGION` | `global` |
| Gemini location | `VERTEX_PROXY_GEMINI_REGION` | `global` |
| Open-model location | `VERTEX_PROXY_MAAS_REGION` | `global` |
| Bind address | `VERTEX_PROXY_HOST`, `VERTEX_PROXY_PORT` | `127.0.0.1`, `8787` |
| Proxy API key | `VERTEX_PROXY_API_KEY` | none |
| Metrics | `VERTEX_PROXY_METRICS_ENABLED` | `false` |
| Extra model names | `VERTEX_PROXY_ANTHROPIC_MODEL_ALIASES`, `..._GEMINI_...`, `..._MAAS_...` | `{}` |
| Ollama servers | `VERTEX_PROXY_OLLAMA_BACKENDS` | `{}` |
| Log level | `VERTEX_PROXY_LOG_LEVEL` | `info` |

A location can be `global`, a multi-region (`us` or `eu`), or a region such as `us-east5`. Claude 4.7 and later are served only from `global`, `us` and `eu`, and regional endpoints cost more. Keep `global` unless you need data residency.

The credentials file can be a service-account key, user credentials, or a workload identity federation config.

## Errors

Errors come back in the shape of the API you called, so SDKs raise their usual exception types. For the common failures, the message also says what to do:

| Vertex returns | The message tells you |
|---|---|
| 401 | The credentials were rejected |
| 403 | To enable the Vertex AI API (with the `gcloud` command), or to grant `roles/aiplatform.user` |
| 404 | The model is not enabled in Model Garden or not served in that location |
| 429 | The quota for that model and location is exhausted; `retry-after` is passed through |

Streaming requests are checked before the stream starts, so a quota error arrives as an HTTP 429 that clients retry. The `/vertex` pass-through forwards Vertex's error bodies unchanged.

## Docker

```bash
docker build -t vertex-proxy .
docker run --rm -p 127.0.0.1:8787:8787 \
  -v ~/keys/vertex.json:/app/credentials.json:ro \
  -e VERTEX_PROXY_CREDENTIALS_PATH=/app/credentials.json \
  vertex-proxy
```

[`docker-compose.yaml`](docker-compose.yaml) does the same with a `.env` file. The image runs as a non-root user and has a health check. Tagged releases publish it as `ghcr.io/prasadus92/vertex-proxy`.

## Run as a service

On macOS, [`launchd/install.sh`](launchd/install.sh) installs a LaunchAgent that starts at login:

```bash
./launchd/install.sh --credentials ~/keys/vertex.json --project my-gcp-project
```

On Linux, adapt [`examples/systemd.service`](examples/systemd.service).

## Security

The proxy binds to `127.0.0.1` and has no authentication by default. Anyone who can reach it spends on your GCP project. Before you expose it beyond localhost, set `VERTEX_PROXY_API_KEY`. Clients can send the key as `Authorization: Bearer`, `x-api-key` or `x-goog-api-key`, which covers the Anthropic, OpenAI and Gemini SDKs. Put TLS in front of it with a reverse proxy.

The proxy does not log request or response bodies.

## A note on GCP credits

Google's own models, such as Gemini and Gemma, are billed as Vertex AI usage. Promotional credits usually cover them. Partner models such as Claude are usually billed through Google Cloud Marketplace, and most promotional credits do not apply there. Each model's page in Model Garden states what applies.

## Troubleshooting

**404 for a Claude model.** Enable the model in Model Garden and accept its terms. If you set a regional location, check that the model is served there, or switch back to `global`.

**429 on every request.** Some projects start with a quota of 0 for partner models. Request an increase on the Quotas page for `aiplatform.googleapis.com`.

**Claude Code reports `Extra inputs are not permitted`.** Claude Code is using the plain Anthropic endpoint. Use the Vertex mode setup above, or set `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`.

**Gemini replies are empty and `reasoning_tokens` is set.** Thinking used up `max_tokens`. Raise it.

**TLS errors on a minimal Linux host.** The proxy verifies TLS against the system trust store. Install `ca-certificates`, or point `SSL_CERT_FILE` at a CA bundle.

## How it compares

| Tool | Choose it when |
|---|---|
| vertex-proxy | An existing tool should run on Vertex AI without code or auth changes |
| LiteLLM | You route across many providers and need budgets, caching and a management UI |
| Anthropic, OpenAI or Google SDKs in Vertex mode | You are writing new code and control the client |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The [changelog](CHANGELOG.md) lists every release.

## License

MIT. See [LICENSE](LICENSE). Built by [Prasad Subrahmanya](https://prasad.tech).
