# vertex-proxy

A FastAPI proxy that lets Anthropic Messages, Gemini generateContent, OpenAI Chat Completions and OpenAI Responses clients use Google Cloud Vertex AI. It holds the Google credential, refreshes the token, and translates requests. `AGENTS.md` is a symlink to this file.

## Commands

```bash
uv sync --extra dev                              # install from uv.lock
uv run pytest                                    # tests (no network, no Google)
uv run ruff check vertex_proxy tests
uv run ruff format vertex_proxy tests
uv run vertex-proxy --credentials KEY.json       # run locally
```

After changing dependencies, run `uv lock` and commit `uv.lock`. CI uses `uv sync --locked` and fails on a stale lockfile.

## Layout

- `vertex_proxy/app.py` wires everything. `build_app(settings, tokens=..., transport=...)` takes a token source and an HTTP transport, so tests swap in fakes.
- `routes/` has 1 router per API surface. Each router declares its error dialect (Anthropic, OpenAI, Google or raw) through `dialect_router`, and `ProxyError` renders in that dialect.
- `catalog.py` holds the built-in model lists and name resolution. `vertex.py` builds Vertex URLs per location (`global`, `us`/`eu`, or a region).
- `errors.py` classifies upstream failures and adds hints. `upstream.py` makes authenticated calls and relays streams.
- `bridge/` translates OpenAI Chat Completions to Anthropic Messages and back, streaming included.
- `chat.py` routes a Chat Completions request to Claude, Gemini, MaaS or Ollama. `responses/` translates the Responses API to Chat Completions and back, which is how Codex is served.

## Rules

- Every behavior change gets a test in `tests/`, using the `FakeVertex` and `make_client` fixtures from `tests/conftest.py`. Check that the test fails without the change.
- Streaming routes check the upstream status before they return a `StreamingResponse`, so upstream errors reach the client as HTTP error statuses. Keep it that way.
- Never forward the proxy's API key or a client's `Authorization` header upstream. The `/vertex` pass-through forwards only the headers in its allowlist.
- Model IDs come from Google's Vertex AI docs. When you change a catalog, update the source date in the comment above it.
- Comments say only what the code cannot: a source, a constraint, a reason. Docstrings state the contract.
- American English, digits for counts, no em dashes, and no emoji in code, docs or commits.
- This repo is public. Never commit credentials, home directory paths, internal hostnames or project IDs.

## Releases

`vertex_proxy/__init__.py` holds the version. `CHANGELOG.md` follows Keep a Changelog. Pushing a `v*` tag publishes to PyPI and GHCR, so only the maintainer tags releases. See `CONTRIBUTING.md`.
