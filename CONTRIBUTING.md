# Contributing

Pull requests are welcome. For a new feature, open an issue first so we can agree on the design.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and a committed `uv.lock`.

```bash
git clone https://github.com/prasadus92/vertex-proxy.git
cd vertex-proxy
uv sync --extra dev
```

## Checks

```bash
uv run pytest
uv run ruff check vertex_proxy tests
uv run ruff format --check vertex_proxy tests
```

CI runs these on Python 3.11 to 3.15, audits the locked dependencies with pip-audit, and builds the Docker image.

The tests never call Google. `tests/conftest.py` provides a fake Vertex AI upstream (`FakeVertex`, backed by `httpx2.MockTransport`) and a fake token source. Every behavior change needs a test that fails without the change.

## Running against live Vertex AI

```bash
uv run vertex-proxy --credentials /path/to/key.json --project-id your-project
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemini-3.5-flash", "messages": [{"role": "user", "content": "hi"}]}'
```

## Code layout

| Path | Job |
|---|---|
| `vertex_proxy/config.py` | Settings and the TOML config file |
| `vertex_proxy/catalog.py` | Built-in models and model name resolution |
| `vertex_proxy/vertex.py` | Vertex AI URLs per location |
| `vertex_proxy/auth.py` | Credentials and token refresh |
| `vertex_proxy/errors.py` | Upstream errors, hints and per-API error shapes |
| `vertex_proxy/upstream.py` | Authenticated upstream calls and stream relaying |
| `vertex_proxy/bridge/` | OpenAI Chat Completions to Anthropic Messages translation |
| `vertex_proxy/routes/` | 1 router per API surface |

## Adding a model

Add it to the matching dictionary in `vertex_proxy/catalog.py` and update the source date in the comment above it. Take model IDs from Google's Vertex AI model pages; they sometimes differ from the Anthropic and Gemini API IDs. Users can also add models without a release through the config file.

## Scope

The proxy does not cache responses, log request or response bodies, or rewrite requests beyond what Vertex needs.

## Releasing (maintainer)

Pushing a `v*` tag runs `.github/workflows/release.yml`. It builds the package and publishes it to PyPI through [Trusted Publishing](https://docs.pypi.org/trusted-publishers/), and it publishes the Docker image to `ghcr.io/prasadus92/vertex-proxy`. No API tokens are stored.

1. Set `__version__` in `vertex_proxy/__init__.py`. `pyproject.toml` and the app read it from there.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a dated version heading.
3. Merge that change, then tag the merge commit on `main` and push the tag:
   ```bash
   git tag -s v0.5.0 -m "v0.5.0"
   git push origin v0.5.0
   ```
4. Create the GitHub release from the tag, with the changelog section as its notes.
