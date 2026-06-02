# Contributing

Thanks for your interest. This is a small project maintained in spare time. Contributions welcome via pull request.

## Dev setup

```
git clone https://github.com/prasadus92/vertex-proxy.git
cd vertex-proxy
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Running tests

```
.venv/bin/pytest
```

Tests are pure unit / mocked; they don't hit real GCP. To smoke-test against live Vertex AI you need real credentials, so do it manually (see "Running locally against real Vertex" below); CI stays mock-only.

## Running locally against real Vertex

```
export VERTEX_PROXY_CREDENTIALS_PATH=/path/to/gcp-key.json
export VERTEX_PROXY_PROJECT_ID=your-project
.venv/bin/vertex-proxy
```

Then in another terminal:

```
curl http://127.0.0.1:8787/health
curl -X POST http://127.0.0.1:8787/gemini/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}'
```

## Style

- Line length: 100 chars
- Format: `ruff format`
- Lint: `ruff check`

## Scope

This project intentionally does not:

- Authenticate incoming requests by default (it's a local-loopback proxy; optional bearer-token auth is available via `VERTEX_PROXY_API_KEY` for remote deploys)
- Do request transformation beyond what Vertex requires (e.g., Anthropic `model` field → URL path)
- Cache responses
- Log request bodies (privacy + credit safety)

If you want any of the above, file an issue first so we can discuss design.

## Adding a new model

Most additions only require editing `vertex_proxy/config.py`:

- Claude model → add to `anthropic_model_aliases`
- Gemini model → add to `gemini_model_aliases`
- MaaS partner model (Kimi, GLM, MiniMax, Qwen, Grok, …) → add to `maas_model_aliases`

For a genuinely new model family with a different API shape, open an issue first.

## Releasing (maintainer)

Releases publish to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so there is no API token to store as a secret.

One-time setup:

1. On PyPI, add a trusted publisher at https://pypi.org/manage/account/publishing/ with:
   - PyPI project name: `vertex-proxy`
   - Owner: `prasadus92`
   - Repository: `vertex-proxy`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
2. (Recommended) Create a GitHub environment named `pypi` under repo Settings to gate the publish job.

To cut a release:

1. Bump `__version__` in `vertex_proxy/__init__.py`. That is the single source of truth; `pyproject.toml` and the running app's `version` both read from it.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new dated version heading.
3. Commit, then tag and push:
   ```
   git tag v0.2.0
   git push origin v0.2.0
   ```
4. The `Release` workflow builds the sdist + wheel, runs `twine check`, and publishes to PyPI.
