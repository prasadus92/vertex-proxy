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

Tests are pure unit / mocked — they don't hit real GCP. Integration smoke tests against live Vertex AI live in `tests/integration/` and require real credentials (run manually, not in CI).

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

- Authenticate incoming requests (it's a local-loopback proxy)
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
