#!/usr/bin/env bash
# Point Claude Code at vertex-proxy. Source this file before running `claude`.
#
# Claude Code runs in its Vertex AI mode and the proxy adds the Google credentials, so
# Claude Code sends only the features Vertex supports. The proxy uses its own project,
# so ANTHROPIC_VERTEX_PROJECT_ID only needs to be non-empty.

export CLAUDE_CODE_USE_VERTEX=1
export CLAUDE_CODE_SKIP_VERTEX_AUTH=1
export ANTHROPIC_VERTEX_BASE_URL="http://127.0.0.1:8787/vertex"
export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-vertex-proxy}"
export CLOUD_ML_REGION="${CLOUD_ML_REGION:-global}"

# If the proxy requires VERTEX_PROXY_API_KEY:
# export ANTHROPIC_CUSTOM_HEADERS="x-api-key: your-proxy-key"

# Optional model pins:
# export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-5-5"
# export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-5-5"
# export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-5-5"
