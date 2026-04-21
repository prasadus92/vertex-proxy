#!/usr/bin/env bash
# Point the Claude Code CLI at vertex-proxy so it routes through Vertex AI.
# Source this file (or paste into your shell profile) before running `claude`.

# The proxy's Anthropic route
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787/anthropic"

# Claude Code wants a bearer token. vertex-proxy ignores it (Vertex auth is
# server-side via service account), but the client needs something non-empty.
export ANTHROPIC_AUTH_TOKEN="bypass"

# Optional: pick a specific model for this session
# export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"

echo "Claude Code now routes to vertex-proxy: $ANTHROPIC_BASE_URL"
