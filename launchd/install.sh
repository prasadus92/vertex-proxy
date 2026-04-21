#!/usr/bin/env bash
# Install vertex-proxy as a launchd service on macOS.
#
# Usage:
#   ./install.sh --credentials /path/to/gcp-key.json --project my-gcp-project
#
# After install, the service runs as a LaunchAgent under your user account.
# Listens on http://127.0.0.1:8787. Auto-restarts on crash.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_NAME="ai.hermes.vertex-proxy.plist"
PLIST_TEMPLATE="$SCRIPT_DIR/${PLIST_NAME}.template"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

CREDENTIALS_PATH=""
PROJECT_ID=""

usage() {
    cat <<EOF
usage: ./install.sh --credentials PATH --project PROJECT_ID

Options:
  --credentials PATH   Absolute path to GCP service-account JSON
  --project PROJECT    GCP project ID
  -h, --help           Show this help

Requires:
  - vertex-proxy installed under \$REPO_ROOT/.venv
    (run: python -m venv .venv && .venv/bin/pip install -e .)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --credentials) CREDENTIALS_PATH="$2"; shift 2;;
        --project)     PROJECT_ID="$2"; shift 2;;
        -h|--help)     usage; exit 0;;
        *) echo "unknown arg: $1" >&2; usage; exit 2;;
    esac
done

if [[ -z "$CREDENTIALS_PATH" || -z "$PROJECT_ID" ]]; then
    echo "ERROR: --credentials and --project are required" >&2
    usage
    exit 2
fi

if [[ ! -f "$CREDENTIALS_PATH" ]]; then
    echo "ERROR: credentials file not found: $CREDENTIALS_PATH" >&2
    exit 1
fi

if [[ ! -x "$REPO_ROOT/.venv/bin/vertex-proxy" ]]; then
    echo "ERROR: vertex-proxy not installed under $REPO_ROOT/.venv" >&2
    echo "run: python -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
fi

if [[ ! -f "$PLIST_TEMPLATE" ]]; then
    echo "ERROR: plist template missing: $PLIST_TEMPLATE" >&2
    exit 1
fi

# Render template
rendered="$(mktemp)"
trap 'rm -f "$rendered"' EXIT
sed \
    -e "s|{{INSTALL_DIR}}|$REPO_ROOT|g" \
    -e "s|{{CREDENTIALS_PATH}}|$CREDENTIALS_PATH|g" \
    -e "s|{{PROJECT_ID}}|$PROJECT_ID|g" \
    -e "s|{{HOME}}|$HOME|g" \
    "$PLIST_TEMPLATE" > "$rendered"

mkdir -p "$HOME/Library/LaunchAgents"
cp "$rendered" "$PLIST_DEST"

# Idempotent reload
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

sleep 2
if curl -fsS http://127.0.0.1:8787/health >/dev/null 2>&1; then
    echo "vertex-proxy installed and running"
    echo "  plist: $PLIST_DEST"
    echo "  logs:  $HOME/Library/Logs/vertex-proxy.log"
    echo "  stop:  launchctl unload $PLIST_DEST"
else
    echo "WARNING: vertex-proxy loaded but health check failed"
    echo "  check: $HOME/Library/Logs/vertex-proxy.err"
    exit 1
fi
