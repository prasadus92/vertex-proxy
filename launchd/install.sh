#!/usr/bin/env bash
# Install vertex-proxy as a macOS LaunchAgent that starts at login and restarts on crash.
#
# Usage:
#   ./install.sh --credentials /path/to/key.json --project my-gcp-project [--bin /path/to/vertex-proxy]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="io.github.prasadus92.vertex-proxy"
PLIST_TEMPLATE="$SCRIPT_DIR/vertex-proxy.plist.template"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
# Releases up to 0.4.0 installed the agent under this label.
LEGACY_PLIST="$HOME/Library/LaunchAgents/ai.hermes.vertex-proxy.plist"

CREDENTIALS_PATH=""
PROJECT_ID=""
BIN="$(command -v vertex-proxy || true)"

usage() {
    cat <<EOF
usage: ./install.sh --credentials PATH --project PROJECT_ID [--bin PATH]

  --credentials PATH   Google Cloud credentials JSON file
  --project PROJECT    GCP project ID
  --bin PATH           vertex-proxy executable (default: the one on PATH)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --credentials) CREDENTIALS_PATH="$2"; shift 2;;
        --project)     PROJECT_ID="$2"; shift 2;;
        --bin)         BIN="$2"; shift 2;;
        -h|--help)     usage; exit 0;;
        *) echo "unknown argument: $1" >&2; usage; exit 2;;
    esac
done

if [[ -z "$CREDENTIALS_PATH" || -z "$PROJECT_ID" ]]; then
    echo "error: --credentials and --project are required" >&2
    usage
    exit 2
fi
if [[ ! -f "$CREDENTIALS_PATH" ]]; then
    echo "error: credentials file not found: $CREDENTIALS_PATH" >&2
    exit 1
fi
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
    echo "error: vertex-proxy not found; install it (uv tool install vertex-proxy) or pass --bin" >&2
    exit 1
fi

if [[ -f "$LEGACY_PLIST" ]]; then
    launchctl unload "$LEGACY_PLIST" 2>/dev/null || true
    rm -f "$LEGACY_PLIST"
fi

mkdir -p "$HOME/Library/LaunchAgents"
sed \
    -e "s|{{BIN}}|$BIN|g" \
    -e "s|{{CREDENTIALS_PATH}}|$CREDENTIALS_PATH|g" \
    -e "s|{{PROJECT_ID}}|$PROJECT_ID|g" \
    -e "s|{{HOME}}|$HOME|g" \
    "$PLIST_TEMPLATE" > "$PLIST_DEST"

launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

sleep 2
if curl -fsS http://127.0.0.1:8787/health >/dev/null; then
    echo "vertex-proxy is running"
    echo "  plist: $PLIST_DEST"
    echo "  logs:  $HOME/Library/Logs/vertex-proxy.log"
    echo "  stop:  launchctl unload $PLIST_DEST"
else
    echo "warning: vertex-proxy was loaded but /health failed; see $HOME/Library/Logs/vertex-proxy.err" >&2
    exit 1
fi
