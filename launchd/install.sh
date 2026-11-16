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

absolute() {
    (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")")
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --credentials|--project|--bin)
            if [[ $# -lt 2 ]]; then
                echo "error: $1 needs a value" >&2
                exit 2
            fi
            case "$1" in
                --credentials) CREDENTIALS_PATH="$2";;
                --project)     PROJECT_ID="$2";;
                --bin)         BIN="$2";;
            esac
            shift 2;;
        -h|--help) usage; exit 0;;
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
CREDENTIALS_PATH="$(absolute "$CREDENTIALS_PATH")"
BIN="$(absolute "$BIN")"

if [[ -f "$LEGACY_PLIST" ]]; then
    launchctl unload "$LEGACY_PLIST" 2>/dev/null || true
    rm -f "$LEGACY_PLIST"
fi

mkdir -p "$HOME/Library/LaunchAgents"
launchctl unload "$PLIST_DEST" 2>/dev/null || true
cp "$PLIST_TEMPLATE" "$PLIST_DEST"
plutil -replace Label -string "$LABEL" "$PLIST_DEST"
plutil -replace ProgramArguments -array "$PLIST_DEST"
plutil -insert ProgramArguments -string "$BIN" -append "$PLIST_DEST"
plutil -replace EnvironmentVariables.VERTEX_PROXY_CREDENTIALS_PATH -string "$CREDENTIALS_PATH" "$PLIST_DEST"
plutil -replace EnvironmentVariables.VERTEX_PROXY_PROJECT_ID -string "$PROJECT_ID" "$PLIST_DEST"
plutil -replace StandardOutPath -string "$HOME/Library/Logs/vertex-proxy.log" "$PLIST_DEST"
plutil -replace StandardErrorPath -string "$HOME/Library/Logs/vertex-proxy.err" "$PLIST_DEST"
plutil -lint -s "$PLIST_DEST"
if grep -q '{{' "$PLIST_DEST"; then
    echo "error: $PLIST_DEST still has template placeholders" >&2
    exit 1
fi
launchctl load "$PLIST_DEST"

for _ in $(seq 1 15); do
    if curl -fsS http://127.0.0.1:8787/health >/dev/null 2>&1; then
        echo "vertex-proxy is running"
        echo "  plist: $PLIST_DEST"
        echo "  logs:  $HOME/Library/Logs/vertex-proxy.log"
        echo "  stop:  launchctl unload $PLIST_DEST"
        exit 0
    fi
    sleep 1
done
echo "warning: vertex-proxy was loaded but /health did not answer; see $HOME/Library/Logs/vertex-proxy.err" >&2
exit 1
