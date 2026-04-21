"""Entry point: `python -m vertex_proxy` or `vertex-proxy` CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from .config import load_settings
from .main import build_app


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vertex-proxy",
        description="Anthropic + Gemini proxy for Google Cloud Vertex AI",
    )
    parser.add_argument("--host", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="bind port (default: 8787)")
    parser.add_argument("--credentials", help="path to GCP service-account JSON")
    parser.add_argument("--project-id", help="GCP project ID (inferred from creds if unset)")
    parser.add_argument(
        "--log-level",
        default=None,
        help="uvicorn log level (default: info)",
    )
    args = parser.parse_args()

    # Merge CLI into environment so Settings picks them up.
    if args.host:
        _set_env("VERTEX_PROXY_HOST", args.host)
    if args.port:
        _set_env("VERTEX_PROXY_PORT", str(args.port))
    if args.credentials:
        _set_env("VERTEX_PROXY_CREDENTIALS_PATH", str(Path(args.credentials).expanduser()))
    if args.project_id:
        _set_env("VERTEX_PROXY_PROJECT_ID", args.project_id)
    if args.log_level:
        _set_env("VERTEX_PROXY_LOG_LEVEL", args.log_level)

    cfg = load_settings()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = build_app(cfg)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
        access_log=True,
    )
    return 0


def _set_env(key: str, val: str) -> None:
    import os

    os.environ[key] = val


if __name__ == "__main__":
    sys.exit(main())
