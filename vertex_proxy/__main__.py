"""Command line entry point: ``vertex-proxy`` or ``python -m vertex_proxy``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from . import __version__
from .app import build_app
from .auth import CredentialsError, TokenManager
from .config import ConfigError, default_config_path, load_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vertex-proxy",
        description="Anthropic, Gemini and OpenAI API-compatible proxy for Google Cloud Vertex AI",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=f"TOML config file (default: $VERTEX_PROXY_CONFIG or {default_config_path()})",
    )
    parser.add_argument("--host", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="bind port (default: 8787)")
    parser.add_argument("--credentials", type=Path, help="Google Cloud credentials JSON file")
    parser.add_argument("--project-id", help="GCP project (default: from the credentials)")
    parser.add_argument("--log-level", help="debug, info, warning or error (default: info)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(
            args.config,
            host=args.host,
            port=args.port,
            credentials_path=args.credentials,
            project_id=args.project_id,
            log_level=args.log_level,
        )
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        tokens = TokenManager.from_file(settings.credentials_path, settings.token_refresh_seconds)
        tokens.verify()
        app = build_app(settings, tokens=tokens)
    except (ConfigError, CredentialsError) as exc:
        print(f"vertex-proxy: {exc}", file=sys.stderr)
        return 2
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
