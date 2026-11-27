"""Application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import Response

from . import __version__
from .auth import TokenManager, TokenSource
from .catalog import ModelCatalog
from .config import ConfigError, Settings, load_settings
from .errors import Dialect, ProxyError
from .metrics import Metrics
from .routes import ROUTERS
from .services import Services
from .upstream import DEFAULT_TIMEOUT, Upstream
from .vertex import VertexEndpoints

logger = logging.getLogger(__name__)


def build_app(
    settings: Settings | None = None,
    *,
    tokens: TokenSource | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create the app. ``tokens`` and ``transport`` replace Google auth and the network in tests."""
    settings = settings or load_settings()
    if tokens is None:
        tokens = TokenManager.from_file(settings.credentials_path, settings.token_refresh_seconds)
    project_id = settings.project_id or tokens.project_id
    if not project_id:
        raise ConfigError(
            "no Google Cloud project: set VERTEX_PROXY_PROJECT_ID (or project_id in the config "
            "file), or use credentials that name a project"
        )
    catalog = ModelCatalog(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await tokens.start()
        try:
            async with httpx2.AsyncClient(timeout=DEFAULT_TIMEOUT, transport=transport) as http:
                await catalog.discover_ollama(http)
                await catalog.discover_custom_providers(http)
                app.state.services = Services(
                    settings=settings,
                    catalog=catalog,
                    endpoints=VertexEndpoints(project_id),
                    upstream=Upstream(http, tokens),
                    metrics=Metrics(),
                )
                logger.info("vertex-proxy %s ready; project=%s", __version__, project_id)
                yield
        finally:
            await tokens.stop()

    app = FastAPI(
        title="vertex-proxy",
        description="Anthropic, Gemini and OpenAI API-compatible proxy for Google Cloud Vertex AI",
        version=__version__,
        lifespan=lifespan,
        openapi_url=None,
    )

    @app.exception_handler(ProxyError)
    async def proxy_error(request: Request, exc: ProxyError) -> Response:
        if exc.status >= 500 or exc.status in (401, 403, 429):
            logger.warning("%s %s -> %s", request.method, request.url.path, exc)
        return exc.to_response(getattr(request.state, "dialect", Dialect.OPENAI))

    for router in ROUTERS:
        app.include_router(router)
    return app
