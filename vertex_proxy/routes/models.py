"""Model discovery endpoints, mirrored wherever clients look for them."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from ..catalog import Provider
from ..errors import Dialect, ErrorKind, ProxyError
from .common import Services, dialect_router, get_services

LIST_PATHS = (
    "/v1/models",
    "/models",
    "/openai/v1/models",
    "/openai/models",
    "/gemini/models",
    "/api/v1/models",
    "/openai/api/v1/models",
    "/gemini/api/v1/models",
)
PROBE_PATHS = (
    "/v1/models/{model_id:path}",
    "/openai/v1/models/{model_id:path}",
    "/gemini/v1/models/{model_id:path}",
)

router = dialect_router(Dialect.OPENAI)


def _listing(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"object": "list", "data": entries, "has_more": False}


async def list_models(services: Services = Depends(get_services)) -> dict[str, Any]:
    return _listing(services.catalog.entries())


async def probe_model(model_id: str, services: Services = Depends(get_services)) -> dict[str, Any]:
    if not services.catalog.knows(model_id):
        raise ProxyError(404, ErrorKind.NOT_FOUND, f"model '{model_id}' not found")
    return {"id": model_id, "object": "model", "owned_by": "vertex-proxy"}


@router.get("/anthropic/v1/models")
async def list_anthropic_models(services: Services = Depends(get_services)) -> dict[str, Any]:
    return _listing(services.catalog.entries(Provider.ANTHROPIC))


@router.get("/gemini/v1/models")
@router.get("/gemini/v1beta/models")
async def list_gemini_models(services: Services = Depends(get_services)) -> dict[str, Any]:
    return _listing(services.catalog.entries(Provider.GEMINI))


for _path in LIST_PATHS:
    router.add_api_route(_path, list_models, methods=["GET"])
for _path in PROBE_PATHS:
    router.add_api_route(_path, probe_model, methods=["GET"])
