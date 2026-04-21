"""vertex-proxy FastAPI app.

Exposes:
  - POST /anthropic/v1/messages                    : Anthropic-compatible, forwards to Vertex.
  - POST /gemini/v1beta/models/{m}:generateContent : Gemini-compatible, forwards to Vertex.
  - GET  /health                                   : liveness + token status.
  - GET  /v1/models                                : list routable models.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import TokenManager
from .config import Settings, load_settings

logger = logging.getLogger(__name__)


# --- app factory ------------------------------------------------------------


def build_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    token_mgr = TokenManager(
        credentials_path=cfg.credentials_path,
        refresh_seconds=cfg.token_refresh_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await token_mgr.start()
        # Resolve project ID from credentials if not explicitly configured.
        if cfg.project_id is None:
            cfg.project_id = token_mgr.project_id
        if not cfg.project_id:
            raise RuntimeError(
                "no GCP project_id: set VERTEX_PROXY_PROJECT_ID "
                "or use a service-account key that includes project_id"
            )
        logger.info("vertex-proxy ready; project=%s", cfg.project_id)
        app.state.token_mgr = token_mgr
        app.state.cfg = cfg
        app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        try:
            yield
        finally:
            await app.state.http.aclose()
            await token_mgr.stop()

    app = FastAPI(
        title="vertex-proxy",
        description="Anthropic + Gemini API-compatible proxy for Google Cloud Vertex AI",
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- health ----------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        try:
            # Try to get a token; proves auth is working.
            await token_mgr.get_token()
            return {"status": "ok", "project": cfg.project_id}
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(exc)},
            )

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "vertex_model_id": real,
                    "provider": "anthropic-vertex",
                    "region": cfg.anthropic_region,
                }
                for alias, real in cfg.anthropic_model_aliases.items()
            ]
            + [
                {
                    "id": alias,
                    "object": "model",
                    "vertex_model_id": real,
                    "provider": "gemini-vertex",
                    "region": cfg.gemini_region,
                }
                for alias, real in cfg.gemini_model_aliases.items()
            ]
            + [
                {
                    "id": alias,
                    "object": "model",
                    "vertex_model_id": path,
                    "provider": "maas-vertex",
                    "region": cfg.maas_region,
                }
                for alias, path in cfg.maas_model_aliases.items()
            ],
        }

    # --- Anthropic routes ------------------------------------------------------

    @app.post("/anthropic/v1/messages")
    async def anthropic_messages(request: Request) -> Any:
        return await _handle_anthropic(request, cfg, token_mgr)

    # Also accept /v1/messages directly (some clients won't let you override path).
    @app.post("/v1/messages")
    async def anthropic_messages_root(request: Request) -> Any:
        return await _handle_anthropic(request, cfg, token_mgr)

    # --- Gemini routes ---------------------------------------------------------
    # Gemini SDK hits /v1beta/models/{model}:generateContent and :streamGenerateContent.
    # We pass-through both.

    @app.post("/gemini/v1beta/models/{model_and_action:path}")
    async def gemini_generate(model_and_action: str, request: Request) -> Any:
        return await _handle_gemini(model_and_action, request, cfg, token_mgr)

    @app.post("/v1beta/models/{model_and_action:path}")
    async def gemini_generate_root(model_and_action: str, request: Request) -> Any:
        return await _handle_gemini(model_and_action, request, cfg, token_mgr)

    # --- OpenAI-compatible route for Vertex MaaS models ------------------------
    # Kimi K2.5, GLM 5, MiniMax-M2.5, Qwen 3.5, Grok 4.20, etc.
    # Vertex exposes these through an OpenAI Chat Completions-compatible
    # endpoint at /v1beta1/.../endpoints/openapi/chat/completions.

    @app.post("/openai/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Any:
        return await _handle_openai(request, cfg, token_mgr)

    @app.post("/v1/chat/completions")
    async def openai_chat_completions_root(request: Request) -> Any:
        return await _handle_openai(request, cfg, token_mgr)

    # Some OpenAI clients (notably Hermes's internal one) drop the /v1 prefix
    # when you set base_url to the server root. Accept that shape too.
    @app.post("/chat/completions")
    async def openai_chat_completions_bare(request: Request) -> Any:
        return await _handle_openai(request, cfg, token_mgr)

    # /v1/models/{model} — some clients probe for a specific model's existence
    # before dispatching. Return minimal metadata so they don't bail.
    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str) -> dict[str, Any]:
        if (
            model_id in cfg.anthropic_model_aliases
            or model_id in cfg.gemini_model_aliases
            or model_id in cfg.maas_model_aliases
            or model_id.startswith("google/")
        ):
            return {"id": model_id, "object": "model", "owned_by": "vertex-proxy"}
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")

    return app


# --- Anthropic handler ------------------------------------------------------


async def _handle_anthropic(request: Request, cfg: Settings, tm: TokenManager) -> Any:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc

    requested_model = (body.get("model") or "").strip()
    if not requested_model:
        raise HTTPException(status_code=400, detail="missing 'model' in request body")

    # Alias resolution.
    vertex_model = cfg.anthropic_model_aliases.get(requested_model, requested_model)
    if "@" not in vertex_model:
        # Accept a bare name only if it's an exact match; otherwise fail loud.
        raise HTTPException(
            status_code=400,
            detail=f"unknown anthropic model '{requested_model}'. "
            f"known aliases: {sorted(cfg.anthropic_model_aliases.keys())}",
        )

    # Anthropic-on-Vertex wants `anthropic_version` and removes `model`.
    upstream_body = {k: v for k, v in body.items() if k != "model"}
    upstream_body.setdefault("anthropic_version", "vertex-2023-10-16")

    streaming = bool(body.get("stream"))
    # Vertex endpoint: :streamRawPredict for streaming, :rawPredict for one-shot.
    action = "streamRawPredict" if streaming else "rawPredict"
    url = (
        f"https://{cfg.anthropic_region}-aiplatform.googleapis.com/v1/projects/"
        f"{cfg.project_id}/locations/{cfg.anthropic_region}/publishers/anthropic/"
        f"models/{vertex_model}:{action}"
    )

    token = await tm.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info(
        "anthropic: model=%s → vertex_model=%s streaming=%s",
        requested_model,
        vertex_model,
        streaming,
    )

    http: httpx.AsyncClient = request.app.state.http
    if streaming:
        return StreamingResponse(
            _stream_bytes(http, url, headers, upstream_body),
            media_type="text/event-stream",
        )

    try:
        resp = await http.post(url, headers=headers, json=upstream_body)
    except httpx.HTTPError as exc:
        logger.error("anthropic upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    return _passthrough_response(resp)


# --- Gemini handler ---------------------------------------------------------


async def _handle_gemini(
    model_and_action: str, request: Request, cfg: Settings, tm: TokenManager
) -> Any:
    # model_and_action is like "gemini-2.5-pro:generateContent" or
    # "gemini-2.5-flash:streamGenerateContent".
    if ":" not in model_and_action:
        raise HTTPException(
            status_code=400,
            detail="gemini path must include action (e.g., ':generateContent')",
        )
    requested_model, action = model_and_action.rsplit(":", 1)
    vertex_model = cfg.gemini_model_aliases.get(requested_model, requested_model)
    streaming = "stream" in action.lower()

    try:
        body = await request.json()
    except Exception:
        body = {}

    url = (
        f"https://{cfg.gemini_region}-aiplatform.googleapis.com/v1/projects/"
        f"{cfg.project_id}/locations/{cfg.gemini_region}/publishers/google/"
        f"models/{vertex_model}:{action}"
    )
    # Pass through query params (e.g., alt=sse).
    if request.url.query:
        url = f"{url}?{request.url.query}"

    token = await tm.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info(
        "gemini: model=%s action=%s streaming=%s",
        requested_model,
        action,
        streaming,
    )

    http: httpx.AsyncClient = request.app.state.http
    if streaming:
        return StreamingResponse(
            _stream_bytes(http, url, headers, body),
            media_type="text/event-stream",
        )

    try:
        resp = await http.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        logger.error("gemini upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    return _passthrough_response(resp)


# --- OpenAI-compatible (Vertex MaaS) handler -------------------------------


async def _handle_openai(request: Request, cfg: Settings, tm: TokenManager) -> Any:
    """Forward OpenAI Chat Completions requests to Vertex AI MaaS models.

    Supports Moonshot (Kimi), Zhipu (GLM), MiniMax, Alibaba (Qwen), xAI (Grok).
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc

    requested_model = (body.get("model") or "").strip()
    if not requested_model:
        raise HTTPException(status_code=400, detail="missing 'model' in request body")

    streaming = bool(body.get("stream"))
    token = await tm.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # --- routing: Gemini via Vertex OpenAI-compat, or MaaS partner model. ---
    if requested_model in cfg.gemini_model_aliases or requested_model.startswith("google/"):
        # Gemini models through Vertex's OpenAI-compat endpoint.
        # See: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library
        bare_model = requested_model.removeprefix("google/")
        vertex_model = cfg.gemini_model_aliases.get(bare_model, bare_model)
        url = (
            f"https://{cfg.gemini_region}-aiplatform.googleapis.com/v1beta1/projects/"
            f"{cfg.project_id}/locations/{cfg.gemini_region}/endpoints/openapi/chat/completions"
        )
        upstream_body = dict(body)
        upstream_body["model"] = f"google/{vertex_model}"
        logger.info(
            "openai→gemini: model=%s → %s streaming=%s",
            requested_model,
            upstream_body["model"],
            streaming,
        )
    else:
        # MaaS partner models (Kimi, GLM, MiniMax, Qwen, Grok).
        path_fragment = cfg.maas_model_aliases.get(requested_model)
        if path_fragment is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown MaaS model '{requested_model}'. "
                    f"known aliases: {sorted(cfg.maas_model_aliases.keys())} "
                    f"or gemini: {sorted(cfg.gemini_model_aliases.keys())}"
                ),
            )
        url = (
            f"https://{cfg.maas_region}-aiplatform.googleapis.com/v1beta1/projects/"
            f"{cfg.project_id}/locations/{cfg.maas_region}/{path_fragment}/chat/completions"
        )
        upstream_body = dict(body)
        upstream_body["model"] = path_fragment.rsplit("/", 1)[-1]
        logger.info(
            "openai→maas: model=%s → path=%s streaming=%s",
            requested_model,
            path_fragment,
            streaming,
        )

    http: httpx.AsyncClient = request.app.state.http
    if streaming:
        return StreamingResponse(
            _stream_bytes(http, url, headers, upstream_body),
            media_type="text/event-stream",
        )

    try:
        resp = await http.post(url, headers=headers, json=upstream_body)
    except httpx.HTTPError as exc:
        logger.error("maas upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    return _passthrough_response(resp)


# --- helpers ----------------------------------------------------------------


async def _stream_bytes(
    http: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    async with http.stream("POST", url, headers=headers, json=body) as r:
        if r.status_code >= 400:
            # Drain the error body so we can surface it.
            err_body = b""
            async for chunk in r.aiter_bytes():
                err_body += chunk
            detail = err_body.decode("utf-8", errors="replace")[:2000]
            raise HTTPException(status_code=r.status_code, detail=detail)
        async for chunk in r.aiter_bytes():
            yield chunk


def _passthrough_response(resp: httpx.Response) -> JSONResponse:
    """Forward upstream status + JSON body to the client."""
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        # Not JSON; forward as text wrapped.
        payload = {"raw": resp.text[:4000]}
    return JSONResponse(status_code=resp.status_code, content=payload)
