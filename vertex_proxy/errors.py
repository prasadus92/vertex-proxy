"""Errors returned to clients in the shape of the API they called, with hints for common failures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

from fastapi.responses import JSONResponse, Response

# Upstream headers a client uses to decide whether and when to retry.
RETRY_HEADERS = ("retry-after", "x-should-retry")


class Dialect(StrEnum):
    """The API shape a route speaks. RAW forwards upstream error bodies untouched."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    RAW = "raw"


class ErrorKind(Enum):
    INVALID_REQUEST = ("invalid_request_error", "INVALID_ARGUMENT")
    AUTHENTICATION = ("authentication_error", "UNAUTHENTICATED")
    PERMISSION = ("permission_error", "PERMISSION_DENIED")
    NOT_FOUND = ("not_found_error", "NOT_FOUND")
    RATE_LIMIT = ("rate_limit_error", "RESOURCE_EXHAUSTED")
    OVERLOADED = ("overloaded_error", "UNAVAILABLE")
    UPSTREAM = ("api_error", "INTERNAL")

    @property
    def api_type(self) -> str:
        return self.value[0]

    @property
    def google_status(self) -> str:
        return self.value[1]

    @classmethod
    def for_api_type(cls, api_type: str | None) -> ErrorKind:
        return next((kind for kind in cls if kind.api_type == api_type), cls.UPSTREAM)

    @classmethod
    def for_status(cls, status: int) -> ErrorKind:
        return _KIND_BY_STATUS.get(status) or (
            cls.UPSTREAM if status >= 500 else cls.INVALID_REQUEST
        )


_KIND_BY_STATUS = {
    400: ErrorKind.INVALID_REQUEST,
    401: ErrorKind.AUTHENTICATION,
    403: ErrorKind.PERMISSION,
    404: ErrorKind.NOT_FOUND,
    429: ErrorKind.RATE_LIMIT,
    503: ErrorKind.OVERLOADED,
    529: ErrorKind.OVERLOADED,
}


@dataclass(frozen=True)
class UpstreamContext:
    """What the proxy was calling, used to write hints."""

    project_id: str | None = None
    location: str | None = None
    model: str | None = None


@dataclass(eq=False)
class ProxyError(Exception):
    status: int
    kind: ErrorKind
    message: str
    headers: dict[str, str] = field(default_factory=dict)
    raw_body: bytes | None = None
    raw_content_type: str | None = None

    def __str__(self) -> str:
        return f"{self.status} {self.kind.api_type}: {self.message}"

    @classmethod
    def bad_request(cls, message: str) -> ProxyError:
        return cls(400, ErrorKind.INVALID_REQUEST, message)

    @classmethod
    def from_upstream(
        cls,
        status: int,
        body: bytes,
        headers: dict[str, str],
        context: UpstreamContext,
    ) -> ProxyError:
        kind = ErrorKind.for_status(status)
        message = _upstream_message(body) or f"upstream returned HTTP {status}"
        hint = _hint(kind, message, context)
        return cls(
            status=status,
            kind=kind,
            message=f"{message} {hint}" if hint else message,
            headers={k: v for k, v in headers.items() if k.lower() in RETRY_HEADERS},
            raw_body=body,
            raw_content_type=headers.get("content-type"),
        )

    def payload(self, dialect: Dialect) -> dict[str, Any]:
        if dialect is Dialect.ANTHROPIC:
            return {"type": "error", "error": {"type": self.kind.api_type, "message": self.message}}
        if dialect in (Dialect.GOOGLE, Dialect.RAW):
            return {
                "error": {
                    "code": self.status,
                    "message": self.message,
                    "status": self.kind.google_status,
                }
            }
        return {
            "error": {
                "message": self.message,
                "type": self.kind.api_type,
                "code": self.kind.google_status.lower(),
            }
        }

    def to_response(self, dialect: Dialect) -> Response:
        if dialect is Dialect.RAW and self.raw_body is not None:
            return Response(
                self.raw_body,
                status_code=self.status,
                headers=self.headers,
                media_type=self.raw_content_type or "application/json",
            )
        return JSONResponse(self.payload(dialect), status_code=self.status, headers=self.headers)

    def to_sse(self, dialect: Dialect) -> bytes:
        data = json.dumps(self.payload(dialect))
        if dialect is Dialect.ANTHROPIC:
            return f"event: error\ndata: {data}\n\n".encode()
        return f"data: {data}\n\n".encode()


def _upstream_message(body: bytes) -> str | None:
    """The human-readable message from a Google or Anthropic error body."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        text = body.decode("utf-8", errors="replace").strip()
        return text[:2000] or None
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        message = data["error"].get("message")
        return str(message) if message else None
    return None


def _hint(kind: ErrorKind, message: str, ctx: UpstreamContext) -> str | None:
    project = ctx.project_id or "YOUR_PROJECT"
    model = f"'{ctx.model}'" if ctx.model else "this model"
    location = f"'{ctx.location}'" if ctx.location else "the configured location"
    if kind is ErrorKind.AUTHENTICATION:
        return (
            "vertex-proxy: Vertex AI rejected the access token. Check that the service-account "
            "key is valid and not disabled."
        )
    if kind is ErrorKind.PERMISSION:
        lowered = message.lower()
        if (
            "has not been used" in lowered
            or "service_disabled" in lowered
            or "is disabled" in lowered
        ):
            return (
                "vertex-proxy: enable the Vertex AI API with "
                f"`gcloud services enable aiplatform.googleapis.com --project {project}`."
            )
        return (
            "vertex-proxy: grant the credentials roles/aiplatform.user on project "
            f"{project}, and enable {model} in Model Garden."
        )
    if kind is ErrorKind.NOT_FOUND:
        return (
            f"vertex-proxy: {model} is not available in location {location} for project "
            f"{project}. Enable it in Model Garden, or use the 'global' location."
        )
    if kind is ErrorKind.RATE_LIMIT:
        return (
            f"vertex-proxy: Vertex AI quota is exhausted for {model} in {location}. Retry "
            "later, request a quota increase, or use the 'global' location."
        )
    return None
