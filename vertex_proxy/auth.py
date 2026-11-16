"""Google Cloud credentials and a background-refreshed access token."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import google.auth
from google.auth import (
    aws,
    external_account_authorized_user,
    identity_pool,
    impersonated_credentials,
    pluggable,
)
from google.auth import exceptions as google_exceptions
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import credentials as user_credentials
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

NO_CREDENTIALS_HINT = (
    "Set VERTEX_PROXY_CREDENTIALS_PATH (or --credentials) to a service-account key, "
    "or run `gcloud auth application-default login`."
)


class CredentialsError(Exception):
    """Credentials are missing, unreadable or rejected by Google."""


class TokenSource(Protocol):
    @property
    def project_id(self) -> str | None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def get_token(self) -> str: ...


Loaded = tuple[Credentials, str | None]


def _service_account(info: dict[str, Any]) -> Loaded:
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return creds, creds.project_id


def _authorized_user(info: dict[str, Any]) -> Loaded:
    return user_credentials.Credentials.from_authorized_user_info(info, scopes=SCOPES), None


def _external_account(info: dict[str, Any]) -> Loaded:
    source = info.get("credential_source") or {}
    if str(source.get("environment_id", "")).startswith("aws"):
        return aws.Credentials.from_info(info, scopes=SCOPES), None
    if "executable" in source:
        return pluggable.Credentials.from_info(info, scopes=SCOPES), None
    return identity_pool.Credentials.from_info(info, scopes=SCOPES), None


def _external_account_authorized_user(info: dict[str, Any]) -> Loaded:
    return external_account_authorized_user.Credentials.from_info(info), None


def _impersonated_service_account(info: dict[str, Any]) -> Loaded:
    creds = impersonated_credentials.Credentials.from_impersonated_service_account_info(
        info, scopes=SCOPES
    )
    return creds, None


# google-auth deprecates its generic file loader, so each supported type is loaded explicitly.
_LOADERS: dict[str, Callable[[dict[str, Any]], Loaded]] = {
    "service_account": _service_account,
    "authorized_user": _authorized_user,
    "external_account": _external_account,
    "external_account_authorized_user": _external_account_authorized_user,
    "impersonated_service_account": _impersonated_service_account,
}


def _load_file(path: Path) -> Loaded:
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CredentialsError(f"cannot read credentials file {path}: {exc}") from exc
    kind = info.get("type") if isinstance(info, dict) else None
    loader = _LOADERS.get(kind) if isinstance(kind, str) else None
    if loader is None:
        raise CredentialsError(
            f"{path} has credential type {kind!r}; supported types are {', '.join(_LOADERS)}"
        )
    return loader(info)


def load_credentials(path: Path | None) -> Loaded:
    """Load a credentials file, or Application Default Credentials when ``path`` is None.

    Makes no network call.
    """
    try:
        if path is None:
            return google.auth.default(scopes=SCOPES)
        if not path.is_file():
            raise CredentialsError(f"credentials file not found: {path}")
        return _load_file(path)
    except (google_exceptions.DefaultCredentialsError, ValueError) as exc:
        if path is None:
            raise CredentialsError(
                f"no Google Cloud credentials found. {NO_CREDENTIALS_HINT}"
            ) from exc
        raise CredentialsError(f"cannot use credentials file {path}: {exc}") from exc


class TokenManager:
    """Keeps an access token fresh in the background; ``get_token`` refreshes on demand."""

    def __init__(
        self,
        credentials: Credentials,
        project_id: str | None,
        refresh_seconds: int = 3000,
    ) -> None:
        self._credentials = credentials
        self._project_id = project_id
        self._refresh_seconds = refresh_seconds
        self._refresh_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._refresh_lock = asyncio.Lock()

    @classmethod
    def from_file(cls, path: Path | None, refresh_seconds: int = 3000) -> TokenManager:
        credentials, project_id = load_credentials(path)
        return cls(credentials, project_id, refresh_seconds)

    @property
    def project_id(self) -> str | None:
        return self._project_id

    def verify(self) -> None:
        """Fetch a token now, blocking, so bad credentials fail before the server starts."""
        self._refresh_sync()

    async def start(self) -> None:
        if not self._credentials.valid:
            await self._refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop(), name="token-refresh")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

    async def get_token(self) -> str:
        if not self._credentials.valid:
            async with self._refresh_lock:
                if not self._credentials.valid:
                    await self._refresh()
        return self._credentials.token or ""

    async def _refresh(self) -> None:
        await asyncio.to_thread(self._refresh_sync)

    def _refresh_sync(self) -> None:
        try:
            self._credentials.refresh(GoogleAuthRequest())
        except google_exceptions.RefreshError as exc:
            raise CredentialsError(
                f"Google rejected the credentials: {exc}. Check that the key is active and not "
                "deleted, and that this machine's clock is correct."
            ) from exc
        except google_exceptions.TransportError as exc:
            raise CredentialsError(f"could not reach Google to refresh the token: {exc}") from exc
        logger.info("refreshed access token; expires=%s", self._credentials.expiry)

    async def _refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._refresh_seconds)
                return
            except TimeoutError:
                try:
                    await self._refresh()
                except CredentialsError as exc:
                    logger.error("token refresh failed, will retry: %s", exc)
