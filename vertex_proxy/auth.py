"""GCP service-account auth with background token refresh.

Vertex AI uses short-lived OAuth access tokens (60-minute TTL) derived from a
service-account JSON key. This module handles the refresh loop so callers
always see a valid token.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# Vertex AI only needs cloud-platform scope.
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class TokenManager:
    """Holds a refreshing GCP access token.

    Call ``start()`` once at app startup. Access the current token via
    ``token`` or ``await get_token()``. Call ``stop()`` at shutdown.
    """

    def __init__(
        self,
        credentials_path: Path | None,
        refresh_seconds: int = 3000,
    ) -> None:
        self._credentials_path = credentials_path
        self._refresh_seconds = refresh_seconds
        self._credentials: service_account.Credentials | None = None
        self._project_id: str | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._last_refresh: float = 0.0

    @property
    def project_id(self) -> str | None:
        """GCP project ID extracted from the service-account key."""
        return self._project_id

    @property
    def token(self) -> str:
        """Current access token. Raises if uninitialised."""
        if self._credentials is None or self._credentials.token is None:
            raise RuntimeError("TokenManager not initialised; call start() first")
        return self._credentials.token

    async def start(self) -> None:
        """Load credentials and kick off the background refresh loop."""
        if self._credentials is not None:
            return

        if self._credentials_path is None:
            # Fall back to Application Default Credentials
            import google.auth

            creds, project = google.auth.default(scopes=SCOPES)
            self._credentials = creds  # type: ignore[assignment]
            self._project_id = project
            logger.info("loaded Application Default Credentials; project=%s", project)
        else:
            path = Path(self._credentials_path).expanduser()
            self._credentials = service_account.Credentials.from_service_account_file(
                str(path), scopes=SCOPES
            )
            self._project_id = self._credentials.project_id
            logger.info(
                "loaded service-account key from %s; project=%s",
                path,
                self._project_id,
            )

        # Initial refresh; blocks until we have a token.
        await self._do_refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop(), name="token-refresh")

    async def stop(self) -> None:
        """Signal the refresh loop to stop and await it."""
        self._stop_event.set()
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):
                pass

    async def get_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if self._credentials is None:
            raise RuntimeError("TokenManager not initialised")
        # If the token is close to expiry, force a refresh now.
        if self._credentials.expired or self._credentials.token is None:
            await self._do_refresh()
        return self._credentials.token or ""

    # --- internal ----------------------------------------------------------

    async def _do_refresh(self) -> None:
        """Run the blocking google-auth refresh in a worker thread."""
        if self._credentials is None:
            raise RuntimeError("no credentials loaded")

        def _sync_refresh() -> None:
            request = GoogleAuthRequest()
            assert self._credentials is not None
            self._credentials.refresh(request)

        await asyncio.get_running_loop().run_in_executor(None, _sync_refresh)
        self._last_refresh = time.time()
        expiry = getattr(self._credentials, "expiry", None)
        logger.info("refreshed access token; expires=%s", expiry)

    async def _refresh_loop(self) -> None:
        """Background task: refresh the token every N seconds until stopped."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._refresh_seconds)
                # If we got here without timeout, stop was requested.
                return
            except TimeoutError:
                # Normal path: time to refresh.
                try:
                    await self._do_refresh()
                except Exception as exc:  # noqa: BLE001
                    logger.error("token refresh failed: %s", exc, exc_info=True)
                    # Don't crash the loop; try again next interval.
