"""A fake Vertex AI upstream and a test client wired to it; nothing here reaches Google."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from vertex_proxy.app import build_app
from vertex_proxy.auth import CredentialsError
from vertex_proxy.config import Settings

Handler = Callable[[httpx2.Request], httpx2.Response]


class FakeTokens:
    project_id = "test-project"

    def __init__(self) -> None:
        self.error: CredentialsError | None = None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def get_token(self) -> str:
        if self.error is not None:
            raise self.error
        return "fake-token"


def _default_reply(request: httpx2.Request) -> httpx2.Response:
    url = str(request.url)
    if "publishers/anthropic" in url:
        payload: Any = {
            "id": "msg_1",
            "type": "message",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    elif "publishers/google" in url:
        payload = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    elif url.endswith("/api/tags"):
        payload = {"models": []}
    else:
        payload = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    return httpx2.Response(200, json=payload)


@dataclass
class FakeVertex:
    """Records every upstream request and answers with ``reply``."""

    requests: list[httpx2.Request] = field(default_factory=list)
    reply: Handler = _default_reply

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.reply(request)

    @property
    def last(self) -> httpx2.Request:
        return self.requests[-1]

    @property
    def last_json(self) -> Any:
        return json.loads(self.last.content)

    def respond(self, status: int, payload: Any = None, **kwargs: Any) -> None:
        self.reply = lambda _request: httpx2.Response(status, json=payload, **kwargs)

    def respond_sse(self, events: list[str]) -> None:
        body = "".join(f"{event}\n\n" for event in events).encode()
        self.reply = lambda _request: httpx2.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )


@pytest.fixture
def vertex() -> FakeVertex:
    return FakeVertex()


@pytest.fixture
def tokens() -> FakeTokens:
    return FakeTokens()


@pytest.fixture
def make_client(vertex: FakeVertex, tokens: FakeTokens) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(**settings: Any) -> TestClient:
        cfg = Settings(_env_file=None, **{"project_id": "test-project", **settings})
        app = build_app(cfg, tokens=tokens, transport=httpx2.MockTransport(vertex.handle))
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Keep the developer's own config file and VERTEX_PROXY_* variables out of tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in list(os.environ):
        if name.startswith("VERTEX_PROXY_"):
            monkeypatch.delenv(name)
