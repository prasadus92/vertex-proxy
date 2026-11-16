"""Route behavior against a fake Vertex AI upstream."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeTokens, FakeVertex
from vertex_proxy.auth import CredentialsError

GLOBAL = "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global"


def _messages(model: str = "claude-opus-5-5", **extra: object) -> dict[str, object]:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


# --- Anthropic --------------------------------------------------------------


@pytest.mark.parametrize("path", ["/anthropic/v1/messages", "/v1/messages"])
def test_anthropic_messages_forward_to_vertex(
    client: TestClient, vertex: FakeVertex, path: str
) -> None:
    r = client.post(path, json=_messages(context_management={"edits": []}))
    assert r.status_code == 200
    assert r.json()["content"][0]["text"] == "ok"
    assert (
        str(vertex.last.url) == f"{GLOBAL}/publishers/anthropic/models/claude-opus-5-5:rawPredict"
    )
    assert vertex.last.headers["authorization"] == "Bearer fake-token"
    body = vertex.last_json
    assert "model" not in body and "context_management" not in body
    assert body["anthropic_version"] == "vertex-2023-10-16"


def test_dated_model_name_becomes_vertex_snapshot(client: TestClient, vertex: FakeVertex) -> None:
    client.post("/v1/messages", json=_messages("claude-sonnet-4-5-20250929"))
    assert str(vertex.last.url).endswith("/models/claude-sonnet-4-5@20250929:rawPredict")


def test_regional_location_uses_regional_host(
    make_client: Callable[..., TestClient], vertex: FakeVertex
) -> None:
    make_client(anthropic_region="us-east5").post("/v1/messages", json=_messages())
    assert vertex.last.url.host == "us-east5-aiplatform.googleapis.com"


def test_non_claude_model_is_rejected_in_anthropic_shape(
    client: TestClient, vertex: FakeVertex
) -> None:
    r = client.post("/anthropic/v1/messages", json=_messages("gpt-5"))
    assert r.status_code == 400
    assert r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert vertex.requests == []


def test_invalid_json_body_is_a_400(client: TestClient) -> None:
    r = client.post(
        "/v1/messages", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "request body must be JSON"


def test_anthropic_streaming_is_relayed_unchanged(client: TestClient, vertex: FakeVertex) -> None:
    events = [
        'event: message_start\ndata: {"type":"message_start"}',
        'event: message_stop\ndata: {"type":"message_stop"}',
    ]
    vertex.respond_sse(events)
    r = client.post("/v1/messages", json=_messages(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text == "".join(f"{e}\n\n" for e in events)
    assert str(vertex.last.url).endswith(":streamRawPredict")


def test_streaming_upstream_error_is_a_real_http_error(
    client: TestClient, vertex: FakeVertex
) -> None:
    vertex.respond(
        429,
        {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
        headers={"retry-after": "12"},
    )
    r = client.post("/v1/messages", json=_messages(stream=True))
    assert r.status_code == 429
    assert r.headers["retry-after"] == "12"
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert "quota is exhausted" in r.json()["error"]["message"]


def test_stream_interrupted_mid_flight_ends_with_an_sse_error(
    client: TestClient, vertex: FakeVertex
) -> None:
    class Broken(httpx2.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'event: message_start\ndata: {"type":"message_start"}\n\n'
            raise httpx2.ReadError("connection reset")

    vertex.reply = lambda _r: httpx2.Response(
        200, stream=Broken(), headers={"content-type": "text/event-stream"}
    )
    r = client.post("/v1/messages", json=_messages(stream=True))
    assert r.text.startswith("event: message_start")
    last = r.text.strip().split("\n\n")[-1]
    assert last.startswith("event: error\ndata: ")
    assert json.loads(last.split("data: ", 1)[1])["error"]["type"] == "api_error"


def test_count_tokens_uses_the_vertex_endpoint(client: TestClient, vertex: FakeVertex) -> None:
    vertex.respond(200, {"input_tokens": 9})
    r = client.post(
        "/v1/messages/count_tokens", json=_messages("claude-haiku-4-5", context_management={})
    )
    assert r.json() == {"input_tokens": 9}
    assert str(vertex.last.url) == f"{GLOBAL}/publishers/anthropic/models/count-tokens:rawPredict"
    assert vertex.last_json["model"] == "claude-haiku-4-5@20251001"
    assert "context_management" not in vertex.last_json


def test_upstream_unreachable_is_a_502(client: TestClient, vertex: FakeVertex) -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    vertex.reply = down
    r = client.post("/v1/messages", json=_messages())
    assert r.status_code == 502
    assert "could not reach aiplatform.googleapis.com" in r.json()["error"]["message"]


def test_rejected_credentials_are_explained(
    client: TestClient, tokens: FakeTokens, vertex: FakeVertex
) -> None:
    tokens.error = CredentialsError("Google rejected the credentials: invalid_grant")
    r = client.post("/v1/messages", json=_messages())
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "authentication_error"
    assert "invalid_grant" in r.json()["error"]["message"]
    assert vertex.requests == []


# --- Gemini -----------------------------------------------------------------


def test_gemini_generate_content(client: TestClient, vertex: FakeVertex) -> None:
    r = client.post(
        "/gemini/v1beta/models/gemini-3.5-flash:generateContent?key=client-key",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    assert r.status_code == 200
    assert (
        str(vertex.last.url)
        == f"{GLOBAL}/publishers/google/models/gemini-3.5-flash:generateContent"
    )


def test_gemini_streaming_keeps_alt_sse(client: TestClient, vertex: FakeVertex) -> None:
    vertex.respond_sse(['data: {"candidates":[]}'])
    r = client.post("/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse", json={})
    assert r.status_code == 200
    assert vertex.last.url.query == b"alt=sse"


def test_gemini_errors_use_the_google_shape(client: TestClient, vertex: FakeVertex) -> None:
    vertex.respond(
        403, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}
    )
    r = client.post("/v1beta/models/gemini-3.5-flash:generateContent", json={})
    assert r.status_code == 403
    assert r.json()["error"]["status"] == "PERMISSION_DENIED"
    assert "roles/aiplatform.user" in r.json()["error"]["message"]


@pytest.mark.parametrize("target", ["gemini-3.5-flash", "..%2F..%2Fx:generateContent"])
def test_gemini_rejects_malformed_targets(
    client: TestClient, vertex: FakeVertex, target: str
) -> None:
    r = client.post(f"/v1beta/models/{target}", json={})
    assert r.status_code == 400
    assert vertex.requests == []


# --- OpenAI -----------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/v1/chat/completions",
        "/chat/completions",
        "/openai/v1/chat/completions",
        "/openai/chat/completions",
        "/gemini/v1/chat/completions",
        "/gemini/chat/completions",
    ],
)
def test_openai_paths_route_gemini_to_the_openai_compatible_endpoint(
    client: TestClient, vertex: FakeVertex, path: str
) -> None:
    r = client.post(
        path, json={"model": "gemini-3.5-flash", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert str(vertex.last.url) == f"{GLOBAL}/endpoints/openapi/chat/completions"
    assert vertex.last_json["model"] == "google/gemini-3.5-flash"


def test_openai_maas_model_uses_publisher_model_string(
    client: TestClient, vertex: FakeVertex
) -> None:
    client.post("/v1/chat/completions", json={"model": "glm-5.2", "messages": []})
    assert str(vertex.last.url) == f"{GLOBAL}/endpoints/openapi/chat/completions"
    assert vertex.last_json["model"] == "zai-org/glm-5.2-maas"


def test_openai_unknown_model_is_rejected_in_openai_shape(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"model": "mystery", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "GET /v1/models" in r.json()["error"]["message"]


def test_openai_claude_goes_through_the_anthropic_bridge(
    client: TestClient, vertex: FakeVertex
) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5-5",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    assert str(vertex.last.url).endswith("/publishers/anthropic/models/claude-opus-5-5:rawPredict")
    assert vertex.last_json["system"] == "be brief"


def test_ollama_models_skip_google_auth(
    make_client: Callable[..., TestClient], vertex: FakeVertex
) -> None:
    client = make_client(ollama_backends={"llama3.3:70b": "http://ollama:11434"})
    client.post("/v1/chat/completions", json={"model": "llama3.3:70b", "messages": []})
    assert str(vertex.last.url) == "http://ollama:11434/v1/chat/completions"
    assert "authorization" not in vertex.last.headers


# --- Vertex pass-through ------------------------------------------------------


@pytest.mark.parametrize("prefix", ["/vertex/v1/", "/vertex/"])
def test_vertex_passthrough_injects_credentials_and_project(
    client: TestClient, vertex: FakeVertex, prefix: str
) -> None:
    r = client.post(
        f"{prefix}projects/client-side/locations/us/publishers/anthropic/models/claude-opus-5-5:rawPredict",
        content=b'{"anthropic_version":"vertex-2023-10-16","messages":[]}',
        headers={
            "anthropic-beta": "interleaved-thinking-2025-05-14",
            "content-type": "application/json",
            "x-api-key": "dropped",
        },
    )
    assert r.status_code == 200
    assert str(vertex.last.url) == (
        "https://aiplatform.us.rep.googleapis.com/v1/projects/test-project/locations/us"
        "/publishers/anthropic/models/claude-opus-5-5:rawPredict"
    )
    assert vertex.last.headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"
    assert vertex.last.headers["authorization"] == "Bearer fake-token"
    assert "x-api-key" not in vertex.last.headers
    assert vertex.last.content == b'{"anthropic_version":"vertex-2023-10-16","messages":[]}'


def test_vertex_passthrough_streams(client: TestClient, vertex: FakeVertex) -> None:
    vertex.respond_sse(['event: message_stop\ndata: {"type":"message_stop"}'])
    r = client.post(
        "/vertex/v1/projects/p/locations/global/publishers/anthropic/models/claude-opus-5-5:streamRawPredict",
        content=b"{}",
    )
    assert r.text == 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def test_vertex_passthrough_forwards_error_bodies_unchanged(
    client: TestClient, vertex: FakeVertex
) -> None:
    payload = {
        "error": {"code": 404, "message": "Publisher Model not found", "status": "NOT_FOUND"}
    }
    vertex.respond(404, payload)
    r = client.post(
        "/vertex/v1/projects/p/locations/global/publishers/anthropic/models/claude-x:rawPredict",
        content=b"{}",
    )
    assert r.status_code == 404
    assert r.json() == payload


def test_vertex_passthrough_rejects_other_paths(client: TestClient, vertex: FakeVertex) -> None:
    r = client.post("/vertex/v1/projects/p/locations/global/endpoints/x:predict", content=b"{}")
    assert r.status_code == 404
    assert r.json()["error"]["status"] == "NOT_FOUND"
    assert vertex.requests == []


# --- Models, health, metrics, proxy API key ------------------------------------


def test_model_listing(client: TestClient) -> None:
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert {"claude-opus-5-5", "gemini-3.5-flash", "glm-5.2"} <= ids
    anthropic = client.get("/anthropic/v1/models").json()
    assert anthropic["has_more"] is False
    assert all("claude" in m["id"] for m in anthropic["data"])
    assert all(m["type"] == "model" for m in anthropic["data"])


@pytest.mark.parametrize(
    "path", ["/models", "/openai/v1/models", "/gemini/models", "/api/v1/models"]
)
def test_model_listing_is_mirrored(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_model_probe(client: TestClient) -> None:
    assert client.get("/v1/models/claude-opus-5-5").status_code == 200
    r = client.get("/openai/v1/models/nope")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


def test_health(client: TestClient, tokens: FakeTokens) -> None:
    assert client.get("/health").json()["status"] == "ok"
    tokens.error = CredentialsError("expired")
    r = client.get("/health")
    assert r.status_code == 503
    assert "expired" in r.json()["error"]


def test_metrics_are_opt_in(make_client: Callable[..., TestClient]) -> None:
    assert make_client().get("/metrics").status_code == 404
    client = make_client(metrics_enabled=True)
    client.post("/v1/messages", json=_messages())
    body = client.get("/metrics").text
    assert (
        'vertex_proxy_requests_total{route="anthropic",model="claude-opus-5-5",status="200"} 1'
        in body
    )
    assert 'vertex_proxy_tokens_in_total{model="claude-opus-5-5"} 3' in body


def test_failed_requests_are_counted(
    make_client: Callable[..., TestClient], vertex: FakeVertex
) -> None:
    client = make_client(metrics_enabled=True)
    vertex.respond(429, {"error": {"message": "quota"}})
    client.post("/v1/messages", json=_messages())
    assert 'status="429"} 1' in client.get("/metrics").text


@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": "Bearer s3cret"},
        {"x-api-key": "s3cret"},
        {"x-goog-api-key": "s3cret"},
    ],
)
def test_api_key_is_accepted_in_every_client_header(
    make_client: Callable[..., TestClient], headers: dict[str, str]
) -> None:
    client = make_client(api_key="s3cret")
    assert client.post("/v1/messages", json=_messages(), headers=headers).status_code == 200


def test_api_key_rejections_use_the_route_dialect(
    make_client: Callable[..., TestClient], vertex: FakeVertex
) -> None:
    client = make_client(api_key="s3cret")
    anthropic = client.post("/v1/messages", json=_messages(), headers={"x-api-key": "wrong"})
    assert anthropic.status_code == 401
    assert anthropic.json()["error"]["type"] == "authentication_error"
    gemini = client.post("/v1beta/models/gemini-3.5-flash:generateContent", json={})
    assert gemini.json()["error"]["status"] == "UNAUTHENTICATED"
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models/claude-opus-5-5").status_code == 401
    assert client.get("/health").status_code == 200
    assert vertex.requests == []


def test_api_key_in_query_is_not_forwarded(
    make_client: Callable[..., TestClient], vertex: FakeVertex
) -> None:
    client = make_client(api_key="s3cret")
    r = client.post("/v1beta/models/gemini-3.5-flash:generateContent?key=s3cret&alt=sse", json={})
    assert r.status_code == 200
    assert vertex.last.url.query == b"alt=sse"
