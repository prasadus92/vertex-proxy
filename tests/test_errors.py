from __future__ import annotations

import json

from vertex_proxy.errors import Dialect, ErrorKind, ProxyError, UpstreamContext

CTX = UpstreamContext(project_id="p1", location="us-east5", model="claude-opus-5-5")


def _google(status: int, message: str, code: str) -> bytes:
    return json.dumps({"error": {"code": status, "message": message, "status": code}}).encode()


def test_quota_error_explains_what_to_do_and_keeps_retry_headers() -> None:
    err = ProxyError.from_upstream(
        429,
        _google(429, "Quota exceeded for aiplatform.googleapis.com", "RESOURCE_EXHAUSTED"),
        {"retry-after": "7", "content-type": "application/json", "x-other": "dropped"},
        CTX,
    )
    assert err.kind is ErrorKind.RATE_LIMIT
    assert err.message.startswith("Quota exceeded for aiplatform.googleapis.com")
    assert "quota is exhausted for 'claude-opus-5-5' in 'us-east5'" in err.message
    assert err.headers == {"retry-after": "7"}


def test_permission_error_suggests_the_role() -> None:
    err = ProxyError.from_upstream(
        403, _google(403, "Permission denied", "PERMISSION_DENIED"), {}, CTX
    )
    assert "roles/aiplatform.user on project p1" in err.message


def test_disabled_api_suggests_enabling_it() -> None:
    body = _google(403, "Vertex AI API has not been used in project p1", "PERMISSION_DENIED")
    err = ProxyError.from_upstream(403, body, {}, CTX)
    assert "gcloud services enable aiplatform.googleapis.com --project p1" in err.message


def test_not_found_points_at_model_garden_and_location() -> None:
    err = ProxyError.from_upstream(
        404, _google(404, "Publisher model not found", "NOT_FOUND"), {}, CTX
    )
    assert "not available in location 'us-east5'" in err.message
    assert "Model Garden" in err.message


def test_validation_errors_keep_the_upstream_wording() -> None:
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "x: Extra inputs are not permitted",
            },
        }
    ).encode()
    err = ProxyError.from_upstream(400, body, {}, CTX)
    assert err.message == "x: Extra inputs are not permitted"


def test_list_wrapped_google_errors_and_plain_text_bodies() -> None:
    wrapped = json.dumps([{"error": {"code": 400, "message": "bad field"}}]).encode()
    assert ProxyError.from_upstream(400, wrapped, {}, CTX).message == "bad field"
    assert ProxyError.from_upstream(502, b"Bad Gateway", {}, CTX).message == "Bad Gateway"


def test_payload_shapes_per_dialect() -> None:
    err = ProxyError(429, ErrorKind.RATE_LIMIT, "slow down")
    assert err.payload(Dialect.ANTHROPIC) == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "slow down"},
    }
    assert err.payload(Dialect.OPENAI)["error"]["type"] == "rate_limit_error"
    assert err.payload(Dialect.GOOGLE)["error"]["status"] == "RESOURCE_EXHAUSTED"


def test_raw_dialect_returns_the_upstream_body_unchanged() -> None:
    body = _google(404, "nope", "NOT_FOUND")
    response = ProxyError.from_upstream(
        404, body, {"content-type": "application/json"}, CTX
    ).to_response(Dialect.RAW)
    assert response.body == body
    assert response.status_code == 404


def test_anthropic_sse_errors_carry_an_event_line() -> None:
    sse = ProxyError(502, ErrorKind.UPSTREAM, "gone").to_sse(Dialect.ANTHROPIC)
    assert sse.startswith(b"event: error\ndata: ")
