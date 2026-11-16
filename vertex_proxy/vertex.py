"""Vertex AI endpoint URLs for global, multi-region and regional locations."""

from __future__ import annotations

MULTI_REGIONS = frozenset({"us", "eu"})


def vertex_host(location: str) -> str:
    if location == "global":
        return "aiplatform.googleapis.com"
    if location in MULTI_REGIONS:
        return f"aiplatform.{location}.rep.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


class VertexEndpoints:
    """Builds request URLs for one GCP project."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id

    def location_base(self, location: str, version: str = "v1") -> str:
        return (
            f"https://{vertex_host(location)}/{version}/projects/{self.project_id}"
            f"/locations/{location}"
        )

    def publisher_model(
        self, location: str, publisher: str, model: str, method: str, version: str = "v1"
    ) -> str:
        return (
            f"{self.location_base(location, version)}/publishers/{publisher}"
            f"/models/{model}:{method}"
        )

    def anthropic(self, location: str, model: str, *, stream: bool) -> str:
        method = "streamRawPredict" if stream else "rawPredict"
        return self.publisher_model(location, "anthropic", model, method)

    def anthropic_count_tokens(self, location: str) -> str:
        return self.publisher_model(location, "anthropic", "count-tokens", "rawPredict")

    def gemini(self, location: str, model: str, action: str) -> str:
        return self.publisher_model(location, "google", model, action)

    def openai_chat(self, location: str) -> str:
        return f"{self.location_base(location)}/endpoints/openapi/chat/completions"
