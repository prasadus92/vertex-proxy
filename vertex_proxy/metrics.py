"""In-memory Prometheus counters for requests and tokens."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .errors import ProxyError


def usage_tokens(payload: Any) -> tuple[int, int]:
    """(prompt, completion) token counts from an OpenAI, Anthropic or Gemini response body."""
    if not isinstance(payload, dict):
        return 0, 0
    usage = payload.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        return int(prompt), int(completion)
    gemini = payload.get("usageMetadata")
    if isinstance(gemini, dict):
        return int(gemini.get("promptTokenCount") or 0), int(
            gemini.get("candidatesTokenCount") or 0
        )
    return 0, 0


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, str]] = Counter()
        self._tokens_in: Counter[str] = Counter()
        self._tokens_out: Counter[str] = Counter()
        self._started_at = time.time()

    def record(
        self, route: str, model: str, status: int, prompt: int = 0, completion: int = 0
    ) -> None:
        with self._lock:
            self._requests[(route, model, str(status))] += 1
            self._tokens_in[model] += prompt
            self._tokens_out[model] += completion

    @contextmanager
    def counting_failures(self, route: str, model: str) -> Iterator[None]:
        """Record a request that fails with a ProxyError before re-raising it."""
        try:
            yield
        except ProxyError as exc:
            self.record(route, model, exc.status)
            raise

    def render(self) -> str:
        """Prometheus text exposition format."""
        lines = [
            "# HELP vertex_proxy_uptime_seconds Seconds since proxy start",
            "# TYPE vertex_proxy_uptime_seconds gauge",
            f"vertex_proxy_uptime_seconds {time.time() - self._started_at:.0f}",
            "# HELP vertex_proxy_requests_total Total requests by route, model, and status",
            "# TYPE vertex_proxy_requests_total counter",
        ]
        with self._lock:
            lines += [
                f'vertex_proxy_requests_total{{route="{route}",model="{model}",status="{status}"}} {count}'
                for (route, model, status), count in self._requests.items()
            ]
            lines += [
                "# HELP vertex_proxy_tokens_in_total Prompt tokens forwarded",
                "# TYPE vertex_proxy_tokens_in_total counter",
            ]
            lines += [
                f'vertex_proxy_tokens_in_total{{model="{model}"}} {count}'
                for model, count in self._tokens_in.items()
                if count
            ]
            lines += [
                "# HELP vertex_proxy_tokens_out_total Completion tokens returned",
                "# TYPE vertex_proxy_tokens_out_total counter",
            ]
            lines += [
                f'vertex_proxy_tokens_out_total{{model="{model}"}} {count}'
                for model, count in self._tokens_out.items()
                if count
            ]
        return "\n".join(lines) + "\n"
