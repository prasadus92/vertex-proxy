# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY vertex_proxy/ vertex_proxy/

RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------

FROM python:3.12-slim

RUN useradd --system --uid 1000 --no-create-home --home-dir /app vertex \
    && mkdir -p /app && chown vertex:vertex /app

COPY --from=builder /install /usr/local
WORKDIR /app
USER vertex

ENV VERTEX_PROXY_HOST=0.0.0.0 \
    VERTEX_PROXY_PORT=8787 \
    PYTHONUNBUFFERED=1

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["vertex-proxy"]
