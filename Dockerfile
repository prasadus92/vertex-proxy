# syntax=docker/dockerfile:1

FROM python:3.15-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-editable --no-install-project

COPY README.md ./
COPY vertex_proxy/ vertex_proxy/
RUN uv sync --locked --no-editable

FROM python:3.15-slim

RUN useradd --uid 1000 --no-create-home --home-dir /app --shell /usr/sbin/nologin vertex \
    && mkdir -p /app && chown vertex:vertex /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VERTEX_PROXY_HOST=0.0.0.0 \
    VERTEX_PROXY_PORT=8787 \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER vertex
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["vertex-proxy"]
