"""HTTP routes, one module per API surface."""

from . import anthropic, gemini, health, models, openai, vertex

ROUTERS = (
    health.router,
    models.router,
    anthropic.router,
    gemini.router,
    openai.router,
    vertex.router,
)
