"""HTTP routes, one module per API surface."""

from . import anthropic, gemini, health, models, openai, responses, vertex

ROUTERS = (
    health.router,
    models.router,
    anthropic.router,
    gemini.router,
    openai.router,
    responses.router,
    vertex.router,
)
