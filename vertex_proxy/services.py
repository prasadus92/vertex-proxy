"""The objects every route shares for the life of the app."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import ModelCatalog
from .config import Settings
from .metrics import Metrics
from .upstream import Upstream
from .vertex import VertexEndpoints


@dataclass(frozen=True)
class Services:
    settings: Settings
    catalog: ModelCatalog
    endpoints: VertexEndpoints
    upstream: Upstream
    metrics: Metrics
