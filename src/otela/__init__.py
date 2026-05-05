"""otela — OpenTelemetry analytics & data formatting for agent traces."""

from __future__ import annotations

from .api import dims, load, to_arrow, to_dfs, to_parquet
from .schemas import (
    DOCUMENTS_SCHEMA,
    LINKS_SCHEMA,
    MESSAGES_SCHEMA,
    SPANS_SCHEMA,
    SPEC,
    SPEC_VERSION,
    TRACES_SCHEMA,
)

__version__ = "0.1.0"

__all__ = [
    "DOCUMENTS_SCHEMA",
    "LINKS_SCHEMA",
    "MESSAGES_SCHEMA",
    "SPANS_SCHEMA",
    "SPEC",
    "SPEC_VERSION",
    "TRACES_SCHEMA",
    "__version__",
    "dims",
    "load",
    "to_arrow",
    "to_dfs",
    "to_parquet",
]
