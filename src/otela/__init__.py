"""otela — OpenTelemetry analytics & data formatting for agent traces."""

from __future__ import annotations

from .api import dims, load, to_arrow, to_dfs, to_dicts, to_parquet
from .schemas import (
    DOCUMENTS_SCHEMA,
    LINKS_SCHEMA,
    MESSAGES_SCHEMA,
    SPANS_SCHEMA,
    SPEC,
    SPEC_VERSION,
    TRACES_SCHEMA,
)
from .tensors import TensorTable, to_tensors

__version__ = "0.1.0"

__all__ = [
    "DOCUMENTS_SCHEMA",
    "LINKS_SCHEMA",
    "MESSAGES_SCHEMA",
    "SPANS_SCHEMA",
    "SPEC",
    "SPEC_VERSION",
    "TRACES_SCHEMA",
    "TensorTable",
    "__version__",
    "dims",
    "load",
    "to_arrow",
    "to_dfs",
    "to_dicts",
    "to_parquet",
    "to_tensors",
]
