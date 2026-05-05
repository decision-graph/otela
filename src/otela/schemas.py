"""Arrow schemas for the otela agent-trace (at/v1) spec.

Stable, versioned schemas. Every row in every table carries `spec` and
`spec_version` so downstream pipelines can detect schema drift.

Design notes
------------
- `spans` is the canonical row-per-span table. It is fixed-width (no nested
  columns) so it round-trips cleanly through pandas, parquet, and DuckDB.
- `messages`, `documents`, `links` are side tables joined back to `spans` by
  (`trace_id`, `span_id`). This keeps `spans` narrow and lets ML pipelines
  pull only what they need.
- Long free-text fields use `large_string` to avoid the 2 GiB-per-chunk
  limit of regular `string`.
- `raw_attributes_json` is the JSON-encoded leftover OTel attributes that
  the normalizer did not promote into a typed column. It preserves fidelity
  without forcing a heterogeneously-typed map.
"""

from __future__ import annotations

import pyarrow as pa

SPEC = "at"
SPEC_VERSION = "v1"


SPANS_SCHEMA = pa.schema(
    [
        pa.field("spec", pa.string(), nullable=False),
        pa.field("spec_version", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("span_id", pa.string(), nullable=False),
        pa.field("parent_span_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("convention", pa.string()),
        pa.field("status_code", pa.string()),
        pa.field("status_message", pa.string()),
        pa.field("start_time_unix_nano", pa.int64()),
        pa.field("end_time_unix_nano", pa.int64()),
        pa.field("duration_ns", pa.int64()),
        pa.field("service_name", pa.string()),
        pa.field("scope_name", pa.string()),
        pa.field("scope_version", pa.string()),
        pa.field("model_name", pa.string()),
        pa.field("tool_name", pa.string()),
        pa.field("agent_name", pa.string()),
        pa.field("input_tokens", pa.int64()),
        pa.field("output_tokens", pa.int64()),
        pa.field("total_tokens", pa.int64()),
        pa.field("io_format", pa.string()),
        pa.field("input_text", pa.large_string()),
        pa.field("output_text", pa.large_string()),
        pa.field("raw_attributes_json", pa.large_string()),
    ]
)


MESSAGES_SCHEMA = pa.schema(
    [
        pa.field("spec", pa.string(), nullable=False),
        pa.field("spec_version", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("span_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("direction", pa.string()),
        pa.field("role", pa.string()),
        pa.field("content", pa.large_string()),
        pa.field("tool_call_id", pa.string()),
    ]
)


DOCUMENTS_SCHEMA = pa.schema(
    [
        pa.field("spec", pa.string(), nullable=False),
        pa.field("spec_version", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("span_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("document_id", pa.string()),
        pa.field("content", pa.large_string()),
        pa.field("score", pa.float64()),
    ]
)


LINKS_SCHEMA = pa.schema(
    [
        pa.field("spec", pa.string(), nullable=False),
        pa.field("spec_version", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("span_id", pa.string(), nullable=False),
        pa.field("linked_trace_id", pa.string()),
        pa.field("linked_span_id", pa.string()),
    ]
)


# Trace-level rollup. One row per trace. Computed by aggregating spans, so
# `traces` is not flushed per batch in streaming writers — it's emitted once
# at end-of-load from per-trace accumulators in TableBuilder.
TRACES_SCHEMA = pa.schema(
    [
        pa.field("spec", pa.string(), nullable=False),
        pa.field("spec_version", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("root_span_id", pa.string()),
        pa.field("root_span_name", pa.string()),
        pa.field("service_name", pa.string()),
        pa.field("start_time_unix_nano", pa.int64()),
        pa.field("end_time_unix_nano", pa.int64()),
        pa.field("duration_ns", pa.int64()),
        pa.field("span_count", pa.int64()),
        pa.field("error_count", pa.int64()),
        pa.field("status", pa.string()),
        pa.field("total_input_tokens", pa.int64()),
        pa.field("total_output_tokens", pa.int64()),
        pa.field("total_tokens", pa.int64()),
    ]
)


# Tables that the streaming writer flushes per batch.
TABLE_NAMES = ("spans", "messages", "documents", "links")

# Trace rollup is materialized once at end-of-load.
TRACES_TABLE_NAME = "traces"

# Every table the library produces.
ALL_TABLE_NAMES = (*TABLE_NAMES, TRACES_TABLE_NAME)

SCHEMAS: dict[str, pa.Schema] = {
    "spans": SPANS_SCHEMA,
    "messages": MESSAGES_SCHEMA,
    "documents": DOCUMENTS_SCHEMA,
    "links": LINKS_SCHEMA,
    "traces": TRACES_SCHEMA,
}
