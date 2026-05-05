"""Public Python API for otela."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .builder import TableBuilder
from .normalize import normalize_span
from .reader import RawSpan, iter_otlp_spans
from .schemas import (
    ALL_TABLE_NAMES,
    SCHEMAS,
    SPEC,
    SPEC_VERSION,
    TABLE_NAMES,
    TRACES_TABLE_NAME,
)


def load(
    path: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    spec: str = "at/v1",
) -> dict[str, pa.Table]:
    """Load OTLP/JSON traces from `path` into Arrow tables.

    `path` may be a single file, a directory of OTLP/JSON files, or an
    iterable of either. Returns a dict with keys `spans`, `messages`,
    `documents`, `links`, `traces` — all joinable on `trace_id` (and
    `span_id` for the per-span side tables).

    The `spec` argument is currently fixed at "at/v1"; the parameter
    is reserved for a future `wg/v1` workflow-graph spec.
    """
    _check_spec(spec)
    builder = TableBuilder()
    for raw in _iter_paths(path):
        builder.add(normalize_span(raw))
    tables = builder.build()
    tables[TRACES_TABLE_NAME] = builder.build_traces()
    return tables


def to_arrow(
    path: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    spec: str = "at/v1",
) -> dict[str, pa.Table]:
    """Alias for `load()`. Kept for symmetry with `to_dfs()`."""
    return load(path, spec=spec)


def to_dfs(tables: dict[str, pa.Table]):
    """Convert the table dict into pandas DataFrames (Arrow-backed dtypes).

    `pandas` is an optional dependency; install with `pip install otela[pandas]`.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pandas is not installed. Install with: pip install otela[pandas]"
        ) from e
    return {name: t.to_pandas(types_mapper=_pandas_types_mapper) for name, t in tables.items()}


def to_parquet(
    path: str | os.PathLike | Iterable[str | os.PathLike],
    output_dir: str | os.PathLike,
    *,
    spec: str = "at/v1",
    batch_size: int = 10_000,
    compression: str = "zstd",
    row_group_size: int | None = None,
) -> dict[str, Path]:
    """Stream OTLP/JSON traces to parquet files with bounded memory.

    Writes one parquet file per table:

        output_dir/spans.parquet
        output_dir/messages.parquet
        output_dir/documents.parquet
        output_dir/links.parquet

    Memory is bounded by `batch_size` (number of spans buffered before each
    flush). The same code path handles a 10 MB file and a 1 TB directory.

    Parameters
    ----------
    path
        File, directory, or iterable of paths containing OTLP/JSON traces.
    output_dir
        Directory to write parquet files into. Created if missing.
    spec
        Spec version (currently only `at/v1`).
    batch_size
        Spans per flush. Larger = bigger row groups, fewer flushes,
        more peak memory. 10k is a reasonable default.
    compression
        Parquet compression. `zstd` gives the best ratio for analytics;
        use `snappy` for faster decompression.
    row_group_size
        Row group size override; defaults to `batch_size` worth of rows
        for spans, and the natural batch size for child tables.

    Returns
    -------
    Mapping of table name to written file path.
    """
    _check_spec(spec)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"{name}.parquet" for name in ALL_TABLE_NAMES}

    # Spans/messages/documents/links stream per batch. Traces is a rollup,
    # written once at end-of-load from the builder's per-trace accumulators.
    batch_writers: dict[str, pq.ParquetWriter] = {
        name: pq.ParquetWriter(paths[name], SCHEMAS[name], compression=compression)
        for name in TABLE_NAMES
    }
    builder = TableBuilder()
    try:
        for raw in _iter_paths(path):
            builder.add(normalize_span(raw))
            if builder.num_spans >= batch_size:
                _flush_batch(builder, batch_writers, row_group_size)
        if builder.num_spans > 0:
            _flush_batch(builder, batch_writers, row_group_size)
    finally:
        for w in batch_writers.values():
            w.close()

    # Traces rollup written as a single table; its row count is bounded by
    # the number of distinct trace_ids, which is much smaller than the
    # number of spans.
    pq.write_table(
        builder.build_traces(),
        paths[TRACES_TABLE_NAME],
        compression=compression,
    )
    return paths


def to_dicts(tables: dict[str, pa.Table]) -> list[dict[str, Any]]:
    """Convert the table dict into a list of nested trace records.

    Each record is a single trace with all its spans nested underneath, and
    messages/documents/links nested under each span. Trace-level fields
    (`trace_id`, `start_time_unix_nano`, `total_tokens`, ...) sit at the
    record root.

    Per-span fields shadow nothing — the span-level versions are nested
    under `"spans"`. Redundant FK columns (`trace_id` on spans,
    `trace_id`/`span_id` on messages/documents/links) are stripped because
    they're implied by the nesting. `raw_attributes_json` is decoded back
    into a `raw_attributes` dict (or `None`).

    This materializes the entire dataset into Python dicts. For
    billion-trace scale, write parquet first and process partitions with
    DuckDB or Polars rather than calling this directly.
    """
    spans_rows = tables["spans"].to_pylist()
    messages_rows = tables["messages"].to_pylist()
    documents_rows = tables["documents"].to_pylist()
    links_rows = tables["links"].to_pylist()
    traces_rows = tables["traces"].to_pylist()

    # Index span children by (trace_id, span_id).
    msgs_by_span: dict[tuple[str, str], list[dict]] = {}
    for m in messages_rows:
        msgs_by_span.setdefault((m["trace_id"], m["span_id"]), []).append(
            _strip_keys(m, _CHILD_FK_KEYS)
        )
    docs_by_span: dict[tuple[str, str], list[dict]] = {}
    for d in documents_rows:
        docs_by_span.setdefault((d["trace_id"], d["span_id"]), []).append(
            _strip_keys(d, _CHILD_FK_KEYS)
        )
    links_by_span: dict[tuple[str, str], list[dict]] = {}
    for ln in links_rows:
        links_by_span.setdefault((ln["trace_id"], ln["span_id"]), []).append(
            _strip_keys(ln, _CHILD_FK_KEYS)
        )

    # Index spans by trace_id, preserving input order so callers see a
    # natural ordering instead of something hash-derived.
    spans_by_trace: dict[str, list[dict]] = {}
    for s in spans_rows:
        spans_by_trace.setdefault(s["trace_id"], []).append(s)

    records: list[dict[str, Any]] = []
    for trace in traces_rows:
        tid = trace["trace_id"]
        nested_spans: list[dict[str, Any]] = []
        for span in spans_by_trace.get(tid, []):
            sid = span["span_id"]
            key = (tid, sid)
            raw_json = span.get("raw_attributes_json")
            raw_attrs = orjson.loads(raw_json) if raw_json else None
            nested_span = _strip_keys(span, _SPAN_FK_KEYS)
            nested_span["raw_attributes"] = raw_attrs
            nested_span["messages"] = msgs_by_span.get(key, [])
            nested_span["documents"] = docs_by_span.get(key, [])
            nested_span["links"] = links_by_span.get(key, [])
            nested_spans.append(nested_span)
        record = dict(trace)
        record["spans"] = nested_spans
        records.append(record)
    return records


# Columns dropped from nested span dicts (redundant with parent trace).
_SPAN_FK_KEYS: frozenset[str] = frozenset({"trace_id", "raw_attributes_json"})

# Columns dropped from nested message/document/link dicts.
_CHILD_FK_KEYS: frozenset[str] = frozenset({"trace_id", "span_id", "spec", "spec_version"})


def _strip_keys(d: dict[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in keys}


def dims(tables: dict[str, pa.Table]) -> dict[str, pa.Table]:
    """Materialize on-demand dimension tables from the spans table.

    Returns a dict with keys `tools`, `agents`, `models`, `services`, each
    an Arrow table with columns `(spec, spec_version, name, span_count,
    trace_count)`.

    These are derived views — not written to disk by `load()` or
    `to_parquet()`. They exist for users who want a star-schema layout
    on top of the canonical (denormalized) span columns. For 1B-scale
    datasets, prefer running this against parquet via DuckDB rather than
    loading everything into pyarrow first.
    """
    spans = tables["spans"]
    return {
        "tools": _dim(spans, "tool_name"),
        "agents": _dim(spans, "agent_name"),
        "models": _dim(spans, "model_name"),
        "services": _dim(spans, "service_name"),
    }


def _dim(spans: pa.Table, col: str) -> pa.Table:
    """`SELECT name, COUNT(*) AS span_count, COUNT(DISTINCT trace_id) AS trace_count
    FROM spans WHERE name IS NOT NULL GROUP BY name`."""
    if spans.num_rows == 0:
        return _empty_dim()
    keep = pc.is_valid(spans[col])
    filtered = spans.filter(keep)
    if filtered.num_rows == 0:
        return _empty_dim()
    grouped = filtered.select([col, "span_id", "trace_id"]).group_by(col).aggregate(
        [("span_id", "count"), ("trace_id", "count_distinct")]
    )
    n = grouped.num_rows
    name_col = grouped.column(col)
    span_count_col = grouped.column("span_id_count").cast(pa.int64())
    trace_count_col = grouped.column("trace_id_count_distinct").cast(pa.int64())
    return pa.table(
        {
            "spec": pa.array([SPEC] * n, type=pa.string()),
            "spec_version": pa.array([SPEC_VERSION] * n, type=pa.string()),
            "name": name_col.cast(pa.string()),
            "span_count": span_count_col,
            "trace_count": trace_count_col,
        }
    )


def _empty_dim() -> pa.Table:
    return pa.table(
        {
            "spec": pa.array([], type=pa.string()),
            "spec_version": pa.array([], type=pa.string()),
            "name": pa.array([], type=pa.string()),
            "span_count": pa.array([], type=pa.int64()),
            "trace_count": pa.array([], type=pa.int64()),
        }
    )


def _flush_batch(
    builder: TableBuilder,
    writers: dict[str, pq.ParquetWriter],
    row_group_size: int | None,
) -> None:
    tables = builder.build()
    for name, table in tables.items():
        if table.num_rows == 0:
            continue
        # row_group_size is per-table; passing None lets pyarrow pick a default.
        writers[name].write_table(table, row_group_size=row_group_size)
    builder.reset()


def _check_spec(spec: str) -> None:
    if spec not in ("at/v1", "at"):
        raise ValueError(f"unsupported spec: {spec}")


def _pandas_types_mapper(arrow_type: pa.DataType):
    # Arrow-backed pandas dtypes preserve nullability for ints/floats and
    # avoid silently widening int64 to float64 when nulls appear.
    import pandas as pd

    return pd.ArrowDtype(arrow_type)


def _iter_paths(
    path: str | os.PathLike | Iterable[str | os.PathLike],
) -> Iterable[RawSpan]:
    if isinstance(path, (str, os.PathLike)):
        yield from iter_otlp_spans(path)
        return
    for p in path:
        yield from iter_otlp_spans(p)
