"""Arrow record-batch builders.

Accumulates `NormalizedSpan` records into column-oriented Python lists, then
materializes them as `pyarrow.Table` instances against the canonical schemas.

Why list-of-columns rather than streaming `RecordBatchBuilder`: pyarrow's
type-specific builder API is verbose and slower for mixed-type rows in
Python. For the v0 in-memory path, building columnar lists and handing them
to `pa.array()` per field is both faster and simpler. A future
`stream_to_parquet(path, batch_size=...)` will reuse the same builder by
flushing a batch every N spans.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from .normalize import NormalizedSpan
from .schemas import (
    DOCUMENTS_SCHEMA,
    LINKS_SCHEMA,
    MESSAGES_SCHEMA,
    SPANS_SCHEMA,
    SPEC,
    SPEC_VERSION,
    TRACES_SCHEMA,
)

# Status worst-of: ERROR > OK > UNSET. Tracked per-trace so we can roll up
# without re-scanning spans.
_STATUS_RANK = {"UNSET": 0, "OK": 1, "ERROR": 2}
_STATUS_BY_RANK = {0: "UNSET", 1: "OK", 2: "ERROR"}


@dataclass(slots=True)
class _SpanCols:
    spec: list[str] = field(default_factory=list)
    spec_version: list[str] = field(default_factory=list)
    trace_id: list[str] = field(default_factory=list)
    span_id: list[str] = field(default_factory=list)
    parent_span_id: list[str | None] = field(default_factory=list)
    name: list[str | None] = field(default_factory=list)
    kind: list[str | None] = field(default_factory=list)
    convention: list[str | None] = field(default_factory=list)
    status_code: list[str | None] = field(default_factory=list)
    status_message: list[str | None] = field(default_factory=list)
    start_time_unix_nano: list[int | None] = field(default_factory=list)
    end_time_unix_nano: list[int | None] = field(default_factory=list)
    duration_ns: list[int | None] = field(default_factory=list)
    service_name: list[str | None] = field(default_factory=list)
    scope_name: list[str | None] = field(default_factory=list)
    scope_version: list[str | None] = field(default_factory=list)
    model_name: list[str | None] = field(default_factory=list)
    tool_name: list[str | None] = field(default_factory=list)
    agent_name: list[str | None] = field(default_factory=list)
    input_tokens: list[int | None] = field(default_factory=list)
    output_tokens: list[int | None] = field(default_factory=list)
    total_tokens: list[int | None] = field(default_factory=list)
    io_format: list[str | None] = field(default_factory=list)
    input_text: list[str | None] = field(default_factory=list)
    output_text: list[str | None] = field(default_factory=list)
    raw_attributes_json: list[str | None] = field(default_factory=list)


@dataclass(slots=True)
class _MessageCols:
    spec: list[str] = field(default_factory=list)
    spec_version: list[str] = field(default_factory=list)
    trace_id: list[str] = field(default_factory=list)
    span_id: list[str] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    direction: list[str | None] = field(default_factory=list)
    role: list[str | None] = field(default_factory=list)
    content: list[str | None] = field(default_factory=list)
    tool_call_id: list[str | None] = field(default_factory=list)


@dataclass(slots=True)
class _DocumentCols:
    spec: list[str] = field(default_factory=list)
    spec_version: list[str] = field(default_factory=list)
    trace_id: list[str] = field(default_factory=list)
    span_id: list[str] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    document_id: list[str | None] = field(default_factory=list)
    content: list[str | None] = field(default_factory=list)
    score: list[float | None] = field(default_factory=list)


@dataclass(slots=True)
class _LinkCols:
    spec: list[str] = field(default_factory=list)
    spec_version: list[str] = field(default_factory=list)
    trace_id: list[str] = field(default_factory=list)
    span_id: list[str] = field(default_factory=list)
    linked_trace_id: list[str | None] = field(default_factory=list)
    linked_span_id: list[str | None] = field(default_factory=list)


@dataclass(slots=True)
class _TraceAccum:
    """Per-trace rollup state. Folded over spans as they arrive.

    Only the root_span fields are picked deterministically (earliest start
    time among parentless spans). Everything else is min/max/sum.
    """

    root_span_id: str | None = None
    root_span_name: str | None = None
    root_start_ns: int | None = None
    service_name: str | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    span_count: int = 0
    error_count: int = 0
    status_rank: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    tot_tokens: int = 0
    saw_in_tokens: bool = False
    saw_out_tokens: bool = False
    saw_tot_tokens: bool = False


class TableBuilder:
    """Accumulate normalized spans into column buffers, then build Arrow tables.

    The builder can be flushed and reused: after `build()`, call `reset()`
    to clear span/message/doc/link buffers and continue feeding the next
    batch. This is the primitive `to_parquet()` uses to keep memory bounded.

    Per-trace accumulators are *not* cleared on `reset()` — the trace
    rollup spans the entire load and is materialized once via
    `build_traces()`.
    """

    def __init__(self) -> None:
        self._spans = _SpanCols()
        self._messages = _MessageCols()
        self._documents = _DocumentCols()
        self._links = _LinkCols()
        self._traces: dict[str, _TraceAccum] = {}

    def reset(self) -> None:
        """Clear span/message/doc/link buffers. Trace accumulators persist."""
        self._spans = _SpanCols()
        self._messages = _MessageCols()
        self._documents = _DocumentCols()
        self._links = _LinkCols()

    @property
    def num_spans(self) -> int:
        return len(self._spans.span_id)

    @property
    def num_traces(self) -> int:
        return len(self._traces)

    def add(self, span: NormalizedSpan) -> None:
        s = self._spans
        s.spec.append(SPEC)
        s.spec_version.append(SPEC_VERSION)
        s.trace_id.append(span.trace_id)
        s.span_id.append(span.span_id)
        s.parent_span_id.append(span.parent_span_id)
        s.name.append(span.name)
        s.kind.append(span.kind)
        s.convention.append(span.convention)
        s.status_code.append(span.status_code)
        s.status_message.append(span.status_message)
        s.start_time_unix_nano.append(span.start_time_unix_nano)
        s.end_time_unix_nano.append(span.end_time_unix_nano)
        s.duration_ns.append(span.duration_ns)
        s.service_name.append(span.service_name)
        s.scope_name.append(span.scope_name)
        s.scope_version.append(span.scope_version)
        s.model_name.append(span.model_name)
        s.tool_name.append(span.tool_name)
        s.agent_name.append(span.agent_name)
        s.input_tokens.append(span.input_tokens)
        s.output_tokens.append(span.output_tokens)
        s.total_tokens.append(span.total_tokens)
        s.io_format.append(span.io_format)
        s.input_text.append(span.input_text)
        s.output_text.append(span.output_text)
        s.raw_attributes_json.append(span.raw_attributes_json)

        for msg in span.messages:
            m = self._messages
            m.spec.append(SPEC)
            m.spec_version.append(SPEC_VERSION)
            m.trace_id.append(span.trace_id)
            m.span_id.append(span.span_id)
            m.position.append(msg.position)
            m.direction.append(msg.direction)
            m.role.append(msg.role)
            m.content.append(msg.content)
            m.tool_call_id.append(msg.tool_call_id)

        for doc in span.documents:
            d = self._documents
            d.spec.append(SPEC)
            d.spec_version.append(SPEC_VERSION)
            d.trace_id.append(span.trace_id)
            d.span_id.append(span.span_id)
            d.position.append(doc.position)
            d.document_id.append(doc.document_id)
            d.content.append(doc.content)
            d.score.append(doc.score)

        for ln in span.links:
            lk = self._links
            lk.spec.append(SPEC)
            lk.spec_version.append(SPEC_VERSION)
            lk.trace_id.append(span.trace_id)
            lk.span_id.append(span.span_id)
            lk.linked_trace_id.append(ln.linked_trace_id)
            lk.linked_span_id.append(ln.linked_span_id)

        self._fold_into_trace(span)

    def _fold_into_trace(self, span: NormalizedSpan) -> None:
        accum = self._traces.get(span.trace_id)
        if accum is None:
            accum = _TraceAccum()
            self._traces[span.trace_id] = accum

        accum.span_count += 1

        # Status rollup: ERROR > OK > UNSET, plus separate error counter.
        rank = _STATUS_RANK.get(span.status_code, 0)
        if rank > accum.status_rank:
            accum.status_rank = rank
        if span.status_code == "ERROR":
            accum.error_count += 1

        # Timing min/max across all spans (robust to clock skew / async).
        start = span.start_time_unix_nano
        end = span.end_time_unix_nano
        if start is not None and (accum.start_ns is None or start < accum.start_ns):
            accum.start_ns = start
        if end is not None and (accum.end_ns is None or end > accum.end_ns):
            accum.end_ns = end

        # Root span: earliest-starting parentless span.
        if span.parent_span_id is None:
            new_root_start = span.start_time_unix_nano
            existing_start = accum.root_start_ns
            is_first = accum.root_span_id is None
            is_earlier = (
                new_root_start is not None
                and existing_start is not None
                and new_root_start < existing_start
            )
            existing_unknown = existing_start is None and new_root_start is not None
            if is_first or is_earlier or existing_unknown:
                accum.root_span_id = span.span_id
                accum.root_span_name = span.name
                accum.root_start_ns = new_root_start
                # service_name follows the root span — the entry-point service.
                accum.service_name = span.service_name

        # Token sums. Track whether we ever saw a non-null so we can emit NULL
        # when no span carried the field rather than misleading 0.
        if span.input_tokens is not None:
            accum.in_tokens += span.input_tokens
            accum.saw_in_tokens = True
        if span.output_tokens is not None:
            accum.out_tokens += span.output_tokens
            accum.saw_out_tokens = True
        if span.total_tokens is not None:
            accum.tot_tokens += span.total_tokens
            accum.saw_tot_tokens = True

    def extend(self, spans: Iterable[NormalizedSpan]) -> None:
        for s in spans:
            self.add(s)

    def build(self) -> dict[str, pa.Table]:
        """Build the four batch-level tables. Excludes `traces` (use `build_traces()`)."""
        return {
            "spans": _to_table(self._spans, SPANS_SCHEMA),
            "messages": _to_table(self._messages, MESSAGES_SCHEMA),
            "documents": _to_table(self._documents, DOCUMENTS_SCHEMA),
            "links": _to_table(self._links, LINKS_SCHEMA),
        }

    def build_traces(self) -> pa.Table:
        """Materialize the trace rollup table from accumulators."""
        n = len(self._traces)
        cols: dict[str, list] = {
            "spec": [SPEC] * n,
            "spec_version": [SPEC_VERSION] * n,
            "trace_id": [],
            "root_span_id": [],
            "root_span_name": [],
            "service_name": [],
            "start_time_unix_nano": [],
            "end_time_unix_nano": [],
            "duration_ns": [],
            "span_count": [],
            "error_count": [],
            "status": [],
            "total_input_tokens": [],
            "total_output_tokens": [],
            "total_tokens": [],
        }
        for trace_id, a in self._traces.items():
            cols["trace_id"].append(trace_id)
            cols["root_span_id"].append(a.root_span_id)
            cols["root_span_name"].append(a.root_span_name)
            cols["service_name"].append(a.service_name)
            cols["start_time_unix_nano"].append(a.start_ns)
            cols["end_time_unix_nano"].append(a.end_ns)
            duration = (
                a.end_ns - a.start_ns
                if a.start_ns is not None and a.end_ns is not None
                else None
            )
            cols["duration_ns"].append(duration)
            cols["span_count"].append(a.span_count)
            cols["error_count"].append(a.error_count)
            cols["status"].append(_STATUS_BY_RANK[a.status_rank])
            cols["total_input_tokens"].append(a.in_tokens if a.saw_in_tokens else None)
            cols["total_output_tokens"].append(a.out_tokens if a.saw_out_tokens else None)
            cols["total_tokens"].append(a.tot_tokens if a.saw_tot_tokens else None)

        arrays = [pa.array(cols[f.name], type=f.type) for f in TRACES_SCHEMA]
        return pa.Table.from_arrays(arrays, schema=TRACES_SCHEMA)


def _to_table(cols: Any, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field_ in schema:
        data = getattr(cols, field_.name)
        arrays.append(pa.array(data, type=field_.type))
    return pa.Table.from_arrays(arrays, schema=schema)
