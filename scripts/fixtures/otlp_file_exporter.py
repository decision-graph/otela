"""OpenTelemetry SpanExporter that writes OTLP/JSON to a file.

This is the bridge between an in-process OTel SDK pipeline and the
on-disk format `otela.load()` consumes. Buffers spans in memory and emits
a single OTLP/JSON document at shutdown so a test fixture is one
deterministic file rather than a stream of fragments.

Why we don't use a stock exporter
---------------------------------
The OTel Python SDK ships gRPC and HTTP-protobuf exporters, but no
"OTLP/JSON to file" exporter. Using `MessageToDict` on the protobuf
encoder almost works, but it base64-encodes `traceId` / `spanId` (because
they're protobuf `bytes`) and our reader expects hex strings — the OTLP
v1.x JSON profile. So we do the conversion ourselves; it's not much code
and stays stable across SDK versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, StatusCode

# OTLP/JSON canonical names for the protobuf SpanKind enum.
_SPAN_KIND_NAME = {
    SpanKind.INTERNAL: "SPAN_KIND_INTERNAL",
    SpanKind.SERVER: "SPAN_KIND_SERVER",
    SpanKind.CLIENT: "SPAN_KIND_CLIENT",
    SpanKind.PRODUCER: "SPAN_KIND_PRODUCER",
    SpanKind.CONSUMER: "SPAN_KIND_CONSUMER",
}

_STATUS_CODE_NAME = {
    StatusCode.UNSET: "STATUS_CODE_UNSET",
    StatusCode.OK: "STATUS_CODE_OK",
    StatusCode.ERROR: "STATUS_CODE_ERROR",
}


class OTLPFileExporter(SpanExporter):
    """Buffer spans, emit a single OTLP/JSON document at shutdown."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._buffer: list[ReadableSpan] = []
        self._flushed = False

    def export(self, spans):
        # SimpleSpanProcessor calls export() once per span; BatchSpanProcessor
        # passes a sequence. Either works.
        self._buffer.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        if self._flushed:
            return
        self._flushed = True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = _spans_to_otlp(self._buffer)
        # OPT_INDENT_2 makes the fixture diffable; orjson is still ~10x faster
        # than stdlib json even with indentation enabled.
        self._path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    @property
    def num_spans_buffered(self) -> int:
        return len(self._buffer)


def _spans_to_otlp(spans: list[ReadableSpan]) -> dict[str, Any]:
    """Group spans by (resource, scope) and emit the OTLP/JSON tree shape."""
    # Group by id(resource) — within a single SDK process, all spans from
    # one tracer share the same Resource object. If two tracers happen to
    # have equal-but-distinct Resources we duplicate the resource block;
    # that's correct semantically (the wire format permits it).
    by_resource: dict[int, dict[str, Any]] = {}
    for span in spans:
        rid = id(span.resource)
        if rid not in by_resource:
            by_resource[rid] = {
                "resource": {"attributes": _attrs_to_kv(span.resource.attributes)},
                "scopeSpans": [],
                "_scope_index": {},  # internal helper, removed before emit
            }
        bucket = by_resource[rid]
        scope = span.instrumentation_scope
        scope_key = (scope.name, scope.version or "")
        if scope_key not in bucket["_scope_index"]:
            scope_block = {
                "scope": {"name": scope.name, "version": scope.version or ""},
                "spans": [],
            }
            bucket["scopeSpans"].append(scope_block)
            bucket["_scope_index"][scope_key] = scope_block
        bucket["_scope_index"][scope_key]["spans"].append(_span_to_dict(span))

    # Strip the bookkeeping field before serialization.
    resource_spans = []
    for bucket in by_resource.values():
        bucket.pop("_scope_index", None)
        resource_spans.append(bucket)
    return {"resourceSpans": resource_spans}


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    parent = span.parent
    return {
        "traceId": format(ctx.trace_id, "032x"),
        "spanId": format(ctx.span_id, "016x"),
        "parentSpanId": format(parent.span_id, "016x") if parent is not None else "",
        "name": span.name,
        "kind": _SPAN_KIND_NAME.get(span.kind, "SPAN_KIND_INTERNAL"),
        "startTimeUnixNano": str(span.start_time or 0),
        "endTimeUnixNano": str(span.end_time or 0),
        "attributes": _attrs_to_kv(span.attributes or {}),
        "events": [_event_to_dict(e) for e in span.events],
        "links": [_link_to_dict(ln) for ln in span.links],
        "status": _status_to_dict(span.status),
    }


def _attrs_to_kv(attrs) -> list[dict[str, Any]]:
    if not attrs:
        return []
    return [{"key": k, "value": _any_value(v)} for k, v in attrs.items()]


def _any_value(v: Any) -> dict[str, Any]:
    """Wrap a Python value in an OTLP `AnyValue`."""
    if v is None:
        return {}
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        # OTLP int64 is encoded as a string in JSON to preserve precision.
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, bytes):
        # OTLP/JSON encodes bytes as base64 strings. Rare in practice for traces.
        import base64
        return {"bytesValue": base64.b64encode(v).decode("ascii")}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_any_value(x) for x in v]}}
    if isinstance(v, dict):
        return {"kvlistValue": {"values": [{"key": k, "value": _any_value(val)} for k, val in v.items()]}}
    # Last-resort coercion — better to round-trip something than to fail.
    return {"stringValue": str(v)}


def _status_to_dict(status) -> dict[str, Any]:
    out: dict[str, Any] = {"code": _STATUS_CODE_NAME.get(status.status_code, "STATUS_CODE_UNSET")}
    if status.description:
        out["message"] = status.description
    return out


def _event_to_dict(event) -> dict[str, Any]:
    return {
        "name": event.name,
        "timeUnixNano": str(event.timestamp or 0),
        "attributes": _attrs_to_kv(event.attributes or {}),
    }


def _link_to_dict(link) -> dict[str, Any]:
    ctx = link.context
    return {
        "traceId": format(ctx.trace_id, "032x"),
        "spanId": format(ctx.span_id, "016x"),
        "attributes": _attrs_to_kv(link.attributes or {}),
    }
