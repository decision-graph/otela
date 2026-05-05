"""OTLP/JSON parsing helpers.

OTLP/JSON is the wire format the OTel Collector emits when configured with a
`file` exporter or the `otlp/json` exporter. It mirrors the protobuf shape
exactly: `resourceSpans -> scopeSpans -> spans`, with attributes encoded as
`[{key, value: AnyValue}, ...]` lists and int64 timestamps encoded as
strings (because JSON numbers cannot represent int64 precisely).

These helpers convert that wire shape into ergonomic Python: attribute lists
become dicts, AnyValue becomes a native Python value, and timestamps become
plain ints.
"""

from __future__ import annotations

from typing import Any

# OTLP status code -> canonical short form
_STATUS_CODE_MAP = {
    "STATUS_CODE_OK": "OK",
    "STATUS_CODE_ERROR": "ERROR",
    "STATUS_CODE_UNSET": "UNSET",
    # Some exporters omit the prefix.
    "OK": "OK",
    "ERROR": "ERROR",
    "UNSET": "UNSET",
    "": "UNSET",
}


def any_value(v: Any) -> Any:
    """Decode an OTLP `AnyValue` dict into a native Python value.

    Returns None for an unset/empty value. Nested arrays and key-value lists
    are decoded recursively.
    """
    if v is None:
        return None
    if not isinstance(v, dict):
        # Some exporters strip the AnyValue wrapper and emit raw scalars.
        return v
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        iv = v["intValue"]
        # int64 in OTLP/JSON is a string; SDKs sometimes emit a JSON number.
        return int(iv) if isinstance(iv, str) else iv
    if "doubleValue" in v:
        return v["doubleValue"]
    if "boolValue" in v:
        return v["boolValue"]
    if "bytesValue" in v:
        return v["bytesValue"]
    if "arrayValue" in v:
        items = (v["arrayValue"] or {}).get("values", []) or []
        return [any_value(i) for i in items]
    if "kvlistValue" in v:
        kvs = (v["kvlistValue"] or {}).get("values", []) or []
        return {kv.get("key", ""): any_value(kv.get("value")) for kv in kvs}
    return None


def attributes_to_dict(attrs: list[dict] | None) -> dict[str, Any]:
    """Flatten an OTLP attributes list into a `{key: value}` dict."""
    if not attrs:
        return {}
    out: dict[str, Any] = {}
    for kv in attrs:
        key = kv.get("key")
        if not key:
            continue
        out[key] = any_value(kv.get("value"))
    return out


def parse_nano(v: Any) -> int | None:
    """Parse an OTLP nano-timestamp; tolerates str (canonical) or int."""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def normalize_status_code(code: Any) -> str:
    if code is None:
        return "UNSET"
    return _STATUS_CODE_MAP.get(str(code), str(code))


def parent_span_id_or_none(v: Any) -> str | None:
    if not v:
        return None
    return str(v)
