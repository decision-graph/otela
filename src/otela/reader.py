"""Streaming reader for OTLP/JSON files and directories.

`iter_otlp_spans` yields one `RawSpan` per OTel span, with resource and scope
metadata already propagated. The reader does not normalize semantic
conventions — that is the normalizer's job. Keeping these stages separate
means the reader can stay zero-allocation-heavy and the normalizer can be
swapped per spec.

Memory model
------------
Files are loaded one at a time and parsed with `orjson` (fast C JSON). The
function is a generator so a 1 TB directory of files works the same as a
10 MB single file: callers can pipe straight into the Arrow builder without
holding the whole dataset in memory.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .otlp import attributes_to_dict


@dataclass(slots=True)
class RawSpan:
    """One OTLP span plus the resource/scope context it was emitted under."""

    span: dict[str, Any]
    resource_attributes: dict[str, Any]
    scope_name: str | None
    scope_version: str | None
    source_file: str | None = None


def iter_otlp_files(path: str | os.PathLike) -> Iterator[Path]:
    """Yield OTLP/JSON files from a path. Path may be a file or directory.

    Directories are walked recursively for `*.json` files. Order is sorted
    for determinism.
    """
    p = Path(path)
    if p.is_file():
        yield p
        return
    if not p.is_dir():
        raise FileNotFoundError(f"path does not exist: {p}")
    for child in sorted(p.rglob("*.json")):
        if child.is_file():
            yield child


def iter_otlp_spans(path: str | os.PathLike) -> Iterator[RawSpan]:
    """Stream every span under `path` as a `RawSpan`.

    Accepts a single file or a directory of OTLP/JSON files. Resource
    attributes and instrumentation scope are propagated onto each span so
    downstream code never has to walk back up the tree.
    """
    for file_path in iter_otlp_files(path):
        with file_path.open("rb") as f:
            payload = orjson.loads(f.read())
        yield from _iter_payload_spans(payload, source_file=str(file_path))


def iter_payloads(payloads: Iterable[dict[str, Any]]) -> Iterator[RawSpan]:
    """Stream spans from already-parsed OTLP/JSON payloads (e.g. from tests)."""
    for payload in payloads:
        yield from _iter_payload_spans(payload, source_file=None)


def _iter_payload_spans(payload: dict[str, Any], source_file: str | None) -> Iterator[RawSpan]:
    resource_spans = payload.get("resourceSpans") or []
    for rs in resource_spans:
        resource = rs.get("resource") or {}
        resource_attrs = attributes_to_dict(resource.get("attributes"))
        for ss in rs.get("scopeSpans") or []:
            scope = ss.get("scope") or {}
            scope_name = scope.get("name")
            scope_version = scope.get("version")
            for span in ss.get("spans") or []:
                yield RawSpan(
                    span=span,
                    resource_attributes=resource_attrs,
                    scope_name=scope_name,
                    scope_version=scope_version,
                    source_file=source_file,
                )
