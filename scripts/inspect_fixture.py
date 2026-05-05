"""Print a summary of any otela-readable trace file.

Useful as a smoke test after generating a new fixture, and as a quick
"what does this look like" tool when triaging real production traces.

Usage
-----
    uv run python scripts/inspect_fixture.py path/to/trace.json
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import otela


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: inspect_fixture.py <path>", file=sys.stderr)
        return 2

    path = Path(args[0])
    if not path.exists():
        print(f"error: not found: {path}", file=sys.stderr)
        return 1

    tables = otela.load(path)
    spans = tables["spans"]
    traces = tables["traces"]

    kinds = Counter(spans.column("kind").to_pylist())
    conventions = Counter(spans.column("convention").to_pylist())
    statuses = Counter(spans.column("status_code").to_pylist())

    print(f"file: {path}")
    print(f"traces:    {traces.num_rows}")
    print(f"spans:     {spans.num_rows}")
    print(f"messages:  {tables['messages'].num_rows}")
    print(f"documents: {tables['documents'].num_rows}")
    print(f"links:     {tables['links'].num_rows}")
    print()
    print("kinds:        " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))
    print("conventions:  " + ", ".join(f"{k}={v}" for k, v in conventions.most_common()))
    print("statuses:     " + ", ".join(f"{k}={v}" for k, v in statuses.most_common()))
    print()

    services = set(spans.column("service_name").to_pylist())
    models = sorted(m for m in set(spans.column("model_name").to_pylist()) if m)
    tools = sorted(t for t in set(spans.column("tool_name").to_pylist()) if t)
    agents = sorted(a for a in set(spans.column("agent_name").to_pylist()) if a)
    print(f"services: {sorted(services - {None})}")
    print(f"models:   {models}")
    print(f"tools:    {tools}")
    print(f"agents:   {agents}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
