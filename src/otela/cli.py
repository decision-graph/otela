"""Command-line interface for otela.

Subcommands:

    otela totables  INPUT OUTPUT_DIR [--format parquet|csv|arrow|json|jsonl] [--spec at/v1]
    otela torecords INPUT OUTPUT_DIR [--format json|jsonl] [--spec at/v1]

`totables` writes one file per table (spans, messages, documents, links,
traces). Parquet streams to disk with bounded memory; the other formats
materialize the full tableset.

`torecords` writes a single file (`traces.json` or `traces.jsonl`) where
each record is a trace with all of its spans nested inside, and
messages/documents/links nested under each span. JSONL is preferable for
larger datasets — each line is a complete trace, so downstream consumers
can stream the file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.feather as pa_feather

from .api import load, to_dicts, to_parquet

TOTABLES_FORMATS = ("parquet", "csv", "arrow", "json", "jsonl")
TOTABLES_DEFAULT = "parquet"

TORECORDS_FORMATS = ("json", "jsonl")
TORECORDS_DEFAULT = "json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "totables":
        return _cmd_totables(args)
    if args.command == "torecords":
        return _cmd_torecords(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="otela",
        description="OpenTelemetry analytics & data formatting for agent traces.",
    )
    sub = parser.add_subparsers(dest="command")

    totables = sub.add_parser(
        "totables",
        help="Convert OTLP/JSON traces into tabular files (one per table).",
    )
    totables.add_argument("input", help="OTLP/JSON file or directory of files.")
    totables.add_argument("output_dir", help="Output directory.")
    totables.add_argument(
        "--format",
        choices=TOTABLES_FORMATS,
        default=TOTABLES_DEFAULT,
        help=f"Output format (default: {TOTABLES_DEFAULT}).",
    )
    totables.add_argument(
        "--spec",
        default="at/v1",
        help="Spec/version to apply (default: at/v1).",
    )
    totables.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Spans per parquet flush (parquet only). Default: 10000.",
    )
    totables.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec (parquet only). Default: zstd.",
    )

    torecords = sub.add_parser(
        "torecords",
        help="Convert OTLP/JSON traces into nested records (one per trace).",
    )
    torecords.add_argument("input", help="OTLP/JSON file or directory of files.")
    torecords.add_argument("output_dir", help="Output directory.")
    torecords.add_argument(
        "--format",
        choices=TORECORDS_FORMATS,
        default=TORECORDS_DEFAULT,
        help=f"Output format (default: {TORECORDS_DEFAULT}).",
    )
    torecords.add_argument(
        "--spec",
        default="at/v1",
        help="Spec/version to apply (default: at/v1).",
    )

    return parser


def _cmd_totables(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if not in_path.exists():
        print(f"error: input path does not exist: {in_path}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == "parquet":
        paths = to_parquet(
            in_path,
            out_dir,
            spec=args.spec,
            batch_size=args.batch_size,
            compression=args.compression,
        )
        for name, p in paths.items():
            print(f"{name}: {p}")
        return 0

    # Non-streaming formats: materialize once, write per table.
    tables = load(in_path, spec=args.spec)
    for name, table in tables.items():
        out_path = out_dir / f"{name}.{_extension_for(fmt)}"
        _write_table(table, out_path, fmt)
        print(f"{name}: {out_path}")
    return 0


def _cmd_torecords(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if not in_path.exists():
        print(f"error: input path does not exist: {in_path}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load(in_path, spec=args.spec)
    records = to_dicts(tables)

    fmt = args.format
    out_path = out_dir / f"traces.{fmt}"
    with out_path.open("wb") as f:
        if fmt == "jsonl":
            for r in records:
                f.write(orjson.dumps(r))
                f.write(b"\n")
        else:  # json
            f.write(orjson.dumps(records))
    print(f"traces: {out_path} ({len(records)} records)")
    return 0


def _extension_for(fmt: str) -> str:
    if fmt == "arrow":
        return "arrow"
    return fmt  # csv, json, jsonl


def _write_table(table: pa.Table, path: Path, fmt: str) -> None:
    if fmt == "csv":
        pa_csv.write_csv(table, path)
        return
    if fmt == "arrow":
        pa_feather.write_feather(table, path)
        return
    if fmt in ("json", "jsonl"):
        _write_json(table, path, lines=(fmt == "jsonl"))
        return
    raise ValueError(f"unsupported format: {fmt}")


def _write_json(table: pa.Table, path: Path, *, lines: bool) -> None:
    rows = table.to_pylist()
    with path.open("wb") as f:
        if lines:
            for row in rows:
                f.write(orjson.dumps(row))
                f.write(b"\n")
        else:
            f.write(orjson.dumps(rows))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
