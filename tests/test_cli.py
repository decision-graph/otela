"""CLI tests."""

import json
from pathlib import Path

import pyarrow.feather as pa_feather
import pyarrow.parquet as pq
import pytest

from otela.cli import main


def test_totables_parquet_default(fixtures_dir, tmp_path: Path, capsys):
    rc = main(["totables", str(fixtures_dir), str(tmp_path)])
    assert rc == 0
    for name in ("spans", "messages", "documents", "links"):
        assert (tmp_path / f"{name}.parquet").exists()
    # CLI should print one line per table.
    out = capsys.readouterr().out
    assert "spans:" in out
    assert "links:" in out


def test_totables_csv(fixtures_dir, tmp_path: Path):
    import pyarrow.csv as pa_csv

    rc = main(["totables", str(fixtures_dir), str(tmp_path), "--format", "csv"])
    assert rc == 0
    spans_csv = tmp_path / "spans.csv"
    assert spans_csv.exists()
    # Parse properly — content fields may contain embedded newlines.
    table = pa_csv.read_csv(spans_csv)
    assert table.num_rows == 13
    assert table.column_names[:4] == ["spec", "spec_version", "trace_id", "span_id"]


def test_totables_arrow(fixtures_dir, tmp_path: Path):
    rc = main(["totables", str(fixtures_dir), str(tmp_path), "--format", "arrow"])
    assert rc == 0
    table = pa_feather.read_table(tmp_path / "spans.arrow")
    assert table.num_rows == 13


def test_totables_jsonl(fixtures_dir, tmp_path: Path):
    rc = main(["totables", str(fixtures_dir), str(tmp_path), "--format", "jsonl"])
    assert rc == 0
    spans_jsonl = (tmp_path / "spans.jsonl").read_text().splitlines()
    assert len(spans_jsonl) == 13
    parsed = [json.loads(line) for line in spans_jsonl]
    assert all(row["spec"] == "at" for row in parsed)


def test_totables_json(fixtures_dir, tmp_path: Path):
    rc = main(["totables", str(fixtures_dir), str(tmp_path), "--format", "json"])
    assert rc == 0
    rows = json.loads((tmp_path / "spans.json").read_text())
    assert isinstance(rows, list)
    assert len(rows) == 13


def test_totables_batch_size_applied(fixtures_dir, tmp_path: Path):
    rc = main(
        [
            "totables",
            str(fixtures_dir),
            str(tmp_path),
            "--format",
            "parquet",
            "--batch-size",
            "3",
        ]
    )
    assert rc == 0
    pf = pq.ParquetFile(tmp_path / "spans.parquet")
    assert pf.num_row_groups == 5


def test_totables_missing_input_returns_error(tmp_path: Path, capsys):
    rc = main(["totables", str(tmp_path / "does-not-exist"), str(tmp_path)])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_no_subcommand_prints_help(capsys):
    rc = main([])
    assert rc == 2
    assert "totables" in capsys.readouterr().out


def test_invalid_format_rejected(fixtures_dir, tmp_path: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "totables",
                str(fixtures_dir),
                str(tmp_path),
                "--format",
                "xml",
            ]
        )
