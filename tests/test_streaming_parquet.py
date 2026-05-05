"""Streaming parquet writer tests."""

from pathlib import Path

import pyarrow.parquet as pq

import otela
from otela.schemas import (
    ALL_TABLE_NAMES,
    DOCUMENTS_SCHEMA,
    LINKS_SCHEMA,
    MESSAGES_SCHEMA,
    SPANS_SCHEMA,
    TRACES_SCHEMA,
)


def test_to_parquet_writes_all_tables(fixtures_dir, tmp_path: Path):
    paths = otela.to_parquet(fixtures_dir, tmp_path)
    assert set(paths) == set(ALL_TABLE_NAMES)
    for name in ALL_TABLE_NAMES:
        assert paths[name].exists()


def test_parquet_schemas_round_trip(fixtures_dir, tmp_path: Path):
    paths = otela.to_parquet(fixtures_dir, tmp_path)
    expected = {
        "spans": SPANS_SCHEMA,
        "messages": MESSAGES_SCHEMA,
        "documents": DOCUMENTS_SCHEMA,
        "links": LINKS_SCHEMA,
        "traces": TRACES_SCHEMA,
    }
    for name, schema in expected.items():
        t = pq.read_table(paths[name])
        assert t.schema.equals(schema), f"{name} schema drifted"


def test_parquet_row_counts_match_load(fixtures_dir, tmp_path: Path):
    in_memory = otela.load(fixtures_dir)
    paths = otela.to_parquet(fixtures_dir, tmp_path)
    for name, expected_table in in_memory.items():
        on_disk = pq.read_table(paths[name])
        assert on_disk.num_rows == expected_table.num_rows


def test_batch_size_creates_multiple_row_groups(fixtures_dir, tmp_path: Path):
    # 13 spans total; batch_size=3 should produce 5 row groups in spans.
    paths = otela.to_parquet(fixtures_dir, tmp_path, batch_size=3)
    pf = pq.ParquetFile(paths["spans"])
    assert pf.num_row_groups == 5


def test_empty_links_table_writes_valid_parquet(fixtures_dir, tmp_path: Path):
    paths = otela.to_parquet(fixtures_dir, tmp_path)
    t = pq.read_table(paths["links"])
    assert t.num_rows == 0
    assert t.schema.equals(LINKS_SCHEMA)


def test_compression_codec_is_applied(fixtures_dir, tmp_path: Path):
    paths = otela.to_parquet(fixtures_dir, tmp_path, compression="snappy")
    md = pq.ParquetFile(paths["spans"]).metadata
    rg = md.row_group(0)
    # All columns should report snappy compression.
    codecs = {rg.column(i).compression for i in range(rg.num_columns)}
    assert codecs == {"SNAPPY"}


def test_invalid_spec_raises(tmp_path: Path, fixtures_dir):
    import pytest

    with pytest.raises(ValueError, match="unsupported spec"):
        otela.to_parquet(fixtures_dir, tmp_path, spec="bogus/v9")
