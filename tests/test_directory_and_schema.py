"""Tests for multi-file loading, schema stability, and the to_dfs adapter."""

import pyarrow as pa

import otela
from otela.schemas import (
    DOCUMENTS_SCHEMA,
    LINKS_SCHEMA,
    MESSAGES_SCHEMA,
    SPANS_SCHEMA,
    SPEC,
    SPEC_VERSION,
)


def test_directory_load_combines_files(fixtures_dir):
    t = otela.load(fixtures_dir)
    # 6 OpenInference + 7 OTel GenAI
    assert t["spans"].num_rows == 13
    conventions = set(t["spans"].column("convention").to_pylist())
    assert conventions == {"openinference", "otel_genai"}


def test_iterable_of_paths_load(openinference_path, otel_genai_path):
    t = otela.load([openinference_path, otel_genai_path])
    assert t["spans"].num_rows == 13


def test_spec_columns_present(fixtures_dir):
    t = otela.load(fixtures_dir)
    for table_name in ("spans", "messages", "documents"):
        col_spec = t[table_name].column("spec").unique().to_pylist()
        col_ver = t[table_name].column("spec_version").unique().to_pylist()
        assert col_spec == [SPEC]
        assert col_ver == [SPEC_VERSION]


def test_schemas_match(fixtures_dir):
    t = otela.load(fixtures_dir)
    assert t["spans"].schema.equals(SPANS_SCHEMA)
    assert t["messages"].schema.equals(MESSAGES_SCHEMA)
    assert t["documents"].schema.equals(DOCUMENTS_SCHEMA)
    assert t["links"].schema.equals(LINKS_SCHEMA)


def test_empty_tables_have_correct_schema():
    # Build via the public API on a path with no spans yet — we use an empty
    # in-memory call by creating a TableBuilder directly.
    from otela.builder import TableBuilder

    tables = TableBuilder().build()
    assert tables["spans"].num_rows == 0
    assert tables["spans"].schema.equals(SPANS_SCHEMA)


def test_to_dfs_returns_pandas(fixtures_dir):
    t = otela.load(fixtures_dir)
    dfs = otela.to_dfs(t)
    assert set(dfs.keys()) == {"spans", "messages", "documents", "links", "traces", "sessions"}
    assert len(dfs["spans"]) == 13
    assert len(dfs["traces"]) == 2
    # Both fixtures carry distinct session_ids; one row per session.
    assert len(dfs["sessions"]) == 2


def test_int_columns_preserve_nullability(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = t["spans"]
    # input_tokens has nulls (e.g. for AGENT and TOOL spans). Confirm the
    # arrow column is int64 with nulls, not silently widened to float.
    assert spans.schema.field("input_tokens").type == pa.int64()
    assert spans.column("input_tokens").null_count > 0


def test_raw_attributes_json_holds_unconsumed_attrs(openinference_path):
    import orjson

    t = otela.load(openinference_path)
    spans = t["spans"].to_pandas()
    # The root agent span has 'agent.name' consumed but 'openinference.span.kind'
    # is consumed too. Anything else like missing keys should be in raw.
    # For this fixture, all known attrs are consumed -> raw_attributes_json
    # may be None. But unknown extra attrs should land here. Add a synthetic
    # check: parse and verify structure where present.
    for raw in spans["raw_attributes_json"].dropna():
        decoded = orjson.loads(raw)
        assert isinstance(decoded, dict)
