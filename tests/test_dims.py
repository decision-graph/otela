"""Tests for the otela.dims() helper."""

import otela


def test_dims_returns_four_tables(fixtures_dir):
    t = otela.load(fixtures_dir)
    d = otela.dims(t)
    assert set(d.keys()) == {"tools", "agents", "models", "services"}


def test_dim_columns(fixtures_dir):
    t = otela.load(fixtures_dir)
    d = otela.dims(t)
    for table in d.values():
        assert table.column_names == ["spec", "spec_version", "name", "span_count", "trace_count"]


def test_models_dim(fixtures_dir):
    t = otela.load(fixtures_dir)
    models = otela.dims(t)["models"].to_pandas().set_index("name")
    assert set(models.index) == {"gpt-4o", "gemini-2.0-flash"}
    assert models.loc["gpt-4o", "span_count"] == 2
    assert models.loc["gpt-4o", "trace_count"] == 1
    assert models.loc["gemini-2.0-flash", "span_count"] == 2


def test_tools_dim(fixtures_dir):
    t = otela.load(fixtures_dir)
    tools = otela.dims(t)["tools"].to_pandas().set_index("name")
    assert "web_search" in tools.index
    assert "lookup_order" in tools.index
    assert tools.loc["web_search", "trace_count"] == 1


def test_services_dim(fixtures_dir):
    t = otela.load(fixtures_dir)
    services = otela.dims(t)["services"].to_pandas().set_index("name")
    assert services.loc["research-agent", "span_count"] == 6
    assert services.loc["refund-agent", "span_count"] == 7


def test_agents_dim(fixtures_dir):
    t = otela.load(fixtures_dir)
    agents = otela.dims(t)["agents"].to_pandas().set_index("name")
    assert set(agents.index) == {"research_agent", "refund_agent"}


def test_dims_skips_null_names(fixtures_dir):
    """Spans without a tool_name (e.g. LLM/AGENT spans) must not produce
    a row in the tools dim."""
    t = otela.load(fixtures_dir)
    tools = otela.dims(t)["tools"].to_pandas()
    assert tools["name"].notna().all()


def test_dims_on_empty_spans():
    import pyarrow as pa

    from otela.schemas import SPANS_SCHEMA

    empty = pa.Table.from_arrays(
        [pa.array([], type=f.type) for f in SPANS_SCHEMA], schema=SPANS_SCHEMA
    )
    d = otela.dims({"spans": empty})
    for table in d.values():
        assert table.num_rows == 0
        assert table.column_names == ["spec", "spec_version", "name", "span_count", "trace_count"]
