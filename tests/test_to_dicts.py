"""Tests for otela.to_dicts() — nested trace records."""

import otela


def test_returns_one_record_per_trace(fixtures_dir):
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    assert len(recs) == 2


def test_record_carries_trace_rollup_fields(fixtures_dir):
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    by_id = {r["trace_id"]: r for r in recs}
    research = by_id["4bf92f3577b34da6a3ce929d0e0e4736"]
    assert research["root_span_name"] == "research_agent.run"
    assert research["service_name"] == "research-agent"
    assert research["span_count"] == 6
    assert research["status"] == "ERROR"
    assert research["total_input_tokens"] == 245 + 1840


def test_spans_nested_under_record(fixtures_dir):
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    research = next(r for r in recs if r["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736")
    assert isinstance(research["spans"], list)
    assert len(research["spans"]) == 6
    span_names = {s["name"] for s in research["spans"]}
    assert "planner.chat" in span_names
    assert "retrieve_arxiv" in span_names


def test_messages_nested_under_span(fixtures_dir):
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    research = next(r for r in recs if r["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736")
    planner = next(s for s in research["spans"] if s["name"] == "planner.chat")
    assert len(planner["messages"]) == 3
    assert planner["messages"][0]["role"] == "system"
    assert planner["messages"][2]["direction"] == "output"


def test_documents_nested_under_span(fixtures_dir):
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    research = next(r for r in recs if r["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736")
    web = next(s for s in research["spans"] if s["name"] == "retrieve_web")
    assert len(web["documents"]) == 2
    assert web["documents"][0]["score"] == 0.94


def test_redundant_fk_keys_stripped(fixtures_dir):
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    research = recs[0]
    # The trace record itself keeps trace_id (it's the primary key).
    assert "trace_id" in research
    # Spans nested under a trace omit trace_id (implied by nesting).
    span = research["spans"][0]
    assert "trace_id" not in span
    # raw_attributes_json gets decoded into raw_attributes.
    assert "raw_attributes_json" not in span
    assert "raw_attributes" in span
    # Messages omit trace_id, span_id, spec, spec_version.
    if research["spans"][1]["messages"]:
        msg = research["spans"][1]["messages"][0]
        assert "trace_id" not in msg
        assert "span_id" not in msg
        assert "spec" not in msg


def test_raw_attributes_decoded(fixtures_dir):
    """Any span with non-null raw_attributes_json must produce a dict in the
    nested record."""
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    for rec in recs:
        for span in rec["spans"]:
            if span["raw_attributes"] is not None:
                assert isinstance(span["raw_attributes"], dict)


def test_links_field_present_even_when_empty(fixtures_dir):
    """The fixtures have no span links, but every span dict should still
    expose a `links: []` slot so consumers don't have to defensively check."""
    t = otela.load(fixtures_dir)
    recs = otela.to_dicts(t)
    for rec in recs:
        for span in rec["spans"]:
            assert "links" in span
            assert span["links"] == []
