"""Tests for session_id detection, session_turn ordering, and the sessions rollup."""

from __future__ import annotations

import otela
from otela.builder import TableBuilder
from otela.normalize import (
    CONV_GENERIC,
    KIND_LLM,
    NormalizedSpan,
    _detect_session_id,
)
from otela.schemas import SESSIONS_SCHEMA

# ---------------------------------------------------------------------------
# Detection precedence
# ---------------------------------------------------------------------------


def test_detect_session_id_otel_genai():
    consumed: set[str] = set()
    sid = _detect_session_id({"gen_ai.conversation.id": "conv-1"}, consumed)
    assert sid == "conv-1"
    assert "gen_ai.conversation.id" in consumed


def test_detect_session_id_openinference_session_id():
    consumed: set[str] = set()
    sid = _detect_session_id({"session.id": "sess-1"}, consumed)
    assert sid == "sess-1"
    assert "session.id" in consumed


def test_detect_session_id_adk_vertex():
    consumed: set[str] = set()
    sid = _detect_session_id({"gcp.vertex.agent.session_id": "adk-99"}, consumed)
    assert sid == "adk-99"
    assert "gcp.vertex.agent.session_id" in consumed


def test_detect_session_id_vercel():
    consumed: set[str] = set()
    sid = _detect_session_id({"ai.telemetry.metadata.sessionId": "vercel-x"}, consumed)
    assert sid == "vercel-x"


def test_detect_session_id_mlflow():
    consumed: set[str] = set()
    sid = _detect_session_id({"mlflow.trace.session": "mlf-session"}, consumed)
    assert sid == "mlf-session"


def test_detect_session_id_traceloop():
    consumed: set[str] = set()
    sid = _detect_session_id(
        {"traceloop.association.properties.session_id": "tl-7"}, consumed
    )
    assert sid == "tl-7"


def test_detect_session_id_precedence_otel_over_openinference():
    """When both keys are present, OTel GenAI's conversation.id wins —
    it's the official upstream spec."""
    consumed: set[str] = set()
    sid = _detect_session_id(
        {"gen_ai.conversation.id": "otel", "session.id": "oi"}, consumed
    )
    assert sid == "otel"


def test_detect_session_id_none_when_absent():
    consumed: set[str] = set()
    sid = _detect_session_id({"unrelated.key": "x"}, consumed)
    assert sid is None
    assert consumed == set()


def test_detect_session_id_empty_string_is_none():
    """Empty strings shouldn't masquerade as a real session id, but the
    matched key is still consumed so it doesn't pollute raw_attributes."""
    consumed: set[str] = set()
    sid = _detect_session_id({"session.id": ""}, consumed)
    assert sid is None
    assert "session.id" in consumed


# ---------------------------------------------------------------------------
# Promotion: source attribute removed from raw_attributes_json
# ---------------------------------------------------------------------------


def test_session_id_not_in_raw_attributes(openinference_path):
    import orjson

    t = otela.load(openinference_path)
    spans = t["spans"].to_pandas()
    for raw in spans["raw_attributes_json"].dropna():
        decoded = orjson.loads(raw)
        # session.id was promoted to its own column; it should not also
        # land in the leftovers bag.
        assert "session.id" not in decoded


def test_genai_conversation_id_not_in_raw_attributes(otel_genai_path):
    import orjson

    t = otela.load(otel_genai_path)
    spans = t["spans"].to_pandas()
    for raw in spans["raw_attributes_json"].dropna():
        decoded = orjson.loads(raw)
        assert "gen_ai.conversation.id" not in decoded


# ---------------------------------------------------------------------------
# Span and trace propagation from real fixtures
# ---------------------------------------------------------------------------


def test_session_id_on_every_span(openinference_path):
    """Every span in the OpenInference fixture carries session.id, so the
    spans table column should be uniformly populated."""
    t = otela.load(openinference_path)
    sids = set(t["spans"].column("session_id").to_pylist())
    assert sids == {"research-session-001"}


def test_session_id_on_traces_rollup(fixtures_dir):
    t = otela.load(fixtures_dir)
    df = t["traces"].to_pandas().set_index("trace_id")
    assert df.loc["4bf92f3577b34da6a3ce929d0e0e4736", "session_id"] == "research-session-001"
    assert df.loc["5cf03f4688c45eb7b4df939e1f1f5847", "session_id"] == "refund-conv-7a3f"


def test_session_turn_zero_for_single_trace_session(fixtures_dir):
    t = otela.load(fixtures_dir)
    df = t["traces"].to_pandas().set_index("trace_id")
    # Each fixture is one trace per session, so session_turn = 0 for both.
    assert df.loc["4bf92f3577b34da6a3ce929d0e0e4736", "session_turn"] == 0
    assert df.loc["5cf03f4688c45eb7b4df939e1f1f5847", "session_turn"] == 0


# ---------------------------------------------------------------------------
# Sessions rollup
# ---------------------------------------------------------------------------


def test_sessions_table_schema(fixtures_dir):
    t = otela.load(fixtures_dir)
    assert t["sessions"].schema.equals(SESSIONS_SCHEMA)


def test_sessions_rollup_one_row_per_session(fixtures_dir):
    t = otela.load(fixtures_dir)
    sessions = t["sessions"].to_pandas().set_index("session_id")
    assert set(sessions.index) == {"research-session-001", "refund-conv-7a3f"}


def test_sessions_rollup_aggregates(fixtures_dir):
    t = otela.load(fixtures_dir)
    sessions = t["sessions"].to_pandas().set_index("session_id")
    research = sessions.loc["research-session-001"]
    assert research["trace_count"] == 1
    assert research["span_count"] == 6
    assert research["error_count"] == 1
    assert research["total_input_tokens"] == 245 + 1840
    assert research["total_output_tokens"] == 89 + 421

    refund = sessions.loc["refund-conv-7a3f"]
    assert refund["trace_count"] == 1
    assert refund["span_count"] == 7
    assert refund["error_count"] == 0
    assert refund["total_input_tokens"] == 312 + 478


# ---------------------------------------------------------------------------
# session_turn ordering (synthetic, full control over start times)
# ---------------------------------------------------------------------------


def _mk_span(
    trace_id: str,
    span_id: str,
    *,
    start_ns: int,
    session_id: str | None,
) -> NormalizedSpan:
    """Minimal NormalizedSpan for builder-driven tests."""
    return NormalizedSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        name="root",
        kind=KIND_LLM,
        convention=CONV_GENERIC,
        status_code="OK",
        status_message=None,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=start_ns + 1000,
        duration_ns=1000,
        service_name="svc",
        scope_name=None,
        scope_version=None,
        session_id=session_id,
        model_name=None,
        tool_name=None,
        agent_name=None,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        io_format="text",
        input_text=None,
        output_text=None,
        raw_attributes_json=None,
    )


def test_session_turn_orders_by_start_time():
    builder = TableBuilder()
    # Insert traces out of chronological order to verify sort behavior.
    builder.add(_mk_span("t-c", "s-c", start_ns=300, session_id="sess-A"))
    builder.add(_mk_span("t-a", "s-a", start_ns=100, session_id="sess-A"))
    builder.add(_mk_span("t-b", "s-b", start_ns=200, session_id="sess-A"))

    traces = builder.build_traces().to_pandas().set_index("trace_id")
    assert traces.loc["t-a", "session_turn"] == 0
    assert traces.loc["t-b", "session_turn"] == 1
    assert traces.loc["t-c", "session_turn"] == 2


def test_session_turn_per_session_independent():
    """Two sessions each get their own 0-indexed sequence."""
    builder = TableBuilder()
    builder.add(_mk_span("t-1", "s-1", start_ns=100, session_id="A"))
    builder.add(_mk_span("t-2", "s-2", start_ns=200, session_id="A"))
    builder.add(_mk_span("t-3", "s-3", start_ns=150, session_id="B"))
    builder.add(_mk_span("t-4", "s-4", start_ns=250, session_id="B"))

    traces = builder.build_traces().to_pandas().set_index("trace_id")
    assert traces.loc["t-1", "session_turn"] == 0
    assert traces.loc["t-2", "session_turn"] == 1
    assert traces.loc["t-3", "session_turn"] == 0
    assert traces.loc["t-4", "session_turn"] == 1


def test_session_turn_tiebreak_uses_trace_id():
    """When two traces start at the same nanosecond, lex-order trace_id breaks
    the tie deterministically."""
    builder = TableBuilder()
    builder.add(_mk_span("t-zzz", "s-z", start_ns=100, session_id="X"))
    builder.add(_mk_span("t-aaa", "s-a", start_ns=100, session_id="X"))
    builder.add(_mk_span("t-mmm", "s-m", start_ns=100, session_id="X"))

    traces = builder.build_traces().to_pandas().set_index("trace_id")
    assert traces.loc["t-aaa", "session_turn"] == 0
    assert traces.loc["t-mmm", "session_turn"] == 1
    assert traces.loc["t-zzz", "session_turn"] == 2


def test_session_turn_null_when_no_session():
    """Traces with no session_id get a null session_turn and are excluded
    from the sessions rollup."""
    import pandas as pd

    builder = TableBuilder()
    builder.add(_mk_span("t-1", "s-1", start_ns=100, session_id=None))
    builder.add(_mk_span("t-2", "s-2", start_ns=200, session_id="A"))

    traces = builder.build_traces().to_pandas().set_index("trace_id")
    assert pd.isna(traces.loc["t-1", "session_id"])
    assert pd.isna(traces.loc["t-1", "session_turn"])
    assert traces.loc["t-2", "session_turn"] == 0

    sessions = builder.build_sessions().to_pandas()
    assert list(sessions["session_id"]) == ["A"]


def test_session_first_non_null_wins_within_trace():
    """If only some spans in a trace carry session_id, the trace inherits the
    first non-null one observed (deterministic; spec says all should match)."""
    builder = TableBuilder()
    # Root has no session_id; a child does.
    builder.add(_mk_span("t-1", "s-root", start_ns=100, session_id=None))
    builder.add(_mk_span("t-1", "s-child", start_ns=110, session_id="late-tag"))
    builder.add(_mk_span("t-1", "s-grand", start_ns=120, session_id="other-tag"))

    traces = builder.build_traces().to_pandas().set_index("trace_id")
    assert traces.loc["t-1", "session_id"] == "late-tag"


# ---------------------------------------------------------------------------
# Streaming parquet preserves the rollup
# ---------------------------------------------------------------------------


def test_streaming_parquet_emits_sessions(fixtures_dir, tmp_path):
    import pyarrow.parquet as pq

    paths = otela.to_parquet(fixtures_dir, tmp_path, batch_size=2)
    assert paths["sessions"].exists()
    on_disk = pq.read_table(paths["sessions"])
    assert on_disk.schema.equals(SESSIONS_SCHEMA)
    in_memory = otela.load(fixtures_dir)["sessions"]
    a = in_memory.sort_by("session_id").to_pylist()
    b = on_disk.sort_by("session_id").to_pylist()
    assert a == b
