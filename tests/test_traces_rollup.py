"""Tests for the traces rollup table."""

from pathlib import Path

import pyarrow.parquet as pq

import otela
from otela.schemas import TRACES_SCHEMA


def test_traces_table_returned_by_load(fixtures_dir):
    t = otela.load(fixtures_dir)
    assert "traces" in t
    assert t["traces"].schema.equals(TRACES_SCHEMA)
    assert t["traces"].num_rows == 2  # one per fixture file


def test_trace_identity_uses_root_span(fixtures_dir):
    t = otela.load(fixtures_dir)
    df = t["traces"].to_pandas().set_index("trace_id")
    research = df.loc["4bf92f3577b34da6a3ce929d0e0e4736"]
    assert research["root_span_id"] == "00f067aa0ba902b7"
    assert research["root_span_name"] == "research_agent.run"
    assert research["service_name"] == "research-agent"

    refund = df.loc["5cf03f4688c45eb7b4df939e1f1f5847"]
    assert refund["root_span_id"] == "11e178bb1cb013c8"
    assert refund["root_span_name"] == "invoke_agent refund_agent"
    assert refund["service_name"] == "refund-agent"


def test_timing_min_max(fixtures_dir):
    t = otela.load(fixtures_dir)
    df = t["traces"].to_pandas().set_index("trace_id")
    research = df.loc["4bf92f3577b34da6a3ce929d0e0e4736"]
    assert research["start_time_unix_nano"] == 1714838400000000000
    assert research["end_time_unix_nano"] == 1714838408500000000
    assert research["duration_ns"] == 8_500_000_000


def test_span_count_and_error_count(fixtures_dir):
    t = otela.load(fixtures_dir)
    df = t["traces"].to_pandas().set_index("trace_id")
    research = df.loc["4bf92f3577b34da6a3ce929d0e0e4736"]
    refund = df.loc["5cf03f4688c45eb7b4df939e1f1f5847"]
    assert research["span_count"] == 6
    assert research["error_count"] == 1  # retrieve_arxiv ERROR
    assert refund["span_count"] == 7
    assert refund["error_count"] == 0


def test_status_worst_of(fixtures_dir):
    t = otela.load(fixtures_dir)
    df = t["traces"].to_pandas().set_index("trace_id")
    assert df.loc["4bf92f3577b34da6a3ce929d0e0e4736", "status"] == "ERROR"
    assert df.loc["5cf03f4688c45eb7b4df939e1f1f5847", "status"] == "OK"


def test_token_sums_with_null_handling(fixtures_dir):
    t = otela.load(fixtures_dir)
    df = t["traces"].to_pandas().set_index("trace_id")
    research = df.loc["4bf92f3577b34da6a3ce929d0e0e4736"]
    refund = df.loc["5cf03f4688c45eb7b4df939e1f1f5847"]

    # Research trace: planner.chat (245/89/334) + synthesize.chat (1840/421/2261).
    assert research["total_input_tokens"] == 245 + 1840
    assert research["total_output_tokens"] == 89 + 421
    assert research["total_tokens"] == 334 + 2261

    # Refund trace: OTel GenAI sample reports input/output but never total.
    assert refund["total_input_tokens"] == 312 + 478
    assert refund["total_output_tokens"] == 47 + 92
    # No span carried gen_ai.usage.total_tokens, so the rollup should be NULL,
    # not an erroneous 0.
    assert refund["total_tokens"] is None or (
        refund["total_tokens"] != refund["total_tokens"]  # NaN
    )


def test_traces_written_to_parquet(fixtures_dir, tmp_path: Path):
    paths = otela.to_parquet(fixtures_dir, tmp_path)
    assert paths["traces"].exists()
    t = pq.read_table(paths["traces"])
    assert t.num_rows == 2
    assert t.schema.equals(TRACES_SCHEMA)


def test_streaming_parquet_traces_match_in_memory(fixtures_dir, tmp_path: Path):
    """Trace rollup must be identical whether produced via load() or to_parquet()
    with a small batch_size — the streaming flushes don't reset the trace
    accumulators."""
    in_memory = otela.load(fixtures_dir)["traces"]
    paths = otela.to_parquet(fixtures_dir, tmp_path, batch_size=2)
    on_disk = pq.read_table(paths["traces"])
    # Sort both by trace_id for comparison.
    a = in_memory.sort_by("trace_id").to_pylist()
    b = on_disk.sort_by("trace_id").to_pylist()
    assert a == b
