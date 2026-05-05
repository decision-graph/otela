"""Tests for otela.to_tensors() — PyTorch tensor view."""

import pytest

import otela
from otela.tensors import NULL_CATEGORY_CODE, TensorTable

torch = pytest.importorskip("torch")


def test_returns_spans_and_traces_by_default(fixtures_dir):
    t = otela.load(fixtures_dir)
    tensors = otela.to_tensors(t)
    assert set(tensors) == {"spans", "traces"}
    assert isinstance(tensors["spans"], TensorTable)


def test_n_rows_matches_table(fixtures_dir):
    t = otela.load(fixtures_dir)
    tensors = otela.to_tensors(t)
    assert tensors["spans"].n_rows == 13
    assert tensors["traces"].n_rows == 2


def test_numeric_dtype_is_float32_by_default(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    for tensor in spans.numeric.values():
        assert tensor.dtype == torch.float32


def test_numeric_dtype_override_to_float64(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t, dtype=torch.float64)["spans"]
    for tensor in spans.numeric.values():
        assert tensor.dtype == torch.float64


def test_nulls_in_numeric_become_nan(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    # input_tokens is null on AGENT/TOOL spans; expect NaNs.
    nan_count = int(torch.isnan(spans.numeric["input_tokens"]).sum().item())
    assert nan_count > 0


def test_categorical_dtype_is_int64(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    for tensor in spans.categorical.values():
        assert tensor.dtype == torch.int64


def test_categorical_nulls_become_sentinel(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    # tool_name is null on LLM/AGENT spans.
    null_count = int((spans.categorical["tool_name"] == NULL_CATEGORY_CODE).sum().item())
    assert null_count > 0


def test_vocab_round_trip(fixtures_dir):
    """Code i in the categorical tensor maps to vocab[i] in the original column."""
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    codes = spans.categorical["kind"].tolist()
    vocab = spans.vocab["kind"]
    decoded = [vocab[c] if c != NULL_CATEGORY_CODE else None for c in codes]
    expected = t["spans"].column("kind").to_pylist()
    assert decoded == expected


def test_identity_columns_kept_as_lists(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    assert "trace_id" in spans.ids
    assert "span_id" in spans.ids
    assert "parent_span_id" in spans.ids
    # Same row count as the source table.
    assert len(spans.ids["span_id"]) == 13
    # Verify no encoding occurred.
    assert isinstance(spans.ids["span_id"][0], str)


def test_large_string_columns_skipped(fixtures_dir):
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    assert "input_text" in spans.skipped
    assert "output_text" in spans.skipped
    assert "raw_attributes_json" in spans.skipped


def test_spec_columns_dropped(fixtures_dir):
    """spec/spec_version are constants — not useful as features."""
    t = otela.load(fixtures_dir)
    spans = otela.to_tensors(t)["spans"]
    all_cols = set(spans.numeric) | set(spans.categorical) | set(spans.ids) | set(spans.skipped)
    assert "spec" not in all_cols
    assert "spec_version" not in all_cols


def test_traces_table_tensors(fixtures_dir):
    t = otela.load(fixtures_dir)
    traces = otela.to_tensors(t)["traces"]
    # span_count and error_count have no nulls (always int).
    assert torch.isnan(traces.numeric["span_count"]).sum().item() == 0
    # total_tokens has one NaN: the OTel-genai trace doesn't carry total.
    assert int(torch.isnan(traces.numeric["total_tokens"]).sum().item()) == 1
    # Status worst-of categorical with two values.
    assert set(traces.vocab["status"]) <= {"OK", "ERROR", "UNSET"}


def test_can_select_more_tables(fixtures_dir):
    t = otela.load(fixtures_dir)
    tensors = otela.to_tensors(
        t,
        table_names=("spans", "traces", "messages", "documents", "links"),
    )
    assert set(tensors) == {"spans", "traces", "messages", "documents", "links"}
    assert tensors["messages"].n_rows == 15
    # Identity columns stripped from messages.
    assert "trace_id" in tensors["messages"].ids
    assert "span_id" in tensors["messages"].ids


def test_unknown_table_silently_skipped(fixtures_dir):
    t = otela.load(fixtures_dir)
    tensors = otela.to_tensors(t, table_names=("spans", "nope"))
    assert set(tensors) == {"spans"}


def test_all_tensors_share_n_rows(fixtures_dir):
    """Every numeric and categorical tensor in a TensorTable has shape (n_rows,)."""
    t = otela.load(fixtures_dir)
    for bundle in otela.to_tensors(t).values():
        for tensor in bundle.numeric.values():
            assert tensor.shape == (bundle.n_rows,)
        for tensor in bundle.categorical.values():
            assert tensor.shape == (bundle.n_rows,)
