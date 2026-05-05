"""Unit tests for OTLP/JSON helpers."""

from otela.otlp import (
    any_value,
    attributes_to_dict,
    normalize_status_code,
    parent_span_id_or_none,
    parse_nano,
)


def test_any_value_decodes_scalars():
    assert any_value({"stringValue": "x"}) == "x"
    assert any_value({"intValue": "42"}) == 42  # int64-as-string is canonical
    assert any_value({"intValue": 42}) == 42
    assert any_value({"doubleValue": 1.5}) == 1.5
    assert any_value({"boolValue": True}) is True
    assert any_value(None) is None
    assert any_value({}) is None


def test_any_value_decodes_arrays_and_kvlists():
    arr = {"arrayValue": {"values": [{"stringValue": "a"}, {"intValue": "1"}]}}
    assert any_value(arr) == ["a", 1]

    kv = {"kvlistValue": {"values": [{"key": "k", "value": {"stringValue": "v"}}]}}
    assert any_value(kv) == {"k": "v"}


def test_attributes_to_dict_skips_keyless_entries():
    attrs = [
        {"key": "a", "value": {"stringValue": "x"}},
        {"key": "", "value": {"stringValue": "y"}},  # skipped
        {"value": {"stringValue": "z"}},  # skipped
    ]
    assert attributes_to_dict(attrs) == {"a": "x"}
    assert attributes_to_dict(None) == {}


def test_parse_nano_handles_str_and_int():
    assert parse_nano("1714838400000000000") == 1714838400000000000
    assert parse_nano(1714838400000000000) == 1714838400000000000
    assert parse_nano("") is None
    assert parse_nano(None) is None
    assert parse_nano("not-a-number") is None


def test_normalize_status_code_canonicalizes():
    assert normalize_status_code("STATUS_CODE_OK") == "OK"
    assert normalize_status_code("STATUS_CODE_ERROR") == "ERROR"
    assert normalize_status_code("STATUS_CODE_UNSET") == "UNSET"
    assert normalize_status_code("OK") == "OK"
    assert normalize_status_code(None) == "UNSET"
    assert normalize_status_code("") == "UNSET"


def test_parent_span_id_empty_becomes_none():
    assert parent_span_id_or_none("") is None
    assert parent_span_id_or_none(None) is None
    assert parent_span_id_or_none("abc") == "abc"
