"""End-to-end tests against the OTel GenAI fixture (events-based I/O)."""

import otela


def test_loads_all_spans(otel_genai_path):
    t = otela.load(otel_genai_path)
    assert t["spans"].num_rows == 7


def test_convention_detected(otel_genai_path):
    t = otela.load(otel_genai_path)
    assert set(t["spans"].column("convention").to_pylist()) == {"otel_genai"}


def test_kinds_from_operation_name(otel_genai_path):
    t = otela.load(otel_genai_path)
    spans = t["spans"].to_pandas()
    assert spans[spans["name"] == "invoke_agent refund_agent"]["kind"].iloc[0] == "AGENT"
    assert (spans[spans["name"].str.startswith("chat ")]["kind"] == "LLM").all()
    assert (spans[spans["name"].str.startswith("execute_tool ")]["kind"] == "TOOL").all()


def test_messages_from_span_events(otel_genai_path):
    t = otela.load(otel_genai_path)
    msgs = t["messages"].to_pandas()
    chat1 = msgs[msgs["span_id"] == "11e178bb1cb013c9"].sort_values("position")
    assert list(chat1["role"]) == ["system", "user", "assistant"]
    assert list(chat1["direction"]) == ["input", "input", "output"]


def test_tool_messages_carry_call_id(otel_genai_path):
    t = otela.load(otel_genai_path)
    msgs = t["messages"].to_pandas()
    lookup = msgs[msgs["span_id"] == "11e178bb1cb013ca"]
    assert lookup.iloc[0]["role"] == "tool"
    assert lookup.iloc[0]["tool_call_id"] == "call_a1b2c3"


def test_token_usage_extracted(otel_genai_path):
    t = otela.load(otel_genai_path)
    spans = t["spans"].to_pandas()
    chat = spans[spans["name"] == "chat gemini-2.0-flash"].iloc[0]
    assert chat["input_tokens"] == 312
    assert chat["output_tokens"] == 47


def test_tool_arguments_become_input_text(otel_genai_path):
    t = otela.load(otel_genai_path)
    spans = t["spans"].to_pandas()
    lookup = spans[spans["name"] == "execute_tool lookup_order"].iloc[0]
    assert "84291" in (lookup["input_text"] or "")
    assert lookup["io_format"] == "tool_call"


def test_agent_root_uses_input_value_fallback(otel_genai_path):
    t = otela.load(otel_genai_path)
    spans = t["spans"].to_pandas()
    root = spans[spans["name"] == "invoke_agent refund_agent"].iloc[0]
    assert root["agent_name"] == "refund_agent"
    assert "refund" in root["input_text"].lower()
    assert root["io_format"] == "text"
