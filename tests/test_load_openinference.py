"""End-to-end tests against the OpenInference fixture."""

import otela


def test_loads_all_spans(openinference_path):
    t = otela.load(openinference_path)
    assert t["spans"].num_rows == 6


def test_kinds_detected(openinference_path):
    t = otela.load(openinference_path)
    by_name = dict(
        zip(
            t["spans"].column("name").to_pylist(),
            t["spans"].column("kind").to_pylist(),
            strict=True,
        )
    )
    assert by_name["research_agent.run"] == "AGENT"
    assert by_name["planner.chat"] == "LLM"
    assert by_name["retrieve_web"] == "RETRIEVER"
    assert by_name["synthesize.chat"] == "LLM"


def test_convention_set(openinference_path):
    t = otela.load(openinference_path)
    conventions = set(t["spans"].column("convention").to_pylist())
    assert conventions == {"openinference"}


def test_token_counts_extracted(openinference_path):
    t = otela.load(openinference_path)
    spans = t["spans"].to_pandas()
    planner = spans[spans["name"] == "planner.chat"].iloc[0]
    assert planner["input_tokens"] == 245
    assert planner["output_tokens"] == 89
    assert planner["total_tokens"] == 334


def test_messages_extracted_with_direction(openinference_path):
    t = otela.load(openinference_path)
    msgs = t["messages"].to_pandas()
    planner_msgs = msgs[msgs["span_id"] == "00f067aa0ba902b8"].sort_values("position")
    assert list(planner_msgs["direction"]) == ["input", "input", "output"]
    assert list(planner_msgs["role"]) == ["system", "user", "assistant"]


def test_documents_extracted_with_score(openinference_path):
    t = otela.load(openinference_path)
    docs = t["documents"].to_pandas()
    web = docs[docs["span_id"] == "00f067aa0ba902b9"].sort_values("position")
    assert len(web) == 2
    assert web.iloc[0]["score"] == 0.94
    assert web.iloc[0]["document_id"] == "hai.stanford.edu/ai-index-2026"


def test_resource_service_name_propagated(openinference_path):
    t = otela.load(openinference_path)
    services = set(t["spans"].column("service_name").to_pylist())
    assert services == {"research-agent"}


def test_scope_propagated(openinference_path):
    t = otela.load(openinference_path)
    scopes = set(
        zip(
            t["spans"].column("scope_name").to_pylist(),
            t["spans"].column("scope_version").to_pylist(),
            strict=True,
        )
    )
    assert scopes == {("openinference.instrumentation.langchain", "0.1.0")}


def test_io_format_classification(openinference_path):
    t = otela.load(openinference_path)
    spans = t["spans"].to_pandas().set_index("name")
    assert spans.loc["research_agent.run", "io_format"] == "text"
    assert spans.loc["planner.chat", "io_format"] == "text"
    assert spans.loc["retrieve_web", "io_format"] == "retrieval"


def test_error_status_preserved(openinference_path):
    t = otela.load(openinference_path)
    spans = t["spans"].to_pandas()
    arxiv = spans[spans["name"] == "retrieve_arxiv"].iloc[0]
    assert arxiv["status_code"] == "ERROR"
    assert arxiv["status_message"] == "rate limit exceeded"


def test_parent_span_id_empty_string_normalized(openinference_path):
    t = otela.load(openinference_path)
    spans = t["spans"].to_pandas()
    root = spans[spans["name"] == "research_agent.run"].iloc[0]
    assert root["parent_span_id"] is None or (root["parent_span_id"] != root["parent_span_id"])  # NaN
