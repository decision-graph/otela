"""Generate a real LangGraph agent trace as an otela fixture.

Builds a minimal LangGraph React agent with one tool, instruments it
with OpenInference, runs a deterministic query, and writes the resulting
OTLP/JSON trace to `tests/fixtures/real/langgraph_research_agent.json`.

Usage
-----
    export OPENAI_API_KEY=...
    uv sync --group fixtures
    uv run python scripts/generate_fixtures.py langgraph

Determinism
-----------
`temperature=0` keeps the model output stable across runs, but trace IDs,
span IDs, and timestamps will of course change. Tests must not assert on
those — they should assert on shapes, kinds, conventions, and whether
identifiable fields (model name, tool name, token counts) were extracted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .otlp_file_exporter import OTLPFileExporter

DEFAULT_OUTPUT = Path("tests/fixtures/real/langgraph_research_agent.json")
DEFAULT_PROMPT = "What's the weather in San Francisco?"
DEFAULT_MODEL = "gpt-4o-mini"


def generate(
    output_path: Path = DEFAULT_OUTPUT,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Run a small LangGraph agent and write its trace to `output_path`."""
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY is not set. Export your key first:\n"
            "    export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Lazy imports — these only resolve when the script actually runs, so
    # `import scripts.fixtures` doesn't blow up in a default environment.
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from openinference.instrumentation.langchain import LangChainInstrumentor

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Configure tracing ---------------------------------------------
    resource = Resource.create(
        {
            "service.name": "langgraph-research-agent",
            "telemetry.sdk.language": "python",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPFileExporter(output_path)
    # SimpleSpanProcessor flushes synchronously on each span end — exactly
    # what we want for a one-shot fixture run (no batch backlog at exit).
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)

    # ---- Build the agent -----------------------------------------------
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        # Hardcoded so the fixture content stays deterministic.
        return f"The weather in {city} is sunny, 72F."

    llm = ChatOpenAI(model=model, temperature=0)
    agent = create_react_agent(llm, tools=[get_weather])

    # ---- Run the trace -------------------------------------------------
    print(f"running LangGraph agent with prompt: {prompt!r}")
    result = agent.invoke({"messages": [("user", prompt)]})
    final = result["messages"][-1].content
    print(f"agent response: {final!r}")

    # ---- Flush -------------------------------------------------------
    LangChainInstrumentor().uninstrument()
    provider.shutdown()
    print(f"wrote {exporter.num_spans_buffered} spans to {output_path}")
    return output_path
