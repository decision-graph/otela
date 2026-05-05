"""Generate a real Google ADK agent trace as an otela fixture.

Drives Google's Agent Development Kit (ADK) through one tool-using
conversation and captures the OTel GenAI semconv spans it emits. Uses
LiteLLM to route to OpenAI under the hood so the script doesn't require
GCP, Vertex AI, or a Gemini API key — your existing `OPENAI_API_KEY`
suffices. ADK's OTel instrumentation lives in its agent/runner/tool
layer, so the same `gen_ai.*` attributes and span events get emitted
regardless of which model backend serves the request.

Usage
-----
    export OPENAI_API_KEY=...
    uv sync --group fixtures
    uv run python scripts/generate_fixtures.py adk

Why this exists
---------------
Our synthetic `otel_genai_sample.json` fixture covers the spec on paper.
This fixture covers what a real SDK actually emits — span events for
messages, `gen_ai.tool.call.id` correlation, ADK-specific attribute
extensions, and any deviations from semconv that real implementations
sneak in. If something here surprises us, that's exactly the value of
testing against real output.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .otlp_file_exporter import OTLPFileExporter

DEFAULT_OUTPUT = Path("tests/fixtures/real/adk_research_agent.json")
DEFAULT_PROMPT = "What's the weather in San Francisco?"
DEFAULT_MODEL = "openai/gpt-4o-mini"
APP_NAME = "adk-weather-agent"
USER_ID = "fixture-user"


def generate(
    output_path: Path = DEFAULT_OUTPUT,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Run a small ADK agent and write its OTel GenAI trace to `output_path`."""
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY is not set. ADK + LiteLLM still needs it.\n"
            "    export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Lazy imports — keep `import scripts.fixtures` working in default envs.
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Configure tracing BEFORE creating the agent so ADK picks it up.
    resource = Resource.create(
        {
            "service.name": APP_NAME,
            "telemetry.sdk.language": "python",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPFileExporter(output_path)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # ---- Define the tool. ADK auto-wraps plain Python callables.
    def get_weather(city: str) -> dict:
        """Get the current weather for a city.

        Args:
            city: The name of the city to look up.

        Returns:
            A dict with `city`, `weather`, and `temperature_f`.
        """
        return {"city": city, "weather": "sunny", "temperature_f": 72}

    # ---- Build the agent using LiteLLM-routed OpenAI.
    agent = LlmAgent(
        name="weather_agent",
        model=LiteLlm(model=model),
        description="A weather assistant for fixture generation.",
        instruction=(
            "You are a helpful weather assistant. "
            "Use the get_weather tool to look up weather for any city the user asks about. "
            "Reply concisely with the result."
        ),
        tools=[get_weather],
    )

    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)

    async def _run() -> str:
        session = await runner.session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID
        )
        message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        final_text: str | None = None
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                # The final response is the assistant's last message.
                part = event.content.parts[0]
                final_text = getattr(part, "text", None)
        return final_text or "(no final response)"

    print(f"running ADK agent with prompt: {prompt!r}")
    final = asyncio.run(_run())
    print(f"agent response: {final!r}")

    # ---- Flush.
    provider.shutdown()
    print(f"wrote {exporter.num_spans_buffered} spans to {output_path}")
    return output_path
