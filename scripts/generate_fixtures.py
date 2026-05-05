"""Generate real-trace fixtures for the otela test suite.

Subcommands one per real-world SDK. Each generator is responsible for
running its workflow, exporting OTLP/JSON, and writing the result under
`tests/fixtures/real/`.

Usage
-----
    uv sync --group fixtures
    export OPENAI_API_KEY=...
    uv run python scripts/generate_fixtures.py langgraph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `from fixtures.<x>` resolve regardless of where this is launched from.
# Python prepends a script's own dir to sys.path when invoked as
# `python scripts/generate_fixtures.py`, but not when invoked some other way
# (e.g., from a parent dir or as a module). Explicit insertion is safer.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_fixtures",
        description="Generate real-trace fixtures for otela tests.",
    )
    sub = parser.add_subparsers(dest="command")

    lg = sub.add_parser("langgraph", help="LangGraph React agent (OpenInference).")
    lg.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to tests/fixtures/real/langgraph_research_agent.json.",
    )
    lg.add_argument("--prompt", type=str, default=None, help="Prompt for the agent.")
    lg.add_argument("--model", type=str, default=None, help="OpenAI model name.")

    adk = sub.add_parser("adk", help="Google ADK agent via LiteLLM (OTel GenAI semconv).")
    adk.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to tests/fixtures/real/adk_research_agent.json.",
    )
    adk.add_argument("--prompt", type=str, default=None, help="Prompt for the agent.")
    adk.add_argument(
        "--model",
        type=str,
        default=None,
        help="LiteLLM model identifier, e.g. 'openai/gpt-4o-mini' (default).",
    )

    args = parser.parse_args(argv)
    if args.command == "langgraph":
        from fixtures.langgraph_agent import (
            DEFAULT_MODEL,
            DEFAULT_OUTPUT,
            DEFAULT_PROMPT,
            generate,
        )

        generate(
            output_path=args.output or DEFAULT_OUTPUT,
            prompt=args.prompt or DEFAULT_PROMPT,
            model=args.model or DEFAULT_MODEL,
        )
        return 0

    if args.command == "adk":
        from fixtures.adk_agent import (
            DEFAULT_MODEL,
            DEFAULT_OUTPUT,
            DEFAULT_PROMPT,
            generate,
        )

        generate(
            output_path=args.output or DEFAULT_OUTPUT,
            prompt=args.prompt or DEFAULT_PROMPT,
            model=args.model or DEFAULT_MODEL,
        )
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
