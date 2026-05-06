# otela

Format OpenTelemetry agent traces for analytics, ML, and context graphs.

`otela` (OTel + analytics) reads raw OTel traces (OTLP/JSON) and emits
analytics-ready data in whatever shape you need: Arrow tables, Parquet,
Pandas, nested JSON records, or PyTorch tensors. It auto-detects between
OpenInference and OTel GenAI semantic conventions and gives you a single
uniform schema you can run cross-trace analytics, dashboards, ML data
prep, and context-graph builds against.

## Try it (no install)

Grab a sample OTel agent trace and run otela's CLI against it via `uvx` —
nothing gets installed permanently:

```bash
curl -sL https://raw.githubusercontent.com/zach-blumenfeld/otela/main/examples/sample.json \
  -o trace.json

uvx otela totables trace.json otela-out
```

This writes one Parquet file per table:

```
otela-out/
├── sessions.parquet    # one row per session (multi-trace conversation rollup)
├── traces.parquet      # one row per trace (rollup), with session_id + session_turn
├── spans.parquet       # one row per span
├── messages.parquet    # LLM messages, joinable on (trace_id, span_id)
├── documents.parquet   # retrieved docs from RETRIEVER spans
└── links.parquet       # OTel span links
```

Want nested-record output instead?

```bash
uvx otela torecords trace.json otela-records --format jsonl
```

## Install

```bash
pip install otela                    # core
pip install "otela[pandas]"          # adds the to_dfs() pandas adapter
pip install "otela[ml]"              # adds torch for to_tensors()
pip install "otela[pandas,ml]"       # everything
```

Python 3.11+.

## Basic Usage

### CLI

Create tabular data for analytics:

```bash
otela totables path/to/otel/trace.json path/to/output/
```

Create records with otela formatting — each record is a trace/workflow
with nested spans, tool calls, etc.:

```bash
otela torecords path/to/otel/trace.json path/to/output/
```

The input can be a single OTLP/JSON file or a directory of files
(walked recursively).

### Python

```python
import otela

# load one or multiple json files (file or directory)
traces = otela.load('path/to/otel/trace.json')

# dicts / json maps in otela format
trace_dicts = otela.to_dicts(traces)

# tabular (Pandas)
dfs = otela.to_dfs(traces)

# tensors (PyTorch)
tensors = otela.to_tensors(traces)

# stream straight to parquet without holding everything in memory
otela.to_parquet('path/to/traces/', 'out/', batch_size=10_000)

# on-demand dim tables: tools, agents, models, services
dims = otela.dims(traces)
```

## Specs

To make Otel logs useful for analytics & DS/ML, we need to format a bit
differently. otela has two specs for this:

1. `agent-trace (at)` (default): A minimal normalization between
   [OpenInference](https://arize-ai.github.io/openinference/spec/) and
   [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
   that gives you uniform records for analytics. Stays close to OTel
   naming conventions while surfacing input & output attributes and using
   a schema for tabular/batch processing.
2. `workflow-graph (wg)`: A more opinionated spec for representing the
   structural decisions and actions in a workflow — agent, human, or
   hybrid. It's prescriptive about node and relationship types in a graph
   schema, optimized for [context graphs](https://neo4j.com/blog/agentic-ai/hands-on-with-context-graphs-and-neo4j/),
   reinforcement learning, and other research.

Both take either OpenInference or OTel GenAI semantic conventions as
input and also tolerate Vercel AI SDK (`ai.*`), MLflow (`mlflow.*`), and
Traceloop (`traceloop.*`) attributes as fallbacks.

To specify the spec:

```bash
otela totables path/to/otel/trace.json path/to/output/ \
  --spec wg/v1
```

```python
trace_dicts = otela.to_dicts(otela.load('path/...', spec='at/v2'))
```

Forward-slash notation (`at/v2`) calls the spec at a specific version.
This is recommended as specs may change in non-backward-compatible ways.
Omitting the version (`at`) calls the latest. Specification type and
version are always embedded in the output records (`spec`,
`spec_version` columns or fields). Migration utilities will be added as
needed.

> **Status:** `at/v2` is implemented today (adds session_id, session_turn,
> and the sessions rollup; supersedes `at/v1`). `wg/v1` is on the
> roadmap — see [Status](#status) below.

## Output Formats

`totables` output formats: `parquet (default), csv, arrow, json, jsonl`

`torecords` output formats: `json (default), jsonl`

Specify the output format:

```bash
otela totables path/to/otel/trace.json path/to/output/ \
  --format parquet
```

`parquet` streams to disk with bounded memory (controlled by
`--batch-size`, default 10 000 spans per row group). The other tabular
formats materialize the full tableset in memory before writing — fine for
development, but for billions of spans use `parquet`.

## Status

| Area                                            | Status         |
| ----------------------------------------------- | -------------- |
| `agent-trace` spec, version `at/v2`             | implemented    |
| Session detection + multi-turn `sessions` rollup | implemented   |
| OTLP/JSON file + directory ingestion            | implemented    |
| OpenInference convention                        | implemented    |
| OTel GenAI semconv (events + attrs)             | implemented    |
| Vercel AI SDK / MLflow / Traceloop              | implemented    |
| Generic `input.value` / `output.value` fallback | implemented    |
| Streaming Parquet writer                        | implemented    |
| `load` / `to_dfs` / `to_dicts` / `to_tensors` / `to_parquet` / `dims` | implemented    |
| `otela totables` / `otela torecords` CLI        | implemented    |
| Real-trace fixtures: LangGraph (OpenInference)  | implemented    |
| Real-trace fixtures: Google ADK (OTel GenAI)    | implemented    |
| Real-trace fixtures: LlamaIndex (RETRIEVER / EMBEDDING) | planned — next |
| HuggingFace dataset adapters                    | planned — after LlamaIndex |
| Phoenix / Langfuse native export readers        | planned — opportunistic |
| `workflow-graph` spec, `wg/v1`                  | not yet started |
| Tokenized-text tensors for LLM fine-tuning      | not yet started |
| Streaming nested-record (`torecords`) writer    | not yet started |
| Parquet directory partitioning (Hive style)     | not yet started |

## Schema Reference (`agent-trace`, `at/v2`)

Every `otela.load()` call returns a dict of six Arrow tables. Schemas
are stable and versioned — every row carries `spec` and `spec_version`,
and `spec_version` only changes on a non-backward-compatible schema
change. Schemas are importable: `otela.SPANS_SCHEMA`,
`otela.TRACES_SCHEMA`, `otela.SESSIONS_SCHEMA`, etc.

```
output/
├── sessions.parquet    # one row per session (multi-trace rollup)
├── traces.parquet      # one row per trace (rollup) — carries session_id, session_turn
├── spans.parquet       # one row per span — the fact table
├── messages.parquet    # one row per LLM message; joins on (trace_id, span_id)
├── documents.parquet   # one row per retrieved document
└── links.parquet       # OTel span links
```

### `sessions`

Session-level rollup, one row per distinct `session_id`. A session
groups multiple traces (turns of one conversation). Traces with no
`session_id` are excluded.

| Column                   | Type    | Notes                                           |
| ------------------------ | ------- | ----------------------------------------------- |
| `session_id`             | string  | primary key                                     |
| `trace_count`            | int64   |                                                 |
| `span_count`             | int64   | sum across traces                               |
| `error_count`            | int64   | sum across traces                               |
| `start_time_unix_nano`   | int64   | min over traces                                 |
| `end_time_unix_nano`     | int64   | max over traces                                 |
| `duration_ns`            | int64   |                                                 |
| `total_input_tokens`     | int64   | sum across traces; `NULL` if no trace had it    |
| `total_output_tokens`    | int64   | "                                               |
| `total_tokens`           | int64   | "                                               |

### `traces`

Trace-level rollup, one row per trace.

| Column                   | Type    | Notes                                                              |
| ------------------------ | ------- | ------------------------------------------------------------------ |
| `trace_id`               | string  | primary key                                                        |
| `session_id`             | string  | nullable; promoted from any recognized session attribute (see below) |
| `session_turn`           | int32   | 0-indexed position within session, ordered by `start_time_unix_nano` ASC, `trace_id` lex tiebreak. Null when `session_id` is null. otela-derived; batch-scoped. |
| `root_span_id`           | string  | earliest parentless span                                           |
| `root_span_name`         | string  |                                                                    |
| `service_name`           | string  | from the root span's resource                                      |
| `start_time_unix_nano`   | int64   | min over spans                                                     |
| `end_time_unix_nano`     | int64   | max over spans                                                     |
| `duration_ns`            | int64   |                                                                    |
| `span_count`             | int64   |                                                                    |
| `error_count`            | int64   | spans with `status = ERROR`                                        |
| `status`                 | string  | worst-of: `ERROR > OK > UNSET`                                     |
| `total_input_tokens`     | int64   | sum across spans; `NULL` if no span had it                         |
| `total_output_tokens`    | int64   | "                                                                  |
| `total_tokens`           | int64   | "                                                                  |

### `spans`

Canonical row-per-span fact table. Joins to `traces` on `trace_id` and
to the side tables on `(trace_id, span_id)`.

- **Identification:** `trace_id`, `span_id`, `parent_span_id`, `name`
- **Classification:** `kind` (`AGENT | LLM | TOOL | CHAIN | RETRIEVER |
  EMBEDDING | RERANKER | GUARDRAIL | EVALUATOR | UNKNOWN`),
  `convention` (which semconv this span came from), `status_code`,
  `status_message`
- **Timing:** `start_time_unix_nano`, `end_time_unix_nano`, `duration_ns`
- **Resource:** `service_name`, `scope_name`, `scope_version`
- **Agent-trace canonical:** `model_name`, `tool_name`, `agent_name`,
  `input_tokens`, `output_tokens`, `total_tokens`, `io_format` (`text |
  tool_call | retrieval | unknown`), `input_text`, `output_text`
- **Session:** `session_id` — promoted from any recognized session/
  conversation attribute (see "Source Conventions Accepted" below).
  Carried on every span the source instrumentation tagged so
  `WHERE session_id = X` queries hit the spans table without a join.
- **Fidelity:** `raw_attributes_json` — JSON-encoded leftover attrs the
  normalizer didn't promote into a typed column. No information is
  silently dropped.

### `messages`

One row per LLM message (system / user / assistant / tool). Sourced from
either OpenInference indexed attributes (`llm.input_messages.N.message.*`)
or OTel GenAI span events (`gen_ai.user.message`,
`gen_ai.assistant.message`, etc.).

Columns: `trace_id`, `span_id`, `position` (order within the span),
`direction` (`input` | `output`), `role`, `content`, `tool_call_id`.

### `documents`

Retrieved documents from `RETRIEVER` spans.

Columns: `trace_id`, `span_id`, `position`, `document_id`, `content`,
`score`.

### `links`

OTel span links (one span pointing at another span outside its parent
chain).

Columns: `trace_id`, `span_id`, `linked_trace_id`, `linked_span_id`.

### Mapping to a graph schema

The 4-FK + 1-rollup tabular layout corresponds 1:1 to the property-graph
schema in [zach-blumenfeld/otel-to-neo4j](https://github.com/zach-blumenfeld/otel-to-neo4j).
The `(:Tool)`, `(:Agent)`, `(:Model)`, `(:Service)` nodes are
denormalized into name columns on `spans`; recover them as dim tables on
demand via `otela.dims(traces)`.

## Source Conventions Accepted

Spans are auto-classified per-span. A single trace can mix conventions —
e.g. an OpenInference LangChain instrumentation alongside an OTel GenAI
model call.

| Convention                     | Detection signal                                          | Session attribute                                |
| ------------------------------ | --------------------------------------------------------- | ------------------------------------------------ |
| [OpenInference](https://arize-ai.github.io/openinference/spec/) | `openinference.span.kind`, `llm.*`, `tool.*`, `retrieval.*`, `embedding.*` | `session.id`                          |
| [OTel GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | any `gen_ai.*` attribute or span event       | `gen_ai.conversation.id`                         |
| Vercel AI SDK                  | `ai.*` attributes                                          | `ai.telemetry.metadata.sessionId`                |
| MLflow                         | `mlflow.*` attributes                                      | `session.id` or `mlflow.trace.session`           |
| Traceloop / OpenLLMetry        | `traceloop.*` attributes                                   | `traceloop.association.properties.session_id`   |
| Google ADK (OTel GenAI extension) | `gcp.vertex.agent.*` attributes                          | `gcp.vertex.agent.session_id`                    |
| Generic                        | `input.value` / `output.value` only                        | —                                                |

When multiple session attributes are present on the same span, otela
picks one with this precedence (most-standard first):
`gen_ai.conversation.id` → `session.id` → `gcp.vertex.agent.session_id`
→ `ai.telemetry.metadata.sessionId` → `mlflow.trace.session` →
`traceloop.association.properties.session_id`. The matched attribute is
promoted into the `session_id` column and removed from
`raw_attributes_json` to avoid duplication.

## Design Principles

1. **Built for scale.** PyArrow + Parquet is the canonical internal
   representation. The reader is generator-based; `to_parquet()` streams
   with bounded memory (a TB of input works the same as a 10 MB file).
   Pandas is a thin convenience adapter for the in-memory case.
2. **ML-focused output.** The eventual goal is a clean data-prep layer
   for training agentic models. That motivates the multi-table
   normalized layout, explicit nullability on numeric columns, and the
   `to_tensors()` adapter.
3. **Schema-stable.** Every row carries `spec` and `spec_version`.
   Schemas are tested for drift on every fixture run.
4. **No information loss.** Anything the convention extractor doesn't
   promote into a typed column lands in `raw_attributes_json`.
5. **Zero-cost optional deps.** `import otela` works without pandas or
   torch installed; calling the adapter raises a clear `ImportError`
   pointing at the right extra.

## Roadmap

### Next up

- **LlamaIndex real-trace fixture (OpenInference).** Closes the last
  major SDK coverage gap: `RETRIEVER` and `EMBEDDING` span kinds have
  only been validated against synthetic fixtures so far. LangGraph +
  ADK don't exercise them. Same `scripts/generate_fixtures.py` harness;
  expected to surface deeper indexed attributes
  (`retrieval.documents.N.document.metadata.*`) that may warrant
  promotion from `raw_attributes_json`.
- **HuggingFace dataset adapters, after LlamaIndex.** Most agent-trace
  datasets on HF aren't OTLP-shaped — typically conversation logs in
  parquet/jsonl, sometimes OTel exports in vendor-specific JSON. Each
  dataset usually needs a small adapter that reshapes its rows into our
  `RawSpan` iterator; the rest of the pipeline (`normalize` → `builder`
  → schemas) doesn't change. Sequenced after LlamaIndex so the spec is
  rock-solid before discovering data-shape issues at scale.

### Later

- **Phoenix / Langfuse native export readers.** Both speak OTLP on
  ingest, but their export formats are vendor-shaped. If you have
  traces in those backends today, the fastest path is configuring an
  OTLP file dump on the backend; native readers are a convenience
  layer worth adding once we see real demand.
- `wg/v1` workflow-graph spec
- Tokenizer-aware `to_tensors()` mode for LLM fine-tuning
  (`input_ids` / `attention_mask` per message)
- Streaming `torecords` writer (per-trace flush as soon as a trace is
  observably complete)
- Hive-partitioned Parquet output (`/service=foo/date=2026-04-22/...`)
  for direct DuckDB / Spark consumption
- Migration utilities once a second `at` version exists

If you have production traces that would make a good test fixture,
please open an issue.

## Development

```bash
git clone https://github.com/zach-blumenfeld/otela
cd otela
uv sync                  # installs runtime + dev deps (pytest, ruff, torch, pandas)
```

Project layout:

```
src/otela/
├── schemas.py     # Arrow schemas — single source of truth
├── otlp.py        # OTLP/JSON parsing helpers
├── reader.py      # Streaming OTLP/JSON file/directory iterator
├── normalize.py   # Convention detection + at/v1 extraction
├── builder.py     # Column buffers + per-trace accumulators -> Arrow tables
├── api.py         # load(), to_dfs(), to_dicts(), to_parquet(), dims()
├── tensors.py     # to_tensors() — optional torch dependency
└── cli.py         # otela totables / otela torecords
```

### Running tests

```bash
uv run pytest -q                      # full suite (synthetic fixtures only)
uv run pytest -v                      # verbose, shows each test name
uv run pytest tests/test_to_dicts.py  # one file
uv run pytest -k traces               # by name pattern
```

Tests against **real-trace fixtures** (`tests/test_real_traces.py`) are
skipped automatically when their fixture file doesn't exist. To run
them, generate the fixture first — see "Generating real-trace fixtures"
below. Without the fixture you'll see something like:

```
92 passed, 11 skipped
```

That's expected — the suite is green; the skips are real-trace tests
waiting on a regenerated fixture.

### Linting

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff check --fix src/ tests/ scripts/   # auto-fix
```

CI runs both `pytest` and `ruff check` — both must be green.

### Generating real-trace fixtures

Synthetic fixtures (`examples/sample.json`, `examples/sample_genai.json`)
cover the spec, but production SDKs surface shape edge cases that
hand-written fixtures don't. The `scripts/generate_fixtures.py` harness
runs minimal example agents under real instrumentation and commits the
resulting OTLP/JSON to `tests/fixtures/real/` so the test suite can
assert against them.

Currently supported sources:

- `langgraph` — LangGraph React agent + OpenInference instrumentation
- `adk` — Google ADK agent via LiteLLM (OTel GenAI semconv)

Generate fixtures:

```bash
uv sync --group fixtures
export OPENAI_API_KEY=sk-...
uv run python scripts/generate_fixtures.py langgraph
uv run python scripts/generate_fixtures.py adk
```

Outputs:

- `tests/fixtures/real/langgraph_research_agent.json`
- `tests/fixtures/real/adk_research_agent.json`

The ADK generator routes through LiteLLM to OpenAI under the hood — ADK's
OTel emission is independent of the model backend, so a `gpt-4o-mini`-served
trace exercises the same `gen_ai.*` events code path as a Vertex Gemini
one. No GCP/Vertex setup required; your `OPENAI_API_KEY` is enough.

The corresponding tests in `tests/test_real_traces.py` skip
automatically if the fixture is missing — contributors who don't
regenerate fixtures still get a green suite.

Quick-summary any otela-readable trace file:

```bash
uv run python scripts/inspect_fixture.py tests/fixtures/real/langgraph_research_agent.json
```

## License

[Apache 2.0](LICENSE).
