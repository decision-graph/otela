# Changelog

All notable changes to otela are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
HEADING FORMAT IS LOAD-BEARING.
The release workflow (.github/workflows/release.yml) extracts the body
of the GitHub Release by awk-matching `## [X.Y.Z]` at the start of a
line, up to the next `## [` heading. Keep version sections under that
exact pattern. Editorial prose is fine; just don't change the heading.
-->

## [0.2.0] — 2026-05-06

`agent-trace` spec bumped to **`at/v2`**. The bump is driven by
first-class session support: a new `sessions` rollup table, plus
`session_id` and `session_turn` columns where they belong. Pure
additive on the existing tables; old code that doesn't touch those
columns keeps working.

### Added

- **`session_id` column on `spans` and `traces`.** Promoted from the
  first source attribute matching this precedence: `gen_ai.conversation.id`
  (OTel GenAI), `session.id` (OpenInference / MLflow),
  `gcp.vertex.agent.session_id` (Google ADK),
  `ai.telemetry.metadata.sessionId` (Vercel AI SDK),
  `mlflow.trace.session` (MLflow alternate),
  `traceloop.association.properties.session_id` (Traceloop). The matched
  attribute is removed from `raw_attributes_json` to avoid duplication.
- **`session_turn` column on `traces`.** 0-indexed ordinal of a trace
  within its session, ordered by `start_time_unix_nano` ASC with
  `trace_id` lex tiebreak. Null when `session_id` is null. otela-derived;
  scoped to the current load (cross-load global ordering is not
  guaranteed).
- **`sessions` rollup table.** One row per distinct `session_id`. Same
  end-of-load materialization pattern as `traces`. Schema importable as
  `otela.SESSIONS_SCHEMA`. Traces with null `session_id` are excluded.
- **Multi-turn fixture generation.** `scripts/generate_fixtures.py`
  langgraph and adk subcommands now drive three turns under a single
  session so generated fixtures exercise `session_turn` ordering.

### Changed

- **Spec bumped:** `SPEC_VERSION` is now `"v2"`. `at/v2` is the default
  resolution for `spec="at"`.
- **`load()`, `to_arrow()`, `to_parquet()` default `spec="at/v2"`.**
- **README schema reference and quickstart snippets** updated to `at/v2`
  and to include the `sessions` table.

### Removed (breaking)

- **`spec="at/v1"` is no longer accepted.** Calls raise `ValueError`
  with a pointer to this changelog. v0.x was pre-stable and we have no
  external users; the schemas are close enough that maintaining a v1
  read path was more debt than value. `raw_attributes_json` preserved
  full fidelity in v1 outputs, so re-running extraction from raw OTLP/JSON
  recovers everything plus the new session columns.

### Migration

If you have stored `at/v1` parquet, re-run `otela load` / `otela totables`
from your raw OTLP/JSON. The v0.2.0 normalizer will populate
`session_id` and `session_turn` automatically wherever a recognized
session attribute is present. There is no automated parquet-level upgrade
path; building one for a purely additive change is strictly weaker than
re-running the extractor against the source.

If you don't have raw OTLP/JSON anymore but do have v1 parquet, the
`session_id` source value is still recoverable from `raw_attributes_json`
(otela v1 wrote unconsumed attrs there). A custom one-off script is the
right tool for that case; it isn't a path otela ships.

## [0.1.1] — pre-release

See git log.
