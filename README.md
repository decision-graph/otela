# otela
OpenTelemetry (Otel) Analytics and Data Formatting, Focused on Agent Traces.

## Basic Usage

### CLI
Create tabular data for analytics
```bash
otela totables path/to/otel/trace.json path/to/output/ 
```

create records with otela formatting - each record is a trace/workflow with nested spans, tool calls, etc.

```bash
otela torecords path/to/otel/trace.json path/to/output/
```
#### CLI Output Formats
`totables` output formats include `csv (default), parquet, arrow, json, jsonl` 

`torecords` output formats include `json (default), jsonl` 

to specify the output format:
```bash
otela totables path/to/otel/trace.json path/to/output/ \
  --format parquet
```

### Python
```python
import otela

#load one or multiple json files
traces = otela.load('path/to/otel/trace.json')

# dicts / json maps in otela format
trace_dicts = otela.to_dicts(traces)

# tabular (Pandas)
dfs = otela.to_dfs(traces)

# tensors (PyTorch)
tensors = otela.to_tensors(traces)
```

## Specs for Analytics
To make Otel logs useful for analytics & DS/ML, we need to format a bit differently.   otela has two specs for this:
1. `agent-trace (at)` (default): A minimal normalization between [OpenInference](https://arize-ai.github.io/openinference/spec/) and [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) that gives you uniform records for analytics. Stays close to OTel naming conventions while surfaces input & output attributes and using a schema for tabular/batch processing.
- `workflow-graph (wg)` : A more opinionated spec for representing the structural decisions and actions in a workflow — agent, human, or hybrid. It's prescriptive about node and relationship types in a graph schema, optimized for [context graphs](https://neo4j.com/blog/agentic-ai/hands-on-with-context-graphs-and-neo4j/), reinforcement learning, and other research. 

Both of the above take either [OpenInference](https://arize-ai.github.io/openinference/spec/) or [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) as input and also tolerate Vercel AI SDK (`ai.*`), MLflow (`mlflow.*`), and Traceloop (`traceloop.*`) attributes as fallbacks.

To specify the spec
```bash
otela totables path/to/otel/trace.json path/to/output/ \
  --spec wg/v1 
```

```python
# dicts / json maps in otela format
trace_dicts = otela.to_dicts(traces, spec='at/v1')
```

Using the forward slash notation calls the spec at the specific version number. This is recommended as specs may change in non-backward compatible ways. Ommiting version specification calls the latest version. Specification type and version are always embedded in the output records. Migration utilities will be added to the library as needed.



