"""Agent-trace (at/v2) normalizer.

Reads a `RawSpan` (an OTLP span plus resource/scope context) and emits a
`NormalizedSpan`: the canonical row that the spans table is built from, plus
optional side records for messages, documents, and links.

Session detection is independent of convention detection: see
`_detect_session_id` for the precedence list across OTel GenAI,
OpenInference, ADK, Vercel, MLflow, and Traceloop.

Convention handling
-------------------
Real-world traces mix conventions. A single trace can have OpenInference
spans from a LangChain instrumentation alongside OTel GenAI spans from a
newer SDK. Each span is independently classified.

Detection priority (most specific first):
    1. OpenInference   — `openinference.span.kind` or `llm.*` / `tool.*` / `retrieval.*`
    2. OTel GenAI      — any `gen_ai.*` attribute or event
    3. Vercel AI SDK   — `ai.*`
    4. MLflow          — `mlflow.*`
    5. Traceloop       — `traceloop.*`
    6. Generic         — `input.value` / `output.value` only

Once a convention is chosen, the corresponding extractor populates the
canonical fields. Anything the extractor leaves unconsumed lands in
`raw_attributes_json` for fidelity.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import orjson

from .otlp import (
    any_value,
    attributes_to_dict,
    normalize_status_code,
    parent_span_id_or_none,
    parse_nano,
)
from .reader import RawSpan
from .schemas import SPEC, SPEC_VERSION

# Canonical span kinds.
KIND_LLM = "LLM"
KIND_TOOL = "TOOL"
KIND_AGENT = "AGENT"
KIND_CHAIN = "CHAIN"
KIND_RETRIEVER = "RETRIEVER"
KIND_EMBEDDING = "EMBEDDING"
KIND_RERANKER = "RERANKER"
KIND_GUARDRAIL = "GUARDRAIL"
KIND_EVALUATOR = "EVALUATOR"
KIND_UNKNOWN = "UNKNOWN"

# Conventions.
CONV_OPENINFERENCE = "openinference"
CONV_OTEL_GENAI = "otel_genai"
CONV_VERCEL_AI = "vercel_ai"
CONV_MLFLOW = "mlflow"
CONV_TRACELOOP = "traceloop"
CONV_GENERIC = "generic"
CONV_UNKNOWN = "unknown"

# I/O formats.
IO_TEXT = "text"
IO_TOOL_CALL = "tool_call"
IO_RETRIEVAL = "retrieval"
IO_UNKNOWN = "unknown"

# Direction tags on messages.
DIR_INPUT = "input"
DIR_OUTPUT = "output"


@dataclass(slots=True)
class NormalizedMessage:
    position: int
    direction: str | None
    role: str | None
    content: str | None
    tool_call_id: str | None = None


@dataclass(slots=True)
class NormalizedDocument:
    position: int
    document_id: str | None
    content: str | None
    score: float | None


@dataclass(slots=True)
class NormalizedLink:
    linked_trace_id: str | None
    linked_span_id: str | None


@dataclass(slots=True)
class NormalizedSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str | None
    kind: str
    convention: str
    status_code: str
    status_message: str | None
    start_time_unix_nano: int | None
    end_time_unix_nano: int | None
    duration_ns: int | None
    service_name: str | None
    scope_name: str | None
    scope_version: str | None
    session_id: str | None
    model_name: str | None
    tool_name: str | None
    agent_name: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    io_format: str
    input_text: str | None
    output_text: str | None
    raw_attributes_json: str | None
    messages: list[NormalizedMessage] = field(default_factory=list)
    documents: list[NormalizedDocument] = field(default_factory=list)
    links: list[NormalizedLink] = field(default_factory=list)


def normalize_span(raw: RawSpan) -> NormalizedSpan:
    """Normalize one OTLP span into the at/v2 canonical shape."""
    span = raw.span
    attrs = attributes_to_dict(span.get("attributes"))
    events = span.get("events") or []

    convention = _detect_convention(attrs, events)
    consumed: set[str] = set()
    session_id = _detect_session_id(attrs, consumed)

    if convention == CONV_OPENINFERENCE:
        extracted = _extract_openinference(attrs, consumed)
    elif convention == CONV_OTEL_GENAI:
        extracted = _extract_otel_genai(attrs, events, consumed)
    elif convention == CONV_VERCEL_AI:
        extracted = _extract_vercel(attrs, consumed)
    elif convention == CONV_MLFLOW:
        extracted = _extract_mlflow(attrs, consumed)
    elif convention == CONV_TRACELOOP:
        extracted = _extract_traceloop(attrs, consumed)
    else:
        extracted = _Extracted()

    # Generic input.value/output.value backfill — works under every convention.
    if "input.value" in attrs and not extracted.input_text:
        extracted.input_text = _coerce_text(attrs.get("input.value"))
        consumed.add("input.value")
    if "output.value" in attrs and not extracted.output_text:
        extracted.output_text = _coerce_text(attrs.get("output.value"))
        consumed.add("output.value")

    # If we've got text but the convention extractor couldn't pick an
    # io_format (e.g. AGENT/CHAIN spans whose I/O lives only in input.value),
    # default to text. RETRIEVER and TOOL keep their kind-specific format.
    if not extracted.io_format and (extracted.input_text or extracted.output_text):
        if extracted.kind == KIND_RETRIEVER:
            extracted.io_format = IO_RETRIEVAL
        elif extracted.kind == KIND_TOOL:
            extracted.io_format = IO_TOOL_CALL
        else:
            extracted.io_format = IO_TEXT

    # Resource attributes are not span attributes; resolve service_name here.
    service_name = raw.resource_attributes.get("service.name")

    start_ns = parse_nano(span.get("startTimeUnixNano"))
    end_ns = parse_nano(span.get("endTimeUnixNano"))
    duration = end_ns - start_ns if (start_ns is not None and end_ns is not None) else None

    status = span.get("status") or {}

    raw_remaining = {k: v for k, v in attrs.items() if k not in consumed}
    raw_json = orjson.dumps(raw_remaining).decode() if raw_remaining else None

    links = list(_extract_links(span.get("links")))

    return NormalizedSpan(
        trace_id=str(span.get("traceId", "")),
        span_id=str(span.get("spanId", "")),
        parent_span_id=parent_span_id_or_none(span.get("parentSpanId")),
        name=span.get("name"),
        kind=extracted.kind or KIND_UNKNOWN,
        convention=convention,
        status_code=normalize_status_code(status.get("code")),
        status_message=status.get("message") or None,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        duration_ns=duration,
        service_name=service_name,
        scope_name=raw.scope_name,
        scope_version=raw.scope_version,
        session_id=session_id,
        model_name=extracted.model_name,
        tool_name=extracted.tool_name,
        agent_name=extracted.agent_name,
        input_tokens=extracted.input_tokens,
        output_tokens=extracted.output_tokens,
        total_tokens=extracted.total_tokens,
        io_format=extracted.io_format or IO_UNKNOWN,
        input_text=extracted.input_text,
        output_text=extracted.output_text,
        raw_attributes_json=raw_json,
        messages=extracted.messages,
        documents=extracted.documents,
        links=links,
    )


# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------


# Source-attribute keys carrying a session/conversation identifier, in
# precedence order. The first key present in the span's attributes wins.
# OTel GenAI is preferred over OpenInference because it's the official
# upstream spec; the others are vendor-specific or alternates that share
# the same semantic.
_SESSION_ID_KEYS: tuple[str, ...] = (
    "gen_ai.conversation.id",
    "session.id",
    "gcp.vertex.agent.session_id",
    "ai.telemetry.metadata.sessionId",
    "mlflow.trace.session",
    "traceloop.association.properties.session_id",
)


def _detect_session_id(attrs: dict[str, Any], consumed: set[str]) -> str | None:
    """Pick a session id from any recognized source attribute.

    Adds the matched key to `consumed` so it is not duplicated into
    `raw_attributes_json`. Empty strings are treated as not-present.
    """
    for key in _SESSION_ID_KEYS:
        if key in attrs:
            value = _coerce_text(attrs[key])
            consumed.add(key)
            if value:
                return value
    return None


# ---------------------------------------------------------------------------
# Convention detection
# ---------------------------------------------------------------------------


def _detect_convention(attrs: dict[str, Any], events: list[dict[str, Any]]) -> str:
    if "openinference.span.kind" in attrs:
        return CONV_OPENINFERENCE
    # OpenInference indexed-attribute style (e.g. llm.input_messages.N.*).
    # Disambiguate from OTel GenAI which uses the gen_ai.* prefix.
    if any(
        k.startswith(("llm.", "tool.", "retrieval.", "embedding.", "reranker."))
        for k in attrs
    ) and not any(k.startswith("gen_ai.") for k in attrs):
        return CONV_OPENINFERENCE
    if any(k.startswith("gen_ai.") for k in attrs) or any(
        (e.get("name") or "").startswith("gen_ai.") for e in events
    ):
        return CONV_OTEL_GENAI
    if any(k.startswith("ai.") for k in attrs):
        return CONV_VERCEL_AI
    if any(k.startswith("mlflow.") for k in attrs):
        return CONV_MLFLOW
    if any(k.startswith("traceloop.") for k in attrs):
        return CONV_TRACELOOP
    if "input.value" in attrs or "output.value" in attrs:
        return CONV_GENERIC
    return CONV_UNKNOWN


# ---------------------------------------------------------------------------
# Extractor results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Extracted:
    kind: str | None = None
    model_name: str | None = None
    tool_name: str | None = None
    agent_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    io_format: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    messages: list[NormalizedMessage] = field(default_factory=list)
    documents: list[NormalizedDocument] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenInference
# ---------------------------------------------------------------------------


_OPENINFERENCE_KIND_MAP = {
    "AGENT": KIND_AGENT,
    "LLM": KIND_LLM,
    "CHAIN": KIND_CHAIN,
    "TOOL": KIND_TOOL,
    "RETRIEVER": KIND_RETRIEVER,
    "EMBEDDING": KIND_EMBEDDING,
    "RERANKER": KIND_RERANKER,
    "GUARDRAIL": KIND_GUARDRAIL,
    "EVALUATOR": KIND_EVALUATOR,
}


def _extract_openinference(attrs: dict[str, Any], consumed: set[str]) -> _Extracted:
    out = _Extracted()

    if "openinference.span.kind" in attrs:
        out.kind = _OPENINFERENCE_KIND_MAP.get(
            str(attrs["openinference.span.kind"]).upper(), KIND_UNKNOWN
        )
        consumed.add("openinference.span.kind")

    for key in ("llm.model_name", "llm.model"):
        if key in attrs:
            out.model_name = _coerce_text(attrs[key])
            consumed.add(key)
            break

    if "tool.name" in attrs:
        out.tool_name = _coerce_text(attrs["tool.name"])
        consumed.add("tool.name")

    if "agent.name" in attrs:
        out.agent_name = _coerce_text(attrs["agent.name"])
        consumed.add("agent.name")

    # Token counts.
    if "llm.token_count.prompt" in attrs:
        out.input_tokens = _coerce_int(attrs["llm.token_count.prompt"])
        consumed.add("llm.token_count.prompt")
    if "llm.token_count.completion" in attrs:
        out.output_tokens = _coerce_int(attrs["llm.token_count.completion"])
        consumed.add("llm.token_count.completion")
    if "llm.token_count.total" in attrs:
        out.total_tokens = _coerce_int(attrs["llm.token_count.total"])
        consumed.add("llm.token_count.total")

    # Indexed messages: llm.input_messages.N.message.role / .content
    in_msgs = _collect_openinference_messages(attrs, "llm.input_messages", consumed)
    out_msgs = _collect_openinference_messages(attrs, "llm.output_messages", consumed)

    for i, m in enumerate(in_msgs):
        out.messages.append(
            NormalizedMessage(position=i, direction=DIR_INPUT, role=m.get("role"), content=m.get("content"))
        )
    base = len(in_msgs)
    for i, m in enumerate(out_msgs):
        out.messages.append(
            NormalizedMessage(position=base + i, direction=DIR_OUTPUT, role=m.get("role"), content=m.get("content"))
        )

    # Retrieval documents: retrieval.documents.N.document.{id,content,score}
    docs = _collect_openinference_documents(attrs, consumed)
    out.documents = docs

    # Tool I/O.
    if "tool.parameters" in attrs:
        out.input_text = _coerce_text(attrs["tool.parameters"])
        consumed.add("tool.parameters")
    if "tool.output_value" in attrs:
        out.output_text = _coerce_text(attrs["tool.output_value"])
        consumed.add("tool.output_value")

    if "retrieval.query" in attrs:
        out.input_text = out.input_text or _coerce_text(attrs["retrieval.query"])
        consumed.add("retrieval.query")

    # Set io_format, then synthesize input_text/output_text from messages if missing.
    if out.kind == KIND_RETRIEVER or docs:
        out.io_format = IO_RETRIEVAL
    elif out.kind == KIND_TOOL or out.tool_name:
        out.io_format = IO_TOOL_CALL
    elif in_msgs or out_msgs:
        out.io_format = IO_TEXT
        if not out.input_text and in_msgs:
            out.input_text = _join_messages(in_msgs)
        if not out.output_text and out_msgs:
            out.output_text = _join_messages(out_msgs)
    return out


def _collect_openinference_messages(
    attrs: dict[str, Any], prefix: str, consumed: set[str]
) -> list[dict[str, str | None]]:
    """Gather indexed messages of the form `{prefix}.N.message.{role,content}`."""
    by_index: dict[int, dict[str, str | None]] = {}
    pre = prefix + "."
    for k, v in attrs.items():
        if not k.startswith(pre):
            continue
        rest = k[len(pre) :]
        # rest looks like "0.message.role" or "0.message.content"
        try:
            idx_str, _, suffix = rest.partition(".")
            idx = int(idx_str)
        except ValueError:
            continue
        if suffix == "message.role":
            by_index.setdefault(idx, {"role": None, "content": None})["role"] = _coerce_text(v)
            consumed.add(k)
        elif suffix == "message.content":
            by_index.setdefault(idx, {"role": None, "content": None})["content"] = _coerce_text(v)
            consumed.add(k)
        else:
            # Other indexed fields (e.g. tool_calls) — leave for raw_attributes.
            continue
    return [by_index[i] for i in sorted(by_index)]


def _collect_openinference_documents(
    attrs: dict[str, Any], consumed: set[str]
) -> list[NormalizedDocument]:
    pre = "retrieval.documents."
    by_index: dict[int, dict[str, Any]] = {}
    for k, v in attrs.items():
        if not k.startswith(pre):
            continue
        rest = k[len(pre) :]
        try:
            idx_str, _, suffix = rest.partition(".")
            idx = int(idx_str)
        except ValueError:
            continue
        slot = by_index.setdefault(idx, {})
        if suffix == "document.id":
            slot["id"] = _coerce_text(v)
        elif suffix == "document.content":
            slot["content"] = _coerce_text(v)
        elif suffix == "document.score":
            slot["score"] = _coerce_float(v)
        else:
            continue
        consumed.add(k)
    return [
        NormalizedDocument(
            position=i,
            document_id=by_index[i].get("id"),
            content=by_index[i].get("content"),
            score=by_index[i].get("score"),
        )
        for i in sorted(by_index)
    ]


# ---------------------------------------------------------------------------
# OTel GenAI
# ---------------------------------------------------------------------------

# gen_ai.operation.name -> canonical kind
_GENAI_OP_KIND = {
    "chat": KIND_LLM,
    "text_completion": KIND_LLM,
    "generate_content": KIND_LLM,
    "embeddings": KIND_EMBEDDING,
    "execute_tool": KIND_TOOL,
    "invoke_agent": KIND_AGENT,
    "create_agent": KIND_AGENT,
}


def _extract_otel_genai(
    attrs: dict[str, Any], events: list[dict[str, Any]], consumed: set[str]
) -> _Extracted:
    out = _Extracted()

    op = attrs.get("gen_ai.operation.name")
    if op is not None:
        out.kind = _GENAI_OP_KIND.get(str(op), KIND_UNKNOWN)
        consumed.add("gen_ai.operation.name")

    for key in ("gen_ai.request.model", "gen_ai.response.model"):
        if key in attrs and not out.model_name:
            out.model_name = _coerce_text(attrs[key])
            consumed.add(key)

    # Some SDKs (Google ADK among them) emit `call_llm`-style internal spans
    # with `gen_ai.request.model` and `gen_ai.usage.*` but no
    # `gen_ai.operation.name`. Treat those as LLM spans rather than UNKNOWN.
    if out.kind is None and (
        out.model_name or any(k.startswith("gen_ai.usage.") for k in attrs)
    ):
        out.kind = KIND_LLM

    if "gen_ai.tool.name" in attrs:
        out.tool_name = _coerce_text(attrs["gen_ai.tool.name"])
        consumed.add("gen_ai.tool.name")

    if "gen_ai.agent.name" in attrs:
        out.agent_name = _coerce_text(attrs["gen_ai.agent.name"])
        consumed.add("gen_ai.agent.name")

    # Token counts. The semconv has gone through several names.
    out.input_tokens = _first_int(attrs, consumed, ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"))
    out.output_tokens = _first_int(attrs, consumed, ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"))
    out.total_tokens = _first_int(attrs, consumed, ("gen_ai.usage.total_tokens",))

    # Tool call I/O on attribute side.
    if "gen_ai.tool.call.arguments" in attrs:
        out.input_text = _coerce_text(attrs["gen_ai.tool.call.arguments"])
        consumed.add("gen_ai.tool.call.arguments")
    if "gen_ai.tool.call.id" in attrs:
        # Stash for messages but keep as raw too — tool_call_id doesn't have
        # its own column on spans.
        consumed.add("gen_ai.tool.call.id")
    tool_call_id = attrs.get("gen_ai.tool.call.id")
    tool_call_id_str = _coerce_text(tool_call_id) if tool_call_id is not None else None

    # Messages live on span events for OTel GenAI.
    in_msgs: list[NormalizedMessage] = []
    out_msgs: list[NormalizedMessage] = []
    for ev in events:
        ename = ev.get("name") or ""
        if not ename.startswith("gen_ai."):
            continue
        ev_attrs = attributes_to_dict(ev.get("attributes"))
        content = _coerce_text(ev_attrs.get("content"))
        role = _genai_event_role(ename)
        if ename == "gen_ai.choice":
            out_msgs.append(
                NormalizedMessage(
                    position=0,
                    direction=DIR_OUTPUT,
                    role=role,
                    content=content,
                )
            )
        elif ename == "gen_ai.assistant.message":
            # In a chat operation, assistant.message is typically the response.
            out_msgs.append(
                NormalizedMessage(
                    position=0,
                    direction=DIR_OUTPUT,
                    role=role,
                    content=content,
                )
            )
        elif ename == "gen_ai.tool.message":
            # A tool result delivered back to the model — the tool's *output*.
            out_msgs.append(
                NormalizedMessage(
                    position=0,
                    direction=DIR_OUTPUT,
                    role=role,
                    content=content,
                    tool_call_id=tool_call_id_str,
                )
            )
        else:
            in_msgs.append(
                NormalizedMessage(
                    position=0,
                    direction=DIR_INPUT,
                    role=role,
                    content=content,
                )
            )
    # Google ADK extension: messages are encoded as JSON in
    # `gcp.vertex.agent.llm_request` / `llm_response` attributes (Gemini's
    # `genai.Content` shape) instead of as `gen_ai.*` span events. Fall back
    # to those when no events were present.
    if not in_msgs and not out_msgs:
        adk_in, adk_out = _extract_adk_genai_messages(attrs, consumed)
        in_msgs.extend(adk_in)
        out_msgs.extend(adk_out)

    # ADK also encodes tool I/O as `gcp.vertex.agent.tool_call_args` /
    # `tool_response` rather than `gen_ai.tool.call.arguments`.
    if "gcp.vertex.agent.tool_call_args" in attrs and not out.input_text:
        out.input_text = _coerce_text(attrs["gcp.vertex.agent.tool_call_args"])
        consumed.add("gcp.vertex.agent.tool_call_args")
    if "gcp.vertex.agent.tool_response" in attrs and not out.output_text:
        out.output_text = _coerce_text(attrs["gcp.vertex.agent.tool_response"])
        consumed.add("gcp.vertex.agent.tool_response")

    # Reassign positions in stable order (inputs first, then outputs).
    msgs: list[NormalizedMessage] = []
    for i, m in enumerate(in_msgs):
        m.position = i
        msgs.append(m)
    for i, m in enumerate(out_msgs):
        m.position = len(in_msgs) + i
        msgs.append(m)
    out.messages = msgs

    # Determine io_format and synthesize text from messages.
    if out.kind == KIND_TOOL:
        out.io_format = IO_TOOL_CALL
        if not out.output_text and out_msgs:
            out.output_text = out_msgs[0].content
    elif out.kind == KIND_LLM or out.kind == KIND_AGENT:
        out.io_format = IO_TEXT
        if not out.input_text and in_msgs:
            out.input_text = _join_messages_typed(in_msgs)
        if not out.output_text and out_msgs:
            out.output_text = _join_messages_typed(out_msgs)
    elif out.kind == KIND_EMBEDDING:
        out.io_format = IO_TEXT
    return out


def _extract_adk_genai_messages(
    attrs: dict[str, Any], consumed: set[str]
) -> tuple[list[NormalizedMessage], list[NormalizedMessage]]:
    """Parse Google ADK's `gcp.vertex.agent.llm_request` / `llm_response`.

    The payloads are JSON-encoded `genai.Content` structs (Gemini's shape):

        {"contents": [{"role": "user|model", "parts": [{"text"|"function_call"|"function_response": ...}]}],
         "config": {"system_instruction": "..."}}

    Inputs go to `in_msgs`, the response goes to `out_msgs`. The Gemini
    role `"model"` is normalized to `"assistant"` so messages from
    different SDKs join cleanly.
    """
    in_msgs: list[NormalizedMessage] = []
    out_msgs: list[NormalizedMessage] = []

    req_raw = attrs.get("gcp.vertex.agent.llm_request")
    if req_raw:
        try:
            req = orjson.loads(req_raw) if isinstance(req_raw, str) else req_raw
        except (ValueError, TypeError):
            req = None
        if isinstance(req, dict):
            sys_inst = (req.get("config") or {}).get("system_instruction")
            if sys_inst:
                in_msgs.append(
                    NormalizedMessage(position=0, direction=DIR_INPUT, role="system", content=str(sys_inst))
                )
            for content in req.get("contents") or []:
                in_msgs.extend(_genai_content_to_messages(content, DIR_INPUT))
            consumed.add("gcp.vertex.agent.llm_request")

    resp_raw = attrs.get("gcp.vertex.agent.llm_response")
    if resp_raw:
        try:
            resp = orjson.loads(resp_raw) if isinstance(resp_raw, str) else resp_raw
        except (ValueError, TypeError):
            resp = None
        if isinstance(resp, dict) and isinstance(resp.get("content"), dict):
            out_msgs.extend(_genai_content_to_messages(resp["content"], DIR_OUTPUT))
            consumed.add("gcp.vertex.agent.llm_response")

    return in_msgs, out_msgs


def _genai_content_to_messages(content: dict[str, Any], direction: str) -> list[NormalizedMessage]:
    """Decode a single `genai.Content` block into one or more messages.

    A Content has a role and a list of parts. Each part is one of:
      - `{"text": "..."}`              → message content text
      - `{"function_call": {...}}`     → assistant deciding to call a tool
      - `{"function_response": {...}}` → tool result coming back to the model
    """
    role_raw = content.get("role") or "user"
    role = "assistant" if role_raw == "model" else role_raw
    out: list[NormalizedMessage] = []
    for part in content.get("parts") or []:
        if not isinstance(part, dict):
            continue
        if "text" in part and part["text"] is not None:
            out.append(
                NormalizedMessage(position=0, direction=direction, role=role, content=str(part["text"]))
            )
        elif "function_call" in part:
            fc = part["function_call"] or {}
            out.append(
                NormalizedMessage(
                    position=0,
                    direction=direction,
                    role=role,
                    content=orjson.dumps(fc).decode(),
                    tool_call_id=fc.get("id"),
                )
            )
        elif "function_response" in part:
            fr = part["function_response"] or {}
            payload = fr.get("response", fr)
            out.append(
                NormalizedMessage(
                    position=0,
                    direction=direction,
                    role="tool",
                    content=orjson.dumps(payload).decode(),
                    tool_call_id=fr.get("id"),
                )
            )
    return out


def _genai_event_role(ename: str) -> str | None:
    # gen_ai.user.message -> "user", gen_ai.assistant.message -> "assistant", etc.
    parts = ename.split(".")
    if len(parts) >= 3 and parts[0] == "gen_ai" and parts[2] == "message":
        return parts[1]
    if ename == "gen_ai.choice":
        return "assistant"
    return None


# ---------------------------------------------------------------------------
# Vercel AI SDK / MLflow / Traceloop — minimal extractors
# ---------------------------------------------------------------------------


def _extract_vercel(attrs: dict[str, Any], consumed: set[str]) -> _Extracted:
    out = _Extracted()
    if "ai.model.id" in attrs:
        out.model_name = _coerce_text(attrs["ai.model.id"])
        consumed.add("ai.model.id")
    if "ai.toolCall.name" in attrs:
        out.tool_name = _coerce_text(attrs["ai.toolCall.name"])
        consumed.add("ai.toolCall.name")
        out.kind = KIND_TOOL
    if "ai.toolCall.args" in attrs:
        out.input_text = _coerce_text(attrs["ai.toolCall.args"])
        consumed.add("ai.toolCall.args")
    if "ai.toolCall.result" in attrs:
        out.output_text = _coerce_text(attrs["ai.toolCall.result"])
        consumed.add("ai.toolCall.result")
    out.input_tokens = _first_int(attrs, consumed, ("ai.usage.promptTokens",))
    out.output_tokens = _first_int(attrs, consumed, ("ai.usage.completionTokens",))
    if "ai.prompt" in attrs:
        out.input_text = out.input_text or _coerce_text(attrs["ai.prompt"])
        consumed.add("ai.prompt")
    if "ai.response.text" in attrs:
        out.output_text = out.output_text or _coerce_text(attrs["ai.response.text"])
        consumed.add("ai.response.text")
    if not out.kind and (out.model_name or out.input_text or out.output_text):
        out.kind = KIND_LLM
    if out.kind == KIND_TOOL:
        out.io_format = IO_TOOL_CALL
    elif out.kind == KIND_LLM:
        out.io_format = IO_TEXT
    return out


def _extract_mlflow(attrs: dict[str, Any], consumed: set[str]) -> _Extracted:
    out = _Extracted()
    # MLflow's GenAI tracing uses mlflow.spanType, mlflow.spanInputs, mlflow.spanOutputs.
    span_type = attrs.get("mlflow.spanType")
    if span_type is not None:
        out.kind = _OPENINFERENCE_KIND_MAP.get(str(span_type).upper(), KIND_UNKNOWN)
        consumed.add("mlflow.spanType")
    if "mlflow.spanInputs" in attrs:
        out.input_text = _coerce_text(attrs["mlflow.spanInputs"])
        consumed.add("mlflow.spanInputs")
    if "mlflow.spanOutputs" in attrs:
        out.output_text = _coerce_text(attrs["mlflow.spanOutputs"])
        consumed.add("mlflow.spanOutputs")
    if out.input_text or out.output_text:
        out.io_format = IO_TEXT
    return out


def _extract_traceloop(attrs: dict[str, Any], consumed: set[str]) -> _Extracted:
    out = _Extracted()
    # Traceloop's OpenLLMetry roughly mirrors OpenInference under traceloop.* keys.
    if "traceloop.span.kind" in attrs:
        out.kind = _OPENINFERENCE_KIND_MAP.get(
            str(attrs["traceloop.span.kind"]).upper(), KIND_UNKNOWN
        )
        consumed.add("traceloop.span.kind")
    if "traceloop.entity.input" in attrs:
        out.input_text = _coerce_text(attrs["traceloop.entity.input"])
        consumed.add("traceloop.entity.input")
    if "traceloop.entity.output" in attrs:
        out.output_text = _coerce_text(attrs["traceloop.entity.output"])
        consumed.add("traceloop.entity.output")
    if out.input_text or out.output_text:
        out.io_format = IO_TEXT
    return out


# ---------------------------------------------------------------------------
# Span links
# ---------------------------------------------------------------------------


def _extract_links(links: list[dict[str, Any]] | None) -> Iterator[NormalizedLink]:
    if not links:
        return
    for ln in links:
        yield NormalizedLink(
            linked_trace_id=ln.get("traceId") or None,
            linked_span_id=ln.get("spanId") or None,
        )


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return orjson.dumps(v).decode()
    return str(v)


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_int(attrs: dict[str, Any], consumed: set[str], keys: tuple[str, ...]) -> int | None:
    for k in keys:
        if k in attrs:
            consumed.add(k)
            val = _coerce_int(attrs[k])
            if val is not None:
                return val
    return None


def _join_messages(msgs: list[dict[str, str | None]]) -> str | None:
    if not msgs:
        return None
    return "\n".join(f"[{m.get('role') or '?'}] {m.get('content') or ''}" for m in msgs)


def _join_messages_typed(msgs: list[NormalizedMessage]) -> str | None:
    if not msgs:
        return None
    return "\n".join(f"[{m.role or '?'}] {m.content or ''}" for m in msgs)


# Default any_value re-export for tests.
__all__ = [
    "SPEC",
    "SPEC_VERSION",
    "NormalizedDocument",
    "NormalizedLink",
    "NormalizedMessage",
    "NormalizedSpan",
    "any_value",
    "normalize_span",
]
