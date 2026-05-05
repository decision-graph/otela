"""PyTorch tensor view of the otela tables.

Converts numeric columns into float tensors and string columns into
dictionary-encoded categorical-code tensors, suitable as input to
standard tabular ML models (gradient boosting, MLPs, anomaly detection,
cost prediction).

Long-text columns (`input_text`, `output_text`, `raw_attributes_json`,
message `content`) are *not* tokenized here — that is downstream of a
specific tokenizer, model, and use case. Reach for `otela.to_dicts()`
plus a HuggingFace tokenizer when you need NLP-shaped input.

PyTorch is an optional dependency. Import the package without it, but
calling `to_tensors()` raises a clear `ImportError`. Install with:

    pip install otela[ml]

Numerical precision
-------------------
Default dtype is `float32`, which loses precision for nanosecond
timestamps (`start_time_unix_nano` etc.). Pass `dtype=torch.float64`
when timing accuracy matters, or keep using the Arrow tables directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:  # pragma: no cover
    import torch


# Identity columns kept as Python lists rather than encoded — turning
# trace_id/span_id into category codes would explode the vocab to one
# entry per row and is never what an ML pipeline wants.
_IDENTITY_COLUMNS: dict[str, frozenset[str]] = {
    "spans": frozenset({"trace_id", "span_id", "parent_span_id"}),
    "traces": frozenset({"trace_id", "root_span_id"}),
    "messages": frozenset({"trace_id", "span_id", "tool_call_id"}),
    "documents": frozenset({"trace_id", "span_id", "document_id"}),
    "links": frozenset({"trace_id", "span_id", "linked_trace_id", "linked_span_id"}),
}

# `spec` and `spec_version` are constants on every row — including them
# as features would just be a free constant column.
_DROP_COLUMNS = frozenset({"spec", "spec_version"})

# Code emitted into the categorical tensor for null entries. -1 sits
# outside any vocab index and is the canonical sentinel in PyTorch
# embedding tables (e.g. `padding_idx=-1`).
NULL_CATEGORY_CODE = -1


@dataclass(slots=True)
class TensorTable:
    """Tensor view of a single otela table.

    Attributes
    ----------
    n_rows
        Row count (matches the source Arrow table).
    numeric
        Float tensors keyed by column name. Nulls are encoded as NaN.
    categorical
        Int64 code tensors keyed by column name. Nulls are encoded as
        `NULL_CATEGORY_CODE` (-1). Codes index into the matching `vocab` list.
    vocab
        Vocabulary lists per categorical column. `vocab[col][code]` recovers
        the original string.
    ids
        Identity columns kept as Python lists. Not tensorized because their
        cardinality equals the row count.
    skipped
        Names of columns that were not converted (large_string and any
        unsupported types). They are still available via the source Arrow
        table.
    """

    n_rows: int
    numeric: dict[str, torch.Tensor] = field(default_factory=dict)
    categorical: dict[str, torch.Tensor] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)
    ids: dict[str, list[str | None]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def to_tensors(
    tables: dict[str, pa.Table],
    *,
    table_names: tuple[str, ...] = ("spans", "traces"),
    dtype: Any = None,
) -> dict[str, TensorTable]:
    """Convert otela tables into PyTorch tensor bundles.

    Parameters
    ----------
    tables
        The dict returned by `otela.load()` (or `otela.to_arrow()`).
    table_names
        Which tables to tensorize. Defaults to the two wide tables
        (`spans`, `traces`); pass `("spans", "traces", "messages",
        "documents", "links")` to convert all five.
    dtype
        Numeric tensor dtype. Defaults to `torch.float32`. Use
        `torch.float64` when nanosecond timestamps need full precision.

    Returns
    -------
    `dict[str, TensorTable]` keyed by table name.
    """
    torch = _import_torch()
    if dtype is None:
        dtype = torch.float32

    out: dict[str, TensorTable] = {}
    for name in table_names:
        if name not in tables:
            continue
        out[name] = _table_to_tensors(
            tables[name],
            identity=_IDENTITY_COLUMNS.get(name, frozenset()),
            torch=torch,
            float_dtype=dtype,
        )
    return out


def _table_to_tensors(
    table: pa.Table,
    *,
    identity: frozenset[str],
    torch: Any,
    float_dtype: Any,
) -> TensorTable:
    bundle = TensorTable(n_rows=table.num_rows)

    for field_ in table.schema:
        col_name = field_.name
        col_type = field_.type

        if col_name in _DROP_COLUMNS:
            continue
        if col_name in identity:
            bundle.ids[col_name] = table.column(col_name).to_pylist()
            continue

        col = table.column(col_name)

        if pa.types.is_large_string(col_type):
            bundle.skipped.append(col_name)
            continue

        if pa.types.is_integer(col_type) or pa.types.is_floating(col_type):
            bundle.numeric[col_name] = _numeric_to_tensor(col, torch, float_dtype)
            continue

        if pa.types.is_string(col_type):
            codes, vocab = _string_to_codes(col, torch)
            bundle.categorical[col_name] = codes
            bundle.vocab[col_name] = vocab
            continue

        # Anything else (binary, list, struct, ...) is dropped with a note.
        bundle.skipped.append(col_name)

    return bundle


def _numeric_to_tensor(col: pa.ChunkedArray, torch: Any, dtype: Any) -> Any:
    """Cast an int/float column to a torch tensor of `dtype`, NaN for nulls.

    Uses `safe=False` on the int->float64 cast: Arrow's safe cast errors on
    int64 values above 2^53 (where float64 loses integer precision), but
    those are exactly the unix-nano timestamps users want as features. The
    precision tradeoff is documented at module level.
    """
    casted = pc.cast(col, pa.float64(), safe=False)
    filled = pc.fill_null(casted, float("nan"))
    np_arr = filled.combine_chunks().to_numpy(zero_copy_only=False)
    return torch.from_numpy(np_arr.copy()).to(dtype)


def _string_to_codes(col: pa.ChunkedArray, torch: Any) -> tuple[Any, list[str]]:
    """Dictionary-encode a string column. Returns (int64 codes, vocab list).

    Nulls become `NULL_CATEGORY_CODE` (-1).
    """
    encoded = pc.dictionary_encode(col).combine_chunks()
    dictionary = encoded.dictionary

    # Cast to int64 first — PyArrow chooses uint8/16/32 based on cardinality
    # and -1 won't fit in unsigned types. Then fill nulls with the sentinel
    # before going to numpy: `Array.to_numpy()` on a nullable int falls back
    # to float64 (NaN for nulls), which is not what we want for category
    # codes.
    indices_i64 = pc.cast(encoded.indices, pa.int64())
    filled = pc.fill_null(indices_i64, NULL_CATEGORY_CODE)
    np_codes = filled.to_numpy(zero_copy_only=False).copy()

    return torch.from_numpy(np_codes), dictionary.to_pylist()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "torch is not installed. Install with: pip install otela[ml]"
        ) from e
    return torch
