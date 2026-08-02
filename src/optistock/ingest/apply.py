"""Deterministic execution of a proposed mapping, plus schema validation.

This is the "code disposes" half. An agent proposed a :class:`ColumnMapping`; here it
is executed with plain pandas and the result is validated against the canonical schema.
No LLM involvement, fully reproducible, and importable without ``pydantic-ai``.

Everything in here that looks paranoid is: each guard below was written against a
mapping a real model actually returned.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

from ..preprocessing.canonical import (
    FIELDS,
    QUANTITY,
    OrderValidationError,
    drop_invalid_rows,
    validate,
)
from ..preprocessing.transforms import to_returns
from .agents import ColumnMapping, FieldMapping
from .profile import relational_facts, render_facts

__all__ = ["LOW_CONFIDENCE", "ValidationResult", "apply_mapping", "to_canonical"]

#: Bar below which a field is listed in ``ValidationResult.low_confidence``. Reporting
#: only — nothing here gates on it.
LOW_CONFIDENCE = 0.5

# `dtype` is None for the identity fields, whose dtype is deliberately preserved as-is
# — see the note in optistock.preprocessing.canonical.FIELDS.
_DTYPE_BY_FIELD = {f.name: f.dtype for f in FIELDS}


def _parse_numeric(series: pd.Series, numeric_format: str) -> pd.Series:
    """Turn formatted text like ``'0,90 €'`` or ``'$1,234.50'`` into floats."""
    text = series.astype("string").str.strip()
    if numeric_format == "comma_decimal":
        # Drop thousands separators ('.' and spaces incl. non-breaking), then ',' -> '.'
        text = text.str.replace(r"[^\d,\-]", "", regex=True).str.replace(
            ",", ".", regex=False
        )
    else:  # "currency" — strip everything but digits, sign and the decimal point
        text = text.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _coerce_constant(field_name: str, raw: str):
    """Turn a stringified constant into the right Python type for the target field.

    Deliberately *not* forgiving: a model once returned ``'，1'`` with a fullwidth
    comma, and stripping the stray character to read it as ``1.0`` would turn a garbled
    response into a plausible quantity. A loud crash beats a quiet fabrication.
    """
    if _DTYPE_BY_FIELD.get(field_name) == "float64":
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name}: constant {raw!r} is not a number, so it cannot fill a "
                f"numeric field. Use strategy='direct' with a source column, or "
                f"strategy='missing' if the source has no such value."
            ) from exc
    return raw


def _recover_source_column(name: str, columns) -> str | None:
    """Salvage a column name a model buried its reasoning behind.

    Constrained decoding forces a string, and a model that wants to explain itself
    sometimes overflows into the field — observed from gemini-3-flash, which returned
    1,410 characters beginning with the real column name ``'TRANSACTION_DT'``.

    Recovery is deliberately narrow: the value must *start with* exactly one column
    name, taking the longest match so ``item`` never wins over ``item_id``. Anything
    ambiguous returns None and the caller raises, because guessing which column was
    meant is how the wrong data gets loaded silently.
    """
    candidates = [c for c in map(str, columns) if name.startswith(c) and name != c]
    if not candidates:
        return None
    longest = max(candidates, key=len)
    if sum(1 for c in candidates if len(c) == len(longest)) > 1:
        return None  # two equally-long prefixes: no way to choose
    return longest


def _resolve(df: pd.DataFrame, m: FieldMapping) -> pd.Series | None:
    """Produce the canonical column for a single field mapping (None if 'missing')."""
    if m.strategy == "missing":
        return None

    if m.strategy == "direct":
        if not m.source_column:
            raise KeyError(f"{m.canonical_field}: strategy 'direct' with no column")
        if m.source_column in df.columns:
            return df[m.source_column]

        recovered = _recover_source_column(m.source_column, df.columns)
        if recovered is not None:
            warnings.warn(
                f"{m.canonical_field}: source_column was {len(m.source_column)} "
                f"characters of text beginning with the real column name "
                f"{recovered!r} — the model appears to have written its reasoning "
                f"into the field. Using {recovered!r}. Check this field: a mapping "
                f"that malformed may be wrong in ways that are not recoverable.",
                UserWarning,
                stacklevel=3,
            )
            return df[recovered]

        shown = (
            m.source_column
            if len(m.source_column) <= 80
            else m.source_column[:77] + "..."
        )
        raise KeyError(
            f"{m.canonical_field}: source_column {shown!r} is not in the source "
            f"table. Present columns: {list(df.columns)}."
        )

    if m.strategy == "expression":
        if not m.expression:
            raise ValueError(
                f"{m.canonical_field}: strategy 'expression' but no expression given"
            )
        # df.eval restricts to column references + arithmetic — no arbitrary code
        # execution. TypeError is in the net because adding a text column to a numeric
        # one raises that, not ValueError: a model reaching for
        # 'TRANSACTION_DT + CUSTOMER_ID' is trying to *concatenate* identifiers into a
        # composite key, which arithmetic cannot express.
        try:
            return df.eval(m.expression)
        except (ValueError, TypeError) as exc:
            hint = (
                "Adding a text column to a numeric one usually means trying to join "
                "two identifiers into one key, which arithmetic cannot do. Pick a "
                "single source column with strategy='direct', or use "
                "strategy='missing' if the source has no such field."
                if isinstance(exc, TypeError)
                else "Expressions support arithmetic over columns only; dates must "
                "use strategy='direct' with a 'date_format'."
            )
            raise ValueError(
                f"{m.canonical_field}: expression {m.expression!r} is not evaluable "
                f"({type(exc).__name__}: {exc}). {hint}"
            ) from exc

    if m.strategy == "constant":
        if m.constant is None:
            raise ValueError(
                f"{m.canonical_field}: strategy 'constant' but no constant given"
            )
        return pd.Series(
            [_coerce_constant(m.canonical_field, m.constant)] * len(df), index=df.index
        )

    raise ValueError(f"{m.canonical_field}: unknown strategy {m.strategy!r}")


def _warn_if_unparseable(m: FieldMapping, series: pd.Series, dtype: str) -> None:
    """Flag a numeric or date field whose source had data but parsed to all-NaN.

    The silent-failure case this exists for: a price column of ``'0,90 €'`` mapped
    without ``numeric_format``. The schema coerces it and every value becomes NaN — and
    because the optional fields are nullable, the frame *validates*. A column of NaN,
    no error, and zero revenue downstream.
    """
    if not (dtype.startswith("datetime") or dtype == "float64"):
        return
    if series.isna().all() or series.empty:
        return  # the source was empty to begin with — nothing was lost

    parsed = (
        pd.to_datetime(series, errors="coerce")
        if dtype.startswith("datetime")
        else pd.to_numeric(series, errors="coerce")
    )
    if not parsed.isna().all():
        return

    hint = (
        "set date_format to the strptime pattern of the source text"
        if dtype.startswith("datetime")
        else "set numeric_format='comma_decimal' or 'currency'"
    )
    warnings.warn(
        f"{m.canonical_field!r} was mapped from {m.source_column or m.expression!r} "
        f"but every value fails to parse as {dtype} — e.g. "
        f"{series.dropna().astype(str).head(3).tolist()}. The column will be all-NaN "
        f"and, because it is nullable, validation will still pass. Fix the mapping: "
        f"{hint}.",
        UserWarning,
        stacklevel=3,
    )


def apply_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Build a canonical frame from the source frame and a proposed mapping.

    Only the mapped fields and the columns named in ``unmapped_source_columns``
    survive; anything the mapping does not mention is dropped, and the index is reset.
    If all you need is a rename, use
    :func:`optistock.preprocessing.rename_to_canonical`, which keeps both.
    """
    out = pd.DataFrame(index=df.index)
    for m in mapping.mappings:
        series = _resolve(df, m)
        if series is None:
            continue
        dtype = _DTYPE_BY_FIELD.get(m.canonical_field) or ""
        # Parse dates with the format the agent declared, so an ambiguous dd/mm/yyyy
        # is not silently read as mm/dd/yyyy by pandas' inference.
        if dtype.startswith("datetime") and m.date_format:
            series = pd.to_datetime(series, format=m.date_format, errors="coerce")
        # Numeric columns arriving as formatted text ('0,90 €') need explicit parsing;
        # schema coercion would fail or silently produce NaN.
        elif m.numeric_format:
            series = _parse_numeric(series, m.numeric_format)
        _warn_if_unparseable(m, series, dtype)
        out[m.canonical_field] = series.values

    # Pass through covariates the agent didn't consume.
    for col in mapping.unmapped_source_columns:
        if col in df.columns and col not in out.columns:
            out[col] = df[col].values

    return out.reset_index(drop=True)


@dataclass
class ValidationResult:
    """Outcome of :func:`to_canonical`.

    ``ok=False`` still returns the built frame, so you can inspect what the mapping
    produced rather than only being told it was wrong. ``ok`` is driven by
    :func:`~optistock.preprocessing.validate` **alone** — nothing else here can make a
    mapping invalid.

    ``confidence`` is the agent's own self-assessment per field, exposed as data rather
    than as a gate: across real datasets ``0.00`` has appeared on the worst mapping
    seen *and* on a correct one. ``facts`` is the relational view the review agent
    reads, and usually the fastest way for a human to spot a bad mapping.
    """

    ok: bool
    frame: pd.DataFrame | None
    error: str | None = None
    confidence: dict[str, float] = field(default_factory=dict)
    #: Fields below :data:`LOW_CONFIDENCE`, excluding ``strategy='missing'`` entries —
    #: declining a field the source lacks is a right answer, not a doubt, and a
    #: confident 'missing' would otherwise dominate this list.
    low_confidence: tuple[str, ...] = ()
    facts: dict[str, object] = field(default_factory=dict)
    #: The return lines, captured **before** ``drop_invalid_rows`` removed them.
    #: Computing this here is the only chance: once ``frame`` has been dropped and
    #: validated there is nothing negative left to find, so a caller who wanted the
    #: return rate would silently read 0%.
    returns: pd.DataFrame | None = None

    def render(self) -> str:
        lines = [
            f"ok={self.ok}"
            + ("" if self.frame is None else f"  rows={len(self.frame):,}")
        ]
        if self.error:
            lines.append(f"\n{self.error}")
        if self.low_confidence:
            lines.append(
                f"\nlow confidence (< {LOW_CONFIDENCE}): {list(self.low_confidence)}"
            )
        if self.returns is not None and len(self.returns):
            lines.append(f"\nreturns held back: {len(self.returns):,} line(s)")
        if self.facts:
            lines.append("\nfacts:\n" + render_facts(self.facts))
        return "\n".join(lines)


def to_canonical(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    drop_invalid: bool = True,
) -> ValidationResult:
    """Apply the mapping and validate the result against the canonical schema.

    Returns a result object rather than raising, because a partial mapping is the
    normal case when profiling an unfamiliar source: you want to see the failure and
    the frame side by side. The transforms in :mod:`optistock.preprocessing` are the
    ones that refuse outright.

    ``drop_invalid`` removes the rows the schema would reject — non-positive quantities
    and nulls in required columns — before validating. **On by default**, since returns
    arriving as negative quantities are near-universal and would otherwise block the
    demand panel. Every drop emits a ``UserWarning`` naming the count and the reason;
    the resulting demand is gross of returns, so call
    :func:`~optistock.preprocessing.to_returns` on the source frame if you want them.
    """
    built = apply_mapping(df, mapping)
    facts = relational_facts(built)
    confidence = {m.canonical_field: m.confidence for m in mapping.mappings}
    low = tuple(
        m.canonical_field
        for m in mapping.mappings
        if m.strategy != "missing" and m.confidence < LOW_CONFIDENCE
    )
    # Captured before the drop, which is the only moment they exist.
    returns = to_returns(built) if QUANTITY in built.columns else None

    if drop_invalid:
        built = drop_invalid_rows(built)

    common = dict(
        confidence=confidence, low_confidence=low, facts=facts, returns=returns
    )
    try:
        validated = validate(built)
    except OrderValidationError as exc:
        return ValidationResult(ok=False, frame=built, error=str(exc), **common)
    return ValidationResult(ok=True, frame=validated, **common)
