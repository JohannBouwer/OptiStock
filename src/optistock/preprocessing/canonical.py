"""The canonical order-line schema: names, dtypes, and the validator.

One registry, two consumers. :data:`ORDER_SCHEMA` *enforces* the contract on every
transform in this package, and :func:`describe_fields` *describes* it to the agents in
:mod:`optistock.ingest`. Because both read :data:`FIELDS`, the gate and the prompt can
never drift apart.

Pure pandas + pandera. No PyMC, no LLM.

Three ways to get a frame into this shape, in increasing order of mess: it already uses
the canonical names; only the names differ (:func:`rename_to_canonical`); the *shape*
differs too — a line total to divide, a ``dd/mm/yyyy`` string, a ``'0,90 €'`` price —
which is what :mod:`optistock.ingest` is for.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
import pandera.errors as pa_errors
import pandera.pandas as pa
from pandera.engines import pandas_engine

__all__ = [
    "CUSTOMER_ID",
    "CUSTOMER_SCHEMA",
    "DATE",
    "FIELDS",
    "FIELD_NAMES",
    "ITEM_ID",
    "ORDER_ID",
    "ORDER_SCHEMA",
    "QUANTITY",
    "REQUIRED_COLUMNS",
    "UNIT_COST",
    "UNIT_PRICE",
    "FieldSpec",
    "OrderValidationError",
    "describe_fields",
    "drop_invalid_rows",
    "rename_to_canonical",
    "validate",
]


@dataclass(frozen=True)
class FieldSpec:
    """One canonical field.

    ``dtype=None`` means *accept whatever the caller has* — see the note in
    :data:`FIELDS`. ``description`` is what the agents see, so it carries the guidance
    that stops a plausible-but-wrong mapping.
    """

    name: str
    dtype: str | None
    required: bool
    description: str


FIELDS: tuple[FieldSpec, ...] = (
    # `dtype=None` on the three identity columns is deliberate and load-bearing.
    # Coercing them to pandera's `string` rewrites an integer SKU 101 as "101" and a
    # float SKU 101.0 as "101.0". BaseStockKeep joins histories to item_configs by raw
    # value equality (`histories[item_col] == item.name`, stockkeep.py:371 and :1167),
    # so a hand-built item_configs with integer names would silently match zero rows
    # and fail deep inside the forecaster. Validation must not change what an
    # identifier *is*.
    FieldSpec(
        "order_id",
        None,
        True,
        "Identifier for the order/transaction a line belongs to. Groups lines "
        "bought together, so a 40-line invoice counts as one transaction.",
    ),
    FieldSpec(
        "customer_id",
        None,
        False,
        "Stable identifier for the buyer — a customer number, account id or loyalty "
        "id. Map it whenever the source has one, even if only some rows are "
        "populated: a partly-filled customer column is normal and still usable. "
        "Leave it unmapped only when the source has no buyer identifier at all, and "
        "never substitute a ticket/basket/invoice number, which would make every "
        "basket look like a different customer.",
    ),
    FieldSpec("date", "datetime64[ns]", True, "Date the order line was placed."),
    FieldSpec(
        "item_id",
        None,
        True,
        "Product/SKU identifier for the line. Becomes the 'item' column of to_sales.",
    ),
    FieldSpec(
        "quantity",
        "float64",
        True,
        "Units of the item on this line. Becomes 'sales' after aggregation. Must "
        "be strictly positive — returns and cancellations are not order lines.",
    ),
    FieldSpec(
        "unit_price",
        "float64",
        False,
        "Selling price per unit. Required by to_items and by to_clv(monetary=True). "
        "If the source only has a line total, divide it by quantity.",
    ),
    FieldSpec(
        "unit_cost",
        "float64",
        False,
        "Cost/COGS per unit. Required by to_items, which uses it as cost_price.",
    ),
)

FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)
REQUIRED_COLUMNS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.required)

ORDER_ID = "order_id"
CUSTOMER_ID = "customer_id"
DATE = "date"
ITEM_ID = "item_id"
QUANTITY = "quantity"
UNIT_PRICE = "unit_price"
UNIT_COST = "unit_cost"

QUANTITY_MUST_BE_POSITIVE = (
    "order line(s) have quantity <= 0. Returns and cancellations are not order "
    "lines — filter or net them out before calling."
)

# `pd.to_numeric(errors="coerce")` semantics, restored. pandera's `coerce=True` uses
# astype(), which raises on 'abc', leaves the column as object, and then lets the
# `> 0` check run against strings — surfacing a TypeError *as if it were a quantity
# violation*. Parsing first turns unparseable values into NaN, which the nullability
# check then reports honestly. The trailing astype pins float64 so an integer quantity
# column is not rejected as WRONG_DATATYPE.
_NUMERIC = pa.Parser(lambda s: pd.to_numeric(s, errors="coerce").astype("float64"))


def _column(spec: FieldSpec, *, force_required: bool = False) -> pa.Column:
    required = spec.required or force_required
    nullable = not required

    if spec.dtype is None:
        return pa.Column(
            nullable=nullable, required=required, coerce=False,
            description=spec.description,
        )
    if spec.dtype == "float64":
        checks = (
            [pa.Check.gt(0, error=QUANTITY_MUST_BE_POSITIVE)]
            if spec.name == QUANTITY
            else None
        )
        return pa.Column(
            "float64", nullable=nullable, required=required, coerce=False,
            parsers=_NUMERIC, checks=checks, description=spec.description,
        )
    return pa.Column(
        pandas_engine.Engine.dtype(spec.dtype),
        nullable=nullable, required=required, coerce=True,
        description=spec.description,
    )


# `strict=False` lets unmodeled columns (channel, category, country, ...) through
# untouched — `to_clv(covariates=[...])` depends on that. `coerce=False` at frame level
# matters just as much: a frame-level True would re-impose dtype coercion on the
# identity columns and undo the care taken above.
ORDER_SCHEMA = pa.DataFrameSchema(
    {f.name: _column(f) for f in FIELDS},
    strict=False, coerce=False, name="optistock_order_lines",
)

# `to_clv` and `to_sales(split=True)` additionally need a real buyer id: without one
# there is no such thing as a customer's first purchase day.
CUSTOMER_SCHEMA = pa.DataFrameSchema(
    {f.name: _column(f, force_required=f.name == CUSTOMER_ID) for f in FIELDS},
    strict=False, coerce=False, name="optistock_order_lines_with_customer",
)


class OrderValidationError(ValueError):
    """Order lines failed the canonical schema.

    Subclasses ``ValueError``. :attr:`failure_cases` is pandera's own frame — one row
    per offending cell — for callers that want to repair the data programmatically
    rather than read the rendered message.
    """

    def __init__(self, message: str, failure_cases: pd.DataFrame) -> None:
        super().__init__(message)
        self.failure_cases = failure_cases


def validate(
    orders: pd.DataFrame,
    schema: pa.DataFrameSchema = ORDER_SCHEMA,
) -> pd.DataFrame:
    """Enforce the canonical schema, returning a normalised copy.

    Every transform calls this first, so its rules are the real entry contract. Row
    order and the index are preserved, *orders* is never mutated, and columns outside
    the schema pass through untouched so they stay available as covariates.

    Pass :data:`CUSTOMER_SCHEMA` when a buyer id is mandatory. Raises
    :class:`OrderValidationError` with every violation reported at once.
    """
    try:
        return schema.validate(orders, lazy=True)
    except pa_errors.SchemaErrors as exc:
        raise OrderValidationError(
            _render(exc.failure_cases, orders), exc.failure_cases
        ) from exc


def _render(failure_cases: pd.DataFrame, orders: pd.DataFrame) -> str:
    """Turn pandera's failure-case frame into an actionable message.

    No checks of its own — :data:`ORDER_SCHEMA` stays the single definition of
    validity. This exists because ``str(SchemaErrors)`` is a JSON dump: the counts and
    row indices are all *present* in ``failure_cases`` but none of them are legible.
    """
    parts: list[str] = []
    check = failure_cases["check"].astype(str)

    missing = sorted(
        failure_cases.loc[check == "column_in_dataframe", "failure_case"].tolist()
    )
    if missing:
        parts.append(
            f"orders is missing required column(s): {missing}. "
            f"Present columns: {list(orders.columns)}."
        )

    # One unparseable date raises *two* failure cases — the coercion attempt and the
    # dtype check after it. Match both so the value is reported once.
    is_bad_date = check.str.contains(r"dtype\('datetime", regex=True)
    bad_dates = failure_cases.loc[is_bad_date].drop_duplicates(
        subset=["index", "failure_case"]
    )
    if not bad_dates.empty:
        parts.append(
            f"Could not parse {len(bad_dates)} value(s) in {DATE!r} as dates, e.g. "
            f"{bad_dates['failure_case'].head(5).tolist()}."
        )

    nulls = failure_cases.loc[check == "not_nullable"].groupby("column").size()
    if not nulls.empty:
        parts.append(
            "Required column(s) contain nulls (or unparseable values): "
            f"{ {k: int(v) for k, v in nulls.items()} }."
        )

    bad_qty = failure_cases.loc[check == QUANTITY_MUST_BE_POSITIVE]
    if not bad_qty.empty:
        parts.append(f"{len(bad_qty)} {QUANTITY_MUST_BE_POSITIVE}")

    if not bad_qty.empty or not nulls.empty:
        parts.append("Call drop_invalid_rows(orders) to discard those rows and continue.")

    # Anything this renderer does not recognise still has to surface, or a future
    # pandera failure class would produce a confidently empty error message.
    known = {"column_in_dataframe", "not_nullable", QUANTITY_MUST_BE_POSITIVE}
    unexplained = failure_cases.loc[~check.isin(known) & ~is_bad_date]
    if not unexplained.empty:
        parts.append("Other schema failures:\n" + unexplained.to_string(index=False))

    return "\n\n".join(parts)


def rename_to_canonical(
    orders: pd.DataFrame,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    """Rename source columns onto the canonical names. Nothing is dropped.

    The deterministic, LLM-free path for a frame whose columns are merely *named*
    differently. *columns* is ``{source_name: canonical_name}`` — the same direction as
    ``DataFrame.rename``. Unmapped columns carry through in place, so covariates
    survive.
    """
    unknown = sorted(set(columns) - set(orders.columns))
    if unknown:
        raise KeyError(
            f"source column(s) not in orders: {unknown}. "
            f"Present columns: {list(orders.columns)}. Note the mapping direction "
            "is {source: canonical}, not {canonical: source}."
        )

    not_canonical = sorted(set(columns.values()) - set(FIELD_NAMES))
    if not_canonical:
        # The commonest cause is writing the mapping backwards, which puts a canonical
        # name in the key slot and the source name in the value slot.
        hint = (
            " Every key you passed is already a canonical name — the mapping "
            "direction is {source: canonical}, not {canonical: source}."
            if set(columns) <= set(FIELD_NAMES)
            else ""
        )
        raise ValueError(
            f"not canonical field name(s): {not_canonical}. "
            f"Expected one of {list(FIELD_NAMES)}.{hint}"
        )

    # df.rename will happily produce two columns with the same label, without a
    # warning, and pandera then validates whichever comes first.
    collisions = sorted(set(columns.values()) & (set(orders.columns) - set(columns)))
    if collisions:
        raise ValueError(
            f"renaming would collide with existing column(s): {collisions}. "
            "Drop or rename those first."
        )

    return orders.rename(columns=dict(columns))


def drop_invalid_rows(orders: pd.DataFrame, *, warn: bool = True) -> pd.DataFrame:
    """Drop the rows the canonical schema will reject, so the rest can proceed.

    Two rules, both universal — a non-positive ``quantity`` and a null in a required
    column are never valid order lines. Everything source-specific (credit-note invoice
    prefixes, placeholder SKUs, guest checkouts) stays the caller's job.

    Be aware of what dropping costs: returns arrive as negative quantities in most
    exports, and those units *were* sold and later came back. The resulting demand is
    gross of returns and the return rate reads as 0% rather than "not modelled". Use
    :func:`~optistock.preprocessing.transforms.to_returns` on the **pre-drop** frame to
    keep them.

    The index is preserved, so the dropped rows remain identifiable by difference.
    """
    reasons: dict[str, int] = {}
    keep = pd.Series(True, index=orders.index)

    if QUANTITY in orders.columns:
        quantity = pd.to_numeric(orders[QUANTITY], errors="coerce")
        non_positive = quantity.le(0).fillna(False)
        if non_positive.any():
            reasons["quantity <= 0"] = int(non_positive.sum())
            keep &= ~non_positive

    for column in REQUIRED_COLUMNS:
        if column not in orders.columns:
            continue
        null = orders[column].isna()
        if column == QUANTITY:
            # An unparseable quantity is a null for our purposes.
            null = null | pd.to_numeric(orders[column], errors="coerce").isna()
        if null.any():
            reasons[f"null {column}"] = int(null.sum())
            keep &= ~null

    dropped = int((~keep).sum())
    detail = ", ".join(f"{count:,} {why}" for why, count in reasons.items())

    # Removing every row is not cleaning, it is a broken mapping — most often a
    # required column that parsed to all-NaN. Returning an empty frame would be
    # actively harmful: it validates cleanly, because no rows means no violations.
    if len(orders) and not keep.any():
        raise ValueError(
            f"drop_invalid_rows would remove all {len(orders):,} order lines "
            f"({detail}). That is a broken mapping rather than dirty data — check "
            "that every required column parsed, since a column of NaN takes every "
            "row with it. Nothing was dropped."
        )

    if warn and reasons:
        warnings.warn(
            f"drop_invalid_rows removed {dropped:,} of {len(orders):,} order lines "
            f"({dropped / len(orders):.1%}): {detail}. Negative quantities are "
            "usually returns — the remaining demand is gross of them, and the return "
            "rate now reads as 0% rather than 'not recorded'.",
            UserWarning,
            stacklevel=2,
        )

    return orders[keep].copy()


def describe_fields() -> str:
    """Render the schema as prompt context for the agents.

    Lives here rather than in :mod:`optistock.ingest` so it reads the same
    :data:`FIELDS` registry the validator does — an agent can therefore never be told
    about a field the gate does not enforce, or vice versa.
    """
    lines = []
    for f in FIELDS:
        tag = "required" if f.required else "optional"
        dtype = f.dtype or "any (identifier — dtype is preserved as-is)"
        lines.append(f"- {f.name} ({dtype}, {tag}): {f.description}")
    return "\n".join(lines)
