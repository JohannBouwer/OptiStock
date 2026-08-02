"""What the agents are allowed to see: a compact description of a table.

Two views, both pure description with no judgement, so neither can be wrong:

    profile_dataframe(df)   per column — name, dtype, null rate, cardinality, samples
    relational_facts(df)    how the canonical columns relate — orders per customer,
                            date span, lines per order

We never hand a model the whole frame. That keeps token usage bounded and independent
of row count, and means no customer record leaves the machine: the truncated sample
values are the only data that travels.

The second view is what a per-column profile cannot give you, and it is where a
swapped or collapsed column announces itself. A day counter mapped to ``date`` looks
unremarkable column by column and absurd as ``date_span_days=0.0`` across 2.6M lines.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing.canonical import (
    CUSTOMER_ID,
    DATE,
    ITEM_ID,
    ORDER_ID,
    QUANTITY,
    UNIT_PRICE,
)

__all__ = [
    "ColumnProfile",
    "TableProfile",
    "profile_dataframe",
    "read_table",
    "relational_facts",
    "render_facts",
]


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    non_null: int
    null_rate: float
    n_unique: int
    samples: list[str]


@dataclass(frozen=True)
class TableProfile:
    n_rows: int
    columns: list[ColumnProfile]

    def render(self) -> str:
        """Prompt-ready text description of the table."""
        lines = [f"Source table: {self.n_rows} rows, {len(self.columns)} columns.", ""]
        for c in self.columns:
            sample_str = ", ".join(c.samples) if c.samples else "(all null)"
            lines.append(
                f"- {c.name!r} | dtype={c.dtype} | null_rate={c.null_rate:.0%} "
                f"| unique={c.n_unique} | samples: {sample_str}"
            )
        return "\n".join(lines)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read csv/tsv/xlsx/parquet into a DataFrame (everything as-is, no coercion)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix!r} (path={path})")


def profile_dataframe(df: pd.DataFrame, n_samples: int = 5) -> TableProfile:
    """Summarise each column into a compact, model-friendly profile."""
    columns: list[ColumnProfile] = []
    n_rows = len(df)
    for name in df.columns:
        col = df[name]
        non_null = int(col.notna().sum())
        null_rate = 0.0 if n_rows == 0 else 1.0 - non_null / n_rows
        # First few distinct non-null values, stringified and length-capped.
        distinct = col.dropna().unique()[:n_samples]
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(col.dtype),
                non_null=non_null,
                null_rate=null_rate,
                n_unique=int(col.nunique(dropna=True)),
                samples=[str(v)[:40] for v in distinct],
            )
        )
    return TableProfile(n_rows=n_rows, columns=columns)


def _as_dates(frame: pd.DataFrame) -> pd.Series:
    """The date column coerced the way the schema will coerce it.

    This may run *before* validation, so ``date`` can still hold source strings or,
    more importantly, source integers. Coercing here mirrors what ``ORDER_SCHEMA``
    does later, which is what makes an epoch collapse visible in the facts rather than
    after the damage.

    Pandas' own parsing chatter is suppressed: the coercion exists only to compute a
    statistic, and a describer that emits someone else's warnings is noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(frame[DATE], errors="coerce").dropna()


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def relational_facts(orders: pd.DataFrame) -> dict[str, object]:
    """The facts a per-column profile cannot show.

    :func:`profile_dataframe` covers dtype, null rate, cardinality and sample values.
    What it cannot know is how the columns *relate* — orders per customer, the date
    span, lines per order — and that is where a swapped or collapsed column announces
    itself. Pure description, no judgement.

    Every key is optional: a column that is absent or entirely null is simply not
    described, rather than reported as zero.
    """
    out: dict[str, object] = {"n_order_lines": int(len(orders))}
    if not len(orders):
        return out

    if ORDER_ID in orders:
        out["n_orders"] = int(orders[ORDER_ID].nunique(dropna=True))
        out["lines_per_order"] = round(len(orders) / max(out["n_orders"], 1), 2)
    if CUSTOMER_ID in orders:
        out["n_customers"] = int(orders[CUSTOMER_ID].nunique(dropna=True))
        if ORDER_ID in orders:
            per = (
                orders.dropna(subset=[CUSTOMER_ID, ORDER_ID])
                .groupby(CUSTOMER_ID)[ORDER_ID]
                .nunique()
            )
            if len(per):
                out["orders_per_customer"] = round(float(per.mean()), 2)
                out["share_customers_with_one_order"] = round(float(per.eq(1).mean()), 4)
    if ITEM_ID in orders:
        out["n_items"] = int(orders[ITEM_ID].nunique(dropna=True))

    dates = (
        _as_dates(orders)
        if DATE in orders and orders[DATE].notna().any()
        else pd.Series([], dtype="datetime64[ns]")
    )
    if len(dates):
        out["first_date"] = str(dates.min())
        out["last_date"] = str(dates.max())
        out["date_span_days"] = round(
            (dates.max() - dates.min()).total_seconds() / 86400, 4
        )
        out["n_distinct_dates"] = int(dates.dt.normalize().nunique())

    if QUANTITY in orders and orders[QUANTITY].notna().any():
        q = _as_numeric(orders, QUANTITY).dropna()
        if len(q):
            out["quantity_modal_value"] = float(q.mode().iat[0])
            out["quantity_modal_share"] = round(float(q.eq(q.mode().iat[0]).mean()), 4)
            out["quantity_is_integral"] = bool(np.allclose(q % 1, 0))
            out["quantity_min"] = float(q.min())
            out["quantity_max"] = float(q.max())
    if UNIT_PRICE in orders and orders[UNIT_PRICE].notna().any():
        p = _as_numeric(orders, UNIT_PRICE).replace([np.inf, -np.inf], np.nan).dropna()
        if len(p):
            out["unit_price_median"] = round(float(p.median()), 4)
            out["unit_price_share_non_positive"] = round(float(p.le(0).mean()), 4)
    return out


def render_facts(facts: dict[str, object]) -> str:
    """One fact per line, for a prompt or a console."""
    return "\n".join(f"  {k} = {v}" for k, v in facts.items())
