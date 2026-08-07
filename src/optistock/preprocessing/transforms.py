"""Deterministic transforms from the canonical order-line table to model-ready frames.

One row per order line goes in; the frames the models consume come out:

    to_sales(orders)    → long daily panel (date, item, sales) for the forecasters
                          and the ``BaseStockKeep`` orchestrators
    to_items(orders)    → per-item economics (``item_configs``) for ``Item``/the solver
    to_clv(orders)      → per-customer RFM summary for ``ParetoNBD``/``GammaGammaSpend``
    to_returns(orders)  → the negative-quantity lines, as positive return counts

All four derive from the same source, so the model families can never disagree about
what happened. Every one validates first, so :mod:`~optistock.preprocessing.canonical`
is the entry contract rather than a convention.

Call :func:`available_transforms` first when the source is unfamiliar — real order books
are routinely partial, and it is cheaper to ask than to crash three steps later.

New vs returning customers
--------------------------
``to_sales(split=True)`` splits every item's daily sales into

    sales_new         units bought on a customer's *first* purchase day (acquisition)
    sales_returning   units bought on every later day (retention)

That partition is not arbitrary. ``rfm_summary`` is called with
``include_first_transaction=False`` (its default), so ``frequency`` counts *repeat*
transactions only and the Pareto/NBD likelihood is defined over exactly the transactions
that land in ``sales_returning``. The two branches carve the order book cleanly, with no
double counting. ``rfm_summary`` also treats all of a customer's purchases on one day as
a single transaction, so the split here is made per customer-**day** rather than per
order id — see :func:`split_new_returning`.

Pure pandas + pandera; ``pymc-marketing`` is imported lazily inside :func:`to_clv`.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import pandas as pd

from .canonical import (
    CUSTOMER_ID,
    CUSTOMER_SCHEMA,
    DATE,
    ITEM_ID,
    ORDER_ID,
    ORDER_SCHEMA,
    QUANTITY,
    REQUIRED_COLUMNS,
    UNIT_COST,
    UNIT_PRICE,
    validate,
)

__all__ = [
    "OPTISTOCK_CORE_COLS",
    "available_transforms",
    "split_new_returning",
    "to_clv",
    "to_items",
    "to_returns",
    "to_sales",
]

# Columns OptiStock treats as item economics. ANY other column in item_configs is read
# by `BaseStockKeep._create_items` as a per-item *constraint*, so we emit exactly this
# set and no more. Kept in sync with `BaseStockKeep._CORE_COLS` by
# tests/test_to_items.py::test_core_columns_match_the_orchestrator.
OPTISTOCK_CORE_COLS = [
    "name",
    "cost_price",
    "selling_price",
    "salvage_value",
    "lead_time",
]


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def available_transforms(orders: pd.DataFrame) -> dict[str, str | None]:
    """Report which transforms this frame can feed, and why not.

    Not validated — this is the cheap check you run *before* committing to a transform,
    so it only looks at which columns carry usable data. Returns
    ``{transform_name: None if runnable else reason}``.
    """

    def usable(col: str) -> bool:
        return col in orders.columns and not orders[col].isna().all()

    def blocker(*needed: str, otherwise: str) -> str | None:
        """Report the schema's required columns first, then the transform's own.

        Every transform below calls :func:`validate` before doing anything, so a frame
        missing a *required* canonical column cannot feed any of them however complete
        the rest of it looks. Checking only the transform-specific columns was how a
        Ta Feng export with no ``order_id`` at all was reported as ``to_clv OK``.
        """
        absent = [c for c in REQUIRED_COLUMNS if not usable(c)]
        if absent:
            return f"needs the required column(s) {absent}, which the schema demands"
        return None if all(usable(c) for c in needed) else otherwise

    has_returns = QUANTITY in orders.columns and bool(
        pd.to_numeric(orders[QUANTITY], errors="coerce").lt(0).any()
    )

    return {
        "to_sales": blocker(
            ITEM_ID, DATE, QUANTITY, otherwise=f"needs {ITEM_ID}, {DATE} and {QUANTITY}"
        ),
        "to_sales(split=True)": blocker(
            ITEM_ID, DATE, QUANTITY, CUSTOMER_ID,
            otherwise=f"needs {CUSTOMER_ID} on top of the plain panel — without a "
            "buyer id there is no first purchase day to split on",
        ),
        "to_items": blocker(
            ITEM_ID, UNIT_PRICE, UNIT_COST,
            otherwise=f"needs {UNIT_COST} (and {UNIT_PRICE}, {ITEM_ID}) — no cost "
            "data present",
        ),
        "to_clv": blocker(
            CUSTOMER_ID, DATE, UNIT_PRICE,
            otherwise=f"needs {CUSTOMER_ID} — absent in point-of-sale data without a "
            "loyalty id",
        ),
        # to_returns is the exception: it never validates, because the rows it wants
        # are exactly the ones the schema rejects.
        "to_returns": None
        if has_returns
        else f"no {QUANTITY} < 0 rows — either this source records no returns, or "
        "they were already dropped by drop_invalid_rows",
    }


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------


def _tag_acquisition_lines(
    df: pd.DataFrame,
    observation_start: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Boolean Series (aligned to *df*) flagging lines from each customer's first day.

    Assigned per customer-**day**, never per line: a multi-item first order must stay
    whole, and ``rfm_summary`` collapses same-day transactions into one occasion. An
    order-level rule would count a customer's second same-day order as retention while
    the CLV frame still counted the day as their acquisition.
    """
    days = df[DATE].dt.normalize()
    first_day = days.groupby(df[CUSTOMER_ID]).transform("min")
    is_new = days.eq(first_day)

    if observation_start is not None:
        # Customers already active when observation began are left-censored: their
        # first *observed* purchase is not an acquisition.
        is_new &= first_day > pd.Timestamp(observation_start)

    return is_new


def split_new_returning(
    orders: pd.DataFrame,
    *,
    observation_start: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition order lines into acquisition and retention.

    A line is *new* when it falls on its customer's first purchase day. ``customer_id``
    is mandatory: without one there is no such thing as a first purchase day.

    *observation_start* left-censors — customers whose first purchase falls on or before
    it are assumed to have history predating the data, so all their lines are treated as
    returning. The default treats every customer's first observed purchase as a genuine
    acquisition, correct only when the data covers the full customer lifetime.

    Returns ``(new_lines, returning_lines)``: disjoint, index-preserving, and together
    they reconstruct the validated input.
    """
    df = validate(orders, CUSTOMER_SCHEMA)
    is_new = _tag_acquisition_lines(df, observation_start)
    return df[is_new].copy(), df[~is_new].copy()


def to_sales(
    orders: pd.DataFrame,
    *,
    date_col: str = "date",
    item_col: str = "item",
    target: str = "sales",
    split: bool = True,
    fill_calendar: bool = True,
    freq: str = "D",
    min_nonzero_days: int = 1,
    observation_start: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Order lines → the long panel the forecasting orchestrators consume.

    Output is exactly what ``BaseStockKeep(histories=...)`` expects: one row per
    (period, item), sorted by date then item.

    Parameters
    ----------
    date_col, item_col, target : str
        Column names for the *output*, matching ``BaseStockKeep``'s arguments.
    split : bool
        Also emit ``{target}_new`` and ``{target}_returning``, which sum to ``target``.
        Compute this on the **whole order book**, then subset items from the result:
        filtering to a few SKUs first redefines "first purchase day" within that subset,
        so long-standing customers get relabelled as new and ``{target}_new`` inflates
        silently.
    fill_calendar : bool
        Reindex onto the complete period × item grid, filling gaps with zero. **Leave
        this on.** Order data is sparse and the forecasters use row order as the time
        index, so an unfilled panel distorts the time axis and biases the level upward.
        The filled zeros mean "no demand", not "closed".
    min_nonzero_days : int
        Items with fewer non-zero periods are dropped with a warning. The default of 1
        removes only all-zero series, which would make ``BayesTimeSeries.fit`` divide by
        a ``max_scaler`` of zero and fail opaquely inside the sampler.
    observation_start : pd.Timestamp or str, optional
        Left-censoring cutoff; affects the split columns only. Pass the same value to
        :func:`to_clv` so the two frames agree on who is new.
    """
    df = validate(orders, CUSTOMER_SCHEMA if split else ORDER_SCHEMA)

    new_target = f"{target}_new"
    returning_target = f"{target}_returning"

    def _panel(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return (
            frame.groupby([pd.Grouper(key=DATE, freq=freq), ITEM_ID], sort=True)[
                QUANTITY
            ]
            .sum()
            .rename(name)
            .to_frame()
        )

    panel = _panel(df, target)

    if split:
        is_new = _tag_acquisition_lines(df, observation_start)
        panel = panel.join(
            [_panel(df[is_new], new_target), _panel(df[~is_new], returning_target)],
            how="outer",
        )

    if fill_calendar:
        periods = panel.index.get_level_values(0)
        panel = panel.reindex(
            pd.MultiIndex.from_product(
                [
                    pd.date_range(periods.min(), periods.max(), freq=freq),
                    sorted(df[ITEM_ID].unique()),
                ],
                names=[DATE, ITEM_ID],
            )
        )

    panel = panel.fillna(0.0).astype(float)
    panel.index.names = [date_col, item_col]
    out = panel.reset_index().sort_values([date_col, item_col], kind="mergesort")

    nonzero_days = out.groupby(item_col)[target].apply(lambda s: int((s > 0).sum()))
    dropped = sorted(nonzero_days[nonzero_days < min_nonzero_days].index)
    if dropped:
        warnings.warn(
            f"Dropping item(s) with fewer than {min_nonzero_days} non-zero "
            f"period(s) of {target!r}: {dropped}. A constant-zero series has no "
            "scale for the forecasters to normalise against.",
            UserWarning,
            stacklevel=2,
        )
        out = out[~out[item_col].isin(dropped)]

    if split:
        degenerate = sorted(
            out.groupby(item_col)[new_target]
            .apply(lambda s: int((s > 0).sum()))
            .pipe(lambda s: s[s < min_nonzero_days])
            .index
        )
        if degenerate:
            warnings.warn(
                f"Item(s) {degenerate} have no first-time-buyer sales in this "
                f"window, so {new_target!r} is constant zero for them. They are "
                f"kept — {target!r} is still usable — but forecasting "
                f"{new_target!r} directly for these items will fail.",
                UserWarning,
                stacklevel=2,
            )

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def to_returns(orders: pd.DataFrame) -> pd.DataFrame:
    """The negative-quantity lines, as positive return counts.

    The counterpart to :func:`~optistock.preprocessing.canonical.drop_invalid_rows`,
    which discards these rows so the demand panel can be built. Run this on the
    **pre-drop** frame — after dropping there is nothing left to find, and the return
    rate silently reads as 0% rather than "not recorded".

    Deliberately *not* validated: ``ORDER_SCHEMA`` requires ``quantity > 0``, which is
    precisely what these rows are not. The index is preserved, so a returned line can be
    traced back to the row it came from.

    Returns
    -------
    pd.DataFrame
        The canonical columns of the ``quantity < 0`` rows, with ``quantity`` sign-
        flipped to a positive count of units returned. Empty (with the same columns)
        when the source records no returns.
    """
    if QUANTITY not in orders.columns:
        raise ValueError(
            f"to_returns requires a {QUANTITY!r} column. Present columns: "
            f"{list(orders.columns)}."
        )

    quantity = pd.to_numeric(orders[QUANTITY], errors="coerce")
    out = orders[quantity.lt(0).fillna(False)].copy()
    out[QUANTITY] = quantity[out.index].abs()
    return out


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------


def to_items(
    orders: pd.DataFrame,
    *,
    price_agg: str = "median",
    salvage_value: float | dict[str, float] = 0.0,
    lead_time: int | dict[str, int] = 0,
    constraints: dict[str, float | dict[str, float]] | None = None,
    validate_prices: bool = True,
) -> pd.DataFrame:
    """Order lines → the per-item economics frame the orchestrators consume.

    Emits exactly :data:`OPTISTOCK_CORE_COLS` — the columns
    ``BaseStockKeep._create_items`` reads — sorted by ``name``, plus any *constraints*.

    Demand history and item economics are deliberately separate: ``to_sales`` needs only
    quantities, so a dataset with no cost column can still drive a forecast — it just
    cannot price the decision.

    Parameters
    ----------
    price_agg : str
        How to collapse per-line prices into one number per item. ``"median"`` (default)
        is robust to promotions; ``"mean"`` and ``"last"`` also work.
    salvage_value, lead_time : float or dict
        A scalar applied to every item, or a ``{item_id: value}`` mapping.
    constraints : dict, optional
        Extra per-item constraint columns, e.g. ``{"storage": 20}``. These become
        ``Item.constraints``.
    validate_prices : bool
        Check ``Item``'s invariant ``selling_price > cost_price > salvage_value >= 0``
        up front, so a bad item fails here rather than deep inside the solver.
    """
    df = validate(orders)

    if UNIT_PRICE not in df.columns or df[UNIT_PRICE].isna().all():
        raise ValueError(
            f"to_items requires {UNIT_PRICE!r} to set selling_price, but the frame "
            "has no usable price data. Map it during ingest — demand forecasting "
            "via to_sales() does not need it."
        )
    if UNIT_COST not in df.columns or df[UNIT_COST].isna().all():
        raise ValueError(
            f"to_items requires {UNIT_COST!r} to set cost_price, but the frame has "
            "no usable cost data. Map it during ingest, or build item economics "
            "yourself — demand forecasting via to_sales() does not need it."
        )

    econ = (
        df.groupby(ITEM_ID)
        .agg({UNIT_COST: price_agg, UNIT_PRICE: price_agg})
        .reset_index()
    )
    configs = pd.DataFrame(
        {
            "name": econ[ITEM_ID],
            "cost_price": econ[UNIT_COST].astype(float),
            "selling_price": econ[UNIT_PRICE].astype(float),
        }
    )
    configs["salvage_value"] = (
        configs["name"].map(salvage_value).fillna(0.0).astype(float)
        if isinstance(salvage_value, dict)
        else float(salvage_value)
    )
    configs["lead_time"] = (
        configs["name"].map(lead_time).fillna(0).astype(int)
        if isinstance(lead_time, dict)
        else int(lead_time)
    )
    configs = configs[OPTISTOCK_CORE_COLS]

    if validate_prices:
        _check_item_price_invariant(configs)

    # Constraints are opt-in: OptiStock reads every non-core column as one.
    if constraints:
        configs = configs.copy()
        for name, value in constraints.items():
            if name in OPTISTOCK_CORE_COLS:
                raise ValueError(
                    f"constraint {name!r} collides with a core item column; "
                    "pick another name"
                )
            configs[name] = (
                configs["name"].map(value).astype(float)
                if isinstance(value, dict)
                else float(value)
            )

    return configs.sort_values("name").reset_index(drop=True)


def _check_item_price_invariant(configs: pd.DataFrame) -> None:
    """``Item`` requires selling_price > cost_price > salvage_value >= 0, strictly."""
    bad = configs[
        ~(
            (configs["selling_price"] > configs["cost_price"])
            & (configs["cost_price"] > configs["salvage_value"])
            & (configs["salvage_value"] >= 0)
        )
    ]
    if not bad.empty:
        rows = "\n".join(
            f"  {r['name']}: selling={r['selling_price']:.4g} "
            f"cost={r['cost_price']:.4g} salvage={r['salvage_value']:.4g}"
            for _, r in bad.iterrows()
        )
        raise ValueError(
            "Item requires selling_price > cost_price > salvage_value >= 0 "
            f"(strict). {len(bad)} item(s) violate it:\n{rows}\n"
            "Note the aggregation collapses per-line prices, so an item sold at or "
            "below cost on average will fail here. Pass validate_prices=False to "
            "skip this check."
        )


# ---------------------------------------------------------------------------
# CLV
# ---------------------------------------------------------------------------


def to_clv(
    orders: pd.DataFrame,
    *,
    observation_period_end: pd.Timestamp | str | None = None,
    monetary: bool = True,
    time_unit: str = "D",
    covariates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Order lines → the per-customer RFM summary the CLV models consume.

    Lines are aggregated to one row per order *before* summarising, so a three-item
    order counts as one transaction rather than three. ``customer_id`` is mandatory.

    Parameters
    ----------
    observation_period_end : pd.Timestamp or str, optional
        The "as of" date. Orders after it are discarded and ``T`` is measured up to it.
        Defaults to the latest order date — the last *trading* day, not necessarily the
        calendar date you had in mind.

        **This is the anchor for every CLV prediction**, so it must line up with the
        training window whenever the output feeds a backtest: with the holdout split
        ``BaseStockKeep._date_prep`` performs (train on ``date < split_date``), pass
        ``split_date - pd.Timedelta(days=1)``. Leaving it at the default silently leaks
        holdout orders into the fit and inflates measured performance.
    monetary : bool
        Include ``monetary_value``, the mean spend per repeat transaction, computed as
        ``Σ(quantity × unit_price)`` per order. Requires ``unit_price``.
    time_unit : str
        Unit for ``recency`` and ``T``. ``"D"`` makes both counts of days, the unit
        ``ParetoNBD.expected_purchases(future_t=...)`` speaks. Unrelated to
        ``GammaGammaSpend.expected_customer_lifetime_value``, whose ``future_t`` is
        always in months.
    covariates : sequence of str, optional
        Extra columns to carry through, taken from each customer's **first** order.

    Returns
    -------
    pd.DataFrame
        ``customer_id``, ``frequency``, ``recency``, ``T``, and when requested
        ``monetary_value`` plus covariates. ``frequency`` counts **repeat**
        transactions, so a one-order customer has ``frequency == 0`` and must be
        filtered out before fitting ``GammaGammaSpend``.
    """
    df = validate(orders, CUSTOMER_SCHEMA)

    if monetary and UNIT_PRICE not in df.columns:
        raise ValueError(
            f"monetary=True requires a {UNIT_PRICE!r} column, which is not in "
            f"orders. Present columns: {list(df.columns)}. Map it during ingest, "
            "or call with monetary=False."
        )

    if observation_period_end is not None:
        cutoff = pd.Timestamp(observation_period_end)
        df = df[df[DATE] <= cutoff]
        if df.empty:
            raise ValueError(
                f"No orders on or before observation_period_end={cutoff.date()}."
            )
    else:
        cutoff = df[DATE].max()

    # rfm_summary expects one row per transaction, not per line.
    keys = [CUSTOMER_ID, ORDER_ID]
    transactions = df.groupby(keys, sort=False)[DATE].min().reset_index()

    if monetary:
        revenue = (
            df[QUANTITY]
            .mul(df[UNIT_PRICE])
            .groupby([df[CUSTOMER_ID], df[ORDER_ID]], sort=False)
            .sum()
            .rename("_revenue")
            .reset_index()
        )
        transactions = transactions.merge(revenue, on=keys, how="left")

    try:
        from pymc_marketing.clv import rfm_summary
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "to_clv() needs pymc-marketing for rfm_summary. Install it with "
            "`uv sync`, or use to_sales() alone if you only need the "
            "forecasting frame."
        ) from exc

    rfm = rfm_summary(
        transactions,
        customer_id_col=CUSTOMER_ID,
        datetime_col=DATE,
        monetary_value_col="_revenue" if monetary else None,
        observation_period_end=cutoff,
        time_unit=time_unit,
    )

    if covariates:
        missing = [col for col in covariates if col not in orders.columns]
        if missing:
            raise ValueError(f"covariate column(s) not in orders: {missing}")
        first_line = (
            df.sort_values([DATE, ORDER_ID], kind="mergesort")
            .groupby(CUSTOMER_ID, sort=False)[list(covariates)]
            .first()
        )
        rfm = rfm.merge(first_line, left_on=CUSTOMER_ID, right_index=True, how="left")

    return rfm
