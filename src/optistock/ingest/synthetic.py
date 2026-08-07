"""Synthetic order-history generators for tests and agent evals.

:func:`make_canonical` produces a clean canonical frame with known ground truth.
:func:`scramble` messes it up the way real exports do — renamed columns, a line
total instead of a unit price, string dates — so the agent (or a test's hand-written
mapping) has to recover the canonical form.

Pure numpy + pandas: no LLM, so this is importable without the ``[ingest]`` extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_canonical", "scramble"]


def make_canonical(n_orders: int = 200, seed: int = 7) -> pd.DataFrame:
    """A clean canonical order-line frame with known structure.

    Multi-line orders, repeat customers and a cost below every price, so the whole
    chain — ``to_sales``, ``to_items``, ``to_clv`` — runs off the result.
    """
    rng = np.random.default_rng(seed)
    n_customers = max(5, n_orders // 4)
    items = [f"SKU-{i:03d}" for i in range(12)]
    start = np.datetime64("2025-01-01")

    rows = []
    for oid in range(n_orders):
        cust = f"C{rng.integers(0, n_customers):04d}"
        day = int(rng.integers(0, 365))
        n_lines = int(rng.integers(1, 4))
        for _ in range(n_lines):
            item = items[rng.integers(0, len(items))]
            qty = float(rng.integers(1, 6))
            price = round(float(rng.uniform(5, 80)), 2)
            rows.append(
                {
                    "order_id": f"O{oid:05d}",
                    "customer_id": cust,
                    "date": start + np.timedelta64(day, "D"),
                    "item_id": item,
                    "quantity": qty,
                    "unit_price": price,
                    "unit_cost": round(price * float(rng.uniform(0.4, 0.7)), 2),
                }
            )
    return pd.DataFrame(rows)


def scramble(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Turn a canonical frame into a messy 'source' export.

    Mirrors the ways real exports differ from the canonical schema:

    - informal, inconsistent column names (``Order Ref``, ``cust``, ``sku``)
    - revenue only as a **line total**, so ``unit_price`` must be derived
    - dates as ``dd/mm/yyyy`` strings, which pandas would otherwise read as US order

    Every canonical field remains *recoverable*, so the full chain can run end to end
    off this frame.
    """
    return pd.DataFrame(
        {
            "Order Ref": df["order_id"],
            "cust": df["customer_id"],
            "purchased_on": pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y"),
            "sku": df["item_id"],
            "qty": df["quantity"],
            "line_total": (df["quantity"] * df["unit_price"]).round(2),
            "cost_ea": df["unit_cost"],
        }
    )
