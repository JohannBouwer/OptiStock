# Order-line preprocessing

Everything in this repo consumes one of three frames: the forecasting models want a long daily panel of `(date, item, sales)`, the solver wants per-item economics, and the CLV models want a per-customer RFM summary. All three are derivable from the same order-line table, and `optistock.preprocessing` does that derivation deterministically.

```python
from optistock.preprocessing import to_sales, to_items, to_clv

histories    = to_sales(orders)   # → forecasting / BaseStockKeep
item_configs = to_items(orders)   # → Item / ForecastSolver
rfm          = to_clv(orders)     # → ParetoNBD / GammaGammaSpend
```

There is no modelling here — no PyMC, no fitting. This is the contract layer. What matters is the shape of the output, not how it is computed.

## The canonical schema

One row per **order line**. The registry in `optistock.preprocessing.schema` is the single source of truth: it drives both the validator that enforces the contract and the description handed to the mapping agent in [ingest](ingest.md), so the two can never drift.

| Column | Required | Feeds |
|---|---|---|
| `order_id` | ✓ | grouping lines into transactions |
| `date` | ✓ | every frame |
| `item_id` | ✓ | `to_sales`, `to_items` |
| `quantity` | ✓ | `to_sales` (`sales`) |
| `customer_id` | for CLV and the split | `to_clv`, `to_sales(split=True)` |
| `unit_price` | for monetary CLV and economics | `to_clv`, `to_items` |
| `unit_cost` | for economics | `to_items` (`cost_price`) |
| anything else | ✗ | passes through; `to_clv(covariates=[...])` |

`customer_id` is deliberately **optional**. Point-of-sale exports routinely have no buyer id, and `to_sales(split=False)` genuinely does not need one — so the base gate lets them through and only the transforms that require a customer refuse.

## Getting your columns to match

The canonical names are not negotiable: the transforms address columns by name. Three ways to arrive there, in increasing order of mess.

**Already canonical** — nothing to do.

**Names differ, nothing else does** — `rename_to_canonical`:

```python
from optistock.preprocessing import rename_to_canonical

orders = rename_to_canonical(raw, {"cust": "customer_id", "sku": "item_id"})
```

The direction is `{source: canonical}`, matching `DataFrame.rename`. Nothing is dropped or reordered, and it refuses a rename that would collide with a column already present — plain `df.rename` would silently give you two columns with the same label, and only the first would be validated.

**The shape differs too** — a line total that must be divided by quantity, a `dd/mm/yyyy` string, a `'0,90 €'` price: that is what [`optistock.ingest`](ingest.md) is for.

## Validation

`validate` runs first inside every transform, so its rules are the real entry contract; calling it yourself is for inspection only, and it is idempotent.

```python
from optistock.preprocessing import validate, OrderValidationError

try:
    orders = validate(raw)
except OrderValidationError as exc:
    print(exc)                  # rendered, actionable
    print(exc.failure_cases)    # one row per offending cell
```

It reports **every** violation at once — missing columns, unparseable dates, nulls in required columns, non-positive quantities — rather than one exception per run:

```
orders is missing required column(s): ['item_id']. Present columns: ['order_id', 'date', 'quantity'].
Rename them with optistock.preprocessing.rename_to_canonical(), or map the source table with
optistock.ingest.

Could not parse 1 value(s) in 'date' as dates, e.g. ['not-a-date'] at row(s) [1].

2 order line(s) have quantity <= 0. Returns and cancellations are not order lines — filter or net
them out before calling. First offending row(s): [1, 2].
```

Row order, the index and unmodeled columns are all preserved, and the input is never mutated.

> [!NOTE]
> **Identity columns keep the dtype they arrive with.** `order_id`, `customer_id` and `item_id` are deliberately not coerced. Casting them to a string dtype would rewrite integer SKU `101` as `'101'` — and `BaseStockKeep` joins `histories` to `item_configs` by raw value equality, so a hand-built `item_configs` with integer names would silently match zero rows and fail deep inside the forecaster.

**Returns are not order lines.** Non-positive quantities are rejected outright rather than guessed at, so credit notes and cancellations have to be filtered or netted out beforehand. There is no returns model in the package yet, and nothing here tries to represent one.

## New vs returning customers

`to_sales(split=True)` — the default — also emits two columns that sum to `sales` for every period and item:

| Column | Meaning | Model that should consume it |
|---|---|---|
| `sales_new` | units bought on a customer's **first** purchase day | forecasting classes (acquisition is a time-series process) |
| `sales_returning` | units bought on every later day | CLV classes (retention is a purchase/dropout process) |

This split is not a convention invented here. `to_clv` calls `rfm_summary` with `include_first_transaction=False` (its default), so `frequency` counts **repeat** transactions only and `T` runs from the first purchase. The Pareto/NBD likelihood is therefore defined over exactly the transactions that land in `sales_returning`. The two branches carve the order book with no overlap and no gap:

```python
new, returning = split_new_returning(orders)

len(returning.drop_duplicates(["customer_id", "date"])) == to_clv(orders)["frequency"].sum()
```

### Why the split is per customer-day, not per order id

`rfm_summary` collapses all of a customer's purchases on one date into a single transaction. So `split_new_returning` tags by **first purchase day**. A customer who places two orders on their first day has both counted as acquisition — an order-level rule would call the second one retention while the CLV frame still counted that day as their acquisition, and the two frames would disagree.

### Left-censoring

By default, every customer's first *observed* purchase counts as an acquisition. That is only true if your data covers the full customer lifetime. If it starts mid-stream, a long-standing customer's first visit inside the window looks like a brand-new customer.

`observation_start` is the remedy — customers first seen on or before it are treated as returning throughout:

```python
burn_in_end = orders["date"].min() + pd.Timedelta(days=90)

histories = to_sales(orders, observation_start=burn_in_end)
```

## `to_sales`

```python
to_sales(orders, *, date_col="date", item_col="item", target="sales",
         split=True, fill_calendar=True, freq="D",
         min_nonzero_days=1, observation_start=None)
```

Output columns are named to drop straight into the orchestrators:

```python
from optistock import PeriodicOrderUpTo

keeper = PeriodicOrderUpTo(to_sales(orders), to_items(orders), review_period=7)
```

Three arguments deserve attention.

**`fill_calendar=True` — leave it on.** Order data is sparse: an item with no sale on a day simply has no row. Every forecaster in this repo uses *row order* as the time index, so an unfilled panel silently compresses the time axis and biases the level upward. Filling reindexes onto the complete period × item grid with zeros. Note the zeros mean "no demand", not "closed" — mask non-trading days yourself if that distinction matters.

**`min_nonzero_days`** drops items too sparse to forecast, with a warning naming them. The forecasters normalise by the series maximum (`max_scaler`), so a constant-zero series divides by zero and fails deep inside the sampler rather than at the call site.

**`split=True` needs the whole order book.** Compute the panel on every item, *then* subset the result. "First purchase day" is a customer-level fact, so filtering to a handful of SKUs first redefines it within that subset and relabels long-standing customers as new. `sales` is unaffected, which is exactly what makes this dangerous — on the Online Retail II order book it inflated `sales_new` by 3.6×.

When `split=True` there is a second, non-fatal warning: items with **no first-time-buyer sales** in the window keep their row (`sales` is still fine) but `sales_new` is constant zero for them, so forecasting that column directly will fail. Niche items bought only by regulars land here.

## `to_items`

```python
to_items(orders, *, price_agg="median", salvage_value=0.0, lead_time=0,
         constraints=None, validate_prices=True)
```

Emits exactly the columns `BaseStockKeep._create_items` reads — `name`, `cost_price`, `selling_price`, `salvage_value`, `lead_time` — and nothing more, because OptiStock reads *any* additional column as a per-item constraint. Constraints are therefore opt-in:

```python
item_configs = to_items(orders, salvage_value=0.2, lead_time={"SKU-1": 3, "SKU-2": 7},
                        constraints={"storage": 20})
```

`price_agg="median"` is robust to promotions. `validate_prices` checks `Item`'s invariant `selling_price > cost_price > salvage_value >= 0` up front, so an item sold at or below cost on average fails here with a readable per-item listing instead of deep inside the solver.

Cost is the one number an order book usually does not carry. Without `unit_cost` this raises — but `to_sales` does not need it, so a costless dataset can still drive a forecast, just not a priced decision. `available_transforms(orders)` reports which frames a given order book can feed, and why not:

```python
for name, blocker in available_transforms(orders).items():
    if blocker:
        print(f"skipping {name}: {blocker}")
```

## `to_clv`

```python
to_clv(orders, *, observation_period_end=None, monetary=True,
       time_unit="D", covariates=None)
```

Lines are aggregated to one row per order before summarising, so a three-item order counts as one transaction rather than three. Output is `customer_id`, `frequency`, `recency`, `T` and — with `monetary=True` — `monetary_value`, the mean spend per *repeat* transaction.

`frequency == 0` for one-purchase customers, which `GammaGammaSpend` rejects by design:

```python
from optistock.clv import GammaGammaSpend, ParetoNBD

txn   = ParetoNBD(rfm)
spend = GammaGammaSpend(rfm[rfm["frequency"] > 0])
```

Covariates come from each customer's first order, which is what you want for a cold-start covariate like acquisition channel:

```python
rfm = to_clv(orders, covariates=["channel"])
txn = ParetoNBD(rfm, purchase_covariate_cols=["channel"])
```

### `observation_period_end` is the anchor, and it is easy to get wrong

Every CLV prediction is measured forward from the end of the observation window. `T` is the distance from a customer's first purchase to that date, and `ParetoNBD.expected_purchases(future_t=...)` counts days *after* it.

When the RFM frame feeds a backtest, this must line up with the forecasting split or the CLV branch sees the future. `BaseStockKeep._date_prep` trains on `date < split_date`, so:

```python
split_date = histories["date"].max() - pd.Timedelta(days=holdout_days - 1)
anchor     = split_date - pd.Timedelta(days=1)

rfm = to_clv(orders, observation_period_end=anchor)   # the anchor does the cutting
```

Leaving it at the default silently fits on holdout orders and inflates measured performance. Two details worth internalising: passing the anchor is *enough* — it filters for you, so restricting `orders` beforehand is redundant — and the default is the last **trading** day, not the split boundary, so on a business that does not trade every weekday the two differ.

One unit trap worth restating: with `time_unit="D"`, `recency` and `T` are in **days**, and so is `ParetoNBD.expected_purchases(future_t=...)`. That is unrelated to `GammaGammaSpend.expected_customer_lifetime_value(future_t=...)`, whose horizon is always in **months** — see [clv](clv.md).

## Going deeper

The [Order-line Preprocessing notebook](../notebooks/data_preprocessing.ipynb) runs all of this end to end on the real Online Retail II order book, including each of the traps above.
