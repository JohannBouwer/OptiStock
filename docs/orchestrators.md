# Inventory Orchestrators

Four policy classes in `optistock.stockkeep` cover the standard textbook replenishment systems. All share the same interface — they own the full pipeline (data splitting, forecaster training, demand forecasting, optimisation, holdout validation) so the caller doesn't wire up [ForecastSolver](solver.md) by hand.

| Policy | Notation | Trigger | Order quantity | Output | Best for |
|---|---|---|---|---|---|
| `PeriodicOrderUpTo` | (R, S) | Every R days | Order up to S, no service floor | Profit-optimal single order | Pure expected-profit replenishment |
| `PeriodicBaseStock` | (R, S) | Every R days | Order up to S, floored to CSL quantile | Service-constrained single order | Items with a contractual or business service-level target |
| `ContinuousFixedQuantity` | (s, Q) | Inventory position ≤ s | Fixed Q | Distribution of first-stockout day | Watching when to place the next fixed order |
| `ContinuousOrderUpTo` | (s, S) | Inventory position ≤ s | S − inventory position | Distribution of first-stockout day | Watching when to top up to a target level |

## Shared interface

All four take the same shape of inputs:

- `histories` — long-format DataFrame (`item`, `date`, `sales`).
- `item_configs` — DataFrame with `name`, `cost_price`, `selling_price`, `salvage_value`, and **`lead_time` (integer days, mandatory)**.
- `forecaster_class` — any class from [forecasting.md](forecasting.md).
- `yield_profiles` — optional `{item_name: YieldDistribution}` to model stochastic supply.
- `inventory_state` — optional `{item_name: {"on_hand": int, "on_order": int}}` for net-of-position calculations.

And all four expose two run modes:

- `run_holdout(holdout_days, ...)` — fit on `history[:-holdout_days]`, forecast and decide for the holdout window, then validate against the actuals you held back. Returns metrics, plots, and the decision.
- `run(forecast_days, ...)` — production mode. Fit on the full history and forecast forward; no validation.

For periodic policies the effective planning horizon is always `lead_time + review_period` (the `forecast_days` argument is accepted for API symmetry but doesn't override that horizon).

See [notebook 6](../notebooks/6_Manager_Class.ipynb) for a tour of all four classes side by side.

---

## PeriodicOrderUpTo (R, S)

Profit-optimising periodic-review policy. No service-level floor — the solver maximises expected profit (or CVaR / Utility).

```python
from optistock.stockkeep import PeriodicOrderUpTo
from optistock.forecasting.state_space import UnivariateSSM

sk = PeriodicOrderUpTo(
    histories=df_history,
    item_configs=df_items,
    review_period=7,
    forecaster_class=UnivariateSSM,
    yield_profiles=yield_map,
)

results = sk.run_holdout(
    holdout_days=30,
    objective="Utility",
    risk_aversion=0.3,
    inventory_state={"Tablet Air": {"on_hand": 80, "on_order": 20}},
    fit_kwargs={"draws": 1000, "build_model_kwargs": {"seasonal_period": 7}},
)

print(results["allocation"])      # gross optimal quantity per item
print(results["net_allocation"])  # quantity to order, net of on-hand + on-order
print(results["metrics"])         # per-item profit, service level, SMAPE, stockout flag
fig, ax = sk.plot_forecast("Tablet Air")
```

## PeriodicBaseStock (R, S) with service targets

Same horizon as `PeriodicOrderUpTo`. Items listed in `service_targets` have their order quantity floored to the given cycle-service-level quantile of the planning-horizon demand distribution; everything else is pure-profit-optimised.

```python
from optistock.stockkeep import PeriodicBaseStock

sk = PeriodicBaseStock(
    histories=df_history,
    item_configs=df_items,
    review_period=14,
    service_targets={"Tablet Air": 0.95, "Gaming Mouse": 0.90},
    forecaster_class=BayesTimeSeries,
)
results = sk.run(forecast_days=14, objective="SAA")
```

See [notebook 7](../notebooks/7_Inventory_Policy.ipynb) for a deep-dive on the profit cost of guaranteeing a target service level.

## ContinuousFixedQuantity (s, Q)

An order of size `Q` is placed whenever the inventory position falls to or below reorder point `s`. Rather than optimising a quantity, `run()` simulates demand depletion across all posterior scenarios and returns the **distribution of the first day a stockout would occur**.

```python
from optistock.stockkeep import ContinuousFixedQuantity

sk = ContinuousFixedQuantity(
    histories=df_history,
    item_configs=df_items,
    Q={"Tablet Air": 200, "Gaming Mouse": 150},
    service_level=0.95,   # quantile used to auto-compute reorder point s
    forecaster_class=BayesTimeSeries,
)

results = sk.run(
    forecast_days=60,
    inventory_state={"Tablet Air": {"on_hand": 120, "on_order": 0}},
)

print(sk.report("Tablet Air"))                   # 90% HDI on stockout timing
fig = sk.plot_stockout_distribution("Tablet Air")
rec = sk.recommended_order_day("Tablet Air", risk_tolerance=0.10)
print(rec["action"])  # "Order 200 units of 'Tablet Air' by day N ..."
```

## ContinuousOrderUpTo (s, S)

Like `ContinuousFixedQuantity` but when the reorder point triggers, the order quantity is `S − inventory_position` rather than a fixed `Q`.

```python
from optistock.stockkeep import ContinuousOrderUpTo

sk = ContinuousOrderUpTo(
    histories=df_history,
    item_configs=df_items,
    S={"Tablet Air": 500, "Gaming Mouse": 300},
    service_level=0.95,
    forecaster_class=BayesTimeSeries,
)
results = sk.run(forecast_days=60)
print(sk.report("Gaming Mouse"))
rec = sk.recommended_order_day("Gaming Mouse", risk_tolerance=0.05)
```
