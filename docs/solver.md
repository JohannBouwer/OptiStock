# ForecastSolver

`ForecastSolver` is the unified optimisation engine. It pairs one or more `(Item, BaseForecaster)` tuples, pulls posterior demand samples via `get_demand_distribution`, and solves for optimal stock quantities under the chosen objective.

Any forecaster from [forecasting.md](forecasting.md) plugs in — the solver only requires the forecaster has been fitted and can produce a demand distribution for the planning window.

## Objectives

| Objective | What it optimises | When to use |
|---|---|---|
| `SAA` | Expected profit (sample-average approximation) | Default — risk-neutral newsvendor |
| `Utility` | Exponential utility — penalises variance | When the decision-maker is risk-averse but you don't have an explicit tail-risk target |
| `CVaR` | Conditional Value-at-Risk on profit | When the goal is to limit the worst-α% outcomes (e.g. avoid catastrophic stockouts) |

## Single item

```python
from optistock.items import Item
from optistock.forecasting import BayesTimeSeries
from optistock.solvers import ForecastSolver

item = Item("Gaming Mouse", cost_price=30, selling_price=50, salvage_value=10)

forecaster = BayesTimeSeries(df_history, target_col="sales")
forecaster.fit(target="sales", date_col="date")
forecaster.forecast(scenario={"df_future": df_future, "date_col": "date"})

solver = ForecastSolver(problems=(item, forecaster), objective="SAA")
allocation = solver.solve("2025-03-01", "2025-03-30")
print(solver.summary())
```

See [notebook 1](../notebooks/1_Introduction.ipynb) for the newsvendor framing.

## Multi-item with constraints and risk aversion

`Item` objects can declare per-unit consumption of shared resources via `constraints={"resource": value}`. The solver enforces those resources globally via `limits={"resource": cap}` and returns shadow prices for each binding constraint in `summary()`.

```python
from optistock.solvers import ForecastSolver
from optistock.distributions.yield_distributions import BetaYield

risky_chip = Item("AI Chip", cost_price=100, selling_price=200, salvage_value=20,
                  constraints={"budget": 100},
                  yield_distribution=BetaYield(7, 3))

solver = ForecastSolver(
    problems=[(risky_chip, chip_forecaster), (widget, widget_forecaster)],
    objective="CVaR",
    limits={"budget": 50_000},
    cvar_alpha=0.10,
    cvar_lambda=0.50,
)
allocation = solver.solve("2025-03-01", "2025-03-30")
print(solver.summary())   # includes shadow prices, service level, CVaR
```

`yield_distribution` on an `Item` introduces stochastic supply (e.g. `BetaYield`, `PerfectYield`, `DiscreteYield`) — orders are scaled by a sampled yield draw before being matched against demand.

## Going deeper

- [Notebook 2](../notebooks/2_Mulit_Item_and_constraints.ipynb) — multi-item constrained optimisation, Lagrangian shadow prices.
- [Notebook 3](../notebooks/3_Yield_distributions.ipynb) — stochastic yield and risk-aware objectives (SAA, Utility, CVaR).

For end-to-end pipelines that wrap forecaster training + solving + holdout validation, see [orchestrators.md](orchestrators.md).
