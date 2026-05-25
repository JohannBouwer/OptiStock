
---

# OptiStock: Inventory Optimization with the News Vendor Framework

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in stock supply allocation.

> [!NOTE]
> **Heads up:** This is a project I completed for my own development / learning. The models work and have been tested, but no claims are made about production-readiness, computational efficiency, or suitability for any particular use case.

## Table of Contents

1. [Installation](#installation)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Probabilistic Forecasting](#probabilistic-demand-forecasting)
5. [Hierarchical Forecasting](#hierarchical-forecasting)
6. [State Space Models](#state-space-models)
7. [Marketing Mix Models](#marketing-mix-models)
8. [Causal Calibration](#causal-calibration)
9. [Configurable Priors](#configurable-priors)
10. [The ForecastSolver](#the-forecastsolver)
11. [Inventory Orchestrators](#inventory-orchestrators)
    - [PeriodicOrderUpTo (R, S)](#periodicorderupto-r-s)
    - [PeriodicBaseStock (R, S) with Service Targets](#periodicbasestock-r-s-with-service-targets)
    - [ContinuousFixedQuantity (s, Q)](#continuousfixedquantity-s-q)
    - [ContinuousOrderUpTo (s, S)](#continuousorderupto-s-s)
---

## Installation

**Prerequisite:** Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

### From GitHub (For Users)
You can install OptiStock directly into your current virtual environment using `pip`. This will automatically install all necessary dependencies like PyMC, BART, and Scipy.

```bash
pip install git+https://github.com/JohannBouwer/OptiStock.git
```

> Note: If you are already managing your own project with `uv`, you can simply run `uv add git+https://github.com/JohannBouwer/OptiStock.git` to add it to your dependency tree.

### For Development (Editable Mode)

If you plan to modify the code or contribute to the project, use `uv` to synchronise
 the environment. This command automatically creates a `.venv`, resolves the lockfile, and installs the package in "editable" mode so your changes appear instantly.

```bash
# Clone the repository
# HTTPS:
git clone https://github.com/JohannBouwer/OptiStock.git
# SSH:
git clone git@github.com:JohannBouwer/OptiStock.git

# move into the directory
cd OptiStock

# Create environment and install dependencies (including editable project)
uv sync

# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Quick Verification

After installation, you can verify everything is working by running the installed package:

```bash
uv run python -c "import optistock"
```

## Key Features

* **Bayesian Forecasting**: Implements Fourier-based Time Series (`BayesTimeSeries`), Hybrid BART (`BARTBayesTimeSeries`), Hilbert Space Gaussian Processes (`HSGPBayesTimeSeries`), and Bayesian Structural State Space Models (`UnivariateSSM`).
* **Hierarchical Panel Forecasting**: `HierarchicalBayesTimeSeries` partially pools intercept, growth, seasonality, and event coefficients across items via shared hyper-priors — useful when some items have short or noisy histories.
* **Marketing Mix Models**: Attribution-aware demand modelling with `MediaMixModel` (via pymc-marketing), separating baseline demand from channel spend effects.
* **Causal Calibration**: Bayesian synthetic control (`SyntheticControl`) measures the lift of past interventions and feeds the result back into the forecasters as a soft prior on the corresponding event coefficient via `LiftConstraint`.
* **Configurable Priors**: Every forecaster and the causal estimator accept a typed `*Priors` dataclass — swap distributions, retune hyper-parameters, or describe the model with `model.describe_priors()` without touching the model code.
* **Unified Solver Interface**: `ForecastSolver` accepts any fitted `BaseForecaster` and supports single-item and multi-item constrained optimisation from a common API.
* **End-to-End Orchestration**: Four policy-specific orchestrators (`PeriodicOrderUpTo`, `PeriodicBaseStock`, `ContinuousFixedQuantity`, `ContinuousOrderUpTo`) each manage the full pipeline — data splitting, forecaster training, demand forecasting, optimisation, and hold-out validation.
* **Inventory Policies**: Periodic-review classes solve for optimal order-up-to quantities; continuous-review classes simulate reorder timing via posterior demand scenarios, returning probabilistic stockout distributions and recommended order days.
* **Stochastic Yield Modeling**: Accounts for supply-chain unreliability using Beta and Discrete yield distributions.
* **Risk-Aware Solvers**: Optimizes for Expected Profit (`SAA`), Exponential Utility (`Utility`), or Conditional Value at Risk (`CVaR`) to protect against tail-risk stockouts.

---

## Project Structure

* `optistock/stockkeep.py`: All inventory orchestrators — `PeriodicOrderUpTo`, `PeriodicBaseStock`, `ContinuousFixedQuantity`, `ContinuousOrderUpTo`, and the shared `BaseStockKeep` engine.
* `optistock/solvers.py`: The unified `ForecastSolver` — pairs `Item` objects with fitted `BaseForecaster` instances to solve for optimal stock quantities.
* `optistock/items.py`: The `Item` class encapsulating cost structure, constraints, and yield distributions.
* `optistock/forecasting/`: Bayesian forecasting module:
  * `base.py`: `BaseForecaster` abstract class and `ErrorEstimations` utilities.
  * `linear_regressors.py`: `BayesTimeSeries`, `BARTBayesTimeSeries`, `HSGPBayesTimeSeries`, `HierarchicalBayesTimeSeries`.
  * `state_space.py`: `UnivariateSSM` — flexible Bayesian structural state space model.
  * `mix_media_models.py`: `MediaMixModel` — Bayesian Marketing Mix Model for sales attribution.
  * `priors.py`: `Prior`, `BasePriors`, and the per-model `*Priors` dataclasses.
* `optistock/causal/`: Causal calibration module:
  * `synthetic_control.py`: `SyntheticControl` (Bayesian synthetic control via CausalPy) and the `CausalEffect` headline result.
  * `lift_constraints.py`: `LiftConstraint` — translates a measured causal lift into an observed-Normal soft prior on a forecaster's `beta_event` coefficient.
  * `priors.py`: `SyntheticControlPriors`.
* `optistock/distributions/`: Probabilistic models for demand (`SampledDemand`, `NormalDemand`, ...) and manufacturing yield (`BetaYield`, `PerfectYield`, ...).
* `optistock/plot_suite/`: Visualization tools for forecast validation, profit curves, and portfolio analysis.

## Probabilistic Demand Forecasting

Generate full posterior predictive distributions instead of single-point estimates using typical Bayesian models.

### Fourier & Event-based Forecasting

Capture complex seasonality and holiday effects using `BayesTimeSeries`.

```python
from optistock.forecasting import BayesTimeSeries
from optistock.distributions.demand_distributions import SampledDemand

# Initialize model and define promotional events
model = BayesTimeSeries(df_history, target_col="sales")
model.create_events({"Promo_A": ["2025-01-01", "2025-01-02"]})

# Fit and predict a future window
model.fit(target="sales", date_col="date")
model.forecast(scenario={"df_future": df_future, "date_col": "date"})

# Extract total demand samples for a specific period
demand_dist = model.get_demand_distribution("2025-01-01", "2025-01-07")

```

### Non-linear Seasonality with BART

Use Bayesian Additive Regression Trees to learn patterns directly from data.

```python
from optistock.forecasting import BARTBayesTimeSeries

# Fit BART model (automatically handles trend and seasonality)
bart_model = BARTBayesTimeSeries(df_history)
bart_model.fit(target="sales", date_col="date", trees=50)

# Generate forecast and visualize learned components
bart_model.forecast(scenario={"df_future": df_future, "date_col": "date"})
fig, ax = bart_model.plot_components()

```

---

## Hierarchical Forecasting

`HierarchicalBayesTimeSeries` is a multi-item version of `BayesTimeSeries` with **partial pooling across items** via shared hyper-priors. Each item keeps its own intercept, growth, seasonal, and event coefficients, but those coefficients are drawn from population-level distributions whose mean and spread are themselves learned from the data. Items with short or noisy histories borrow strength from the rest of the panel.

Input is **wide-format**: one `date` column plus one numeric column per item. Ragged histories (NaNs) are supported — the model masks missing observations out of the likelihood. A non-centered parameterisation is used for every per-item coefficient to avoid funnel pathologies under HMC.

```python
from optistock.forecasting import HierarchicalBayesTimeSeries

# df_history: wide-format — columns = ["date", "Tablet Air", "Gaming Mouse", ...]
model = HierarchicalBayesTimeSeries(df_history, date_col="date")

# Events can fire globally or per-item
model.create_events({
    "Black_Friday": ["2025-11-28"],                           # all items
    "Mouse_Promo":  {"Gaming Mouse": ["2025-02-01", "2025-02-02"]},
})

model.fit(samples=1000, chains=4)
model.forecast(scenario={"df_future": df_future})

fig, ax = model.plot_forecast(item="Tablet Air")
fig, axes = model.plot_components(item="Tablet Air")

# Per-item demand distribution — plugs directly into ForecastSolver
demand = model.get_demand_distribution("2025-03-01", "2025-03-30", item="Tablet Air")

```

---

## State Space Models

`UnivariateSSM` is a flexible Bayesian Structural Time Series (BSTS) model built on `pymc-extras`. It composes interpretable components — trend, seasonality, and exogenous regressors — and estimates them jointly via MCMC, keeping parameters that vary with time.

```python
from optistock.forecasting.state_space import UnivariateSSM

# Build with a datetime-indexed DataFrame
model = UnivariateSSM(df_history, target_col="sales", exog={"spend": True})

# Compose and build the structural model
model.build_model(
    trend_order=2,                    # local linear trend
    trend_innovations_order=[0, 1],   # drifting slope, fixed level
    seasonal_period=7,                # weekly seasonality
    seasonal_harmonics=3,
)

# Fit (MCMC via nutpie/JAX)
model.fit(draws=1000, tune=500)

# Run Kalman smoother and extract latent components
model.smooth_and_filter()
fig, axes = model.plot_components()

# Forecast future periods
model.forecast(periods=30, scenario={"data_spend": future_spend})
fig, ax = model.plot_forecast()

# Pull demand distribution for use with ForecastSolver
demand = model.get_demand_distribution("2025-03-01", "2025-03-30")

```

---

## Marketing Mix Models

`MediaMixModel` wraps `pymc-marketing`'s MMM to model sales as a combination of baseline demand, advertising carry-over (adstock), and diminishing returns (saturation). Useful when channel spend data is available and you need attribution alongside demand forecasting.

```python
from optistock.forecasting.mix_media_models import MediaMixModel

model = MediaMixModel(
    df_history,
    target_col="sales",
    channel_cols=["tv_spend", "digital_spend"],
)
model.fit(target="sales", date_col="date")
model.forecast(df_future=df_future)

demand = model.get_demand_distribution("2025-03-01", "2025-03-30")

```

---

## Causal Calibration

When a past intervention (promotion, price change, packaging refresh) is mixed in with normal sales history, a single `beta_event` coefficient in the forecaster can swing wildly on noisy data. The `optistock.causal` module measures the lift with **Bayesian synthetic control** and feeds the result back as a soft prior, anchoring the forecaster's event coefficient to a number that came from a proper counterfactual.

### Step 1 — measure the lift with SyntheticControl

`SyntheticControl` wraps CausalPy's `WeightedSumFitter`: donor items are combined under a Dirichlet (non-negative weights summing to one) to reconstruct the treated item's pre-intervention behaviour, and the post-intervention gap is the causal effect.

```python
from optistock.causal import SyntheticControl

sc = SyntheticControl(
    data=df_wide,                                # wide-format panel
    treated_item="Tablet Air",
    donor_items=["Gaming Mouse", "Keyboard", "USB Hub"],
    treatment_date="2025-02-01",
    intervention_name="Spring_Promo",
)
sc.fit(samples=1000, tune=1000, chains=4)

effect = sc.summary()
print(effect)
# Spring_Promo changed sales of Tablet Air by +12.4% [94% HDI: +8.1%, +16.9%]

fig, _ = sc.plot()           # observed vs synthetic counterfactual + impact band
fig, _ = sc.plot_weights()   # posterior donor weights (expect sparse)

```

### Step 2 — feed the measured lift into the forecaster

`LiftConstraint` adds an observed-Normal likelihood term on the matching `beta_event` coefficient — same pattern `pymc-marketing` uses for lift tests. Values are in raw units; the forecaster handles internal scaling.

```python
from optistock.causal import LiftConstraint
from optistock.forecasting import HierarchicalBayesTimeSeries

# Build the constraint from the fitted synthetic control...
constraint = LiftConstraint.from_synthetic_control(sc, event_name="Spring_Promo")
# ...or equivalently from the CausalEffect:
# constraint = LiftConstraint.from_causal_effect(effect, event_name="Spring_Promo")

model = HierarchicalBayesTimeSeries(
    df_wide,
    lift_constraints=[constraint],
)
model.create_events({"Spring_Promo": {"Tablet Air": ["2025-02-01", "2025-02-02"]}})
model.fit()

```

For single-series forecasters (`BayesTimeSeries`, `BARTBayesTimeSeries`, `HSGPBayesTimeSeries`) the `item=` field on `LiftConstraint` is unused; for `HierarchicalBayesTimeSeries` it must name the treated item.

---

## Configurable Priors

Every forecaster and the causal estimator accept a typed `*Priors` dataclass. Override one field, the entire prior, or swap the distribution outright — no monkey-patching of model internals.

```python
from optistock.forecasting import (
    BayesTimeSeries,
    BayesTimeSeriesPriors,
    HierarchicalBayesTimeSeries,
    HierarchicalBayesTimeSeriesPriors,
    Prior,
)

# Tighten observation noise for a low-variance series
priors = BayesTimeSeriesPriors(
    sigma=Prior("HalfNormal", {"sigma": 0.25}, "Observation noise"),
)
model = BayesTimeSeries(df_history, target_col="sales", priors=priors)
print(model.priors)   # tabular summary of every prior and its description

# Hierarchical hyper-priors — narrow the population spread on growth
hpriors = HierarchicalBayesTimeSeriesPriors(
    growth_sigma=Prior("HalfNormal", {"sigma": 0.05}, "Population spread on growth"),
)
hmodel = HierarchicalBayesTimeSeries(df_wide, priors=hpriors)

```

The same pattern applies to `BARTBayesTimeSeriesPriors`, `HSGPBayesTimeSeriesPriors`, `UnivariateSSMPriors`, and `SyntheticControlPriors`.

---

## The ForecastSolver

`ForecastSolver` is the unified optimization engine. It pairs one or more `(Item, BaseForecaster)` tuples, pulls posterior demand samples via `get_demand_distribution`, and solves for optimal stock quantities under the chosen objective.

### Single Item

```python
from optistock.items import Item
from optistock.forecasting import BayesTimeSeries
from optistock.solvers import ForecastSolver

item = Item("Gaming Mouse", cost_price=30, selling_price=50, salvage_value=10)

# Use any fitted forecaster
forecaster = BayesTimeSeries(df_history, target_col="sales")
forecaster.fit(target="sales", date_col="date")
forecaster.forecast(scenario={"df_future": df_future, "date_col": "date"})

solver = ForecastSolver(problems=(item, forecaster), objective="SAA")
allocation = solver.solve("2025-03-01", "2025-03-30")
print(solver.summary())

```

### Multi-Item Constrained with Risk Aversion

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

---

## Inventory Orchestrators

Four policy classes in `optistock.stockkeep` cover the standard textbook replenishment systems. All share the same `run_holdout` / `run` interface and accept the same forecaster, yield-profile, and objective arguments. `item_configs` must always include a `lead_time` column (integer days).

---

### PeriodicOrderUpTo (R, S)

Profit-optimising periodic-review policy. The planning horizon is `lead_time + review_period`. No mandatory service-level floor — the solver maximises expected profit (or CVaR / Utility).

```python
from optistock.stockkeep import PeriodicOrderUpTo
from optistock.forecasting.state_space import UnivariateSSM

sk = PeriodicOrderUpTo(
    histories=df_history,        # long-format: item, date, sales
    item_configs=df_items,       # columns: name, cost_price, selling_price, salvage_value, lead_time
    review_period=7,
    forecaster_class=UnivariateSSM,
    yield_profiles=yield_map,    # optional: {item_name: YieldDistribution}
)

# Validate on the last 30 days of observed data
results = sk.run_holdout(
    holdout_days=30,
    objective="Utility",
    risk_aversion=0.3,
    inventory_state={"Tablet Air": {"on_hand": 80, "on_order": 20}},
    fit_kwargs={"draws": 1000, "build_model_kwargs": {"seasonal_period": 7}},
)

print(results["allocation"])     # {item_name: gross optimal quantity}
print(results["net_allocation"]) # quantity to actually order (net of on-hand + on-order)
print(results["metrics"])        # per-item profit, service level, SMAPE, stockout flag
print(sk.summary())              # ForecastSolver diagnostics

fig, ax = sk.plot_forecast("Tablet Air")  # forecast vs holdout actuals

```

**Production mode** (fit on full history, forecast forward):

```python
results = sk.run(
    forecast_days=30,   # accepted for API compatibility; horizon = lead_time + review_period
    objective="SAA",
)
print(results["allocation"])
print(results["period"])         # (start_date, end_date) of the planning horizon

```

---

### PeriodicBaseStock (R, S) with Service Targets

Same horizon as `PeriodicOrderUpTo`. Items listed in `service_targets` have their order quantity floored to the given cycle-service-level (CSL) quantile of the planning-horizon demand distribution; all other items are pure-profit-optimised.

```python
from optistock.stockkeep import PeriodicBaseStock

sk = PeriodicBaseStock(
    histories=df_history,
    item_configs=df_items,
    review_period=14,
    service_targets={
        "Tablet Air":   0.95,   # 95 % CSL floor
        "Gaming Mouse": 0.90,
    },
    forecaster_class=BayesTimeSeries,
)

results = sk.run(forecast_days=14, objective="SAA")
print(results["allocation"])

```

---

### ContinuousFixedQuantity (s, Q)

Continuous-review fixed-order-quantity policy. An order of size `Q` is placed whenever the inventory position falls to or below reorder point `s`. Instead of computing a single optimal quantity, `run()` simulates demand depletion across all posterior scenarios and returns a distribution of the first day a stockout would occur.

```python
from optistock.stockkeep import ContinuousFixedQuantity

sk = ContinuousFixedQuantity(
    histories=df_history,
    item_configs=df_items,
    Q={"Tablet Air": 200, "Gaming Mouse": 150},   # fixed order qty per item
    service_level=0.95,   # quantile used to auto-compute reorder point s
    forecaster_class=BayesTimeSeries,
)

results = sk.run(
    forecast_days=60,
    inventory_state={"Tablet Air": {"on_hand": 120, "on_order": 0}},
)

# results["stockout_days"] — {item_name: np.ndarray of first-stockout day per scenario}
print(sk.report("Tablet Air"))   # 90% HDI statement on when a stockout is expected

fig = sk.plot_stockout_distribution("Tablet Air")

rec = sk.recommended_order_day("Tablet Air", risk_tolerance=0.10)
print(rec["action"])  # "Order 200 units of 'Tablet Air' by day N ..."

```

---

### ContinuousOrderUpTo (s, S)

Like `ContinuousFixedQuantity` but when the reorder point is triggered the order quantity is `S − inventory_position` rather than a fixed `Q`.

```python
from optistock.stockkeep import ContinuousOrderUpTo

sk = ContinuousOrderUpTo(
    histories=df_history,
    item_configs=df_items,
    S={"Tablet Air": 500, "Gaming Mouse": 300},   # order-up-to level per item
    service_level=0.95,
    forecaster_class=BayesTimeSeries,
)

results = sk.run(forecast_days=60)
print(sk.report("Gaming Mouse"))

rec = sk.recommended_order_day("Gaming Mouse", risk_tolerance=0.05)
print(rec["action"])  # "Top up 'Gaming Mouse' to 300 units by day N ..."

```

---
