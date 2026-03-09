
---

# OptiStock: Inventory Optimization with the News Vendor Framework

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in stock supply allocation.

## Table of Contents

1. [Installation](#installation)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Probabilistic Forecasting](#probabilistic-forecasting)
5. [State Space Models](#state-space-models)
6. [Marketing Mix Models](#marketing-mix-models)
7. [The ForecastSolver](#the-forecastsolver)
8. [The StockKeep Orchestrator](#the-stockkeep-orchestrator)
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
* **Marketing Mix Models**: Attribution-aware demand modelling with `MediaMixModel` (via pymc-marketing), separating baseline demand from channel spend effects.
* **Unified Solver Interface**: `ForecastSolver` accepts any fitted `BaseForecaster` and supports single-item and multi-item constrained optimisation from a common API.
* **End-to-End Orchestration**: The `StockKeep` class manages the entire pipeline — data splitting, forecaster training, demand forecasting, and hold-out validation — for any supported forecaster type.
* **Stochastic Yield Modeling**: Accounts for supply-chain unreliability using Beta and Discrete yield distributions.
* **Risk-Aware Solvers**: Optimizes for Expected Profit (`SAA`), Exponential Utility (`Utility`), or Conditional Value at Risk (`CVaR`) to protect against tail-risk stockouts.

---

## Project Structure

* `optistock/stockkeep.py`: The main orchestrator for running multi-item simulations and hold-out tests.
* `optistock/solvers.py`: The unified `ForecastSolver` — pairs `Item` objects with fitted `BaseForecaster` instances to solve for optimal stock quantities.
* `optistock/items.py`: The `Item` class encapsulating cost structure, constraints, and yield distributions.
* `optistock/forecasting/`: Bayesian forecasting module:
  * `base.py`: `BaseForecaster` abstract class and `ErrorEstimations` utilities.
  * `linear_regressors.py`: `BayesTimeSeries`, `BARTBayesTimeSeries`, `HSGPBayesTimeSeries`.
  * `state_space.py`: `UnivariateSSM` — flexible Bayesian structural state space model.
  * `mix_media_models.py`: `MediaMixModel` — Bayesian Marketing Mix Model for sales attribution.
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

## The StockKeep Orchestrator

`StockKeep` is the primary entry point for end-to-end inventory planning. It automatically handles training, forecasting, and optimization for a portfolio of items — and supports hold-out validation to evaluate how well your stock decisions would have performed on real data.

```python
from optistock.stockkeep import StockKeep
from optistock.forecasting.state_space import UnivariateSSM

# Initialize with long-form history and item config table
sk = StockKeep(
    histories=df_history,        # long-format: item, date, sales columns
    item_configs=df_items,       # columns: name, cost_price, selling_price, salvage_value
    forecaster_class=UnivariateSSM,
    yield_profiles=yield_map,    # optional: {item_name: YieldDistribution}
)

# Validate on the last 30 days of observed data
results = sk.run_holdout(
    holdout_days=30,
    objective="Utility",
    risk_aversion=0.3,
    fit_kwargs={
        "draws": 1000,
        "build_model_kwargs": {"seasonal_period": 7},
    },
)

print(results["allocation"])     # {item_name: optimal_quantity}
print(results["metrics"])        # per-item profit, service level, SMAPE, stockout
print(sk.summary())              # ForecastSolver diagnostics

# Overlay forecast vs actual holdout sales
fig, ax = sk.plot_forecast("Tablet Air")

```

### Production Mode

When you are confident in the model, run on the full history to get forward-looking allocations:

```python
results = sk.run(
    forecast_days=30,
    objective="SAA",
)
print(results["allocation"])
print(results["period"])         # (start_date, end_date) of the planning horizon

```

---
