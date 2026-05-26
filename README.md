# OptiStock: Inventory Optimization with the News Vendor Framework

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in stock supply allocation.

> [!NOTE]
> **Heads up:** This is a project I completed for my own development / learning. The models work and have been tested, but no claims are made about production-readiness, computational efficiency, or suitability for any particular use case.

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

## Quick start

The full pipeline — forecast demand, optimise stock under uncertainty, validate on a holdout — is wrapped by the orchestrator classes in `optistock.stockkeep`. Here's a periodic-review (R, S) replenishment problem end-to-end:

```python
from optistock.stockkeep import PeriodicOrderUpTo
from optistock.forecasting import BayesTimeSeries

sk = PeriodicOrderUpTo(
    histories=df_history,         # long-format: item, date, sales
    item_configs=df_items,        # name, cost_price, selling_price, salvage_value, lead_time
    review_period=7,
    forecaster_class=BayesTimeSeries,
)

# Validate on the last 30 days of observed data
results = sk.run_holdout(holdout_days=30, objective="SAA")

print(results["allocation"])      # gross optimal quantity per item
print(results["metrics"])         # per-item profit, service level, SMAPE, stockout flag

# Production mode — fit on full history, forecast forward
results = sk.run(forecast_days=7, objective="SAA")
print(results["allocation"])
```

Swap `forecaster_class` for any other forecaster, switch `PeriodicOrderUpTo` for one of the other three policies, or drop down to [ForecastSolver](docs/solver.md) for a one-shot allocation without the orchestration layer.

## What it can do

| Capability | Class(es) | Docs | Notebook |
|---|---|---|---|
| Bayesian demand forecasting (Fourier, BART, HSGP) | `BayesTimeSeries`, `BARTBayesTimeSeries`, `HSGPBayesTimeSeries` | [forecasting](docs/forecasting.md) | [4](notebooks/4_Forecasting_Example.ipynb) |
| Hierarchical panel forecasting with partial pooling | `HierarchicalBayesTimeSeries` | [forecasting](docs/forecasting.md#hierarchicalbayestimeseries--panel-forecasting) | [9](notebooks/9_Hierarchical_Forecasting.ipynb) |
| Bayesian Structural Time Series | `UnivariateSSM` | [forecasting](docs/forecasting.md#univariatessm--bayesian-structural-state-space) | [4](notebooks/4_Forecasting_Example.ipynb), [5](notebooks/5_Stockouts.ipynb) |
| Marketing Mix Model (attribution + demand) | `MediaMixModel` | [forecasting](docs/forecasting.md#mediamixmodel--attribution--demand) | [4](notebooks/4_Forecasting_Example.ipynb) |
| Causal calibration of event effects | `SyntheticControl`, `LiftConstraint` | [causal](docs/causal.md) | [10](notebooks/10_Causal_Calibration.ipynb) |
| Configurable priors on every model | `*Priors`, `Prior` | [priors](docs/priors.md) | [8](notebooks/8_Advanced_Forecasting.ipynb) |
| Unified single- / multi-item optimiser with constraints | `ForecastSolver`, `Item` | [solver](docs/solver.md) | [1](notebooks/1_Introduction.ipynb), [2](notebooks/2_Mulit_Item_and_constraints.ipynb) |
| Stochastic yield (Beta / Discrete) | `BetaYield`, `DiscreteYield`, ... | [solver](docs/solver.md#multi-item-with-constraints-and-risk-aversion) | [3](notebooks/3_Yield_distributions.ipynb) |
| Risk-aware objectives (SAA, Utility, CVaR) | `ForecastSolver` | [solver](docs/solver.md#objectives) | [3](notebooks/3_Yield_distributions.ipynb) |
| End-to-end policy orchestrators (R, S) and (s, Q)/(s, S) | `PeriodicOrderUpTo`, `PeriodicBaseStock`, `ContinuousFixedQuantity`, `ContinuousOrderUpTo` | [orchestrators](docs/orchestrators.md) | [6](notebooks/6_Manager_Class.ipynb), [7](notebooks/7_Inventory_Policy.ipynb) |

## Project structure

- [`optistock/forecasting/`](src/optistock/forecasting/) — Bayesian forecasters and their `*Priors` dataclasses.
- [`optistock/causal/`](src/optistock/causal/) — `SyntheticControl` and the `LiftConstraint` bridge into the forecasters.
- [`optistock/distributions/`](src/optistock/distributions/) — demand and yield distributions.
- [`optistock/solvers.py`](src/optistock/solvers.py), [`optistock/items.py`](src/optistock/items.py), [`optistock/stockkeep.py`](src/optistock/stockkeep.py) — `ForecastSolver`, `Item`, and the four policy orchestrators.
- [`optistock/plot_suite/`](src/optistock/plot_suite/) — visualisation utilities used by the forecasters and orchestrators.

## Where to go next

- **Learn by example** → [notebooks/](notebooks/README.md) — 10 numbered notebooks building from the single-item newsvendor up to hierarchical forecasting and causal calibration.
- **Look up a feature** → [docs/](docs/) — short narrative pages for each component, with pointers to the matching notebook.
- **Use it in your project** → the install commands and quick-start above.
