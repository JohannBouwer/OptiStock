
---

# OptiStock: Inventory Optimization with the News Vendor Framework

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in stock supply allocation. 

## Table of Contents

1. [Installation](#installation)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Probabilistic Forecasting](#probabilistic-forecasting)
5. [The StockKeep Orchestrator](#the-stockkeep-orchestrator)
---

## Installation
### From GitHub (Recommended for users)
You can install OptiStock directly from the source repository using pip. This will automatically install all necessary dependencies like PyMC, BART, and Scipy.
```Bash
pip install git+https://github.com/JohannBouwer/OptiStock.git
```
### For Development (Editable Mode)
If you plan to modify the code or contribute to the project, clone the repository and install it in "editable" mode. This ensures that changes you make to the source code are immediately reflected in your environment without needing a re-install.

```Bash
# Clone the repository
git clone https://github.com/JohannBouwer/OptiStock.git
cd OptiStock

# Install in editable mode
pip install -e .
```

## Key Features

* **Bayesian Forecasting**: Implements Fourier-based Time Series, Hybrid BART (with linear trend), and Hilbert Space Gaussian Processes (HSGP).
* **End-to-End Orchestration**: The `StockKeep` class manages the entire pipeline from data preparation to hold-out validation.
* **Stochastic Yield Modeling**: Accounts for supply-chain unreliability using Beta and Discrete yield distributions.
* **Risk-Aware Solvers**: Optimizes for Expected Profit, Exponential Utility (default), or CVaR to protect against tail-risk stockouts.

---

## Project Structure

* `optistock/stockkeep.py`: The main orchestrator for running multi-item simulations and hold-out tests.
* `optistock/forecasting.py`: Bayesian models (`BayesTimeSeries`, `BARTBayesTimeSeries`, `HSGPBayesTimeSeries`).
* `optistock/solvers.py`: Optimization engines including Scipy Trust-Region and Monte Carlo solvers.
* `optistock/distributions/`: Probabilistic models for demand and manufacturing yield.
* `optistock/plot_suite/`: Visualization tools for forecast validation and profit curves.

## Probabilistic Demand Forecasting

Generate full posterior predictive distributions instead of single-point estimates using typical bayes models.

### Fourier & Event-based Forecasting

Capture complex seasonality and holiday effects using `BayesTimeSeries`.

```python
from optistock.forecasting import BayesTimeSeries
from optistock.distributions.demand_distributions import SampledDemand

# Initialize model and define promotional events
model = BayesTimeSeries(df_history, target_col="sales")
model.create_events({"Promo_A": ["2025-01-01", "2025-01-02"]})

# Fit and predict a future window
model.fit()
model.predict(df_future)

# Extract total demand samples for a specific period
samples = model.get_demand_distribution("2025-01-01", "2025-01-07")
demand_dist = SampledDemand(samples)

```

### Non-linear Seasonality with BART

Use Bayesian Additive Regression Trees to learn patterns directly from data.

```python
from optistock.forecasting import BARTBayesTimeSeries

# Fit BART model (automatically handles trend and seasonality)
bart_model = BARTBayesTimeSeries(df_history)
bart_model.fit(trees=50)

# Generate forecast and visualize learned components
bart_model.predict(df_future)
fig, ax = bart_model.plot_components()

```

## The Basic Single-Item Problem

Find the optimal order quantity () for a risky product.

```python
from optistock.items import Item
from optistock.distributions.demand_distributions import NormalDemand
from optistock.solvers import SingleItemSolver

# Define Economics: Cost 30, Sell 50, Salvage 10 
item = Item("Gaming Mouse", 30, 50, 10)
demand = NormalDemand(mean=100, std_dev=20)

# Solve for Q*
solver = SingleItemSolver(item, demand)
q_star = solver.solve() 

```

## Multi-Item Constrained & Stochastic

Optimize a portfolio with strict constraints and manufacturing yield risks.

```python
from optistock.solvers import StochasticMonteCarloSolver
from optistock.distributions.yield_distributions import BetaYield

# Item with 70% mean yield risk
risky_chip = Item("AI Chip", 100, 200, 20, 
                  yield_distribution=BetaYield(7, 3))

# Setup Solver (Risk Averse using CVaR)
solver = StochasticMonteCarloSolver(problems=[(risky_chip, demand_dist)], 
                                    limits={'budget': 50000})

allocation = solver.solve(method="CVAR", risk_aversion=0.5)

```

## The StockKeep Orchestrator
The `StockKeep` class is the primary entry point for simulating real-world performance. It automatically splits data into training and hold-out sets to validate how much profit your stock levels would have generated.

```Python
from optistock.stockkeep import StockKeep
from optistock.solvers import StochasticMonteCarloSolver

# Initialize with long-form history and item configs
sk = StockKeep(df_history, df_items, yield_profiles=yield_map)

# Run a 30-day hold-out simulation
results = sk.run_simulation(
    forecast_days=30,
    solver_class=StochasticMonteCarloSolver,
    solver_params={'limits': {'storage': 500}}
)

# Visualize forecast vs. actual sales
sk.plot_forecast("Tablet Air")
```
---

