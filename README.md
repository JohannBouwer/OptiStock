
---

# Inventory Optimization (News Vendor Framework)

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in stock supply allocation. 

* Inventory Optimization (News Vendor Framework)
* Key Features
* Project Structure
* Probabilistic Demand Forecasting
* The Basic Single-Item Problem
* Multi-Item Constrained & Stochastic

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

* **Probabilistic Forecasting**: Implements Bayesian Time Series (PyMC) and Bayesian Additive Regression Trees (BART) to generate full demand distributions.
* **Beyond the Mean**: Uses the Newsvendor Model (Critical Fractile) to find optimal order quantities based on margin and volatility.
* **Constrained Optimization**: Solves multi-item portfolios with budget and storage constraints using Greedy ROI or Scipy Trust-Region methods.
* **Risk Aversion**: Optimizes for Conditional Value at Risk (CVaR) or Exponential Utility to penalize tail risks.

## Project Structure

* **`forecasting.py`**: Bayesian models for time series demand forecasting.
* **`items.py`**: Definitions for `Item` costs, critical fractiles, and constraints.
* **`solvers.py`**: Optimization engines (Single-Item, Multi-Item Greedy, Scipy, and Stochastic MC).
* **`distributions/`**: Models for demand and supply (yield) uncertainty.

## Probabilistic Demand Forecasting

Generate full posterior predictive distributions instead of single-point estimates.

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
