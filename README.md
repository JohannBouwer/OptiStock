
---

# Inventory Optimization (News Vendor Framework)

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in supply chains.

* [Inventory Optimization (News Vendor Framework)](https://www.google.com/search?q=%23inventory-optimization-news-vendor-framework)
* [Key Features](https://www.google.com/search?q=%23key-features)
* [Project Structure](https://www.google.com/search?q=%23project-structure)
* [The Basic Single-Item Problem](https://www.google.com/search?q=%23the-basic-single-item-problem)
* [Multi-Item Constrained & Stochastic](https://www.google.com/search?q=%23multi-item-constrained--stochastic)



## Key Features

* **Probabilistic Forecasting**: Implements Bayesian Time Series (PyMC) and Bayesian Additive Regression Trees (BART) to generate full demand distributions instead of point estimates.
* **Beyond the Mean**: Implements the Newsvendor Model (Critical Fractile) to find optimal order quantities based on margin and volatility.
* **Constrained Optimization**: Solves multi-item portfolios with budget and storage constraints using Greedy ROI or Scipy Trust-Region methods.
* **Double Uncertainty**: Simultaneously handles Demand Uncertainty (Sampled Posteriors) and Supply/Yield Uncertainty (Random Yield modeled via Beta distributions).
* **Risk Aversion**: Optimizes for Conditional Value at Risk (CVaR) or Exponential Utility to penalize catastrophic tail risks.

## Project Structure

* **`forecasting.py`**: Bayesian models (BART and Fourier-based) for time series demand forecasting.
* **`items.py`**: Definitions for `Item` properties including costs, critical fractiles, and constraints.
* **`solvers.py`**: Optimization engines (Single-Item, Multi-Item Greedy, Scipy Trust-Region, and Stochastic Monte Carlo).
* **`distributions/`**:
  * `demand_distributions.py`: Normal and Sampled (Bayesian) demand models.
  * `yield_distributions.py`: Models for manufacturing uncertainty (Beta and Perfect yield).
* **`plot_suite/`**: Modular visualization suite for risk profiles, portfolio allocations, and forecast components.

## The Basic Single-Item Problem

Find the optimal order quantity for a risky product based on cost of understocking vs. overstocking.

```python
from inventory_management.items import Item
from inventory_management.distributions.demand_distributions import NormalDemand
from inventory_management.solvers import SingleItemSolver

# Define Economics: Cost 30, Sell 50, Salvage 10 
item = Item("Gaming Mouse", cost_price=30, selling_price=50, salvage_value=10)

# Define Uncertainty (Normal or Sampled from a Forecast)
demand = NormalDemand(mean=100, std_dev=20)

# Solve for Q* (Smallest Q such that P(D <= Q) >= Critical Fractile)
solver = SingleItemSolver(item, demand)
q_star = solver.solve() 

print(f"Optimal Order Quantity: {q_star}")

```

## Multi-Item Constrained & Stochastic

Optimize a portfolio with strict constraints and manufacturing yield risks using Monte Carlo simulation.

```python
from inventory_management.solvers import StochasticMonteCarloSolver
from inventory_management.distributions.yield_distributions import BetaYield

# Define an item with a 70% mean yield (manufacturing risk)
risky_chip = Item(
    "AI Chip", 
    cost_price=100, 
    selling_price=200, 
    salvage_value=20,
    constraints={'storage': 5.0},
    yield_distribution=BetaYield(alpha=7, beta_param=3)
)

# Setup Solver (Risk Averse using CVaR at 5% tail)
solver = StochasticMonteCarloSolver(
    problems=[(risky_chip, demand_dist)], 
    limits={'storage': 500}
)

# Solve with risk_aversion (0.0 to 1.0)
allocation = solver.solve(method="CVAR", risk_aversion=0.5, cvar=0.05)

```