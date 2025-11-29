# Inventory Optimization (News Vendor Framework)

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in supply chains.
- [Inventory Optimization (News Vendor Framework)](#inventory-optimization-news-vendor-framework)
  - [Key Features](#key-features)
  - [Project Structure](#project-structure)
  - [The Basic Single-Item Problem](#the-basic-single-item-problem)
  - [Multi-Item Constrained \& Stochastic](#multi-item-constrained--stochastic)

## Key Features

- Beyond the Mean: Implements the Newsvendor Model (Critical Fractile) to find optimal order quantities based on margin and volatility.
- Constrained Optimization: Solves multi-item portfolios with hard constraints (Budget, Storage Volume, Weight) using Lagrangian Relaxation and Scipy Trust-Region methods.
- Double Uncertainty: Handles both Demand Uncertainty (Normal or Bayesian Posteriors) and Supply/Yield Uncertainty (Random Yield modeled via Beta distributions).
- Risk Aversion: Optimizes for Conditional Value at Risk (CVaR) or Exponential Utility, allowing you to penalize catastrophic tail risks.
- Visualization: Various plots to view trade-offs and optimum results.

## Project Structure

- newsvendor_solver.py: Core logic containing solvers (StochasticMonteCarloSolver, LagrangianConstraintSolver, ScipyOptimizationSolver).
- items.py: Definitions for Item properties (Cost, Price, Salvage, Constraints).
- demand_distribution.py: Classes for modeling demand uncertainty (NormalDemand, SampledDemand).
- yield_distributions.py: Classes for modeling yield/manufacturing uncertainty (NormalDemand, SampledDemand).
- plotting.py: Visualization suite (NewsvendorVisualizer) for profit curves, demand distributions, and risk profiles.
  
## The Basic Single-Item Problem

Find the optimal order quantity ($Q^*$) for a risky product.

```python
from items import Item
from demand_distribution import NormalDemand
from newsvendor_solver import SingleItemNewsvendorSolver
from plotting import NewsvendorVisualizer

# Define Economics
# Cost 30, Sell 50, Salvage 10 
item = Item("Gaming Mouse", cost_price=30, selling_price=50, salvage_value=10)

# Define Uncertainty (Normal or Sampled)
demand = NormalDemand(mean=100, std_dev=20)

# Solve
solver = SingleItemNewsvendorSolver(item, demand)
q_star = solver.solve() # Returns integer Q*

# Visualize
viz = NewsvendorVisualizer()
fig = viz.plot_single_item_analysis(item, demand, q_star)
fig.show()
```

## Multi-Item Constrained & Stochastic

Optimize a portfolio with a strict budget and uncertain manufacturing yield.

```python
from newsvendor_solver import StochasticMonteCarloSolver
from demand_distribution import BetaYield

# Define an item with uncertain supply (70% mean yield)
risky_chip = Item("AI Chip", 100, 200, 20, 
                  constraints={'budget': 100},
                  yield_distribution=BetaYield(7, 3))

# Setup Solver (Risk Averse: Lambda > 0)
solver = StochasticMonteCarloSolver(
    problems=[(risky_chip, demand_dist)], 
    limits={'budget': 50000},
    risk_aversion=2.0  # Penalize volatility
)

allocation = solver.solve()
```