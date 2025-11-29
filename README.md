# Inventory Optimization (News Vendor Framework)

A Python framework for solving inventory optimization problems. This library goes beyond the standard "Order the Mean" approach, offering tools for constrained, stochastic, and risk-aware decision-making in supply chains.

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