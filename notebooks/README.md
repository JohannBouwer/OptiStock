# OptiStock Notebooks

A walkthrough of the OptiStock library, building up from the single-item newsvendor problem to multi-item Bayesian inventory policies with hierarchical forecasting.

| # | Notebook | Goal |
|---|---|---|
| 1 | [Introduction](1_Introduction.ipynb) | Introduces the newsvendor problem and Jensen's inequality, showing why ordering the mean demand is wrong and how the critical fractile gives the profit-optimal stock quantity for a single item. |
| 2 | [Multi-Item and Constraints](2_Mulit_Item_and_constraints.ipynb) | Extends the newsvendor model to multiple items sharing resources (budget, shelf space), solved jointly with `ForecastSolver` and interpreted via Lagrangian shadow prices. |
| 3 | [Yield Distributions](3_Yield_distributions.ipynb) | Adds a second source of uncertainty — random production yield (`BetaYield`) — and compares SAA, CVaR, and exponential-utility objectives for risk-aware allocation. |
| 4 | [Forecasting Example](4_Forecasting_Example.ipynb) | Introduces the three Bayesian forecasters (`BayesTimeSeries`, `MediaMixModel`, `UnivariateSSM`) on tailored synthetic data, and shows the common `fit → forecast → get_demand_distribution` interface that plugs into `ForecastSolver`. |
| 5 | [Stockouts](5_Stockouts.ipynb) | Simulates demand censoring from chronic stockouts and demonstrates two corrections: NaN-masking the observations for a state space model, and `pm.Censored` likelihood for a linear Bayesian model. |
| 6 | [Manager Class](6_Manager_Class.ipynb) | Tour of the four end-to-end policy classes — `PeriodicOrderUpTo`, `PeriodicBaseStock`, `ContinuousFixedQuantity`, `ContinuousOrderUpTo` — covering periodic (R, S) and continuous (s, Q)/(s, S) review with a shared forecasting back-end. |
| 7 | [Inventory Policy](7_Inventory_Policy.ipynb) | Deep-dive on `PeriodicBaseStock` service-level constraints and the `lead_time + review_period` effective horizon, quantifying the profit cost of guaranteeing a target cycle service level. |
| 8 | [Advanced Forecasting](8_Advanced_Forecasting.ipynb) | Shows how to inspect, tighten, or swap the Bayesian priors on every forecaster via the `*Priors` dataclasses and the `Prior` wrapper, including the family-grouped priors on `UnivariateSSM`. |
| 9 | [Hierarchical Forecasting](9_Hierarchical_Forecasting.ipynb) | Demonstrates `HierarchicalBayesTimeSeries` for panel forecasting across many items, with partial pooling letting short-history "new SKUs" borrow strength from long-history items via shared hyper-priors. |
