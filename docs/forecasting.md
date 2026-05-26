# Forecasting

OptiStock's forecasters all share the same `fit → forecast → get_demand_distribution` interface, so any of them can be passed to [ForecastSolver](solver.md) or to one of the [orchestrators](orchestrators.md). What changes between them is the structure of demand they're built to capture.

| Class | Captures | Input shape | Best for |
|---|---|---|---|
| `BayesTimeSeries` | Trend, Fourier seasonality, named events | Long-format single series | Most single-item series with known calendar effects |
| `BARTBayesTimeSeries` | Non-linear seasonality and interactions (via BART) | Long-format single series | When seasonality is not well described by Fourier terms |
| `HSGPBayesTimeSeries` | Smooth non-parametric seasonality (Hilbert-space GP) | Long-format single series | Smooth but irregular cycles |
| `HierarchicalBayesTimeSeries` | Trend + seasonality + events across a panel of items | Wide-format, one column per item | Many items, some with short or noisy histories |
| `UnivariateSSM` | Time-varying trend + structural components + exogenous regressors | Datetime-indexed DataFrame | When components evolve over time and you want a Kalman-smoothed decomposition |
| `MediaMixModel` | Baseline demand + adstock + saturation per channel | Long-format with channel spend columns | Sales attribution alongside forecasting |

---

## BayesTimeSeries — Fourier + events

The workhorse single-series model. Trend plus Fourier-seasonality components, with named events that contribute a per-event `beta_event` coefficient. Returns a full posterior predictive distribution.

```python
from optistock.forecasting import BayesTimeSeries

model = BayesTimeSeries(df_history, target_col="sales")
model.create_events({"Promo_A": ["2025-01-01", "2025-01-02"]})
model.fit(target="sales", date_col="date")
model.forecast(scenario={"df_future": df_future, "date_col": "date"})

demand = model.get_demand_distribution("2025-01-01", "2025-01-07")
```

See [notebook 4](../notebooks/4_Forecasting_Example.ipynb).

## BARTBayesTimeSeries — non-linear seasonality

Drop-in replacement for `BayesTimeSeries` when the seasonal pattern is hard to write down as Fourier terms. Uses Bayesian Additive Regression Trees to learn trend + seasonality directly from the calendar features.

```python
from optistock.forecasting import BARTBayesTimeSeries

model = BARTBayesTimeSeries(df_history)
model.fit(target="sales", date_col="date", trees=50)
model.forecast(scenario={"df_future": df_future, "date_col": "date"})
fig, ax = model.plot_components()
```

## HSGPBayesTimeSeries — smooth non-parametric seasonality

Hilbert-space approximation to a Gaussian process — gives you the smoothness of a GP without the cubic cost. Same interface as `BayesTimeSeries`.

---

## HierarchicalBayesTimeSeries — panel forecasting

`HierarchicalBayesTimeSeries` is a multi-item version of `BayesTimeSeries` with **partial pooling across items** via shared hyper-priors. Each item keeps its own intercept, growth, seasonal, and event coefficients, but those coefficients are drawn from population-level distributions whose mean and spread are themselves learned from the data. Items with short or noisy histories borrow strength from the rest of the panel.

Input is **wide-format**: one `date` column plus one numeric column per item. Ragged histories (NaNs) are supported — the model masks missing observations out of the likelihood. A non-centered parameterisation is used for every per-item coefficient to avoid funnel pathologies under HMC.

```python
from optistock.forecasting import HierarchicalBayesTimeSeries

model = HierarchicalBayesTimeSeries(df_history, date_col="date")
model.create_events({
    "Black_Friday": ["2025-11-28"],                           # all items
    "Mouse_Promo":  {"Gaming Mouse": ["2025-02-01", "2025-02-02"]},
})
model.fit(samples=1000, chains=4)
model.forecast(scenario={"df_future": df_future})

demand = model.get_demand_distribution("2025-03-01", "2025-03-30", item="Tablet Air")
```

`get_demand_distribution` requires the `item=` argument; the result plugs into `ForecastSolver` like any single-series demand.

See [notebook 9](../notebooks/9_Hierarchical_Forecasting.ipynb).

---

## UnivariateSSM — Bayesian structural state space

Built on `pymc-extras`. Composes interpretable components (trend, seasonality, exogenous regressors) and estimates them jointly via MCMC. Keeps parameters that vary with time, so it's a good choice when the level or slope drifts.

```python
from optistock.forecasting.state_space import UnivariateSSM

model = UnivariateSSM(df_history, target_col="sales", exog={"spend": True})
model.build_model(
    trend_order=2,
    trend_innovations_order=[0, 1],
    seasonal_period=7,
    seasonal_harmonics=3,
)
model.fit(draws=1000, tune=500)
model.smooth_and_filter()
model.forecast(periods=30, scenario={"data_spend": future_spend})

demand = model.get_demand_distribution("2025-03-01", "2025-03-30")
```

See [notebook 4](../notebooks/4_Forecasting_Example.ipynb) for a worked example, and [notebook 5](../notebooks/5_Stockouts.ipynb) for how SSMs handle demand censoring via NaN-masking.

---

## MediaMixModel — attribution + demand

Wraps `pymc-marketing`'s MMM. Splits sales into baseline demand, advertising carry-over (adstock), and diminishing returns (saturation). Useful when channel spend is observed and you need both a forecast and an attribution story.

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

See [notebook 4](../notebooks/4_Forecasting_Example.ipynb).

---

## Configuring forecaster behaviour

Every forecaster accepts a typed `*Priors` dataclass — see [priors.md](priors.md). To anchor a noisy `beta_event` coefficient to a measured causal lift, see [causal.md](causal.md).
