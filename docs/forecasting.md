# Forecasting

OptiStock's forecasters all share the same `fit → forecast → get_demand_distribution` interface, so any of them can be passed to [ForecastSolver](solver.md) or to one of the [orchestrators](orchestrators.md). What changes between them is the structure of demand they're built to capture.

| Class | Captures | Input shape | Best for |
|---|---|---|---|
| `BayesTimeSeries` | Trend, Fourier seasonality, named events | Long-format single series | Most single-item series with known calendar effects |
| `BARTBayesTimeSeries` | Non-linear seasonality and interactions (via BART) | Long-format single series | When seasonality is not well described by Fourier terms |
| `HSGPBayesTimeSeries` | Smooth non-parametric seasonality (Hilbert-space GP) | Long-format single series | Smooth but irregular cycles |
| `HierarchicalBayesTimeSeries` | Trend + seasonality + events across a panel of items | Wide-format, one column per item | Many items, some with short or noisy histories |
| `UnivariateSSM` | Time-varying trend + structural components + exogenous regressors | Datetime-indexed DataFrame | When components evolve over time and you want a Kalman-smoothed decomposition |
| `HierarchicalSSM` | The same, across a panel, with seasonality and covariate effects pooled | Wide-format, one column per item | Panels of short or new items that should share a seasonal shape |
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

### `trend_innovations_order` is a mask, not a list of indices

With `trend_order=2` the trend states are `(level, slope)`, and `trend_innovations_order` is a boolean mask over them:

| value | meaning | forecast sd from trend drift |
|---|---|---|
| `[1, 0]` (**default**) | level drifts, slope fixed — random walk with constant drift | `sigma * sqrt(h)` |
| `[0, 1]` | level fixed, slope drifts — smooth/*integrated* trend | `sigma * sqrt(h**3 / 3)` |
| `[1, 1]` | both drift | between the two |
| `0` | no drift at all | none |

The default puts innovations on the level only: uncertainty grows like `sqrt(h)` rather than cubically. Omit the argument to get it.

The mask must have exactly `trend_order` entries — it is *not* a list of state indices, and a wrong length raises a clear error rather than an opaque one from inside pymc-extras.

**Which mask to use, measured on the bakery holdout** (mean SMAPE over 6 items):

| horizon | level `[1, 0]` (default) | slope `[0, 1]` |
|---|---|---|
| 7 days | **25.4%** | 28.8% |
| 14 days | **23.3%** | 27.3% |
| 30 days | 44.7% | 44.2% |

The default wins by 3–4 points at 1–2 weeks — the range that matches a `lead_time + review_period` planning window — and ties at 30 days. Prefer `[0, 1]` only if you are forecasting a month or more ahead *and* care about the mean rather than the median; tighten `priors.process_noise` by roughly an order of magnitude if you do, since it is calibrated for the default.

One thing to know at long horizons: the level random walk has wider predictive variance, which under `log_transform` inflates the back-transformed **mean**, because `E[y] = exp(mu + sigma**2/2)`. That is the correct expected demand (and the right input to the newsvendor), but it makes the mean a poor *point* forecast at 30 days — the posterior median scores far better there (36.3% vs 79.1% for `[0, 1]`).

### `log_transform` — non-negative forecasts

The Kalman-filter likelihood is Gaussian by construction, so a truncated or count likelihood is not available. `log_transform=True` fits on `log1p(y)` instead and maps samples back with `expm1`, which is what makes forecasts non-negative. It also makes trend, seasonality and regressors *multiplicative* in the original scale — usually what demand data wants, since a Friday uplift is "+30%", not "+12 units".

```python
model = UnivariateSSM(df_history, target_col="sales", log_transform=True)
```

Things worth knowing:

- **The back-transform happens per sample**, inside `inverse_transform`, before any summing or averaging. Every plot method and `get_demand_distribution` route through it. If you pull samples out of `forecast_idata` yourself, call `model.inverse_transform(...)` on the draws first — `expm1` does not commute with the mean.
- **No `sigma**2 / 2` lognormal bias correction is applied, and none is wanted.** `forecast_observed` is already a full posterior *predictive* draw (pymc-extras adds the measurement covariance `H`), so averaging `expm1` over draws estimates `E[y]` directly. An analytic correction on top would double-count the observation noise.
- **`log1p` is not scale-free.** The `+1` offset is negligible for an item selling 300/day and dominant for one selling 2/day, which matters when a panel spans both.
- **A Gaussian in `log1p` space is a poor model for genuinely intermittent, low-count items.** For those, expect wide intervals and a downward-biased mean.
- The default is `False`, so existing raw-scale models are unaffected — but a raw-scale fit *can* forecast negative demand.

---

## HierarchicalSSM — panel state space with pooling

A panel version of `UnivariateSSM`: all items are fitted jointly in one multivariate structural model, with **seasonal amplitudes and regression coefficients partially pooled** across items, so a short or noisy item borrows its seasonal shape and covariate response from the population. Everything else (level, trend, noise variances) stays independent per item.

Input is **wide-format**: a date index, one numeric column per item, plus any exogenous columns. Exogenous regressors are shared — the same `x_t` drives every item, but each learns its own pooled coefficient.

```python
from optistock.forecasting.state_space import HierarchicalSSM

model = HierarchicalSSM(panel, items=ITEMS, exog=["promo"], log_transform=True)
model.build_model(seasonal_period=7, seasonal_harmonics=2)
model.fit(target_accept=0.95)
model.forecast(periods=30, scenario={"data_promo": future_promo})

demand = model.get_demand_distribution("2025-03-01", "2025-03-30", item="CROISSANT")
```

`get_demand_distribution` keeps the item dimension when `item` is omitted; passing `item=` returns a single-item distribution shaped exactly like `UnivariateSSM`'s, so it plugs straight into `ForecastSolver`.

Standardisation here **centres per item but scales globally**. In log space a seasonal or promotional effect is a scale-free multiplicative offset, so a common divisor is what keeps the pooled coefficients comparable across items of very different sizes; a per-item scale would quietly break the pooling. Centring per item is what lets one `initial_level` prior serve every item.

Compute scales worse than fitting items independently — roughly 2x for three items and growing — so prefer it when items are few, short, or new. See [notebook 12](../notebooks/12_Hierarchical_StateSpace.ipynb).

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
