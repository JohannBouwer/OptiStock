# Configurable Priors

Every forecaster, the causal estimator, and the CLV models accept a typed `*Priors` dataclass. Override one field, the entire prior, or swap the distribution outright — no monkey-patching of model internals.

```python
from optistock.forecasting import (
    BayesTimeSeries,
    BayesTimeSeriesPriors,
    HierarchicalBayesTimeSeries,
    HierarchicalBayesTimeSeriesPriors,
    Prior,
)

# Tighten observation noise for a low-variance series
priors = BayesTimeSeriesPriors(
    sigma=Prior("HalfNormal", {"sigma": 0.25}, "Observation noise"),
)
model = BayesTimeSeries(df_history, target_col="sales", priors=priors)
print(model.priors)   # tabular summary of every prior and its description

# Hierarchical hyper-priors — narrow the population spread on growth
hpriors = HierarchicalBayesTimeSeriesPriors(
    growth_sigma=Prior("HalfNormal", {"sigma": 0.05}, "Population spread on growth"),
)
hmodel = HierarchicalBayesTimeSeries(df_wide, priors=hpriors)
```

## Available `*Priors` classes

- `BayesTimeSeriesPriors`
- `BARTBayesTimeSeriesPriors`
- `HSGPBayesTimeSeriesPriors`
- `HierarchicalBayesTimeSeriesPriors`
- `UnivariateSSMPriors` — uses family-grouped fields (trend / seasonal / observation)
- `HierarchicalSSMPriors` — same families, plus `<family>_mu` / `<family>_sigma` hyper-priors for the pooled seasonal and regression coefficients
- `SyntheticControlPriors`
- `ParetoNBDPriors`, `GammaGammaPriors` — see [clv](clv.md)

All defined in `optistock/forecasting/priors.py`, `optistock/causal/priors.py`, and `optistock/clv/priors.py`. Each ships sensible defaults — you only override what you want to change.

## What scale are the numbers in?

Most forecasters scale the target before fitting, so prior values are **not** in units of sales. For the state-space models (`UnivariateSSMPriors`, `HierarchicalSSMPriors`) the target is **standardised** — `(y_work - center) / scale`, where `y_work` is the target, optionally `log1p`-transformed. One unit therefore means **one standard deviation of the training data**, and the data is centred on zero.

That is deliberate: it makes the same numbers valid in both raw and `log_transform` mode and across datasets, rather than being implicitly tied to one series' magnitude. Two consequences when overriding:

- `initial_level` is zero-centred because the data is centred. `initial_slope` is separate and much tighter — it is a *per-step drift* that integrates into the level, so values that look small still move the forecast a long way (see the `trend_innovations_order` note in [forecasting.md](forecasting.md)).
- `regression_beta` assumes regressors are scaled to roughly `O(1)`. Under `log_transform` a coefficient is a log-multiplier, so an unscaled spend column in currency units will fight its prior; the model warns at fit time when it spots one.

The other forecasters (`BayesTimeSeries` and friends) divide the target by its maximum instead, so their priors live in `[0, 1]` space.

The CLV models are wrappers around `pymc-marketing`, which configures priors through its own `model_config` dict. `optistock.clv` translates the project `Prior` objects into that format for you, and its defaults reproduce `pymc-marketing`'s own — so omitting `priors=` behaves exactly like using the library directly.

## Inspecting a model's priors

After construction, `model.priors` (or `model.describe_priors()`) returns a tabular summary of every prior, its distribution, and the human-readable description supplied via the `Prior` wrapper. Useful for documenting model variants in a notebook without re-deriving them from the source.

## Going deeper

See [notebook 8](../notebooks/8_Advanced_Forecasting.ipynb) for a tour of prior inspection, single-field overrides, and full-distribution swaps across all the forecasters.
