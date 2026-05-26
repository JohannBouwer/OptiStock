# Configurable Priors

Every forecaster and the causal estimator accept a typed `*Priors` dataclass. Override one field, the entire prior, or swap the distribution outright — no monkey-patching of model internals.

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
- `SyntheticControlPriors`

All defined in `optistock/forecasting/priors.py` and `optistock/causal/priors.py`. Each ships sensible defaults — you only override what you want to change.

## Inspecting a model's priors

After construction, `model.priors` (or `model.describe_priors()`) returns a tabular summary of every prior, its distribution, and the human-readable description supplied via the `Prior` wrapper. Useful for documenting model variants in a notebook without re-deriving them from the source.

## Going deeper

See [notebook 8](../notebooks/8_Advanced_Forecasting.ipynb) for a tour of prior inspection, single-field overrides, and full-distribution swaps across all the forecasters.
