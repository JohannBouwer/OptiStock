# Causal Calibration

When a past intervention (promotion, price change, packaging refresh) is mixed in with normal sales history, a single `beta_event` coefficient in the forecaster can swing wildly on noisy data. The `optistock.causal` module measures the lift with **Bayesian synthetic control** and feeds the result back as a soft prior, anchoring the forecaster's event coefficient to a number that came from a proper counterfactual.

The pattern is two steps: estimate the causal effect with `SyntheticControl`, then translate it into a `LiftConstraint` and pass it to any forecaster.

## Step 1 — measure the lift with `SyntheticControl`

`SyntheticControl` wraps CausalPy's `WeightedSumFitter`: donor items are combined under a Dirichlet (non-negative weights summing to one) to reconstruct the treated item's pre-intervention behaviour, and the post-intervention gap is the causal effect.

```python
from optistock.causal import SyntheticControl

sc = SyntheticControl(
    data=df_wide,
    treated_item="Tablet Air",
    donor_items=["Gaming Mouse", "Keyboard", "USB Hub"],
    treatment_date="2025-02-01",
    intervention_name="Spring_Promo",
)
sc.fit(samples=1000, tune=1000, chains=4)

effect = sc.summary()
fig, _ = sc.plot()           # observed vs synthetic counterfactual + impact band
fig, _ = sc.plot_weights()   # posterior donor weights (expect sparse)
```

## Step 2 — feed the measured lift into the forecaster

`LiftConstraint` adds an observed-Normal likelihood term on the matching `beta_event` coefficient — same pattern `pymc-marketing` uses for lift tests. Values are in raw units; the forecaster handles internal scaling.

```python
from optistock.causal import LiftConstraint
from optistock.forecasting import HierarchicalBayesTimeSeries

constraint = LiftConstraint.from_synthetic_control(sc, event_name="Spring_Promo")
# Equivalent: LiftConstraint.from_causal_effect(effect, event_name="Spring_Promo")

model = HierarchicalBayesTimeSeries(
    df_wide,
    lift_constraints=[constraint],
)
model.create_events({"Spring_Promo": {"Tablet Air": ["2025-02-01", "2025-02-02"]}})
model.fit()
```

For single-series forecasters (`BayesTimeSeries`, `BARTBayesTimeSeries`, `HSGPBayesTimeSeries`) the `item=` field on `LiftConstraint` is unused; for `HierarchicalBayesTimeSeries` it must name the treated item.

## When to use this

- You have a known event in the training window whose effect on demand you don't want the forecaster to learn from raw data alone.
- You can name 3+ donor items that were untreated during the event window and whose pre-treatment behaviour looks similar to the treated item.
- You want the forecaster's `beta_event` posterior to be anchored to a measurement, not just to whatever the likelihood happens to fit.

Skip it when the event has no plausible donor pool, when treatment was simultaneous across all items, or when the goal is purely forecasting (no decision about repeating the intervention).

## Going deeper

See [notebook 10](../notebooks/10_Causal_Calibration.ipynb) for a full walk-through of the synthetic-control fit, donor weight inspection, and the resulting calibrated forecast.

The priors on `SyntheticControl` itself are configurable — see [priors.md](priors.md) for `SyntheticControlPriors`.
