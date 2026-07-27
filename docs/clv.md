# Customer Lifetime Value

Inventory optimisation asks what to stock. CLV asks who is worth stocking it for. The `optistock.clv` module estimates, per customer, how many more purchases they will make and how much each is worth — in a **non-contractual** setting, where customers churn silently and you never observe the moment they leave.

Two models split the job, and are combined at the end:

| Question | Model |
|---|---|
| How often will they buy, and are they still active? | `ParetoNBD` |
| How much do they spend per purchase? | `GammaGammaSpend` |

Both wrap [pymc-marketing](https://www.pymc-marketing.io/) and take the same configurable `*Priors` dataclasses as the rest of the repo.

## Step 1 — build the RFM summary

Both models consume one row per customer: `frequency` (repeat purchases), `recency` (time of the last one), `T` (time observed), and `monetary_value` (**mean** spend per transaction, not total). `rfm_summary` derives all of it from an order-line table and is re-exported for convenience.

```python
from optistock.clv import rfm_summary

rfm = rfm_summary(
    orders,                       # one row per order line
    customer_id_col="customer_id",
    datetime_col="date",
    monetary_value_col="revenue",
)
```

## Step 2 — model purchase frequency

```python
from optistock.clv import ParetoNBD

txn = ParetoNBD(rfm)
txn.fit(method="demz")

txn.expected_purchases(future_t=90)      # posterior, per customer
txn.expected_probability_alive()         # still active?

fig, ax = txn.plot_probability_alive_matrix()
fig, ax = txn.plot_frequency_recency_matrix(future_t=90)
```

Each customer carries a latent purchase rate and a latent dropout time: they buy as a Poisson process until they silently lapse. That is what lets `expected_probability_alive()` separate "quiet because they churned" from "quiet because they were always infrequent" — two customers with identical purchase counts get different answers depending on recency.

### Covariates

Pareto/NBD is the model here (rather than BG/NBD) because it accepts customer attributes that shift the purchase or dropout rate:

```python
txn = ParetoNBD(
    rfm,
    purchase_covariate_cols=["tier"],     # shifts how often they buy
    dropout_covariate_cols=["channel"],   # shifts how fast they churn
)
```

Covariate columns must be numeric and present in `rfm`. Note that the two matrix plots are unavailable on a covariate model — they sweep a synthetic frequency/recency grid with no covariate values to condition on; predict on real customers instead.

## Step 3 — model spend, then combine

```python
from optistock.clv import GammaGammaSpend

spend = GammaGammaSpend(rfm[rfm["frequency"] > 0])   # repeat buyers only
spend.fit(method="demz")

spend.expected_customer_spend()          # shrunk towards the population mean

clv = spend.expected_customer_lifetime_value(
    txn,                 # the fitted transaction model
    future_t=12,         # MONTHS — see the warning below
    discount_rate=0.01,  # per month
)
leaderboard = clv.mean(("chain", "draw")).to_dataframe("clv").sort_values("clv", ascending=False)
```

> [!WARNING]
> **`future_t` means two different things depending on the method.**
>
> | Method | Unit of `future_t` |
> |---|---|
> | `ParetoNBD.expected_purchases` | the time unit of your data (days, if you built RFM from daily dates) |
> | `GammaGammaSpend.expected_customer_lifetime_value` | **always months**, regardless of `time_unit` |
>
> This asymmetry is inherited from `pymc-marketing` (and from `lifetimes` before it). With daily data, `future_t=90` is 90 days of purchases but 90 *months* of lifetime value. For one quarter of CLV, pass `future_t=3`. `discount_rate` is likewise **per month**.

Gamma-Gamma is fitted on repeat buyers only — a customer with one purchase carries no information about how their spend varies — so filter `frequency > 0` first. Its value is the shrinkage: a customer with one £200 basket is pulled hard towards the population mean, while one with thirty is trusted.

It assumes spend per transaction is independent of purchase frequency. Worth checking (`rfm[["frequency", "monetary_value"]].corr()`) — heavy buyers with systematically smaller baskets break the assumption.

## Priors

```python
from optistock.clv import ParetoNBD, ParetoNBDPriors
from optistock.forecasting import Prior

priors = ParetoNBDPriors(
    alpha=Prior("Weibull", {"alpha": 3, "beta": 7}, "Purchase-rate scale"),
)
txn = ParetoNBD(rfm, priors=priors)
txn.describe_priors()      # same tabular summary as the forecasters
```

Defaults reproduce `pymc-marketing`'s own, so omitting `priors=` changes nothing. See [priors.md](priors.md).

## Sampling

`fit()` takes `method="mcmc"` (full NUTS), `"demz"`, or `"map"`, plus the usual `draws` / `chains` / `target_accept`.

| Method | Speed | Use it for |
|---|---|---|
| `"mcmc"` | slowest | the numbers you act on |
| `"demz"` | ~10× faster than NUTS | iterating; diagnostics are looser (expect `r_hat` warnings) |
| `"map"` | fastest | a quick point estimate — no posterior spread, so don't read intervals off it |

Prefer **`method="demz"`** while iterating: on these low-dimensional models it lands in the same place as NUTS. Fit once with `"mcmc"` before you trust the numbers.

> [!NOTE]
> `method="map"` needs `pymc-marketing>=0.19.4`. Earlier releases raise `TypeError: NDArray.record() missing ... 'in_warmup'` against `pymc>=5.28`, which is why `pyproject.toml` pins the floor there.

## When to use this

- Customers transact repeatedly and can leave without telling you (retail, e-commerce, food service).
- You want to rank customers by expected future value, or size a retention budget against it.

Skip it in a **contractual** setting where churn is observed directly (subscriptions with cancellations) — a survival model fits that better. Skip it too when purchases are effectively one-off, since there is no repeat behaviour to learn from.

## Going deeper

See [intro_clv_models.ipynb](../notebooks/intro_clv_models.ipynb) for a full walk-through on synthetic data with a known ground truth: parameter recovery, separating churn from quiet, holdout validation, and where covariates actually change a prediction.

## Anything not wrapped

`model.model` exposes the underlying `pymc-marketing` object, so the full API stays reachable:

```python
txn.model.distribution_new_customer_purchase_rate()
```
