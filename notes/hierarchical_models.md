# Hierarchical Models

## Why pool across items

- Typical models fit one item at a time. Each new item starts from scratch, short or noisy histories produce noisy coefficients, and event coefficients are especially fragile when the event fires only a handful of times. 
- A hierarchical model puts every item into the same PyMC model and lets them "borrow strength" from each other. 
- Items with short history shrink toward the population mean; items with strong history barely move from where they would have landed on their own.
- The hyper-prior $(\mu_\beta, \sigma_\beta)$ is learned from data alongside the per-item $\beta_i$. The likelihood is item-specific: each item's model components are compared against its own sales.

## Non-centered parameterisation

The naive form $\beta_i \sim \text{Normal}(\mu, \sigma)$ creates a funnel in the joint posterior of $(\sigma, \beta_i)$: when $\sigma$ is small, $\beta_i$ lives in a narrow neck around $\mu$, and HMC step sizes that work in the wide part of the funnel are far too large in the neck. The fix is to sample standard normals and rescale:

```python
z_intercept = pm.Normal("z_intercept", 0.0, 1.0, dims="item")
intercept = pm.Deterministic(
    "intercept",
    intercept_mu + intercept_sigma * z_intercept,
    dims="item",
)
```

## Sharing and shrinkage in practice

- $\sigma_\beta$ controls shrinkage: tight → all items behave alike; wide → items can vary.
- The models learns this via the posterior on $\sigma_\beta$, but the prior still plays a roll. If items genuinely differ a lot, $\sigma_\beta$ ends up wide and pooling is weak. If they're similar, $\sigma_\beta$ collapses and short-history items get strongly pulled toward $\mu_\beta$.


## Calibration $\rightarrow$ adding a node from an experiment

- We can add a "constraint" to model if we have extra knowledge about a variable in the form of calibration. The calibration is implemented as one extra observed node added to the PyMC model, for example:

```python
pm.Normal(
    f"extra_coeff_info",
    mu=mu_coeff, # the expected value of the variable
    sigma=sigma_coeff , # thow "certain" we are of the extra info
    observed=data, # what we observed the variable to be
)
```
PyMC treats this as just another likelihood term. 

- This info can still be shared over a hyper-prior if set up correctly.
