"""
Configurable priors for the forecasting models.

Each model has a dedicated ``*Priors`` dataclass whose fields default to the
values previously hard-coded in the model. Users can instantiate a priors
object, tweak any field (or swap distributions entirely), and pass it to the
model's constructor.

Example
-------
>>> from optistock.forecasting import BayesTimeSeries, BayesTimeSeriesPriors, Prior
>>> priors = BayesTimeSeriesPriors(
...     sigma=Prior("HalfNormal", {"sigma": 0.5}, "Observation noise"),
... )
>>> model = BayesTimeSeries(df, priors=priors)
>>> model.describe_priors()
"""

from dataclasses import dataclass, field, fields
from typing import Any

import pymc as pm


@dataclass(frozen=True)
class Prior:
    """
    Lightweight wrapper around a PyMC distribution specification.

    Parameters
    ----------
    distribution : str
        Name of the PyMC distribution class, e.g. ``"Normal"`` or ``"HalfNormal"``.
    params : dict
        Keyword arguments for the distribution (e.g. ``{"mu": 0.0, "sigma": 1.0}``).
    description : str
        Human-readable description shown by ``BasePriors.__repr__``.
    """

    distribution: str
    params: dict
    description: str = ""

    def build(self, name: str, **pymc_kwargs: Any) -> pm.Distribution:
        """
        Instantiate the underlying PyMC distribution inside the active model.

        ``pymc_kwargs`` covers per-call extras like ``dims=...`` or ``shape=...``
        that depend on the surrounding model context, not the prior itself.
        """
        dist_cls = getattr(pm, self.distribution)
        return dist_cls(name, **self.params, **pymc_kwargs)


@dataclass
class BasePriors:
    """
    Base class for model-specific prior configurations.

    Subclasses only declare ``Prior`` fields; introspection (``to_dict`` /
    ``__repr__``) works generically off ``dataclasses.fields``.
    """

    def to_dict(self) -> dict[str, dict]:
        """Return a JSON-serializable dict describing every prior."""
        return {
            f.name: {
                "distribution": getattr(self, f.name).distribution,
                "params": dict(getattr(self, f.name).params),
                "description": getattr(self, f.name).description,
            }
            for f in fields(self)
            if isinstance(getattr(self, f.name), Prior)
        }

    def __str__(self) -> str:
        rows = self.to_dict()
        if not rows:
            return f"{type(self).__name__}()"

        name_w = max(len("Variable"), max(len(n) for n in rows))
        dist_w = max(len("Distribution"), max(len(r["distribution"]) for r in rows.values()))
        param_strs = {
            n: ", ".join(f"{k}={v}" for k, v in r["params"].items())
            for n, r in rows.items()
        }
        param_w = max(len("Parameters"), max(len(p) for p in param_strs.values()))

        header = f"{'Variable':<{name_w}}  {'Distribution':<{dist_w}}  {'Parameters':<{param_w}}  Description"
        sep = "-" * len(header)
        lines = [f"{type(self).__name__}:", header, sep]
        for n, r in rows.items():
            lines.append(
                f"{n:<{name_w}}  {r['distribution']:<{dist_w}}  "
                f"{param_strs[n]:<{param_w}}  {r['description']}"
            )
        return "\n".join(lines)


@dataclass
class BayesTimeSeriesPriors(BasePriors):
    """Priors for :class:`BayesTimeSeries`. All values live in scaled [0, 1] space."""

    intercept: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 1.0}, "Baseline level of the series"
    ))
    growth: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 1.0}, "Linear trend slope"
    ))
    beta_event: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.5}, "Per-event additive effect"
    ))
    beta_fourier: Prior = field(default_factory=lambda: Prior(
        "Laplace", {"mu": 0.0, "b": 1.0}, "Fourier seasonality coefficients"
    ))
    sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.05}, "Observation noise"
    ))


@dataclass
class HierarchicalBayesTimeSeriesPriors(BasePriors):
    """
    Priors for :class:`HierarchicalBayesTimeSeries` — a multi-item version of
    :class:`BayesTimeSeries` where each item has its own coefficients drawn
    from a population-level (hyper) distribution.

    Naming convention: ``<coef>_mu`` and ``<coef>_sigma`` are the hyper-priors
    on the population mean and spread of ``<coef>``. The per-item ``<coef>[i]``
    is then ``Normal(<coef>_mu, <coef>_sigma)`` (non-centered internally).

    For vector-valued coefficients (``beta_fourier``, ``beta_event``) the
    hyper-priors are themselves vector-valued — one ``mu`` and ``sigma`` per
    Fourier feature or per event — so items are partially pooled separately
    for each feature/event.

    All values live in scaled [0, 1] space.
    """

    intercept_mu: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.5, "sigma": 0.5},
        "Population mean of per-item baseline level",
    ))
    intercept_sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.3},
        "Population spread of per-item baseline level",
    ))
    growth_mu: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.05},
        "Population mean of per-item trend slope",
    ))
    growth_sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.05},
        "Population spread of per-item trend slope",
    ))
    beta_fourier_mu: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.3},
        "Population mean of per-item Fourier coefficients (per feature)",
    ))
    beta_fourier_sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.3},
        "Population spread of per-item Fourier coefficients (per feature)",
    ))
    beta_event_mu: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.3},
        "Population mean of per-item event effect (per event)",
    ))
    beta_event_sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.3},
        "Population spread of per-item event effect (per event)",
    ))
    sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.05},
        "Shared observation noise (not pooled across items)",
    ))


@dataclass
class BARTBayesTimeSeriesPriors(BasePriors):
    """Priors for :class:`BARTBayesTimeSeries`. BART tree count ``m`` stays a constructor arg."""

    intercept: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 1.0}, "Baseline level of the series"
    ))
    growth: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 1.0}, "Linear trend slope"
    ))
    sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.1}, "Observation noise"
    ))


@dataclass
class HSGPBayesTimeSeriesPriors(BasePriors):
    """Priors for :class:`HSGPBayesTimeSeries`."""

    ell: Prior = field(default_factory=lambda: Prior(
        "InverseGamma", {"mu": 0.5, "sigma": 0.2}, "GP lengthscale"
    ))
    eta: Prior = field(default_factory=lambda: Prior(
        "Exponential", {"lam": 1.0}, "GP amplitude"
    ))
    intercept: Prior = field(default_factory=lambda: Prior(
        "Normal", {"sigma": 0.5},
        "Baseline level. If `mu` is not provided, fit() injects y_scaled.mean().",
    ))
    sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.1}, "Observation noise"
    ))


@dataclass
class UnivariateSSMPriors(BasePriors):
    """
    Priors for :class:`UnivariateSSM`, grouped by **family** rather than per
    individual parameter (matches the dynamic structure of ``_register_priors``).

    All values live in **standardised space**: the model fits
    ``(y_work - center) / scale``, where ``y_work`` is the target (optionally
    ``log1p``-transformed). One unit therefore means *one standard deviation of
    the training data*, and the data has mean 0 — so these numbers carry over
    unchanged between raw and ``log_transform`` mode, and between datasets.

    Note on ``process_noise``: it is calibrated for the **default** trend mask
    ``innovations_order=[1, 0]`` — the innovation lands on the *level*, giving a
    random walk with constant drift whose forecast sd grows like
    ``sigma * sqrt(h)``. On the bakery holdout this beats a drifting slope by
    3-4 SMAPE points at horizons of 1-2 weeks, which is the range that matches a
    ``lead_time + review_period`` planning window.

    If you switch to ``[0, 1]`` (innovation on the *slope*), the level becomes an
    *integrated* random walk and its forecast sd grows like
    ``sigma * sqrt(h**3 / 3)`` — cubic in the horizon, roughly **17x** wider at
    ``h=30``. The same ``process_noise`` is then far too loose; tighten it by
    about an order of magnitude (``beta`` around 100 rather than 25).

    At long horizons (``h`` approaching 30) the two masks tie on SMAPE, but the
    level random walk's wider predictive variance inflates the back-transformed
    **mean** under ``log_transform`` (``E[y] = exp(mu + sigma**2/2)``). The mean
    is still the right quantity for expected demand and the newsvendor; if you
    want a point forecast at long range, prefer the posterior median.
    """

    initial_state_cov: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 100},
        "Diagonal scale of the initial state covariance (P0_diag). One scalar "
        "covers every state including the slope, so it is kept tight.",
    ))
    initial_level: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 1.0},
        "Initial level state. Zero-centred because the data is centred.",
    ))
    initial_slope: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.02},
        "Initial slope (and any higher trend state). Near zero: this is a "
        "per-step drift that integrates into the level.",
    ))
    observation_noise: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.5},
        "Measurement noise (sigma_obs), in data standard deviations",
    ))
    regression_beta: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.5},
        "Regression coefficients (beta_*); assumes regressors scaled to ~O(1). "
        "Zero-centred so an effect may be negative.",
    ))
    regression_innovation: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 50},
        "Innovation sd for time-varying regression coefs (sigma_beta_*)",
    ))
    process_noise: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 25},
        "Process noise for the trend states (sigma_*). Mean 0.08, calibrated for "
        "the default level-innovation mask on the bakery holdout; only weakly "
        "informative in practice, since the posterior lands near 0.13-0.22 for "
        "beta anywhere in 15-100. See the class note before changing the mask.",
    ))
    seasonal_amplitude: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.3},
        "Initial seasonal amplitudes (params_*)",
    ))
    seasonal_innovation: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 50},
        "Process noise for seasonal amplitudes (sigma_seasonal)",
    ))


@dataclass
class HierarchicalSSMPriors(BasePriors):
    """
    Priors for :class:`HierarchicalSSM` — a multi-item state-space model where
    the **seasonal amplitudes** and **regression (covariate) coefficients** are
    partially pooled across items, while every other parameter stays independent
    per item.

    Two naming conventions coexist, matching the two kinds of family:

    * **Pooled families** (``seasonal_amplitude``, ``regression_beta``) declare
      ``<family>_mu`` and ``<family>_sigma`` hyper-priors. Each per-item
      coefficient is ``Normal(<family>_mu, <family>_sigma)`` built non-centred
      (``mu + sigma * z``, ``z ~ Normal(0, 1)``). The hyper-priors are vector
      valued over the non-item dimension — one ``mu``/``sigma`` per seasonal
      harmonic or per regressor — so items pool separately for each.
    * **Independent families** (everything else) keep the single-``Prior`` form
      of :class:`UnivariateSSMPriors`; each item gets its own draw with no
      shrinkage.

    All values live in **standardised space**, exactly as in
    :class:`UnivariateSSMPriors`. For the panel this means a **per-item centre**
    and a single **global scale**: in log space a seasonal or promotional effect
    is a scale-free multiplicative offset (a 20% Friday uplift is 0.18 log units
    for a 30-unit item and a 700-unit item alike), so a common divisor is what
    keeps the pooled ``params_*`` and ``beta_*`` coefficients comparable across
    items. Centring per item is what lets one ``initial_level`` prior serve every
    item regardless of its size.
    """

    # --- Independent families (per item, no pooling) ---
    initial_state_cov: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 100},
        "Diagonal scale of the initial state covariance (P0_diag). A single "
        "scalar over the whole stacked state — crude for many items.",
    ))
    initial_level: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 1.0},
        "Per-item initial level state. Zero-centred because each item is centred.",
    ))
    initial_slope: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.02},
        "Per-item initial slope (and any higher trend state)",
    ))
    observation_noise: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.5},
        "Per-item measurement noise (sigma_obs)",
    ))
    process_noise: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 25},
        "Per-item process noise for the trend states (sigma_*). Kept in step "
        "with UnivariateSSMPriors.process_noise — see the trend-mask note there "
        "before changing either.",
    ))
    regression_innovation: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 50},
        "Per-item innovation sd for time-varying regression coefs (sigma_beta_*)",
    ))
    seasonal_innovation: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 50},
        "Per-item process noise for seasonal amplitudes (sigma_seasonal)",
    ))

    # --- Pooled families (partial pooling across items) ---
    seasonal_amplitude_mu: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.3},
        "Population mean of per-item seasonal amplitudes (per harmonic)",
    ))
    seasonal_amplitude_sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.3},
        "Population spread of per-item seasonal amplitudes (per harmonic)",
    ))
    regression_beta_mu: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.5},
        "Population mean of per-item regression coefficients (per regressor)",
    ))
    regression_beta_sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.5},
        "Population spread of per-item regression coefficients (per regressor)",
    ))
