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

    All values live in scaled [0, 1] space.
    """

    initial_state_cov: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 10},
        "Diagonal scale of the initial state covariance (P0_diag)",
    ))
    initial_state: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.5, "sigma": 0.5},
        "Initial state values (initial_*)",
    ))
    observation_noise: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.05},
        "Measurement noise (sigma_obs)",
    ))
    regression_beta: Prior = field(default_factory=lambda: Prior(
        "HalfNormal", {"sigma": 0.5},
        "Regression coefficient magnitudes (beta_*); assumes regressors scaled to ~O(1)",
    ))
    regression_innovation: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 50},
        "Innovation variance for time-varying regression coefs (sigma_beta_*)",
    ))
    process_noise: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 50},
        "Process noise for trend / level / slope (sigma_*)",
    ))
    seasonal_amplitude: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0.0, "sigma": 0.15},
        "Initial seasonal amplitudes (params_*)",
    ))
    seasonal_innovation: Prior = field(default_factory=lambda: Prior(
        "Gamma", {"alpha": 2, "beta": 50},
        "Process noise for seasonal amplitudes (sigma_seasonal)",
    ))
