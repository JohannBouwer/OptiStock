"""
Configurable priors for the customer lifetime value models.

Mirrors the pattern in :mod:`optistock.forecasting.priors`: users instantiate the
dataclass, override any ``Prior`` field, and pass it to the model. The model
translates these project ``Prior`` objects into the ``model_config`` format
expected by ``pymc-marketing`` (see :meth:`optistock.clv.base.BaseCLV._model_config`).

Defaults reproduce ``pymc-marketing``'s own defaults exactly, so constructing a
model without a ``priors`` argument behaves identically to using the library
directly.

Example
-------
>>> from optistock.clv import ParetoNBD, ParetoNBDPriors
>>> from optistock.forecasting import Prior
>>> priors = ParetoNBDPriors(
...     alpha=Prior("Weibull", {"alpha": 3, "beta": 7}, "Purchase-rate scale"),
... )
>>> model = ParetoNBD(rfm, priors=priors)
>>> model.describe_priors()
"""

from dataclasses import dataclass, field

from ..forecasting.priors import BasePriors, Prior


@dataclass
class ParetoNBDPriors(BasePriors):
    """
    Priors for :class:`~optistock.clv.models.ParetoNBD`.

    The Pareto/NBD model gives each customer a latent purchase rate and a latent
    dropout rate, both drawn from population-level Gamma distributions:

    .. math::
        \\lambda_i &\\sim \\mathrm{Gamma}(r, \\alpha)  \\quad \\text{(purchase rate)} \\\\
        \\mu_i &\\sim \\mathrm{Gamma}(s, \\beta)   \\quad \\text{(dropout rate)}

    so ``r``/``s`` are the population *shapes* and ``alpha``/``beta`` the
    population *scales*. Weibull priors keep both strictly positive.

    When covariates are supplied, ``pymc-marketing`` reparameterises the two
    scales as ``alpha_scale``/``beta_scale`` and multiplies them by
    ``exp(X @ coefficient)``. The ``alpha`` and ``beta`` fields below still apply
    in that case — the same config keys are read either way — and the two
    ``*_coefficient`` fields become active.
    """

    r: Prior = field(default_factory=lambda: Prior(
        "Weibull", {"alpha": 2, "beta": 1},
        "Population shape of the customer purchase rate",
    ))
    alpha: Prior = field(default_factory=lambda: Prior(
        "Weibull", {"alpha": 2, "beta": 10},
        "Population scale of the customer purchase rate",
    ))
    s: Prior = field(default_factory=lambda: Prior(
        "Weibull", {"alpha": 2, "beta": 1},
        "Population shape of the customer dropout rate",
    ))
    beta: Prior = field(default_factory=lambda: Prior(
        "Weibull", {"alpha": 2, "beta": 10},
        "Population scale of the customer dropout rate",
    ))
    purchase_coefficient: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0, "sigma": 1},
        "Covariate effect on the purchase rate (only used with covariates)",
    ))
    dropout_coefficient: Prior = field(default_factory=lambda: Prior(
        "Normal", {"mu": 0, "sigma": 1},
        "Covariate effect on the dropout rate (only used with covariates)",
    ))


@dataclass
class GammaGammaPriors(BasePriors):
    """
    Priors for :class:`~optistock.clv.models.GammaGammaSpend`.

    Spend per transaction is Gamma distributed with a customer-specific scale
    that is itself Gamma distributed across the population:

    .. math::
        z_i &\\sim \\mathrm{Gamma}(p, \\nu_i) \\\\
        \\nu_i &\\sim \\mathrm{Gamma}(q, v)

    ``HalfFlat`` (improper, positive) defaults let the data determine all three
    shape parameters. They are weak by design; swap in proper priors if the fit
    struggles on a small or low-frequency customer base.
    """

    p: Prior = field(default_factory=lambda: Prior(
        "HalfFlat", {}, "Shape of the within-customer spend distribution",
    ))
    q: Prior = field(default_factory=lambda: Prior(
        "HalfFlat", {}, "Shape of the population spend-scale distribution",
    ))
    v: Prior = field(default_factory=lambda: Prior(
        "HalfFlat", {}, "Scale of the population spend-scale distribution",
    ))
