"""
Configurable priors for the causal estimators.

Mirrors the pattern in :mod:`optistock.forecasting.priors`: users instantiate
the dataclass, override any ``Prior`` field, and pass it to the estimator.
The estimator translates these project ``Prior`` objects into the prior format
expected by the underlying inference library (CausalPy / ``pymc_extras``).
"""

from dataclasses import dataclass, field

from ..forecasting.priors import BasePriors, Prior


@dataclass
class SyntheticControlPriors(BasePriors):
    """
    Priors for :class:`SyntheticControl`.

    The underlying model is

    .. math::
        \\beta &\\sim \\mathrm{Dirichlet}(a, \\ldots, a) \\\\
        \\sigma &\\sim \\mathrm{HalfNormal}(\\sigma_0) \\\\
        \\mu &= X \\cdot \\beta \\\\
        y &\\sim \\mathrm{Normal}(\\mu, \\sigma)

    where :math:`X` are the donor unit series and :math:`y` is the treated
    series. The Dirichlet enforces non-negative donor weights that sum to one
    (convex combination); the scalar ``a`` is broadcast across donors so the
    prior stays uniform regardless of how many donors are supplied.
    """

    donor_weights: Prior = field(default_factory=lambda: Prior(
        "Dirichlet",
        {"a": 1.0},
        "Per-donor weight; scalar concentration broadcast across all donors",
    ))
    sigma: Prior = field(default_factory=lambda: Prior(
        "HalfNormal",
        {"sigma": 1.0},
        "Observation noise on the treated series",
    ))
