"""Shared machinery for the customer lifetime value models."""

from abc import ABC, abstractmethod
from dataclasses import fields

import arviz as az
import pandas as pd
from pymc_extras.prior import Prior as _PyMCMarketingPrior

from ..forecasting.priors import BasePriors, Prior


class BaseCLV(ABC):
    """
    Abstract base class for the customer lifetime value models.

    CLV models have a different lifecycle to the forecasters: there is no future
    date range to project onto, no demand distribution, and the "prediction" is a
    per-customer quantity rather than a time series. What they do share is the
    prior-configuration machinery — a ``*Priors`` dataclass whose fields are
    project :class:`~optistock.forecasting.priors.Prior` objects.

    This class owns that translation. ``pymc-marketing`` configures its CLV models
    through a ``model_config`` dict of ``pymc_extras.prior.Prior`` objects, so
    every project ``Prior`` field on ``self.priors`` is converted across in
    :meth:`_model_config`. Subclasses only implement :meth:`_build`.

    Anything not wrapped explicitly stays reachable through :attr:`model`, which
    exposes the underlying ``pymc-marketing`` object and all of its built-in
    prediction and plotting tools.

    Parameters
    ----------
    data : pd.DataFrame
        Per-customer RFM summary. Required columns depend on the concrete model.
    priors : BasePriors
        Prior configuration for this model.
    sampler_config : dict, optional
        Sampler defaults forwarded to the underlying ``pymc-marketing`` model.
    **extra_config
        Non-prior ``model_config`` entries (e.g. covariate column lists) merged
        into the translated config.
    """

    priors: BasePriors

    def __init__(
        self,
        data: pd.DataFrame,
        priors: BasePriors,
        sampler_config: dict | None = None,
        **extra_config,
    ):
        self.data = data.copy()
        self.priors = priors
        self.sampler_config = sampler_config
        self._extra_config = extra_config
        self._model = self._build()
        self.idata: az.InferenceData | None = None

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def describe_priors(self) -> dict[str, dict]:
        """
        Print and return a structured description of every prior used by the model.

        Returns
        -------
        dict
            ``{prior_name: {"distribution": str, "params": dict, "description": str}}``
        """
        print(self.priors)
        return self.priors.to_dict()

    @property
    def _model_config(self) -> dict:
        """
        Translate the project ``Prior`` fields into ``pymc-marketing``'s format.

        ``pymc_extras.prior.Prior`` takes its distribution parameters as keyword
        arguments where the project ``Prior`` holds them in a ``params`` dict, so
        the conversion is a splat. Non-prior entries (covariate column lists) are
        merged in afterwards.
        """
        config = {}
        for f in fields(self.priors):
            value = getattr(self.priors, f.name)
            if isinstance(value, Prior):
                config[f.name] = _PyMCMarketingPrior(value.distribution, **value.params)

        config.update(self._extra_config)
        return config

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build(self):
        """Construct and return the underlying ``pymc-marketing`` model."""
        pass

    @property
    def model(self):
        """
        The underlying ``pymc-marketing`` model.

        Escape hatch for any built-in method this wrapper does not surface
        directly, e.g. ``model.distribution_new_customer(...)``.
        """
        return self._model

    def fit(self, method: str = "mcmc", **kwargs) -> az.InferenceData:
        """
        Fit the model.

        Parameters
        ----------
        method : str
            Inference method passed through to ``pymc-marketing``:

            - ``"mcmc"`` (default) — full NUTS. Slowest, and the one to trust.
            - ``"demz"`` — DEMetropolisZ. Roughly an order of magnitude faster
              than NUTS and accurate enough for iteration on these
              low-dimensional models, at the cost of looser convergence
              diagnostics.
            - ``"map"`` — maximum a posteriori. Fastest; a point estimate with
              no posterior spread, so use it for quick checks rather than for
              anything that reads an interval.

            ``"map"`` requires ``pymc-marketing>=0.19.4``; earlier releases
            raise ``TypeError`` against ``pymc>=5.28`` (see ``pyproject.toml``).
        **kwargs
            Forwarded to the underlying ``fit`` (e.g. ``draws``, ``chains``,
            ``target_accept``, ``nuts_sampler="numpyro"``).

        Returns
        -------
        az.InferenceData
        """
        self.idata = self._model.fit(method=method, **kwargs)
        return self.idata

    def summary(self, **kwargs):
        """
        Posterior summary table for the fitted parameters.

        Requires :meth:`fit` to have been called first.
        """
        self._require_fit("summary")
        return self._model.fit_summary(**kwargs)

    def _require_fit(self, caller: str) -> None:
        """Raise if the model has not been fitted yet."""
        if self.idata is None:
            raise RuntimeError(f"Call fit() before {caller}().")
