"""Customer lifetime value models built on ``pymc-marketing``."""

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from pymc_marketing.clv import (
    GammaGammaModel,
    ParetoNBDModel,
)
from pymc_marketing.clv import (
    plot_frequency_recency_matrix as _plot_frequency_recency_matrix,
)
from pymc_marketing.clv import (
    plot_probability_alive_matrix as _plot_probability_alive_matrix,
)

from .base import BaseCLV
from .priors import GammaGammaPriors, ParetoNBDPriors


class ParetoNBD(BaseCLV):
    """
    Pareto/NBD model of transaction frequency in a non-contractual setting.

    Answers the two questions behind customer value when customers can lapse
    without telling you: *how many more purchases will this customer make?* and
    *are they still active at all?*

    Each customer carries a latent purchase rate and a latent dropout time.
    Purchases arrive as a Poisson process until the customer silently drops out,
    which happens at an exponentially distributed time. Both rates vary across
    the population via Gamma distributions — see :class:`ParetoNBDPriors`.

    Optionally accepts **covariates**: customer attributes that shift the purchase
    or dropout rate multiplicatively. This is the main reason to reach for
    Pareto/NBD over BG/NBD — an acquisition channel or a plan tier can be given
    its own effect on how often people buy and how quickly they churn.

    Parameters
    ----------
    data : pd.DataFrame
        Per-customer RFM summary with columns ``customer_id``, ``frequency``,
        ``recency``, and ``T``, plus any covariate columns. Build it from an
        order-line table with :func:`~pymc_marketing.clv.rfm_summary`, re-exported
        as ``optistock.clv.rfm_summary``.
    priors : ParetoNBDPriors, optional
        Prior configuration. Defaults to :class:`ParetoNBDPriors`.
    purchase_covariate_cols : list[str], optional
        Columns of ``data`` that shift the purchase rate.
    dropout_covariate_cols : list[str], optional
        Columns of ``data`` that shift the dropout rate.
    sampler_config : dict, optional
        Sampler defaults forwarded to ``pymc-marketing``.

    Examples
    --------
    >>> from optistock.clv import ParetoNBD, rfm_summary
    >>> rfm = rfm_summary(orders, "customer_id", "date", monetary_value_col="revenue")
    >>> model = ParetoNBD(rfm)
    >>> model.fit(method="demz")
    >>> model.expected_purchases(future_t=90).mean("chain").mean("draw")  # 90 days
    """

    def __init__(
        self,
        data: pd.DataFrame,
        priors: ParetoNBDPriors | None = None,
        purchase_covariate_cols: list[str] | None = None,
        dropout_covariate_cols: list[str] | None = None,
        sampler_config: dict | None = None,
    ):
        self.purchase_covariate_cols = purchase_covariate_cols or []
        self.dropout_covariate_cols = dropout_covariate_cols or []

        missing = [
            col
            for col in self.purchase_covariate_cols + self.dropout_covariate_cols
            if col not in data.columns
        ]
        if missing:
            raise ValueError(
                f"Covariate column(s) not found in data: {missing}. "
                f"Available columns: {list(data.columns)}"
            )

        super().__init__(
            data=data,
            priors=priors or ParetoNBDPriors(),
            sampler_config=sampler_config,
            purchase_covariate_cols=self.purchase_covariate_cols,
            dropout_covariate_cols=self.dropout_covariate_cols,
        )

    def _build(self) -> ParetoNBDModel:
        return ParetoNBDModel(
            data=self.data,
            model_config=self._model_config,
            sampler_config=self.sampler_config,
        )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def expected_purchases(
        self,
        data: pd.DataFrame | None = None,
        future_t: int | None = None,
    ) -> xr.DataArray:
        """
        Expected number of purchases over the next ``future_t`` time units.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Customers to predict for. Defaults to the training frame.
        future_t : int, optional
            Length of the forward window, in the same time unit as ``T``.

        Returns
        -------
        xr.DataArray
            Posterior samples, dimensioned ``(chain, draw, customer_id)``.
        """
        self._require_fit("expected_purchases")
        return self._model.expected_purchases(data=data, future_t=future_t)

    def expected_probability_alive(
        self, data: pd.DataFrame | None = None
    ) -> xr.DataArray:
        """
        Posterior probability that each customer has not yet dropped out.

        Requires :meth:`fit` to have been called first.
        """
        self._require_fit("expected_probability_alive")
        return self._model.expected_probability_alive(data=data)

    def expected_purchases_new_customer(
        self, data: pd.DataFrame | None = None, t: int | None = None
    ) -> xr.DataArray:
        """
        Expected purchases over ``t`` time units for a not-yet-observed customer.

        Uses the population-level posterior only, so it is the natural prediction
        for a customer with no history.
        """
        self._require_fit("expected_purchases_new_customer")
        return self._model.expected_purchases_new_customer(data=data, t=t)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _require_no_covariates(self, caller: str) -> None:
        """
        Raise if covariates are configured.

        The matrix plots sweep a synthetic frequency / recency grid, and there is
        no principled covariate value to attach to those made-up customers, so
        ``pymc-marketing`` rejects the call.
        """
        if self.purchase_covariate_cols or self.dropout_covariate_cols:
            raise NotImplementedError(
                f"{caller}() is not available on a model fitted with covariates: "
                "the plot sweeps a synthetic frequency/recency grid, which has no "
                "covariate values to condition on. Predict on real customers with "
                "expected_probability_alive()/expected_purchases() instead, or fit "
                "a second model without covariates for the diagnostic view."
            )

    def plot_probability_alive_matrix(self, **kwargs) -> tuple:
        """
        Heatmap of P(alive) across the frequency / recency plane.

        ``**kwargs`` are forwarded to
        :func:`pymc_marketing.clv.plot_probability_alive_matrix`.

        Not available on a model fitted with covariates — see
        :meth:`_require_no_covariates`.

        Returns
        -------
        tuple
            ``(fig, ax)``
        """
        self._require_fit("plot_probability_alive_matrix")
        self._require_no_covariates("plot_probability_alive_matrix")

        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_probability_alive_matrix(self._model, ax=ax, **kwargs)
        fig.tight_layout()
        return fig, ax

    def plot_frequency_recency_matrix(self, future_t: int = 1, **kwargs) -> tuple:
        """
        Heatmap of expected purchases across the frequency / recency plane.

        ``**kwargs`` are forwarded to
        :func:`pymc_marketing.clv.plot_frequency_recency_matrix`.

        Not available on a model fitted with covariates — see
        :meth:`_require_no_covariates`.

        Returns
        -------
        tuple
            ``(fig, ax)``
        """
        self._require_fit("plot_frequency_recency_matrix")
        self._require_no_covariates("plot_frequency_recency_matrix")

        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_frequency_recency_matrix(self._model, future_t=future_t, ax=ax, **kwargs)
        fig.tight_layout()
        return fig, ax


class GammaGammaSpend(BaseCLV):
    """
    Gamma-Gamma model of monetary value per transaction.

    Pairs with :class:`ParetoNBD`: that model says *how often* a customer buys,
    this one says *how much they spend* when they do. Together they give expected
    lifetime value via :meth:`expected_customer_lifetime_value`.

    The model assumes spend per transaction is independent of purchase frequency
    — worth sanity-checking on your data, since heavy buyers with systematically
    smaller baskets violate it.

    Parameters
    ----------
    data : pd.DataFrame
        Per-customer summary with columns ``customer_id``, ``frequency``, and
        ``monetary_value``, where ``monetary_value`` is the customer's **mean**
        spend per transaction (what :func:`~pymc_marketing.clv.rfm_summary`
        produces), not their total.

        Customers with ``frequency == 0`` carry no repeat-purchase information and
        must be filtered out first; passing them raises ``ValueError``.
    priors : GammaGammaPriors, optional
        Prior configuration. Defaults to :class:`GammaGammaPriors`.
    sampler_config : dict, optional
        Sampler defaults forwarded to ``pymc-marketing``.

    Examples
    --------
    >>> repeat = rfm[rfm["frequency"] > 0]
    >>> spend = GammaGammaSpend(repeat)
    >>> spend.fit(method="demz")
    >>> clv = spend.expected_customer_lifetime_value(transactions, future_t=12)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        priors: GammaGammaPriors | None = None,
        sampler_config: dict | None = None,
    ):
        if "frequency" in data.columns and (data["frequency"] <= 0).any():
            n_zero = int((data["frequency"] <= 0).sum())
            raise ValueError(
                f"{n_zero} customer(s) have frequency == 0. The Gamma-Gamma model "
                "is fitted on repeat purchases only — filter them out first, e.g. "
                "data[data['frequency'] > 0]."
            )

        super().__init__(
            data=data,
            priors=priors or GammaGammaPriors(),
            sampler_config=sampler_config,
        )

    def _build(self) -> GammaGammaModel:
        return GammaGammaModel(
            data=self.data,
            model_config=self._model_config,
            sampler_config=self.sampler_config,
        )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def expected_customer_spend(
        self, data: pd.DataFrame | None = None
    ) -> xr.DataArray:
        """
        Expected spend per transaction for each customer.

        Shrinks each customer's observed mean towards the population mean, by an
        amount that depends on how many transactions they have — the main reason
        to model spend rather than take a raw average.
        """
        self._require_fit("expected_customer_spend")
        return self._model.expected_customer_spend(
            data=self.data if data is None else data
        )

    def expected_new_customer_spend(self) -> xr.DataArray:
        """Expected spend per transaction for a customer with no history."""
        self._require_fit("expected_new_customer_spend")
        return self._model.expected_new_customer_spend()

    def expected_customer_lifetime_value(
        self,
        transaction_model: "ParetoNBD",
        data: pd.DataFrame | None = None,
        future_t: int = 12,
        discount_rate: float = 0.0,
        time_unit: str = "D",
    ) -> xr.DataArray:
        """
        Expected lifetime value per customer over the next ``future_t`` months.

        Combines this spend model with a fitted transaction model: expected
        purchases in each future period, valued at the customer's expected spend,
        discounted back to the present.

        Parameters
        ----------
        transaction_model : ParetoNBD
            A **fitted** transaction model. A raw ``pymc-marketing`` CLV model is
            also accepted.
        data : pd.DataFrame, optional
            Customers to value. Defaults to this model's training frame, which
            must then also carry the ``recency`` and ``T`` columns the transaction
            model needs.
        future_t : int
            Horizon **in months**, and always in months — *not* in ``time_unit``.
            Default 12 (one year).

            .. warning::
               This differs from :meth:`ParetoNBD.expected_purchases`, whose
               ``future_t`` is in the time unit of your data. Here, with daily
               data, ``future_t=90`` means 90 *months*, not 90 days. For one
               quarter, pass ``future_t=3``.
        discount_rate : float
            Discount rate applied **per month**, matching ``future_t``. Default
            0.0 (no discounting). Note that discounting makes the computation
            iterate once per month, so large horizons get slow.
        time_unit : str
            Time unit of the purchase history in ``data`` — ``"D"`` (default),
            ``"W"``, ``"M"``, or ``"H"``. This describes your input; it does not
            change the meaning of ``future_t``.

        Returns
        -------
        xr.DataArray
            Posterior samples of lifetime value per customer.
        """
        self._require_fit("expected_customer_lifetime_value")

        if isinstance(transaction_model, BaseCLV):
            transaction_model._require_fit("expected_customer_lifetime_value")
            underlying = transaction_model.model
        else:
            underlying = transaction_model

        return self._model.expected_customer_lifetime_value(
            transaction_model=underlying,
            data=self.data if data is None else data,
            future_t=future_t,
            discount_rate=discount_rate,
            time_unit=time_unit,
        )
