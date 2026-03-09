"""Marketing Mix Models using pymc-marketing."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
)
from pymc_marketing.mmm.multidimensional import MMM

from .base import BaseForecaster


class MediaMixModel(BaseForecaster):
    """
    Bayesian Marketing Mix Model (MMM) for single-product sales attribution.

    Models sales as::

        sales ≈ baseline
              + Σ saturation(adstock(spend_k))   for each channel k
              + seasonality                        (optional)
              + Σ γ_j · control_j                 (optional)
              + noise

    Each channel contribution applies an adstock transformation (carry-over
    effect) followed by a saturation curve (diminishing returns).

    Sensible defaults produce the simplest valid model:

    ========================  ========================================
    Component                 Default
    ========================  ========================================
    Adstock                   ``GeometricAdstock(l_max=8)`` — exponential decay
    Saturation                ``LogisticSaturation()`` — S-shaped curve
    Seasonality               Off
    Controls                  None
    Time-varying intercept    Off (static baseline)
    Time-varying media        Off (static media effectiveness)
    ========================  ========================================

    Set ``time_varying_intercept=True`` or ``time_varying_media=True`` to let
    those components drift over time via a Hilbert Space Gaussian Process.

    Parameters
    ----------
    data : pd.DataFrame
        Training data containing the date, channel spend, and target columns.
    target_col : str
        Name of the sales / dependent variable column.
    date_col : str
        Name of the date column (must be parseable as dates).
    channel_columns : list[str]
        Names of the marketing spend columns.
    control_columns : list[str], optional
        Additional non-media regressors (e.g. holiday flags, price index).
        These receive a linear treatment with no saturation.
    adstock : AdstockTransformation, optional
        Carry-over transformation applied to each channel.
        Defaults to ``GeometricAdstock(l_max=8)``.
    saturation : SaturationTransformation, optional
        Diminishing-returns curve applied after adstock.
        Defaults to ``LogisticSaturation()``.
    yearly_seasonality : int, optional
        Number of Fourier modes for yearly seasonality.
        ``None`` (default) disables the seasonal component entirely.
    time_varying_intercept : bool
        Allow the baseline to drift over time via a GP. Default ``False``.
    time_varying_media : bool
        Allow per-channel media effectiveness to drift over time via a GP.
        Default ``False``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        date_col: str,
        channel_columns: list[str],
        control_columns: list[str] | None = None,
        adstock=None,
        saturation=None,
        yearly_seasonality: int | None = None,
        time_varying_intercept: bool = False,
        time_varying_media: bool = False,
    ):
        self.data = data.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.channel_columns = channel_columns
        self.control_columns = control_columns

        self._mmm = MMM(
            target_column=self.target_col,
            date_column=date_col,
            channel_columns=channel_columns,
            control_columns=control_columns,
            adstock=adstock or GeometricAdstock(l_max=8),
            saturation=saturation or LogisticSaturation(),
            yearly_seasonality=yearly_seasonality,
            time_varying_intercept=time_varying_intercept,
            time_varying_media=time_varying_media,
        )

        self.idata: az.InferenceData | None = None
        self.predictions: xr.DataArray | None = None

    @property
    def _X(self) -> pd.DataFrame:
        """Feature matrix: date + channels + optional controls."""
        cols = [self.date_col] + self.channel_columns + (self.control_columns or [])
        return self.data[cols]

    @property
    def _y(self) -> pd.Series:
        return self.data[self.target_col]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        target: str = None,
        date_col: str = None,
        draws: int = 1000,
        chains: int = 4,
        **sampler_kwargs,
    ) -> az.InferenceData:
        """
        Sample the posterior of the MMM.

        Parameters
        ----------
        draws : int
            Posterior draws per chain (default 1000).
        chains : int
            Number of MCMC chains (default 4).
        **sampler_kwargs
            Forwarded to ``pm.sample`` (e.g. ``target_accept``,
            ``nuts_sampler="numpyro"``).

        Returns
        -------
        az.InferenceData
        """
        self.idata = self._mmm.fit(
            self._X,
            self._y,
            draws=draws,
            chains=chains,
            **sampler_kwargs,
        )
        return self.idata

    def forecast(
        self,
        df_future: pd.DataFrame,
    ) -> xr.DataArray:
        """
        Sample the posterior predictive for out-of-sample periods.

        Parameters
        ----------
        df_future : pd.DataFrame
            Must contain ``date_col``, all ``channel_columns``, and (if
            specified at construction) all ``control_columns``.

        Returns
        -------
        xr.DataArray
            Shape ``(date, sample)`` — combined chain × draw posterior
            predictive samples.
        """
        result = (
            self._mmm.sample_posterior_predictive(
                df_future, combined=True, extend_idata=False
            )
            * self._mmm.scalers._target.values
        )
        # az.extract returns a Dataset even for a single variable; unwrap to DataArray
        if isinstance(result, xr.Dataset):
            self.predictions = result[self._mmm.output_var]
        else:
            self.predictions = result
        return self.predictions

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_forecast(self) -> tuple:
        """
        Plot the out-of-sample posterior predictive with 94% uncertainty band.

        Requires ``forecast()`` to have been called first.

        Returns
        -------
        tuple
            ``(fig, ax)``
        """
        if self.predictions is None:
            raise RuntimeError("Call forecast() before plot_forecast().")

        dates = self.predictions.coords["date"].values
        mean = self.predictions.mean(dim="sample").values
        lower = self.predictions.quantile(0.03, dim="sample").values
        upper = self.predictions.quantile(0.97, dim="sample").values

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(dates, mean, color="tab:blue", lw=1.5, label="Predicted Mean")
        ax.fill_between(
            dates, lower, upper, color="tab:blue", alpha=0.25, label="94% HDI"
        )
        ax.set_title(f"{self.target_col} — Posterior Predictive Forecast")
        ax.legend()
        fig.tight_layout()

        return fig, ax

    def plot_components(self) -> tuple:
        """
        Plot individual model components from the fitted posterior.

        Produces one subplot per marketing channel (contribution over time),
        plus additional subplots for yearly seasonality and controls if those
        components are active in the model.

        Requires ``fit()`` to have been called first.

        Returns
        -------
        tuple
            ``(fig, axes)``
        """
        if self.idata is None:
            raise RuntimeError("Call fit() before plot_components().")

        posterior = self.idata.posterior
        dates = posterior.coords["date"].values

        # Channel contributions — prefer original-scale Deterministic
        contrib_key = (
            "channel_contribution_original_scale"
            if "channel_contribution_original_scale" in posterior.data_vars
            else "channel_contribution"
        )
        ch_contrib = posterior[contrib_key]

        # Additional time-series components that may be present
        extra: list[tuple[str, str]] = []
        if "yearly_seasonality_contribution" in posterior.data_vars:
            extra.append(("Yearly Seasonality", "yearly_seasonality_contribution"))
        if "control_contribution" in posterior.data_vars:
            extra.append(("Control Contribution", "control_contribution"))
        # Time-varying intercept produces a "date"-dimensioned intercept
        intercept = posterior["intercept_baseline"]
        if "date" in intercept.dims:
            extra.append(("Baseline (Time-varying)", "intercept"))

        n_plots = len(self.channel_columns) + len(extra)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        ax_idx = 0

        # --- Channel subplots ---
        for ch in self.channel_columns:
            ch_data = ch_contrib.sel(channel=ch)
            mean = ch_data.mean(dim=["chain", "draw"]).values
            lower = ch_data.quantile(0.03, dim=["chain", "draw"]).values
            upper = ch_data.quantile(0.97, dim=["chain", "draw"]).values

            axes[ax_idx].plot(dates, mean, color="tab:blue")
            axes[ax_idx].fill_between(dates, lower, upper, color="tab:blue", alpha=0.2)
            axes[ax_idx].set_title(f"{ch} — Channel Contribution")
            ax_idx += 1

        # --- Extra components ---
        for label, var in extra:
            da = posterior[var]
            # control_contribution has a "control" dim — sum across controls
            if "control" in da.dims:
                da = da.sum(dim="control")

            mean = da.mean(dim=["chain", "draw"]).values
            lower = da.quantile(0.03, dim=["chain", "draw"]).values
            upper = da.quantile(0.97, dim=["chain", "draw"]).values

            axes[ax_idx].plot(dates, mean, color="tab:orange")
            axes[ax_idx].fill_between(
                dates, lower, upper, color="tab:orange", alpha=0.2
            )
            axes[ax_idx].set_title(label)
            ax_idx += 1

        fig.tight_layout()
        return fig, np.asarray(axes)

    # ------------------------------------------------------------------
    # Demand distribution
    # ------------------------------------------------------------------

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Return the posterior distribution of total demand over a date window.

        Sums the forecast posterior predictive across every period in
        ``[start_date, end_date]``, yielding a full sample-level distribution
        of total demand for planning purposes.

        Requires ``forecast()`` to have been called first.

        Parameters
        ----------
        start_date : str
            Inclusive start of the window (any format accepted by pandas).
        end_date : str
            Inclusive end of the window (any format accepted by pandas).

        Returns
        -------
        xr.Dataset
            Dataset with a single ``"demand"`` variable of shape ``(sample,)``
            representing the total demand distribution over the window.
        """
        if self.predictions is None:
            raise RuntimeError("Call forecast() before get_demand_distribution().")

        window = self.predictions.sel(date=slice(start_date, end_date))

        if window.sizes["date"] == 0:
            raise ValueError(
                f"No forecast data found between {start_date!r} and {end_date!r}."
            )

        return window.sum(dim="date").to_dataset(name="demand")
