"""Module for bayes state space models. Focused on modelling parameters that vary with time"""

import arviz as az
from arviz import InferenceData
from pymc_extras.statespace import structural as st
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr

from typing import Optional

from .base import BaseForecaster


class UnivariateSSM(BaseForecaster):
    """
    Flexible univariate Bayesian structural time series model via state space.

    Composes a structural SSM from configurable components:
      - LevelTrend  : local level or local linear trend with optional stochastic drift
      - Regression  : one component per exogenous variable, each with optional innovations
      - FrequencySeasonality : trigonometric seasonality with optional innovations
      - MeasurementError : observation noise

    Priors are registered dynamically at fit-time by inspecting the built model's
    ``param_dims``, so the class handles any combination of active components without
    hardcoded branching.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the target column and any exogenous columns.
    target_col : str
        Name of the dependent variable column.
    exog : dict[str, bool], optional
        Mapping of exogenous column name → whether that regressor has stochastic
        innovations (time-varying coefficient).
        Example: ``{"spend": True, "event": False}``
    """

    _MEAS_ERROR_NAME = "obs"

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        exog: Optional[dict[str, bool]] = None,
    ):
        self.data = data.copy()
        self.target = target_col
        self.y = self.data[self.target]
        self.exog = exog or {}
        self.max_scaler: float = 1.0  # set in fit()

        self._seasonal_name: Optional[str] = None

        self.model = None
        self.ssm = None
        self.idata = None
        self.post_idata = None
        self.component_idata = None

    def build_model(
        self,
        trend_order: int = 2,
        trend_innovations_order: int | list[int] = [0, 1],
        seasonal_period: Optional[int] = None,
        seasonal_harmonics: Optional[int] = None,
        seasonal_innovations: bool = True,
    ) -> None:
        """
        Compose and build the structural state space model.

        Parameters
        ----------
        trend_order : int
            Polynomial order of the trend component.
            1 = local level, 2 = local linear trend (level + slope).
        trend_innovations_order : int or list[int]
            Which trend states receive stochastic innovations.
            ``[0, 1]`` lets both level and slope drift; ``[1]`` fixes the level
            but allows the slope to drift (smooth trend); ``0`` removes all drift.
        seasonal_period : int, optional
            Number of time steps per seasonal cycle (e.g. 7 for weekly).
            Omit to exclude a seasonal component.
        seasonal_harmonics : int, optional
            Number of Fourier harmonics. Defaults to ``seasonal_period // 2``.
        seasonal_innovations : bool
            Whether the seasonal amplitudes themselves can drift over time.
        """
        self.model = st.LevelTrend(
            order=trend_order,
            innovations_order=trend_innovations_order,
        )

        for col, innovations in self.exog.items():
            self.model += st.Regression(
                name=col,
                state_names=[col],
                innovations=innovations,
            )

        self._seasonal_innovations = False
        if seasonal_period is not None:
            n_harmonics = seasonal_harmonics or seasonal_period // 2
            self._seasonal_name = "seasonal"
            self._seasonal_innovations = seasonal_innovations
            self.model += st.FrequencySeasonality(
                name=self._seasonal_name,
                season_length=seasonal_period,
                n=n_harmonics,
                innovations=seasonal_innovations,
            )

        self.model += st.MeasurementError(name=self._MEAS_ERROR_NAME)

        self.ssm = self.model.build(name=self.target, mode="JAX")

    def _register_priors(self) -> None:
        """
        Dynamically register PyMC priors for every parameter declared by the
        built SSM.  Prior families follow the naming conventions used by
        pymc-extras structural models:

        ================  =====================================================
        Prefix / name     Prior
        ================  =====================================================
        ``P0``            Diagonal Gamma (scale of initial covariance)
        ``initial_*``     Normal (initial state distribution)
        ``sigma_obs``     HalfNormal (measurement noise — typically small)
        ``sigma_beta_*``  Gamma (innovation variance for regression coefs)
        ``sigma_*``       Gamma (process noise — trend, seasonal)
        ``beta_*``        HalfNormal (regression coefficient magnitude)
        ``params_*``      Normal (initial seasonal amplitudes)
        ``data_*``        Skipped here; registered separately in ``fit``
        ================  =====================================================

        Note: pymc-extras does not include scalar (undimensioned) sigma parameters
        in ``param_dims``.  Those are registered explicitly after the loop.
        """
        # All priors are specified in the scaled [0, 1] space.
        meas_sigma = f"sigma_{self._MEAS_ERROR_NAME}"
        registered: set[str] = set()

        for param, dims in self.ssm.param_dims.items():
            if param.startswith("data_"):
                continue  # exogenous data is registered separately

            dim_kwargs = {"dims": dims} if dims else {}

            if param == "P0":
                # Initial state covariance — small in [0, 1] space
                P0_diag = pm.Gamma("P0_diag", alpha=2, beta=10)  # noqa: F841
                pm.Deterministic(  # noqa: F841
                    "P0",
                    pt.eye(self.model.k_states) * P0_diag,
                    dims=dims,
                )
            elif param.startswith("initial_"):
                # Initial states are in scaled [0, 1] units
                pm.Normal(param, mu=0.5, sigma=1.0, **dim_kwargs)  # noqa: F841
            elif param == meas_sigma:
                # Measurement noise — tight prior for [0, 1] scaled data
                pm.HalfNormal(param, sigma=0.05)  # noqa: F841
            elif param.startswith("sigma_beta_"):
                # Innovation variance for time-varying regression coefs
                pm.Gamma(param, alpha=2, beta=50, **dim_kwargs)  # noqa: F841
            elif param.startswith("sigma_"):
                # Process noise (level, slope, seasonal) in [0, 1] space
                pm.Gamma(param, alpha=2, beta=50, **dim_kwargs)  # noqa: F841
            elif param.startswith("beta_"):
                # Regression coefficients: exog scale unknown, keep relatively wide
                pm.HalfNormal(param, sigma=3, **dim_kwargs)  # noqa: F841
            elif param.startswith("params_"):
                # Initial seasonal amplitudes in [0, 1] space
                pm.Normal(param, sigma=0.5, **dim_kwargs)  # noqa: F841
            else:
                continue
            registered.add(param)

        # pymc-extras omits scalar sigma params from param_dims; register them here
        # if the loop above didn't already handle them.
        if meas_sigma not in registered:
            pm.HalfNormal(meas_sigma, sigma=0.05)  # noqa: F841

        seasonal_sigma = f"sigma_{self._seasonal_name}" if self._seasonal_name else None
        if (
            seasonal_sigma
            and seasonal_sigma not in registered
            and self._seasonal_innovations
        ):
            pm.Gamma(seasonal_sigma, alpha=2, beta=50)  # noqa: F841

    def fit(self, sampler: str = "nutpie", **sampler_kwargs) -> None:
        """
        Register priors, build the statespace graph, and draw posterior samples.

        Parameters
        ----------
        sampler : str
            NUTS sampler backend passed to ``pm.sample``.
            ``"nutpie"`` (default) gives fast CPU performance.
        **sampler_kwargs
            Additional keyword arguments forwarded to ``pm.sample``.
        """
        # Scale target to [0, 1] for better prior specification and MCMC sampling.
        # All posterior quantities are in scaled space; un-scaling happens in the
        # plot methods and get_demand_distribution.
        self.max_scaler = float(self.y.max())
        y_scaled = self.y / self.max_scaler

        with pm.Model(coords=self.ssm.coords):
            self._register_priors()

            for col in self.exog:
                pm.Data(  # noqa: F841
                    f"data_{col}",
                    self.data[[col]].values.astype(float),
                    dims=("time", f"state_{col}"),
                )

            self.ssm.build_statespace_graph(y_scaled, mode="JAX")
            self.idata = pm.sample(nuts_sampler=sampler, **sampler_kwargs)

    def smooth_and_filter(self, method: str = "cholesky") -> None:
        """
        Run the Kalman smoother on the posterior and decompose into components.

        Results are stored as ``self.post_idata`` and ``self.component_idata``.

        Parameters
        ----------
        method : str
            Matrix decomposition method for ``sample_conditional_posterior``.
            ``"cholesky"`` (default) or ``"eigh"``.
        """
        self.post_idata = self.ssm.sample_conditional_posterior(
            self.idata, mvn_method=method
        )
        self.component_idata = self.ssm.extract_components_from_idata(self.post_idata)

    def forecast(
        self,
        periods: int,
        scenario: Optional[dict] = None,
    ) -> InferenceData:
        """
        Generate an out-of-sample forecast.

        Parameters
        ----------
        periods : int
            Number of steps ahead to forecast.
        scenario : dict, optional
            Exogenous variable values for the forecast horizon.
            Keys must match the ``pm.Data`` names used during fitting
            (i.e. ``"data_{col}"``).
            Example: ``{"data_spend": df_future[["spend"]].values}``

        Returns
        -------
        InferenceData
            ArviZ InferenceData object with ``forecast_observed`` variable.
        """

        self.forecast_idata = self.ssm.forecast(
            self.idata,
            start=self.data.index[-1],
            periods=periods,
            scenario=scenario,
        )
        return self.forecast_idata

    def plot_fit(self) -> tuple:
        """
        Plot the in-sample smoothed posterior against the observed training data.

        Requires ``smooth_and_filter()`` to have been called first.

        Returns
        -------
        tuple
            ``(fig, ax)`` — Matplotlib figure and axes objects.
        """
        if self.post_idata is None:
            self.smooth_and_filter()

        obs = self.post_idata.smoothed_posterior_observed.isel(observed_state=0)
        obs_stacked = obs.stack(sample=["chain", "draw"]) * self.max_scaler

        mean = obs_stacked.mean(dim="sample").values
        lower_95 = obs_stacked.quantile(0.025, dim="sample").values
        upper_95 = obs_stacked.quantile(0.975, dim="sample").values
        lower_50 = obs_stacked.quantile(0.25, dim="sample").values
        upper_50 = obs_stacked.quantile(0.75, dim="sample").values

        x = self.data.index

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, self.data[self.target].to_numpy(), color="k", lw=1, label="Observed")
        ax.plot(x, mean, color="tab:blue", label="Smoothed Mean")
        ax.fill_between(
            x, lower_95, upper_95, color="tab:blue", alpha=0.15, label="95% HDI"
        )
        ax.fill_between(
            x, lower_50, upper_50, color="tab:blue", alpha=0.35, label="50% HDI"
        )
        ax.set_title(f"{self.target} — In-sample Smoothed Posterior")
        ax.set_ylabel(self.target)
        ax.legend()
        fig.tight_layout()

        return fig, ax

    def plot_forecast(self) -> tuple:
        """
        Plot the forecasted smoothed posterior.

        Requires ``forecast()`` to have been called first.

        Returns
        -------
        tuple
            ``(fig, ax)`` — Matplotlib figure and axes objects.
        """
        if self.forecast_idata is None:
            raise ValueError("No forecast found. Please call `forecast()` first.")

        obs = self.forecast_idata["forecast_observed"].isel(observed_state=0)
        obs_stacked = obs.stack(sample=["chain", "draw"]) * self.max_scaler

        mean = obs_stacked.mean(dim="sample").values
        lower_95 = obs_stacked.quantile(0.025, dim="sample").values
        upper_95 = obs_stacked.quantile(0.975, dim="sample").values
        lower_50 = obs_stacked.quantile(0.25, dim="sample").values
        upper_50 = obs_stacked.quantile(0.75, dim="sample").values

        x = self.forecast_idata["time"].values

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, mean, color="tab:blue", label="Forecast Mean")
        ax.fill_between(
            x, lower_95, upper_95, color="tab:blue", alpha=0.15, label="95% HDI"
        )
        ax.fill_between(
            x, lower_50, upper_50, color="tab:blue", alpha=0.35, label="50% HDI"
        )
        ax.set_title(f"{self.target} — Out-of-sample Forecast")
        ax.set_ylabel(self.target)
        ax.legend()
        fig.tight_layout()

        return fig, ax

    def plot_components(self) -> tuple:
        """
        Plot each latent state component from the Kalman smoother.

        One subplot per state (e.g. level, slope, seasonal amplitudes, regression
        coefficients). Requires ``smooth_and_filter()`` to have been called first.

        Returns
        -------
        tuple
            ``(fig, axes)`` — Matplotlib figure and array of axes.
        """
        if self.component_idata is None:
            self.smooth_and_filter()

        states = self.component_idata.coords["state"].values
        component_hdi = az.hdi(self.component_idata, hdi_prob=0.95)
        x = self.data.index

        fig, axes = plt.subplots(len(states), 1, figsize=(14, 3 * len(states)))
        if len(states) == 1:
            axes = [axes]

        for ax, state in zip(axes, states):
            mean = (
                self.component_idata.stack(sample=["chain", "draw"])[
                    "smoothed_posterior"
                ]
                .sel(state=state)
                .mean(dim="sample")
                .values
                * self.max_scaler
            )
            hdi_vals = component_hdi.smoothed_posterior.sel(state=state).values * self.max_scaler

            ax.plot(x, mean, color="tab:orange")
            ax.fill_between(
                x, hdi_vals[:, 0], hdi_vals[:, 1], color="tab:orange", alpha=0.2
            )
            ax.set_title(state.replace("_", " ").title())
            ax.set_ylabel(self.target)

        fig.tight_layout()
        return fig, np.asarray(axes)

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Return the posterior distribution of total demand over a date window.

        Sums the smoothed posterior observed across all time steps in
        ``[start_date, end_date]``, yielding a full (chain × draw) distribution
        of total demand for that period.

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
            Dataset with a single ``"demand"`` variable of shape
            ``(chain, draw)`` representing the total demand distribution.
        """
        if self.forecast_idata is None:
            raise ValueError("No forecast found. Please call `forecast()` first.")

        time_values = self.forecast_idata["time"].values
        mask = (time_values >= np.datetime64(start_date)) & (
            time_values <= np.datetime64(end_date)
        )
        time_indices = np.where(mask)[0]

        total = (
            self.forecast_idata["forecast_observed"]
            .isel(observed_state=0, time=time_indices)
            .sum(dim="time")
            * self.max_scaler
        )

        return total.to_dataset(name="demand")


# TODO: Multivariate model with shared components (e.g. common trend) and cross-series regression effects


class MultivariateSSM(BaseForecaster):
    """
    Placeholder for future multivariate SSM implementation.
    """

    def fit(self, target: str, date_col: str) -> InferenceData:
        pass

    def forecast(self, df_future: pd.DataFrame) -> InferenceData:
        pass

    def plot_forecast(self) -> tuple:
        pass

    def plot_components(self) -> tuple:
        pass

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        pass
