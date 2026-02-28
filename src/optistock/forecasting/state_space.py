"""Module for bayes state space models. Focused on modelling parameters that vary with time"""

from arviz import InferenceData
from pymc_extras.statespace import structural as st
import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt

from typing import Optional


class UnivariateSSM:
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

        self._seasonal_name: Optional[str] = None

        self.model = None
        self.ssm = None
        self.idata = None

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
        meas_sigma = f"sigma_{self._MEAS_ERROR_NAME}"
        registered: set[str] = set()

        for param, dims in self.ssm.param_dims.items():
            if param.startswith("data_"):
                continue  # exogenous data is registered separately

            dim_kwargs = {"dims": dims} if dims else {}

            if param == "P0":
                P0_diag = pm.Gamma("P0_diag", alpha=50, beta=1)  # noqa: F841
                pm.Deterministic(  # noqa: F841
                    "P0",
                    pt.eye(self.model.k_states) * P0_diag,
                    dims=dims,
                )
            elif param.startswith("initial_"):
                pm.Normal(param, **dim_kwargs)  # noqa: F841
            elif param == meas_sigma:
                pm.HalfNormal(param, sigma=5)  # noqa: F841
            elif param.startswith("sigma_beta_"):
                pm.Gamma(param, alpha=2, beta=1, **dim_kwargs)  # noqa: F841
            elif param.startswith("sigma_"):
                pm.Gamma(param, alpha=2, beta=1, **dim_kwargs)  # noqa: F841
            elif param.startswith("beta_"):
                pm.HalfNormal(param, sigma=3, **dim_kwargs)  # noqa: F841
            elif param.startswith("params_"):
                pm.Normal(param, sigma=10, **dim_kwargs)  # noqa: F841
            else:
                continue
            registered.add(param)

        # pymc-extras omits scalar sigma params from param_dims; register them here
        # if the loop above didn't already handle them.
        if meas_sigma not in registered:
            pm.HalfNormal(meas_sigma, sigma=5)  # noqa: F841

        seasonal_sigma = f"sigma_{self._seasonal_name}" if self._seasonal_name else None
        if (
            seasonal_sigma
            and seasonal_sigma not in registered
            and self._seasonal_innovations
        ):
            pm.Gamma(seasonal_sigma, alpha=2, beta=1)  # noqa: F841

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
        with pm.Model(coords=self.ssm.coords):
            self._register_priors()

            for col in self.exog:
                pm.Data(  # noqa: F841
                    f"data_{col}",
                    self.data[[col]].values.astype(float),
                    dims=("time", f"state_{col}"),
                )

            self.ssm.build_statespace_graph(self.y, mode="JAX")
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
        return self.ssm.forecast(
            self.idata,
            start=self.data.index[-1],
            periods=periods,
            scenario=scenario,
        )


class BayesStateSpace:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        time_varying_cols: Optional[list[str]] = None,
    ):
        self.data = data.copy()
        self.target = target_col
        self.y = self.data[[self.target]].values

        # Identify regressors (all columns except target)
        self.regressors = [c for c in self.data.columns if c != self.target]
        self.time_varying_cols = time_varying_cols or []

        self.model = None
        self.idata = None

    def build_model(
        self,
        trend_order: int = 2,
        innovations_order: int | list[int] = [0, 1],
        seasonal_periods=None,
    ) -> None:
        """
        Build the SSM model.
        """

        # trend and growth component
        self.model = st.LevelTrend(
            order=trend_order, innovations_order=innovations_order
        )

        # time-varying regressors
        if self.time_varying_cols:

            self.model += st.Regression(
                name="time_varying_exog",
                state_names=self.time_varying_cols,
                innovations=True,
            )

        # time-invariant regressors
        self.model += st.Regression(
            name="constant_exog",
            state_names=[c for c in self.regressors if c not in self.time_varying_cols],
        )

        # noise / measurement error
        self.model += st.MeasurementError()

        # seasonal component
        if seasonal_periods is not None:
            self.model += st.FrequencySeasonality(
                name="season", season_length=seasonal_periods
            )

        self.ssm = self.model.build(name="SSM", mode="JAX")

    def _get_model_coords(self):

        self.dims = self.model.param_dims.values()
        self.coords = self.model.coords

    def fit(self, sampler_kwargs) -> None:
        """Fit/Sample the model"""
        self._get_model_coords()
        with pm.Model(coords=self.ssm.coords):

            # SET PRIORS
            P0_diag = pm.Gamma("P0_diag", alpha=50, beta=1)
            P0 = pm.Deterministic(  # noqa: F841
                "P0", pt.eye(self.model.k_states) * P0_diag, dims=("state", "state_aux")
            )
            # Trend and level
            initial_level_trend = pm.Normal(  # noqa: F841
                "initial_level_trend", dims=("state_level_trend",)
            )
            # Trend and level innovations
            sigma_level_trend = pm.Gamma(  # noqa: F841
                "sigma_level_trend", alpha=2, beta=1, dims=("shock_level_trend",)
            )
            # Time varying regressors data
            data_time_varying_exog = pm.Data(  # noqa: F841
                "data_time_varying_exog",
                self.data[self.regressors].values,
                dims=("time", "state_time_varying_exog"),
            )
            # Time varying regressors coefficients
            beta_time_varying_exog = pm.HalfNormal(  # noqa: F841
                "beta_time_varying_exog",
                sigma=3,
                dims=("state_time_varying_exog",),
            )
            # Time varying innovations
            sigma_beta_time_varying_exog = pm.HalfNormal(  # noqa: F841
                "sigma_beta_time_varying_exog",
                sigma=3,
                dims=("state_time_varying_exog",),
            )
            # Time invariant regressors data
            data_constant_exog = pm.Data(  # noqa: F841
                "data_constant_exog",
                self.data[
                    [c for c in self.regressors if c not in self.time_varying_cols]
                ].values,
                dims=("time", "state_constant_exog"),
            )
            # Time invariant regressors coefficients
            beta_constant_exog = pm.HalfNormal(  # noqa: F841
                "beta_constant_exog",
                sigma=3,
                dims=("state_constant_exog",),
            )
            # Seasonality
            params_season = pm.Normal(  # noqa: F841
                "params_season", sigma=10, dims=("state_season",)
            )
            sigma_season = pm.Gamma("sigma_season", alpha=2, beta=1)  # noqa: F841

            sigma_MeasurementError = pm.HalfNormal(  # noqa: F841
                "sigma_MeasurementError", sigma=5
            )  # noqa: F841

            # BUILD MODEL GRAPH
            self.ssm.build_statespace_graph(self.y, mode="JAX")

            # SAMPLE MODEL
            self.idata = pm.sample(nuts_sampler="nutpie")

    def smooth_and_filter(self, method: str = "eigh") -> None:
        """Smooth output and extract components"""
        self.post_idata = self.ssm.sample_conditional_posterior(
            self.idata, mvn_method=method
        )

        self.component_idata = self.ssm.extract_components_from_idata(self.post_idata)

    def forecast(self, senarios: None) -> InferenceData:
        """forecast with the model"""
        forecast = self.ssm.forecast(
            self.idata, start=self.data.index[-1], periods=10, senario=senarios
        )

        return forecast
