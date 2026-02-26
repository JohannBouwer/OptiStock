"""Module for bayes state space models. Focused on modelling parameters that vary with time"""

from arviz import InferenceData
from pymc_extras.statespace import structural as st
import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt

from typing import Optional


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
