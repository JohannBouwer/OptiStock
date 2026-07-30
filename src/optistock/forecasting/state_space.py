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

import warnings
from typing import Optional

from .base import BaseForecaster
from .priors import HierarchicalSSMPriors, UnivariateSSMPriors


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
    log_transform : bool
        Fit the model on ``log1p(target)`` instead of the raw target. The
        Kalman-filter likelihood is Gaussian by construction, so this is the
        way to guarantee **non-negative predictions**: samples are mapped back
        with ``expm1`` by :meth:`inverse_transform`, which every plot method
        and :meth:`get_demand_distribution` route through. Components become
        multiplicative in the original scale. Default ``False``.

        Caveats worth knowing: ``log1p`` is not scale-free — the ``+1`` offset is
        negligible for an item selling 300/day but dominant for one selling
        2/day — and a Gaussian in ``log1p`` space is a poor approximation for
        genuinely intermittent, low-count items.
    """

    _MEAS_ERROR_NAME = "obs"

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        exog: Optional[dict[str, bool]] = None,
        priors: Optional[UnivariateSSMPriors] = None,
        log_transform: bool = False,
    ):
        self.data = data.copy()
        self.target = target_col
        self.y = self.data[self.target]
        self.exog = exog or {}
        self.priors = priors or UnivariateSSMPriors()
        self.log_transform = log_transform
        # Standardisation constants, set in fit(). `scale` is always a scalar;
        # `center` is a scalar here and a per-item vector in HierarchicalSSM.
        self.center: float = 0.0
        self.scale: float = 1.0

        self._seasonal_name: Optional[str] = None

        self.model = None
        self.ssm = None
        self.idata = None
        self.post_idata = None
        self.component_idata = None

    @staticmethod
    def _resolve_trend_innovations(
        trend_order: int, mask: Optional[int | list[int]]
    ) -> int | list[int]:
        """
        Fill in the default trend-innovations mask and validate its length.

        pymc-extras treats ``innovations_order`` as a boolean mask and indexes
        the trend states with it directly, so a wrong length surfaces as
        ``boolean index did not match indexed array`` from deep inside the build
        — which says nothing about what the caller got wrong. Catch it here.

        ``None`` means innovations on the **level only** (``[1, 0, ...]``): a
        random walk in the level with a fixed slope, whose forecast sd grows
        like ``sqrt(h)`` rather than the ``sqrt(h**3/3)`` of a drifting slope.
        """
        if mask is None:
            return [1] + [0] * (trend_order - 1)
        if isinstance(mask, (int, np.integer)) and not isinstance(mask, bool):
            return int(mask)

        mask = [int(v) for v in mask]
        default = [1] + [0] * (trend_order - 1)
        if len(mask) != trend_order:
            raise ValueError(
                f"trend_innovations_order is a boolean mask over the {trend_order} "
                f"trend state(s), so it needs exactly {trend_order} entries — got "
                f"{mask!r} ({len(mask)}). It is not a list of state indices. "
                f"For innovations on the level only pass {default!r}; "
                f"for a drifting slope pass {[0] + [1] * (trend_order - 1)!r}."
            )
        if any(v not in (0, 1) for v in mask):
            raise ValueError(
                f"trend_innovations_order is a boolean mask, so every entry must "
                f"be 0 or 1 — got {mask!r}."
            )
        return mask

    def build_model(
        self,
        trend_order: int = 2,
        trend_innovations_order: Optional[int | list[int]] = None,
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
        trend_innovations_order : int or list[int], optional
            Boolean **mask** (not a list of indices) over the trend states saying
            which of them receive stochastic innovations. With ``trend_order=2``
            the states are ``(level, slope)``, so:

            * ``[1, 0]`` (**default**) — the level drifts, the slope is fixed.
              A random walk with constant drift: forecast sd grows like
              ``sqrt(h)``. This is the conservative choice.
            * ``[0, 1]`` — the level is fixed and the slope drifts, making the
              level an *integrated* random walk whose forecast sd grows like
              ``sqrt(h**3 / 3)``, i.e. cubic in the horizon. Flexible, but it
              fans out fast; tighten ``priors.process_noise`` if you use it.
            * ``[1, 1]`` — both drift.  * ``0`` — no drift at all.

            Defaults to innovations on the level only. The mask must have exactly
            ``trend_order`` entries; a wrong length raises a clear error here
            rather than an opaque one inside pymc-extras.
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
            innovations_order=self._resolve_trend_innovations(
                trend_order, trend_innovations_order
            ),
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
        built SSM. Prior families follow the naming conventions used by
        pymc-extras structural models, and each family's distribution +
        parameters is sourced from ``self.priors``:

        ================  =====================================================
        Prefix / name     Priors family field
        ================  =====================================================
        ``P0``            ``initial_state_cov``
        ``initial_*``     ``initial_level`` + ``initial_slope`` (split)
        ``sigma_obs``     ``observation_noise``
        ``sigma_beta_*``  ``regression_innovation``
        ``sigma_*``       ``process_noise``
        ``beta_*``        ``regression_beta``
        ``params_*``      ``seasonal_amplitude``
        ``sigma_seasonal``  ``seasonal_innovation`` (scalar; injected post-loop)
        ``data_*``        Skipped here; registered separately in ``fit``
        ================  =====================================================

        Note: pymc-extras does not include scalar (undimensioned) sigma parameters
        in ``param_dims``. Those are registered explicitly after the loop.
        """
        meas_sigma = f"sigma_{self._MEAS_ERROR_NAME}"
        registered: set[str] = set()

        for param, dims in self.ssm.param_dims.items():
            if param.startswith("data_"):
                continue  # exogenous data is registered separately

            dim_kwargs = {"dims": dims} if dims else {}

            if param == "P0":
                P0_diag = self.priors.initial_state_cov.build("P0_diag")  # noqa: F841
                pm.Deterministic(  # noqa: F841
                    "P0",
                    pt.eye(self.model.k_states) * P0_diag,
                    dims=dims,
                )
            elif param.startswith("initial_"):
                self._build_initial_trend(param, dims)
            elif param == meas_sigma:
                self.priors.observation_noise.build(param)
            elif param.startswith("sigma_beta_"):
                self.priors.regression_innovation.build(param, **dim_kwargs)
            elif param.startswith("sigma_"):
                self.priors.process_noise.build(param, **dim_kwargs)
            elif param.startswith("beta_"):
                self.priors.regression_beta.build(param, **dim_kwargs)
            elif param.startswith("params_"):
                self.priors.seasonal_amplitude.build(param, **dim_kwargs)
            else:
                continue
            registered.add(param)

        # pymc-extras omits scalar sigma params from param_dims; register them here
        # if the loop above didn't already handle them.
        if meas_sigma not in registered:
            self.priors.observation_noise.build(meas_sigma)

        seasonal_sigma = f"sigma_{self._seasonal_name}" if self._seasonal_name else None
        if (
            seasonal_sigma
            and seasonal_sigma not in registered
            and self._seasonal_innovations
        ):
            self.priors.seasonal_innovation.build(seasonal_sigma)

    def _build_initial_trend(self, param: str, dims) -> None:
        """
        Build the initial trend state, with the level and slope given separate
        priors.

        ``initial_level_trend`` is a vector over ``state_level_trend``, whose
        entries are ``(level, slope, ...)`` — a single prior across all of them
        is wrong, because the level sits wherever the data sits while the slope
        is a *per-step drift* that has to start near zero. The two are assembled
        into the vector the SSM expects via ``pm.Deterministic``, following the
        same pattern as ``P0`` above.

        Handles the univariate case (``dims == ("state_level_trend",)``) and the
        panel case, where ``dims`` carries a leading item axis.
        """
        n_states = len(self.ssm.coords[dims[-1]])
        lead_shape = tuple(len(self.ssm.coords[d]) for d in dims[:-1])

        level = self.priors.initial_level.build(
            f"{param}_level", shape=lead_shape + (1,)
        )
        if n_states == 1:  # trend_order=1, local level only
            pm.Deterministic(param, level, dims=dims)
            return

        slope = self.priors.initial_slope.build(
            f"{param}_slope", shape=lead_shape + (n_states - 1,)
        )
        pm.Deterministic(param, pt.concatenate([level, slope], axis=-1), dims=dims)

    def _warn_unscaled_exog(self) -> None:
        """
        Warn when an exogenous column is far from O(1).

        ``priors.regression_beta`` is specified on the assumption that a one-unit
        move in the regressor is a meaningful change. Under ``log_transform`` a
        beta is a *log-multiplier*, so feeding in a raw spend column in currency
        units puts the coefficient many orders of magnitude away from its prior.
        """
        for col in self.exog:
            magnitude = float(np.nanmax(np.abs(self.data[col].to_numpy(dtype=float))))
            if magnitude > 0 and not (0.01 <= magnitude <= 100):
                warnings.warn(
                    f"Exogenous column {col!r} has max absolute value {magnitude:.4g}. "
                    "Priors on beta_* assume regressors are scaled to ~O(1); rescale "
                    "the column (or widen priors.regression_beta) or the coefficient "
                    "will be fighting its prior.",
                    UserWarning,
                    stacklevel=3,
                )

    def _standardise(self, y_work):
        """
        Compute ``center`` / ``scale`` and return the standardised target.

        Split out so :class:`HierarchicalSSM` can centre per item while keeping a
        single global scale. NaNs are ignored throughout, so the stockout
        censoring path (masking observations with ``NaN``) still works.
        """
        self.center = float(np.nanmean(y_work.to_numpy(dtype=float)))
        self.scale = float(np.nanstd(y_work.to_numpy(dtype=float)))
        if not self.scale > 0:
            raise ValueError(
                f"{self.target!r} has zero variance after transformation; "
                "cannot standardise a constant series."
            )
        return (y_work - self.center) / self.scale

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
        # Standardise the target (optionally in log1p space, see `log_transform`)
        # so that one unit means one standard deviation of the training data.
        # That is what makes the priors portable between raw and log mode and
        # across datasets. All posterior quantities live in this standardised
        # space; `inverse_transform` maps back to the original units.
        y_work = np.log1p(self.y) if self.log_transform else self.y
        y_scaled = self._standardise(y_work)
        self._warn_unscaled_exog()

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

    def _center_for(self, x):
        """
        The centring constant matching ``x``. Scalar here; overridden in
        :class:`HierarchicalSSM`, where it is selected per item.
        """
        return self.center

    def inverse_transform(self, x):
        """
        Map model-space values back to original units.

        Undoes the standardisation and, when ``log_transform=True``, the
        ``log1p`` transform. Works elementwise on numpy arrays and xarray
        objects, so it must be applied to *samples* (before any summing or
        averaging), not to already-aggregated quantities.

        No ``sigma**2 / 2`` lognormal bias correction is applied, and none is
        wanted: ``forecast_observed`` and ``smoothed_posterior_observed`` are
        already full posterior *predictive* draws (pymc-extras adds the
        measurement covariance ``H``), so averaging ``expm1`` over draws is a
        Monte-Carlo estimate of ``E[y]``. An analytic correction on top would
        double-count the observation noise.

        ``expm1`` maps the Gaussian model space to ``(-1, inf)``, so a deep
        negative tail sample can still come back as a fraction between -1 and
        0; those sub-unit remnants are clamped to 0 (sales are counts).
        """
        unscaled = x * self.scale + self._center_for(x)
        if not self.log_transform:
            return unscaled
        return np.maximum(np.expm1(unscaled), 0.0)

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
        obs_stacked = self.inverse_transform(obs.stack(sample=["chain", "draw"]))

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
        obs_stacked = self.inverse_transform(obs.stack(sample=["chain", "draw"]))

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

        Note: when ``log_transform=True`` the components are additive in
        ``log1p`` space, so they are plotted in that space (exponentiating an
        individual component in isolation is not meaningful).

        Only the level state is re-centred; every other component is a
        *deviation* around it and so takes the scale alone.

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
        units = f"log1p({self.target})" if self.log_transform else self.target

        fig, axes = plt.subplots(len(states), 1, figsize=(14, 3 * len(states)))
        if len(states) == 1:
            axes = [axes]

        for ax, state in zip(axes, states):
            offset = self._component_offset(state)
            mean = (
                self.component_idata.stack(sample=["chain", "draw"])[
                    "smoothed_posterior"
                ]
                .sel(state=state)
                .mean(dim="sample")
                .values
                * self.scale
                + offset
            )
            hdi_vals = (
                component_hdi.smoothed_posterior.sel(state=state).values * self.scale
                + offset
            )

            ax.plot(x, mean, color="tab:orange")
            ax.fill_between(
                x, hdi_vals[:, 0], hdi_vals[:, 1], color="tab:orange", alpha=0.2
            )
            ax.set_title(state.replace("_", " ").title())
            ax.set_ylabel(units)

        fig.tight_layout()
        return fig, np.asarray(axes)

    # Name `extract_components_from_idata` gives the level sub-component. The
    # univariate form is exactly this; the panel form is `_LEVEL_COMPONENT[<item>]]`.
    _LEVEL_COMPONENT = "level_trend[level]"

    def _component_offset(self, state: str) -> float:
        """
        Centring constant to add back when plotting a single latent state.

        Only the level carries the data's centre; slope, seasonal and regression
        states are deviations around it and get none.
        """
        return self.center if state == self._LEVEL_COMPONENT else 0.0

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

        daily = self.inverse_transform(
            self.forecast_idata["forecast_observed"].isel(
                observed_state=0, time=time_indices
            )
        )
        total = daily.sum(dim="time")

        return total.to_dataset(name="demand")


class HierarchicalSSM(UnivariateSSM):
    """
    Multi-item structural state-space forecaster with partial pooling.

    A panel version of :class:`UnivariateSSM`: all items are fitted jointly in a
    single multivariate structural model (``observed_state_names=items``). The
    **seasonal amplitudes** and **regression (covariate) coefficients** are
    partially pooled across items via hierarchical hyper-priors, so an item with
    a short or noisy history borrows its seasonal shape and covariate response
    from the population. Every other parameter (initial level, trend, and all
    noise variances) stays independent per item.

    Input is **wide-format**: a DataFrame indexed by date with one numeric column
    per item, plus any exogenous columns. Items are inferred as all non-exog
    columns if not supplied. Exogenous regressors are *shared* — the same
    ``x_t`` drives every item, but each item learns its own (pooled) coefficient.

    All per-item coefficients use a non-centred parameterisation to avoid funnel
    pathologies under HMC.

    Parameters
    ----------
    data : pd.DataFrame
        Wide panel: date index (or ``date`` column via the index), one column per
        item, plus any exogenous columns named in ``exog``.
    items : list[str], optional
        Item column names. Defaults to every column except those in ``exog``.
    exog : list[str], optional
        Exogenous regressor column names, shared across all items. Their
        coefficients are the pooled ``beta_*`` family.
    priors : HierarchicalSSMPriors, optional
        Prior configuration. Defaults to :class:`HierarchicalSSMPriors`.
    log_transform : bool
        Fit on ``log1p`` of each series for non-negative forecasts (see
        :class:`UnivariateSSM`). Default ``False``.
    target_name : str
        Label used to name the built state-space model. Default ``"sales"``.

    Notes
    -----
    Compute scales worse than fitting items independently — roughly 2x for three
    items and growing with item count, because the Kalman filter runs dense
    operations over the stacked latent state. Prefer this model when items are
    short/new or few; fit independently otherwise. Sampling the hierarchy needs
    ``target_accept >= 0.95`` (pass it through ``fit``). ``P0`` is a single
    scalar diagonal over the whole stacked state — crude for many items, adequate
    for a first pass.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        items: Optional[list[str]] = None,
        exog: Optional[list[str]] = None,
        priors: Optional[HierarchicalSSMPriors] = None,
        log_transform: bool = False,
        target_name: str = "sales",
    ):
        self.data = data.copy()
        exog = list(exog) if exog is not None else []
        if items is None:
            items = [c for c in self.data.columns if c not in exog]
        self.items = list(items)
        # match the parent's {col: innovations} contract; innovations off for v1
        self.exog = {c: False for c in exog}
        self.target = target_name
        self.y = self.data[self.items]  # wide frame (parent's self.y is a Series)
        self.priors = priors or HierarchicalSSMPriors()
        self.log_transform = log_transform
        # `center` is per item (an xr.DataArray over observed_state); `scale` is
        # a single global scalar. See `_standardise` for why.
        self.center: xr.DataArray | float = 0.0
        self.scale: float = 1.0

        self._seasonal_name: Optional[str] = None
        self._seasonal_innovations = False

        self.model = None
        self.ssm = None
        self.idata = None
        self.post_idata = None
        self.component_idata = None
        self.forecast_idata = None

    def build_model(
        self,
        trend_order: int = 2,
        trend_innovations_order: Optional[int | list[int]] = None,
        seasonal_period: Optional[int] = None,
        seasonal_harmonics: Optional[int] = None,
        seasonal_innovations: bool = False,
    ) -> None:
        """
        Compose the multivariate structural model.

        Same components as :class:`UnivariateSSM.build_model`, but every
        component carries ``observed_state_names=self.items`` so each item gets
        its own (block-diagonal) latent states. ``seasonal_innovations``
        defaults to ``False`` here: the pooled seasonal *shape* is static, which
        keeps the first implementation simple.
        """
        self.model = st.LevelTrend(
            order=trend_order,
            innovations_order=self._resolve_trend_innovations(
                trend_order, trend_innovations_order
            ),
            observed_state_names=self.items,
        )

        for col, innovations in self.exog.items():
            self.model += st.Regression(
                name=col,
                state_names=[col],
                observed_state_names=self.items,
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
                observed_state_names=self.items,
            )

        self.model += st.MeasurementError(
            name=self._MEAS_ERROR_NAME, observed_state_names=self.items
        )

        self.ssm = self.model.build(name=self.target, mode="JAX")

    def _build_pooled(self, param: str, dims, mu_prior, sigma_prior) -> None:
        """
        Non-centred hierarchical prior for a parameter pooled across items.

        The leading dimension of ``dims`` is the item (``endog_*``) axis. The
        hyper-priors ``<param>_mu`` / ``<param>_sigma`` live on the *remaining*
        dims (one per harmonic / regressor), and the standardised ``<param>_z``
        spans the full dims. ``param = mu + sigma * z``.
        """
        hyper_dims = tuple(dims[1:])
        hyper_kwargs = {"dims": hyper_dims} if hyper_dims else {}
        mu = mu_prior.build(f"{param}_mu", **hyper_kwargs)
        sigma = sigma_prior.build(f"{param}_sigma", **hyper_kwargs)
        z = pm.Normal(f"{param}_z", 0.0, 1.0, dims=dims)
        pm.Deterministic(param, mu + sigma * z, dims=dims)

    def _register_priors(self) -> None:
        """
        Register priors for every SSM parameter. Mirrors
        :meth:`UnivariateSSM._register_priors`, but the ``params_*`` (seasonal)
        and ``beta_*`` (regression) families are built hierarchically via
        :meth:`_build_pooled`; all other families stay independent per item.
        """
        meas_sigma = f"sigma_{self._MEAS_ERROR_NAME}"
        registered: set[str] = set()

        for param, dims in self.ssm.param_dims.items():
            if param.startswith("data_"):
                continue

            dim_kwargs = {"dims": dims} if dims else {}

            if param == "P0":
                P0_diag = self.priors.initial_state_cov.build("P0_diag")
                pm.Deterministic(
                    "P0", pt.eye(self.model.k_states) * P0_diag, dims=dims
                )
            elif param.startswith("initial_"):
                self._build_initial_trend(param, dims)
            elif param == meas_sigma:
                # Multivariate: sigma_obs is a vector (one per item), so it carries
                # an endog dim here — unlike the univariate scalar case.
                self.priors.observation_noise.build(param, **dim_kwargs)
            elif param.startswith("sigma_beta_"):
                self.priors.regression_innovation.build(param, **dim_kwargs)
            elif param.startswith("sigma_"):
                self.priors.process_noise.build(param, **dim_kwargs)
            elif param.startswith("beta_"):  # POOLED — covariate coefficients
                self._build_pooled(
                    param, dims,
                    self.priors.regression_beta_mu, self.priors.regression_beta_sigma,
                )
            elif param.startswith("params_"):  # POOLED — seasonal amplitudes
                self._build_pooled(
                    param, dims,
                    self.priors.seasonal_amplitude_mu, self.priors.seasonal_amplitude_sigma,
                )
            else:
                continue
            registered.add(param)

        if meas_sigma not in registered:
            self.priors.observation_noise.build(meas_sigma)

        seasonal_sigma = f"sigma_{self._seasonal_name}" if self._seasonal_name else None
        if (
            seasonal_sigma
            and seasonal_sigma not in registered
            and self._seasonal_innovations
        ):
            self.priors.seasonal_innovation.build(seasonal_sigma)

    def _standardise(self, y_work):
        """
        Centre **per item** but scale by a single **global** constant.

        The asymmetry is deliberate. In log space a seasonal or promotional
        effect is a scale-free multiplicative offset — a 20% Friday uplift is
        0.18 log units whether the item sells 30 or 700 a day — so dividing every
        item by the same constant is what keeps the *pooled* ``params_*`` and
        ``beta_*`` coefficients on a common footing. A per-item scale would put
        each item's coefficients in its own units and quietly break the pooling.

        Centring, by contrast, should be per item: the level state is per item
        and unpooled, so centring each series is what lets a single
        ``initial_level`` prior serve items of wildly different sizes.
        """
        values = y_work.to_numpy(dtype=float)
        self.center = xr.DataArray(
            np.nanmean(values, axis=0),
            dims=("observed_state",),
            coords={"observed_state": list(self.items)},
        )
        centered = y_work - self.center.to_series().reindex(y_work.columns).to_numpy()
        self.scale = float(np.nanstd(centered.to_numpy(dtype=float)))
        if not self.scale > 0:
            raise ValueError(
                "Panel has zero variance after transformation; cannot standardise."
            )
        return centered / self.scale

    def _center_for(self, x):
        """
        Select the per-item centre matching ``x``.

        ``x`` may carry ``observed_state`` as a full dimension (all items) or as
        a scalar coordinate (after ``.sel(observed_state=...)``); ``.sel`` handles
        both and keeps xarray from broadcasting the wrong item back in.
        """
        if isinstance(x, (xr.DataArray, xr.Dataset)) and "observed_state" in x.coords:
            return self.center.sel(observed_state=x.coords["observed_state"])
        return self.center

    def _component_offset(self, state: str):
        """
        Per-item centring for the level states, named ``level_trend[level[<item>]]``.
        """
        prefix = f"{self._LEVEL_COMPONENT[:-1]}["  # "level_trend[level["
        if state.startswith(prefix) and state.endswith("]]"):
            item = state[len(prefix) : -2]
            return float(self.center.sel(observed_state=item))
        return 0.0

    def fit(self, sampler: str = "nutpie", **sampler_kwargs) -> None:
        """
        Register priors, build the state-space graph, and sample. Identical to
        :meth:`UnivariateSSM.fit` except the target is a wide panel, so the
        standardisation centres each item separately while sharing one global
        scale (see :meth:`_standardise`).
        """
        y_work = np.log1p(self.y) if self.log_transform else self.y
        y_scaled = self._standardise(y_work)
        self._warn_unscaled_exog()

        with pm.Model(coords=self.ssm.coords):
            self._register_priors()

            for col in self.exog:
                pm.Data(
                    f"data_{col}",
                    self.data[[col]].values.astype(float),
                    dims=("time", f"state_{col}"),
                )

            self.ssm.build_statespace_graph(y_scaled, mode="JAX")
            self.idata = pm.sample(nuts_sampler=sampler, **sampler_kwargs)

    def get_demand_distribution(
        self, start_date: str, end_date: str, item: Optional[str] = None
    ) -> xr.Dataset:
        """
        Posterior total demand over ``[start_date, end_date]`` in original units.

        With ``item=None`` the result keeps the ``observed_state`` (item) dim;
        passing ``item=<name>`` returns the single-item distribution in the same
        shape as :class:`UnivariateSSM`, so it plugs straight into
        ``ForecastSolver``. Non-negativity is enforced per sample by
        ``inverse_transform`` before the time-sum.
        """
        if self.forecast_idata is None:
            raise ValueError("No forecast found. Please call `forecast()` first.")

        time_values = self.forecast_idata["time"].values
        mask = (time_values >= np.datetime64(start_date)) & (
            time_values <= np.datetime64(end_date)
        )
        time_indices = np.where(mask)[0]

        obs = self.forecast_idata["forecast_observed"].isel(time=time_indices)
        if item is not None:
            obs = obs.sel(observed_state=item)
        daily = self.inverse_transform(obs)
        total = daily.sum(dim="time")

        return total.to_dataset(name="demand")

    def plot_forecast(self, item: Optional[str] = None) -> tuple:
        """
        Plot the forecast for one item (or the first item if ``item=None``).

        Selects the named ``observed_state`` before delegating to the univariate
        plotting logic.
        """
        if self.forecast_idata is None:
            raise ValueError("No forecast found. Please call `forecast()` first.")

        sel = item if item is not None else self.items[0]
        obs = self.forecast_idata["forecast_observed"].sel(observed_state=sel)
        obs_stacked = self.inverse_transform(obs.stack(sample=["chain", "draw"]))

        mean = obs_stacked.mean(dim="sample").values
        lower_95 = obs_stacked.quantile(0.025, dim="sample").values
        upper_95 = obs_stacked.quantile(0.975, dim="sample").values
        lower_50 = obs_stacked.quantile(0.25, dim="sample").values
        upper_50 = obs_stacked.quantile(0.75, dim="sample").values

        x = self.forecast_idata["time"].values

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, mean, color="tab:blue", label="Forecast Mean")
        ax.fill_between(x, lower_95, upper_95, color="tab:blue", alpha=0.15, label="95% HDI")
        ax.fill_between(x, lower_50, upper_50, color="tab:blue", alpha=0.35, label="50% HDI")
        ax.set_title(f"{sel} — Out-of-sample Forecast")
        ax.set_ylabel(sel)
        ax.legend()
        fig.tight_layout()
        return fig, ax


# TODO: shared-states / common-factor variant (share_states=True) for a single
# latent level driving all items, plus per-item deviations.


