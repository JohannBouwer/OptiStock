"""
Module for family of linear bayes regressors
"""

from typing import Optional, Union

import pymc as pm
import pymc_bart as pmb
import xarray as xr
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from .base import BaseForecaster
from .priors import (
    BARTBayesTimeSeriesPriors,
    BayesTimeSeriesPriors,
    HierarchicalBayesTimeSeriesPriors,
    HSGPBayesTimeSeriesPriors,
)


default_seasonal_config = {
    "weekly": (7, 3),
    "monthly": (30.5, 1),
    "yearly": (365.25, 4),
}


class BayesTimeSeries(BaseForecaster):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "sales",
        seasonal_config: dict = default_seasonal_config,
        stockout_dates: Optional[Union[pd.DatetimeIndex, pd.Series]] = None,
        priors: Optional[BayesTimeSeriesPriors] = None,
    ) -> None:
        self.data = data
        self.target_col = target_col
        self.seasonal_config = seasonal_config
        self.stockout_dates = (
            pd.DatetimeIndex(pd.to_datetime(stockout_dates))
            if stockout_dates is not None
            else None
        )
        self.priors = priors or BayesTimeSeriesPriors()
        self._upper_bound_scaled = None
        self.model = None
        self.idata = None
        self.forecast_idata = None
        self.item_map = None
        self.fourier_names = None

    def _get_fourier_matrix(self, t):
        """Generate the fourier models based on the seasonal config"""
        components = []
        names = []
        for name, (period, order) in self.seasonal_config.items():
            for k in range(1, order + 1):
                components.append(np.sin(2 * np.pi * k * t / period))
                names.append(f"{name}_sin_{k}")
                components.append(np.cos(2 * np.pi * k * t / period))
                names.append(f"{name}_cos_{k}")
        return np.column_stack(components), names

    def create_events(self, events: dict, date_col: str = "date"):
        """
        Processes the events dictionary and generates the indicator matrix.
        Example dict: {"promo_1": ["2025-01-01", "2025-01-02"], "black_friday": [...]}
        """
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        dates = df[date_col]
        self.events_input = events

        event_indicators = {}

        for name, date_list in events.items():
            event_indicators[name] = dates.isin(pd.to_datetime(date_list)).astype(int)

        self.event_names = sorted(event_indicators.keys())
        self.event_X = np.column_stack(
            [event_indicators[name] for name in self.event_names]
        )

        return self

    def fit(
        self,
        target: str = "sales",
        date_col: str = "date",
        chains=4,
        samples=1000,
    ):
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # min-max scale date
        df[target] = df[target].astype("float")
        self.max_scaler = df[target].max()
        df[target] = df[target].div(self.max_scaler)

        # Build censoring upper bound in scaled space (np.inf = uncensored)
        if self.stockout_dates is not None:
            train_dates = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
            is_stockout = train_dates.isin(self.stockout_dates)
            missing = self.stockout_dates.difference(train_dates)
            if len(missing):
                import warnings

                warnings.warn(
                    f"{len(missing)} stockout_dates not in training data and will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            self._upper_bound_scaled = np.where(
                is_stockout, df[target].to_numpy(dtype=float), np.inf
            )

        # set up time and fourier modes
        t = np.arange(len(df))
        fourier_X, self.fourier_names = self._get_fourier_matrix(t)

        # Handle case where no events were created
        if not hasattr(self, "event_X"):
            self.event_names = ["none"]
            self.event_X = np.zeros((len(df), 1))

        model_coords = {
            "time": df[date_col],
            "fourier_feature": self.fourier_names,
            "event": self.event_names,
        }

        with pm.Model(coords=model_coords) as self.model:
            # set data containers
            t_shared = pm.Data("t", t, dims="time")
            fourier_shared = pm.Data(
                "fourier_X", fourier_X, dims=("time", "fourier_feature")
            )
            events_shared = pm.Data("event_X", self.event_X, dims=("time", "event"))
            y_obs = pm.Data("y_obs", df[target].to_numpy(dtype=float), dims="time")
            upper_bound = None
            if self.stockout_dates is not None:
                upper_bound = pm.Data(
                    "upper_bound", self._upper_bound_scaled, dims="time"
                )

            # Create Variables and priors (values sourced from self.priors)
            intercept = self.priors.intercept.build("intercept")
            growth = self.priors.growth.build("growth")
            trend = pm.Deterministic(
                "trend",
                intercept + growth * t_shared,
                dims="time",
            )

            beta_event = self.priors.beta_event.build("beta_event", dims="event")
            event_effect = pm.Deterministic(
                "event_effect", pm.math.dot(events_shared, beta_event), dims="time"
            )

            # Seasonality
            beta_fourier = self.priors.beta_fourier.build(
                "beta_fourier", dims="fourier_feature"
            )
            seasonality = pm.Deterministic(
                "seasonality",
                pm.math.dot(fourier_shared, beta_fourier),
                dims="time",
            )

            # Observation model
            sigma = self.priors.sigma.build("sigma")

            mu = pm.Deterministic(
                "mu",
                trend + seasonality + event_effect,
                dims="time",
            )

            if self.stockout_dates is not None:
                pm.Censored(
                    "y",
                    dist=pm.Normal.dist(mu=mu, sigma=sigma),
                    lower=None,
                    upper=upper_bound,
                    observed=y_obs,
                    dims="time",
                )
            else:
                pm.Normal(
                    "y",
                    mu=mu,
                    sigma=sigma,
                    observed=y_obs,
                    dims="time",
                )

            self.idata = pm.sample(
                samples, chains=chains, tune=1000, target_accept=0.95, sampler="numpyro"
            )

            self.posterior_predictive = pm.sample_posterior_predictive(
                self.idata
            ).posterior_predictive

            return self.idata

    def forecast(self, scenario: Optional[dict] = None) -> az.InferenceData:
        """
        Generates posterior predictive samples for the future dataframe provided.

        scenario keys:
            df_future (pd.DataFrame): Future dates dataframe.
            date_col (str): Name of the date column. Default "date".
        """
        if scenario is None:
            raise ValueError("A scenario dict with key 'df_future' is required.")
        df_future = scenario["df_future"]
        date_col = scenario.get("date_col", "date")

        # Align time index with training (t must continue from training)
        n_train = len(self.data)
        n_forecast = len(df_future)
        t_future = np.arange(n_train, n_train + n_forecast)

        # Re-generate Fourier features for future time steps
        fourier_X_future, _ = self._get_fourier_matrix(t_future)

        # Add in any events in the prediction period
        df_future[date_col] = pd.to_datetime(df_future[date_col])
        dates_future = df_future[date_col]
        if "none" in self.event_names:
            event_X_future = np.zeros((n_forecast, 1))
        else:
            event_X_future = np.column_stack(
                [
                    dates_future.isin(
                        pd.to_datetime(self.events_input.get(name, []))
                    ).astype(int)
                    for name in self.event_names
                ]
            )

        with self.model:
            set_data_dict = {
                "t": t_future,
                "fourier_X": fourier_X_future,
                "event_X": event_X_future,
                "y_obs": np.zeros(n_forecast),
            }
            if self.stockout_dates is not None:
                set_data_dict["upper_bound"] = np.full(n_forecast, np.inf)
            pm.set_data(set_data_dict, coords={"time": df_future[date_col]})
            self.forecast_idata = pm.sample_posterior_predictive(
                self.idata,
                predictions=True,
            )

        return self.forecast_idata

    def plot_forecast(self):
        """
        Plots the forecast with 94% HDI uncertainty intervals in the original scale.
        """

        dates = self.forecast_idata.predictions.time
        # Extract mean and HDI, then un-scale to original units
        mu_samples = self.forecast_idata.predictions["y"] * self.max_scaler
        mu_mean = mu_samples.mean(dim=["chain", "draw"])
        hdi_data = az.hdi(mu_samples, hdi_prob=0.94)["y"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.plot(dates, mu_mean, label="Forecast Mean", color="C0", lw=2)
        ax.fill_between(
            dates,
            hdi_data[:, 0],
            hdi_data[:, 1],
            alpha=0.3,
            label="94% HDI",
            color="C0",
        )
        ax.set_title("Posterior Predictive Forecast")
        ax.set_ylabel("Sales")
        ax.legend()

        return (fig, ax)

    def plot_components(
        self, components: list[str] = ["trend", "seasonality", "event_effect", "mu"]
    ):
        """
        Visualizes the individual contributions of Trend, Seasonality, and Events
        to the final forecast.
        """

        if not hasattr(self, "idata"):
            raise ValueError("You must run .fit() before plotting components.")

        # Extract inference data
        post = self.idata.posterior  # type: ignore
        dates = self.idata.posterior.time  # type: ignore

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in np.linspace(0, 1, len(components))]

        fig, axes = plt.subplots(len(components), 1, figsize=(12, 12), sharex=True)

        for i, (comp, color) in enumerate(zip(components, colors)):
            # Calculate mean and HDI, un-scaled to original units
            mean = post[comp].mean(dim=["chain", "draw"]) * self.max_scaler
            hdi = az.hdi(post[comp] * self.max_scaler, hdi_prob=0.94)[comp]

            axes[i].plot(
                dates, mean, color=color, lw=2, label=f"{comp.capitalize()} (Mean)"
            )
            axes[i].fill_between(
                dates,
                hdi[:, 0],
                hdi[:, 1],
                color=color,
                alpha=0.2,
                label="94% HDI",
            )
            axes[i].set_title(f"Component: {comp.capitalize()}")
            axes[i].set_ylabel("Sales")
            axes[i].legend(loc="upper left")
            axes[i].grid(axis="y", linestyle="--", alpha=0.5)

        plt.xlabel("Dates")
        plt.tight_layout()
        plt.show()

        return (fig, axes)

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        if self.forecast_idata is None:
            raise RuntimeError("You must call .forecast() before accessing results.")

        demands = self.forecast_idata.predictions["y"].sel(
            time=slice(start_date, end_date)
        )
        return (demands.sum(dim="time") * self.max_scaler).to_dataset(name="demand")


class BARTBayesTimeSeries(BaseForecaster):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "sales",
        priors: Optional[BARTBayesTimeSeriesPriors] = None,
    ) -> None:
        self.data = data
        self.len_df = len(data)
        self.target_col = target_col
        self.priors = priors or BARTBayesTimeSeriesPriors()
        self.model = None
        self.idata = None
        self.forecast_idata = None

    def _prepare_features(self, df: pd.DataFrame, date_col: str):
        """
        BART needs raw time features to 'learn' seasonality.
        Instead of Fourier, we provide time-based integers.
        """
        dates = pd.to_datetime(df[date_col])
        X = np.column_stack(
            [
                np.arange(len(df)) / self.len_df,  # Trend (time index)
                dates.dt.dayofweek.values,  # Weekly seasonality
                dates.dt.dayofyear.values,  # Yearly seasonality
                dates.dt.day.values,  # type: ignore # Monthly seasonality
            ]
        )  # type: ignore
        return X.astype(float)

    def fit(
        self, target: str = "sales", date_col: str = "date", samples=1000, trees=50
    ):
        df = self.data.copy()

        # min-max scaling
        self.max_scaler = df[target].max()
        y_scaled = df[target].div(self.max_scaler).values

        self.X = self._prepare_features(df, date_col)

        model_coords = {
            "time": df[date_col],
        }

        with pm.Model(coords=model_coords) as self.model:
            X_shared = pm.Data("X_data", self.X, dims=("time", "feature"))
            y_shared = pm.Data("y_obs", y_scaled, dims="time")

            intercept = self.priors.intercept.build("intercept")
            growth = self.priors.growth.build("growth")
            trend = pm.Deterministic(
                "trend",
                intercept + growth * X_shared[:, 0],
                dims="time",
            )

            bart = pmb.BART("bart", X_shared[:, 1:], y_shared, dims="time", m=trees)

            sigma = self.priors.sigma.build("sigma")

            mu = pm.Deterministic("mu", bart + trend, dims=("time"))

            pm.Normal("y", mu=mu, sigma=sigma, observed=y_shared, dims="time")

            self.idata = pm.sample(samples)
            return self.idata

    def forecast(self, scenario: Optional[dict] = None) -> az.InferenceData:
        """
        scenario keys:
            df_future (pd.DataFrame): Future dates dataframe.
            date_col (str): Name of the date column. Default "date".
        """
        if scenario is None:
            raise ValueError("A scenario dict with key 'df_future' is required.")
        df_future = scenario["df_future"]
        date_col = scenario.get("date_col", "date")

        X_future = self._prepare_features(df_future, date_col)
        # update future index for trend
        X_future[:, 0] += self.X[-1, 0]

        with self.model:
            pm.set_data(
                {"X_data": X_future, "y_obs": np.zeros(len(df_future))},
                coords={"time": df_future[date_col]},
            )
            self.forecast_idata = pm.sample_posterior_predictive(
                self.idata, predictions=True
            )
        return self.forecast_idata

    def plot_forecast(self):
        """
        Plots the BART forecast with 94% HDI uncertainty intervals.
        """
        if self.forecast_idata is None:
            raise RuntimeError("You must call .predict() before plotting.")

        # Extract dates and predictive samples
        dates = self.forecast_idata.predictions.time
        y_samples = self.forecast_idata.predictions["y"]

        # Calculate summary statistics, un-scaled to original units
        y_samples_unscaled = y_samples * self.max_scaler
        y_mean = y_samples_unscaled.mean(dim=["chain", "draw"])
        hdi_data = az.hdi(y_samples_unscaled, hdi_prob=0.94)["y"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot the mean prediction
        ax.plot(dates, y_mean, label="BART Forecast Mean", color="C0", lw=2)

        # Plot the uncertainty interval
        ax.fill_between(
            dates,
            hdi_data.sel(hdi="lower"),
            hdi_data.sel(hdi="higher"),
            alpha=0.3,
            label="94% HDI (Uncertainty)",
            color="C0",
        )

        ax.set_title("BART Posterior Predictive Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return (fig, ax)

    def plot_components(self) -> tuple:
        """
        Visualizes BART 'components' and Variable Importance.
        This is done at the "average" of all other variables
        """
        if self.idata is None:
            raise RuntimeError("You must call .fit() before plotting.")

        fig, ax = plt.subplots(3, 1, figsize=(12, 12))
        pmb.plot_pdp(
            self.model["bart"],  # type: ignore
            X=self.X[:, 1:],
            func=lambda x: x * self.max_scaler,
            ax=ax,
        )
        labels = ["Weekly", "Yearly", "Monthly"]
        for a, l in zip(ax, labels):
            a.set_xlabel(l)

        fig.tight_layout()

        return (fig, ax)

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        if self.forecast_idata is None:
            raise RuntimeError("You must call .forecast() before accessing results.")

        demands = self.forecast_idata.predictions["y"].sel(
            time=slice(start_date, end_date)
        )
        return (demands.sum(dim="time") * self.max_scaler).to_dataset(name="demand")


class HSGPBayesTimeSeries(BaseForecaster):
    """
    Forecaster using Hilbert Space Gaussian Process (HSGP) approximation.
    Effective for capturing smooth, non-linear trends with high efficiency.

    Not fully tested, but worked for small problems
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "sales",
        m: int = 20,
        L: float = 1.5,
        priors: Optional[HSGPBayesTimeSeriesPriors] = None,
    ) -> None:
        self.data = data
        self.target_col = target_col
        self.m = m  # Number of basis functions
        self.L = L  # Boundary factor
        self.priors = priors or HSGPBayesTimeSeriesPriors()
        self.model = None
        self.idata = None
        self.forecast_idata = None

    def fit(self, target: str = "sales", date_col: str = "date", samples=1000, chain=4):
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Scaling
        self.max_scaler = df[target].max()
        y_scaled = df[target].div(self.max_scaler).values

        # Time index scaled to [0, 1] for GP stability
        t = np.arange(len(df))
        self.t_train_max = t.max()
        t_scaled = t / self.t_train_max

        model_coords = {"time": df[date_col]}

        with pm.Model(coords=model_coords) as self.model:
            t_shared = pm.Data("t", t_scaled[:, None], dims=("time", "feature"))
            y_obs = pm.Data("y_obs", y_scaled, dims="time")

            # HSGP Hyperparameters
            ell = self.priors.ell.build("ell")
            eta = self.priors.eta.build("eta")

            # Covariance and HSGP Prior
            cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)
            gp = pm.gp.HSGP(m=[self.m], L=[self.L], cov_func=cov_func)
            phi = gp.prior("phi", X=t_shared, dims="time")

            # If the user didn't pin `mu`, center the intercept on the data mean.
            intercept_params = dict(self.priors.intercept.params)
            intercept_params.setdefault("mu", float(y_scaled.mean()))
            intercept = getattr(pm, self.priors.intercept.distribution)(
                "intercept", **intercept_params
            )
            mu = pm.Deterministic("mu", intercept + phi, dims="time")

            sigma = self.priors.sigma.build("sigma")
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="time")

            self.idata = pm.sample(samples, chains=chain, sampler="numpyro")
            return self.idata

    def forecast(self, scenario: Optional[dict] = None) -> az.InferenceData:
        """
        scenario keys:
            df_future (pd.DataFrame): Future dates dataframe.
            date_col (str): Name of the date column. Default "date".
        """
        if scenario is None:
            raise ValueError("A scenario dict with key 'df_future' is required.")
        df_future = scenario["df_future"]
        date_col = scenario.get("date_col", "date")

        n_train = len(self.data)
        n_forecast = len(df_future)

        # Continue the scaled time index
        t_future = np.arange(n_train, n_train + n_forecast) / self.t_train_max

        with self.model:
            pm.set_data(
                {"t": t_future[:, None], "y_obs": np.zeros(n_forecast)},
                coords={"time": df_future[date_col]},
            )
            self.forecast_idata = pm.sample_posterior_predictive(
                self.idata, predictions=True
            )
        return self.forecast_idata

    def plot_forecast(self):
        """Standard forecast plot with HDI in the original scale."""
        if self.forecast_idata is None:
            raise RuntimeError("Call .predict() first.")

        dates = self.forecast_idata.predictions.time
        y_samples = self.forecast_idata.predictions["y"] * self.max_scaler

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, y_samples.mean(dim=["chain", "draw"]), label="HSGP Mean")
        az.plot_hdi(dates, y_samples, hdi_prob=0.94, ax=ax, fill_kwargs={"alpha": 0.3})
        ax.set_title("HSGP Time Series Forecast")
        ax.set_ylabel("Sales")
        return fig, ax

    def plot_components(self):
        """Visualizes the Intercept vs the GP Trend (phi) in the original scale."""
        post = self.idata.posterior
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for i, comp in enumerate(["intercept", "phi"]):
            mean = post[comp].mean(dim=["chain", "draw"]) * self.max_scaler
            hdi = az.hdi(post[comp] * self.max_scaler, hdi_prob=0.94)[comp]
            axes[i].plot(post.time, mean, label=comp)
            axes[i].fill_between(post.time, hdi[:, 0], hdi[:, 1], alpha=0.2)
            axes[i].set_title(f"Component: {comp}")
            axes[i].set_ylabel("Sales")

        return fig, axes

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        if self.forecast_idata is None:
            raise RuntimeError("You must call .forecast() before accessing results.")

        demands = self.forecast_idata.predictions["y"].sel(
            time=slice(start_date, end_date)
        )
        return (demands.sum(dim="time") * self.max_scaler).to_dataset(name="demand")


class HierarchicalBayesTimeSeries(BayesTimeSeries):
    """
    Multi-item Bayesian time-series forecaster with partial pooling across
    items via hierarchical (hyper) priors.

    Each item has its own ``intercept``, ``growth``, ``beta_fourier`` and
    ``beta_event`` coefficients, drawn from a shared population-level
    distribution whose mean and spread (hyper-priors) are themselves learned
    from data. Observation noise ``sigma`` is shared across items.

    Input is **wide-format**: one ``date`` column plus one numeric column per
    item. Items are inferred as all non-``date`` columns if not supplied
    explicitly.

    The model uses a non-centered parameterisation for every per-item
    coefficient to avoid funnel pathologies under HMC.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        items: Optional[list] = None,
        seasonal_config: dict = default_seasonal_config,
        priors: Optional[HierarchicalBayesTimeSeriesPriors] = None,
    ) -> None:
        self.data = data.copy()
        self.date_col = date_col
        if items is None:
            items = [c for c in data.columns if c != date_col]
        self.items = list(items)
        self.seasonal_config = seasonal_config
        self.priors = priors or HierarchicalBayesTimeSeriesPriors()
        self.model = None
        self.idata = None
        self.forecast_idata = None
        self.fourier_names = None
        self.event_names = None
        self.event_X = None
        self.events_input = None
        self.max_scaler = None  # pd.Series indexed by item

    def create_events(self, events: dict, date_col: Optional[str] = None):
        """
        Events apply to every item (one shared indicator matrix), but each
        item learns its own coefficient via the hierarchical ``beta_event``.
        """
        date_col = date_col or self.date_col
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        dates = df[date_col]
        self.events_input = events
        event_indicators = {
            name: dates.isin(pd.to_datetime(date_list)).astype(int)
            for name, date_list in events.items()
        }
        self.event_names = sorted(event_indicators.keys())
        self.event_X = np.column_stack(
            [event_indicators[name] for name in self.event_names]
        )
        return self

    def fit(
        self,
        date_col: Optional[str] = None,
        chains: int = 4,
        samples: int = 1000,
    ) -> az.InferenceData:
        date_col = date_col or self.date_col
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # handle ragged data
        Y = df[self.items].astype(float)
        self.max_scaler = float(np.nanmax(Y.to_numpy(dtype=float)))
        Y_scaled = Y.div(self.max_scaler).to_numpy(dtype=float)

        mask = ~np.isnan(Y_scaled)
        obs_time_idx, obs_item_idx = np.where(mask)
        y_flat = Y_scaled[mask]

        t = np.arange(len(df))
        fourier_X, self.fourier_names = self._get_fourier_matrix(t)

        if self.event_X is None:
            self.event_names = ["none"]
            self.event_X = np.zeros((len(df), 1))

        model_coords = {
            "time": df[date_col],
            "item": self.items,
            "fourier_feature": self.fourier_names,
            "event": self.event_names,
            "obs": np.arange(len(y_flat)),
        }

        with pm.Model(coords=model_coords) as self.model:
            t_shared = pm.Data("t", t, dims="time")
            fourier_shared = pm.Data(
                "fourier_X", fourier_X, dims=("time", "fourier_feature")
            )
            event_shared = pm.Data("event_X", self.event_X, dims=("time", "event"))

            # Hyper-priors (population-level).
            intercept_mu = self.priors.intercept_mu.build("intercept_mu")
            intercept_sigma = self.priors.intercept_sigma.build("intercept_sigma")
            growth_mu = self.priors.growth_mu.build("growth_mu")
            growth_sigma = self.priors.growth_sigma.build("growth_sigma")
            beta_fourier_mu = self.priors.beta_fourier_mu.build(
                "beta_fourier_mu", dims="fourier_feature"
            )
            beta_fourier_sigma = self.priors.beta_fourier_sigma.build(
                "beta_fourier_sigma", dims="fourier_feature"
            )
            beta_event_mu = self.priors.beta_event_mu.build(
                "beta_event_mu", dims="event"
            )
            beta_event_sigma = self.priors.beta_event_sigma.build(
                "beta_event_sigma", dims="event"
            )

            # Non-centered per-item coefficients.
            z_intercept = pm.Normal("z_intercept", 0.0, 1.0, dims="item")
            intercept = pm.Deterministic(
                "intercept",
                intercept_mu + intercept_sigma * z_intercept,
                dims="item",
            )
            z_growth = pm.Normal("z_growth", 0.0, 1.0, dims="item")
            growth = pm.Deterministic(
                "growth",
                growth_mu + growth_sigma * z_growth,
                dims="item",
            )
            z_beta_fourier = pm.Normal(
                "z_beta_fourier", 0.0, 1.0, dims=("item", "fourier_feature")
            )
            beta_fourier = pm.Deterministic(
                "beta_fourier",
                beta_fourier_mu + beta_fourier_sigma * z_beta_fourier,
                dims=("item", "fourier_feature"),
            )
            z_beta_event = pm.Normal("z_beta_event", 0.0, 1.0, dims=("item", "event"))
            beta_event = pm.Deterministic(
                "beta_event",
                beta_event_mu + beta_event_sigma * z_beta_event,
                dims=("item", "event"),
            )

            # Likelihood — vectorised over (time, item).
            trend = pm.Deterministic(
                "trend",
                intercept[None, :] + growth[None, :] * t_shared[:, None],
                dims=("time", "item"),
            )
            seasonality = pm.Deterministic(
                "seasonality",
                pm.math.dot(fourier_shared, beta_fourier.T),
                dims=("time", "item"),
            )
            event_effect = pm.Deterministic(
                "event_effect",
                pm.math.dot(event_shared, beta_event.T),
                dims=("time", "item"),
            )
            mu = pm.Deterministic(
                "mu", trend + seasonality + event_effect, dims=("time", "item")
            )

            sigma = self.priors.sigma.build("sigma")

            mu_obs = mu[obs_time_idx, obs_item_idx]
            pm.Normal("y", mu=mu_obs, sigma=sigma, observed=y_flat, dims="obs")

            self.idata = pm.sample(
                samples,
                chains=chains,
                tune=1000,
                target_accept=0.95,
                sampler="nutpie",
            )
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.idata
            ).posterior_predictive
            return self.idata

    def forecast(self, scenario: Optional[dict] = None) -> az.InferenceData:
        """
        scenario keys:
            df_future (pd.DataFrame): Future dates dataframe with a ``date_col``.
            date_col (str): Name of the date column. Defaults to the constructor's.

        Predictions are computed directly from the posterior samples rather
        than via ``pm.sample_posterior_predictive``. The model's likelihood is
        a flat 1-D vector tied to training-time observed indices, so we can't
        reuse the same graph for a (time, item) grid of future predictions -
        and doing it in numpy is also much faster.
        """
        if scenario is None:
            raise ValueError("A scenario dict with key 'df_future' is required.")
        df_future = scenario["df_future"].copy()
        date_col = scenario.get("date_col", self.date_col)

        n_train = len(self.data)
        n_forecast = len(df_future)
        t_future = np.arange(n_train, n_train + n_forecast)
        fourier_X_future, _ = self._get_fourier_matrix(t_future)

        df_future[date_col] = pd.to_datetime(df_future[date_col])
        dates_future = df_future[date_col]
        if "none" in self.event_names:
            event_X_future = np.zeros((n_forecast, 1))
        else:
            event_X_future = np.column_stack(
                [
                    dates_future.isin(
                        pd.to_datetime(self.events_input.get(name, []))
                    ).astype(int)
                    for name in self.event_names
                ]
            )

        post = self.idata.posterior
        intercept = post["intercept"].values  # (C, D, I)
        growth = post["growth"].values  # (C, D, I)
        beta_fourier = post["beta_fourier"].values  # (C, D, I, F)
        beta_event = post["beta_event"].values  # (C, D, I, E)
        sigma = post["sigma"].values  # (C, D)

        trend = (
            intercept[:, :, None, :]
            + growth[:, :, None, :] * t_future[None, None, :, None]
        )  # (C, D, T, I)
        seasonality = np.einsum("tf,cdif->cdti", fourier_X_future, beta_fourier)
        event_effect = np.einsum("te,cdie->cdti", event_X_future, beta_event)
        mu_future = trend + seasonality + event_effect  # (C, D, T, I)

        rng = np.random.default_rng()
        y_future = rng.normal(mu_future, sigma[:, :, None, None])  # (C, D, T, I)

        predictions = xr.Dataset(
            {"y": (("chain", "draw", "time", "item"), y_future)},
            coords={
                "chain": post.chain.values,
                "draw": post.draw.values,
                "time": df_future[date_col].values,
                "item": self.items,
            },
        )
        self.forecast_idata = az.InferenceData(predictions=predictions)
        return self.forecast_idata

    def plot_forecast(self, item: Optional[str] = None):
        """
        Plot the posterior predictive forecast for one or all items.
        """
        if self.forecast_idata is None:
            raise RuntimeError("Call .forecast() before plotting.")

        items_to_plot = [item] if item is not None else self.items
        n = len(items_to_plot)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False
        )

        dates = self.forecast_idata.predictions.time
        y_all = self.forecast_idata.predictions["y"]
        for i, it in enumerate(items_to_plot):
            ax = axes[i // ncols, i % ncols]
            y = y_all.sel(item=it) * self.max_scaler
            mean = y.mean(dim=("chain", "draw"))
            hdi = az.hdi(y, hdi_prob=0.94)["y"]
            ax.plot(dates, mean, color="C0", lw=2, label="Forecast mean")
            ax.fill_between(
                dates, hdi[:, 0], hdi[:, 1], color="C0", alpha=0.3, label="94% HDI"
            )
            ax.set_title(f"Item: {it}")
            ax.set_ylabel("Sales")
            ax.legend()
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        plt.tight_layout()
        return fig, axes

    def plot_components(
        self,
        item: str,
        components: list = ["trend", "seasonality", "event_effect", "mu"],
    ):
        """
        Plot the additive components for a single item, in the original scale.
        """
        if self.idata is None:
            raise ValueError("Call .fit() before plotting components.")

        post = self.idata.posterior
        dates = post.time
        scaler = self.max_scaler

        fig, axes = plt.subplots(
            len(components), 1, figsize=(12, 3 * len(components)), sharex=True
        )
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in np.linspace(0, 1, len(components))]
        for i, (comp, color) in enumerate(zip(components, colors)):
            arr = post[comp].sel(item=item) * scaler
            mean = arr.mean(dim=("chain", "draw"))
            hdi = az.hdi(arr, hdi_prob=0.94)[comp]
            axes[i].plot(dates, mean, color=color, lw=2, label=f"{comp} (mean)")
            axes[i].fill_between(
                dates, hdi[:, 0], hdi[:, 1], color=color, alpha=0.2, label="94% HDI"
            )
            axes[i].set_title(f"{item} — {comp}")
            axes[i].set_ylabel("Sales")
            axes[i].legend(loc="upper left")
        plt.tight_layout()
        return fig, axes

    def get_demand_distribution(
        self,
        start_date: str,
        end_date: str,
        item: Optional[str] = None,
    ) -> xr.Dataset:
        """
        Posterior total demand over ``[start_date, end_date]``.

        With ``item=None`` the result keeps the ``item`` dim and rescales each
        item by its own ``max_scaler``. Passing ``item=<name>`` returns the
        single-item distribution in the same shape as :class:`BayesTimeSeries`'s
        output, so the caller can plug it into ``ForecastSolver`` per item.
        """
        if self.forecast_idata is None:
            raise RuntimeError("You must call .forecast() before accessing results.")

        y = self.forecast_idata.predictions["y"].sel(time=slice(start_date, end_date))
        if item is not None:
            scaled = y.sel(item=item) * self.max_scaler
            return scaled.sum(dim="time").to_dataset(name="demand")

        scaled = y * self.max_scaler
        return scaled.sum(dim="time").to_dataset(name="demand")
