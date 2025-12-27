import pymc as pm
import pymc_bart as pmb
import xarray as xr
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseForecaster(ABC):
    """
    Abstract Base Class for item demand forecasting models.
    """

    @abstractmethod
    def fit(self, target: str, date_col: str) -> az.InferenceData:
        """
        Train the model on the provided dataframe.
        """
        pass

    @abstractmethod
    def predict(self, df_future: pd.DataFrame) -> az.InferenceData:
        """
        Generate posterior predictive distributions for the future period.
        """
        pass

    @abstractmethod
    def plot_forecast(self) -> tuple:
        """
        Every forecaster should be able to visualize its own trend/seasonality.
        """
        pass

    @abstractmethod
    def plot_components(self) -> tuple:
        """
        Every forecaster should be able to visualize its own trend/seasonality.
        """
        pass

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Generate a forecast sampled distribution over a time frame. The demand is summed to generate a total demand in the original scale.
        """
        pass


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
    ) -> None:
        self.data = data
        self.target_col = target_col
        self.seasonal_config = seasonal_config
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
        chain=4,
        samples=1000,
    ):
        df = self.data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # min-max scale date
        df[target] = df[target].astype("float")
        self.max_scaler = df[target].max()
        df[target] = df[target].div(self.max_scaler)

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
            y_obs = pm.Data("y_obs", df[target].values, dims="time")

            # Create Variables and priors
            intercept = pm.HalfNormal(
                "intercept",
                sigma=1.0,
            )
            growth = pm.Normal(
                "growth",
                mu=0.0,
                sigma=1.0,
            )
            trend = pm.Deterministic(
                "trend",
                intercept + growth * t_shared,
                dims="time",
            )

            beta_event = pm.Normal("beta_event", mu=0.0, sigma=0.5, dims="event")
            event_effect = pm.Deterministic(
                "event_effect", pm.math.dot(events_shared, beta_event), dims="time"
            )

            # Seasonality
            beta_fourier = pm.Laplace(
                "beta_fourier",
                mu=0.0,
                b=1,
                dims="fourier_feature",
            )
            seasonality = pm.Deterministic(
                "seasonality",
                pm.math.dot(fourier_shared, beta_fourier),
                dims="time",
            )

            # Observation model
            sigma = pm.HalfNormal("sigma", sigma=0.05)

            mu = pm.Deterministic(
                "mu",
                trend + seasonality + event_effect,
                dims="time",
            )

            pm.Normal(
                "y",
                mu=mu,
                sigma=sigma,
                observed=y_obs,
                dims="time",
            )

            self.idata = pm.sample(
                samples, chain=chain, tune=1000, target_accept=0.95, sampler="numpyro"
            )

            self.posterior_predictive = pm.sample_posterior_predictive(
                self.idata
            ).posterior_predictive

            return self.idata

    def predict(self, df_future: pd.DataFrame, date_col: str = "date"):
        """
        Generates posterior predictive samples for the future dataframe provided.
        """
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
            # We reconstruct the matrix using the same names logic
            event_X_future = np.column_stack(
                [
                    dates_future.isin(
                        pd.to_datetime(self.events_input.get(name, []))
                    ).astype(int)
                    for name in self.event_names
                ]
            )

        with self.model:
            pm.set_data(
                {
                    "t": t_future,
                    "fourier_X": fourier_X_future,
                    "event_X": event_X_future,
                    "y_obs": np.zeros(
                        n_forecast
                    ),  # Placeholder for future observations
                },
                coords={"time": df_future[date_col]},
            )

            # Sample from the posterior predictive
            # Use 'predictions=True' to keep these separate from training samples
            self.forecast_idata = pm.sample_posterior_predictive(
                self.idata,
                predictions=True,
            )

        return self.forecast_idata

    def plot_forecast(self):
        """
        Plots the forecast with 94% HDI uncertainty intervals.
        """

        dates = self.forecast_idata.predictions.time
        # Extract mean and HDI
        mu_samples = self.forecast_idata.predictions["y"]
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
            # Calculate mean and HDI for the component
            mean = post[comp].mean(dim=["chain", "draw"])
            hdi = az.hdi(post[comp], hdi_prob=0.94)[comp]

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
            axes[i].legend(loc="upper left")
            axes[i].grid(axis="y", linestyle="--", alpha=0.5)

        plt.xlabel("Dates")
        plt.tight_layout()
        plt.show()

        return (fig, axes)

    def get_demand_distribution(self, start_date: str, end_date: str) -> xr.Dataset:
        if self.forecast_idata is None:
            raise RuntimeError("You must call .predict() before accessing results")

        demands = self.forecast_idata.predictions.sel(time=slice(start_date, end_date))

        return demands.sum(dim=("time")) * self.max_scaler


class BARTBayesTimeSeries(BaseForecaster):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "sales",
    ) -> None:
        self.data = data
        self.len_df = len(data)
        self.target_col = target_col
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
            
            intercept = pm.HalfNormal(
                "intercept",
                sigma=1.0,
            )
            growth = pm.Normal(
                "growth",
                mu=0.0,
                sigma=1.0,
            )
            trend = pm.Deterministic(
                "trend",
                intercept + growth * X_shared[:,0],
                dims="time",
            )

            bart = pmb.BART("bart", X_shared[:,1:], y_shared, dims="time", m=trees)

            sigma = pm.HalfNormal("sigma", sigma=0.1)
            
            mu = pm.Deterministic("mu", bart + trend, dims=("time"))
            
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_shared, dims="time")

            self.idata = pm.sample(samples)
            return self.idata

    def predict(self, df_future: pd.DataFrame, date_col: str = "date"):
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

        # Calculate summary statistics
        y_mean = y_samples.mean(dim=["chain", "draw"])
        hdi_data = az.hdi(y_samples, hdi_prob=0.94)["y"]

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
        ax.set_ylabel("Sales (Scaled)")
        ax.set_ylim(0, 1)
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
            X=self.X[:,1:],
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
            raise RuntimeError("You must call .predict() before accessing results")

        demands = self.forecast_idata.predictions.y.sel(
            time=slice(start_date, end_date)
        )

        return demands.sum(dim=("time")) * self.max_scaler


class ErrorEstimations:
    def calculate_smape(actual, forecast):
        """
        Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).
        Parameters:
        actual (array-like): The ground truth values.
        forecast (array-like): The predicted idata.

        Returns:
        float: The SMAPE value as a percentage (0 to 200).
        """
        # Convert inputs to numpy arrays for vectorized operations
        actual = np.array(actual)
        forecast = np.array(forecast)

        # Calculate the numerator (absolute difference)
        numerator = np.abs(forecast - actual)

        # Calculate the denominator (mean of absolute values)
        denominator = (np.abs(actual) + np.abs(forecast)) / 2

        # Handle the case where both actual and forecast are 0 to avoid division by zero
        # We return 0 for those specific points
        smape_val = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
        )

        return np.mean(smape_val) * 100
