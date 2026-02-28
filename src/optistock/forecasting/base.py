from abc import ABC, abstractmethod
import arviz as az
import xarray as xr
import pandas as pd


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
    def forecast(self, df_future: pd.DataFrame) -> az.InferenceData:
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
