from abc import ABC, abstractmethod
from typing import Optional
import arviz as az
import xarray as xr
import pandas as pd
import numpy as np


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
    def forecast(self, scenario: Optional[dict] = None) -> az.InferenceData:
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
