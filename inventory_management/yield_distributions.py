import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from typing import Union
from scipy.stats import norm, beta


class YieldDistribution(ABC):
    """
    Abstract Base Class for Yield (percentage of good units).
    """

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Returns n samples of yield rates (0.0 to 1.0)."""
        pass

    @property
    @abstractmethod
    def mean(self) -> float:
        pass


class PerfectYield(YieldDistribution):
    """Represents 100% yield (Standard Newsvendor)."""

    def sample(self, n: int) -> np.ndarray:
        return np.ones(n)

    @property
    def mean(self) -> float:
        return 1.0


class BetaYield(YieldDistribution):
    """
    Bayesian Yield model using Beta distribution.
    Appropriate because support is [0, 1].
    """

    def __init__(self, alpha: float, beta_param: float):
        self.alpha = alpha
        self.beta = beta_param

    def sample(self, n: int) -> np.ndarray:
        return beta.rvs(self.alpha, self.beta, size=n)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
