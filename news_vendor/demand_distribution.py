import math
from abc import ABC, abstractmethod
from scipy.stats import norm

class DemandDistribution(ABC):
    """
    Abstract Base Class for all demand distributions.
    Requires subclasses to implement a method to find the
    quantile for a given probability.
    """
    
    @abstractmethod
    def get_quantile(self, probability: float) -> float:
        """
        Returns the demand quantity Q such that P(Demand <= Q) = probability.
        This is the inverse of the Cumulative Distribution Function (CDF),
        also known as the Percent-Point Function (PPF).

        Args:
            probability (float): The cumulative probability (must be between 0 and 1).

        Returns:
            float: The demand quantile.
        """
        pass

class NormalDemand(DemandDistribution):
    """
    Represents a normally distributed demand.
    """
    def __init__(self, mean: float, std_dev: float):
        """
        Initializes the normal demand distribution.

        Args:
            mean (float): The average (mean) demand.
            std_dev (float): The standard deviation of demand.
        """
        if mean < 0 or std_dev < 0:
            raise ValueError("Mean and Standard Deviation must be non-negative.")
        self.mean = mean
        self.std_dev = std_dev

    def get_quantile(self, probability: float) -> float:
        """
        Finds the demand quantile using the inverse CDF of the normal distribution.
        
        Args:
            probability (float): The cumulative probability (0 to 1).

        Returns:
            float: The demand quantile. Returns 0 if std_dev is 0.
        """
        if self.std_dev == 0:
            return self.mean
        
        # Clamp probability to avoid issues with ppf at 0 or 1
        epsilon = 1e-9
        clamped_prob = max(epsilon, min(1.0 - epsilon, probability))
        
        return norm.ppf(clamped_prob, loc=self.mean, scale=self.std_dev)
