import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from typing import Union
from scipy.stats import norm
from collections import Counter


class DemandDistribution(ABC):
    """
    Abstract Base Class for all demand distributions.
    Requires subclasses to implement a method to find the
    quantile for a given probability.
    """

    @property
    @abstractmethod
    def mean(self) -> float:
        """Subclasses must define a mean."""
        pass

    @property
    @abstractmethod
    def std(self) -> float:
        """Subclasses must define a standard deviation."""
        pass

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

    @abstractmethod
    def get_cdf(self, quantity: float) -> float:
        """
        Cumulative Distribution Function. Returns P(Demand <= quantity).
        Required for marginal analysis in constrained optimization.
        """
        pass

    @abstractmethod
    def get_pdf(self, quantity: float) -> float:
        """
        Probability Distribution Function. Returns P(Demand = quantity).
        Required for marginal analysis in constrained optimization.
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
        self._mean = mean
        self._std_dev = std_dev

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std_dev

    def get_quantile(self, probability: float) -> float:
        """
        Finds the demand quantile using the inverse CDF of the normal distribution.

        Args:
            probability (float): The cumulative probability (0 to 1).

        Returns:
            float: The demand quantile. Returns 0 if std_dev is 0.
        """
        if self._std_dev == 0:
            return self.mean

        # Clamp probability to avoid issues with ppf at 0 or 1
        epsilon = 1e-9
        clamped_prob = max(epsilon, min(1.0 - epsilon, probability))

        return norm.ppf(clamped_prob, loc=self.mean, scale=self.std)

    def get_cdf(self, quantity: float) -> float:
        if self.std == 0:
            return 1.0 if quantity >= self.mean else 0.0
        return norm.cdf(quantity, loc=self.mean, scale=self.std)

    def get_pdf(self, quantity: float) -> float:
        return norm.pdf(quantity)


class SampledDemand(DemandDistribution):
    """
    Represents a sampled distribution. Most likly a postrior from a bayes model.
    """

    def __init__(self, samples: Union[np.ndarray, list, xr.DataArray]):
        if hasattr(samples, "values"):
            samples = samples.values

        # Flatten and SORT samples immediately.
        # Sorting allows O(log N) CDF lookups via binary search.
        self.samples = np.sort(np.asarray(samples).flatten())

        if self.samples.size == 0:
            raise ValueError("Sample array cannot be empty.")

        counts = Counter(self.samples)
        self._pdf_map = {k: v / len(self.samples) for k, v in counts.items()}

    @property
    def mean(self) -> float:
        return np.mean(self.samples)

    @property
    def std(self) -> float:
        return np.std(self.samples)

    def get_pdf(self, quantity) -> float:
        return self._pdf_map.get(quantity, 0.0)

    def get_quantile(self, probability: float) -> float:
        # Uses linear interpolation for values between samples
        return np.quantile(self.samples, probability, method="linear")

    def get_cdf(self, quantity: float) -> float:
        # Uses binary search on sorted samples for speed
        # side='right' ensures P(X <= x) logic
        count = np.searchsorted(self.samples, quantity, side="right")
        return count / self.samples.size


def demand_aggregator(
    distributions: list[Union[NormalDemand, SampledDemand]],
) -> Union[NormalDemand, SampledDemand]:
    """
    Aggregates a list of distributions into a total period distribution.
    Assumes independence between days.
    """
    # Check if all are Normal
    if all(isinstance(d, NormalDemand) for d in distributions):
        total_mean = sum(d.mean for d in distributions)
        # Sum of Variances (Sigma^2)
        total_variance = sum(d.std**2 for d in distributions)
        total_std = np.sqrt(total_variance)
        return NormalDemand(total_mean, total_std)

    # Handle Sampled (or mix) by converting all to samples
    else:
        n_sims = 10000
        totals = np.zeros(n_sims)

        for dist in distributions:
            # Get samples for this specific day
            if isinstance(dist, SampledDemand):
                samples = dist.samples
            else:
                samples = np.random.normal(dist.mean, dist.std, 10000)

            # Randomly draw for this day (Bootstrapping)
            # We use replace=True to ensure we can generate enough samples
            draws = np.random.choice(samples, size=n_sims, replace=True)

            # Add to running total
            totals += draws

        return SampledDemand(totals)
