from abc import ABC, abstractmethod

import arviz as az

from ..forecasting.priors import BasePriors


class BaseCausalEstimator(ABC):
    """
    Abstract base class for Bayesian causal effect estimators.

    Causal estimators answer counterfactual questions ("what would sales of
    item Y have been without intervention X?"). They share the prior-config
    machinery with the forecasting models but have a different lifecycle:
    no future forecast, no demand distribution — just fit, summarize the
    effect, and plot the actual-vs-counterfactual contrast.
    """

    priors: BasePriors

    def describe_priors(self) -> dict[str, dict]:
        """Print and return a structured description of every prior used by the model."""
        print(self.priors)
        return self.priors.to_dict()

    @abstractmethod
    def fit(self) -> az.InferenceData:
        """Sample the posterior and return InferenceData."""
        pass

    @abstractmethod
    def summary(self):
        """Return a CausalEffect dataclass with point estimate and HDI."""
        pass

    @abstractmethod
    def plot(self) -> tuple:
        """Visualize observed series against the counterfactual."""
        pass
