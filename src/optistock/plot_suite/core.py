import numpy as np
import seaborn as sns
from scipy.stats import norm

"""
File of helper functions shared by all plotting functions
"""


def setup_style(style: str = "darkgrid"):
    sns.set_style(style)


# ---------------------------------------------------------------------------
# SampledDemand — bridge between BaseForecaster outputs and plot functions
# ---------------------------------------------------------------------------


class SampledDemand:
    """
    Adapter that wraps posterior demand samples from a BaseForecaster so they
    can be passed to any plot_suite function that expects a demand object.

    All simulation and density code in the plot suite checks
    ``hasattr(demand, "samples")`` first, so raw posterior samples are always
    used directly — no normality assumption is ever made.  The ``.mean`` and
    ``.std`` properties are derived from the samples and are only used for
    axis-range hints (see ``plot_profit_curve_helper``).

    Parameters
    ----------
    samples : array-like
        1-D array of posterior predictive demand draws (any distribution shape).
    """

    def __init__(self, samples: np.ndarray):
        self.samples = np.asarray(samples, dtype=float).flatten()

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    @property
    def std(self) -> float:
        return float(self.samples.std())

    def get_quantile(self, p: float) -> float:
        """Empirical (non-parametric) quantile — no normality assumed."""
        return float(np.quantile(self.samples, p))

    @classmethod
    def from_forecaster(
        cls, forecaster, start_date: str, end_date: str
    ) -> "SampledDemand":
        """
        Convenience constructor that calls ``forecaster.get_demand_distribution``
        and wraps the resulting samples.

        Parameters
        ----------
        forecaster : BaseForecaster
            A fitted forecaster with a ``get_demand_distribution`` method.
        start_date, end_date : str
            Inclusive planning horizon (e.g. ``"2025-01-01"``).
        """
        ds = forecaster.get_demand_distribution(start_date, end_date)
        return cls(ds["demand"].values.flatten())


# ---------------------------------------------------------------------------
# Shared simulation helpers
# ---------------------------------------------------------------------------


def _draw_demand_samples(demand, n_sims: int) -> np.ndarray:
    """Return ``n_sims`` demand samples from *demand* without assuming normality."""
    if hasattr(demand, "samples"):
        return np.random.choice(demand.samples, n_sims)
    return np.maximum(0, np.random.normal(demand.mean, demand.std, n_sims))


def _profit_curve_axis_range(demand, q_star: int) -> tuple[int, int]:
    """
    Return (start_q, end_q) for the profit curve x-axis.

    Uses sample percentiles when samples are available so the axis correctly
    captures skewed or heavy-tailed posteriors.
    """
    if hasattr(demand, "samples"):
        x_lo = np.percentile(demand.samples, 0.5)
        x_hi = np.percentile(demand.samples, 99.5)
        spread = (x_hi - x_lo) / 2
        central_val = (x_lo + x_hi) / 2
    else:
        central_val = demand.mean
        spread = demand.std * 3

    start_q = 0
    end_q = int(max(q_star * 1.5, central_val + spread))
    return start_q, end_q


def calculate_expected_profit(item, demand, q):
    """
    Helper to calculate profit for a specific Q using simulation.
    """
    n_sims = 2000
    sim_demand = _draw_demand_samples(demand, n_sims)

    sold = np.minimum(q, sim_demand)
    unsold = np.maximum(0, q - sim_demand)

    rev = sold * item.selling_price
    salvage = unsold * item.salvage_value
    cost = q * item.cost_price

    return np.mean(rev + salvage - cost)


def plot_demand_distribution_helper(
    ax, item, demand, q_star, color="blue", label_prefix="item"
):
    """
    Plots the demand distribution (Histogram or PDF) on a given axes.
    """
    if hasattr(demand, "samples"):
        sns.histplot(
            demand.samples, kde=True, ax=ax, stat="density", alpha=0.3, color=color
        )
    else:
        x_min = max(0, demand.mean - 4 * demand.std)
        x_max = demand.mean + 4 * demand.std
        x_range = np.linspace(x_min, x_max, 200)
        y_vals = norm.pdf(x_range, demand.mean, demand.std)
        ax.plot(x_range, y_vals, color=color, lw=2, label=label_prefix)
        ax.fill_between(x_range, y_vals, alpha=0.2, color=color)

    ax.axvline(
        q_star,
        color=color,
        linestyle="--",
        linewidth=2,
        label=f"Order Qty (Q*) = {q_star}",
    )
    ax.set_title(f"Demand vs. Inventory: {item.name}")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Probability Density")
    ax.legend()
    return ax


def plot_profit_curve_helper(
    ax, item, demand, q_star, color="green", label_prefix="item"
):
    """
    Plots the expected profit vs order quantity curve.
    """
    start_q, end_q = _profit_curve_axis_range(demand, q_star)
    q_range = np.unique(np.linspace(start_q, end_q, 75).astype(int))

    # Vectorized simulation — resamples from posterior if available
    n_sims = 2000
    sim_demand = _draw_demand_samples(demand, n_sims)

    # Calculate profit for range of Qs
    profits = []
    for q in q_range:
        sold = np.minimum(q, sim_demand)
        unsold = np.maximum(0, q - sim_demand)
        revenue = sold * item.selling_price
        salvage = unsold * item.salvage_value
        cost = q * item.cost_price
        profits.append(np.mean(revenue + salvage - cost))

    ax.plot(q_range, profits, color=color, lw=2, label=label_prefix)
    ax.axvline(
        q_star,
        color=color,
        linestyle="--",
        alpha=0.5,
        label=f"Allocated: {q_star}",
    )

    max_profit_idx = np.argmax(profits)
    peak_q = q_range[max_profit_idx]
    ax.scatter([peak_q], [profits[max_profit_idx]], color="green", s=30)

    ax.set_title(f"Expected Profit Curve: {item.name}")
    ax.set_xlabel("Order Quantity")
    ax.set_ylabel("Profit")
    ax.legend(fontsize="small")
    return ax
