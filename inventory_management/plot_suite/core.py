import numpy as np
import seaborn as sns
from scipy.stats import norm

"""
File of helper functions shared by all plotting functions
"""


def setup_style(style: str = "darkgrid"):
    sns.set_style(style)


def calculate_expected_profit(item, demand, q):
    """
    Helper to calculate profit for a specific Q using simulation.
    """
    n_sims = 2000
    if hasattr(demand, "samples"):
        sim_demand = np.random.choice(demand.samples, n_sims)
    else:
        sim_demand = np.random.normal(demand.mean, demand.std, n_sims)
        sim_demand = np.maximum(0, sim_demand)

    sold = np.minimum(q, sim_demand)
    unsold = np.maximum(0, q - sim_demand)

    rev = sold * item.selling_price
    salvage = unsold * item.salvage_value
    cost = q * item.cost_price

    return np.mean(rev + salvage - cost)


def plot_demand_distribution_helper(ax, item, demand, q_star):
    """
    Plots the demand distribution (Histogram or PDF) on a given axes.
    """
    if hasattr(demand, "samples"):
        x_min, x_max = demand.samples.min(), demand.samples.max()
        x_range = np.linspace(x_min, x_max, 200)
        sns.histplot(
            demand.samples, kde=True, ax=ax, stat="density", alpha=0.3, color="blue"
        )
    else:
        x_min = max(0, demand.mean - 4 * demand.std)
        x_max = demand.mean + 4 * demand.std
        x_range = np.linspace(x_min, x_max, 200)
        y_vals = norm.pdf(x_range, demand.mean, demand.std)
        ax.plot(x_range, y_vals, color="blue", lw=2)
        ax.fill_between(x_range, y_vals, alpha=0.2, color="blue")

    ax.axvline(
        q_star,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Order Qty (Q*) = {q_star}",
    )
    ax.set_title(f"Demand vs. Inventory: {item.name}")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Probability Density")
    ax.legend()
    return ax


def plot_profit_curve_helper(ax, item, demand, q_star):
    """
    Plots the expected profit vs order quantity curve.
    """
    if hasattr(demand, "mean"):
        central_val = demand.mean
        spread = demand.std * 3
    else:
        central_val = np.mean(demand.samples)
        spread = (np.max(demand.samples) - np.min(demand.samples)) / 2

    start_q = 0
    end_q = int(max(q_star * 1.5, central_val + spread))
    q_range = np.unique(np.linspace(start_q, end_q, 75).astype(int))

    profits = []
    # Vectorized simulation for curve generation
    n_sims = 2000
    if hasattr(demand, "samples"):
        sim_demand = np.random.choice(demand.samples, n_sims)
    else:
        sim_demand = np.random.normal(demand.mean, demand.std, n_sims)
        sim_demand = np.maximum(0, sim_demand)

    # Calculate profit for range of Qs
    for q in q_range:
        sold = np.minimum(q, sim_demand)
        unsold = np.maximum(0, q - sim_demand)
        revenue = sold * item.selling_price
        salvage = unsold * item.salvage_value
        cost = q * item.cost_price
        profits.append(np.mean(revenue + salvage - cost))

    ax.plot(q_range, profits, color="green", lw=2)
    ax.axvline(
        q_star,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Allocated: {q_star}",
    )

    max_profit_idx = np.argmax(profits)
    peak_q = q_range[max_profit_idx]
    ax.scatter(
        [peak_q], [profits[max_profit_idx]], color="green", s=30, label="Ideal Peak"
    )

    ax.set_title(f"Expected Profit Curve: {item.name}")
    ax.set_xlabel("Order Quantity")
    ax.set_ylabel("Profit")
    ax.legend(fontsize="small")
    return ax
