import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import norm
from .core import (
    SampledDemand,
    plot_demand_distribution_helper,
    plot_profit_curve_helper,
    calculate_expected_profit,
)


# ---------------------------------------------------------------------------
# Solver convenience helper
# ---------------------------------------------------------------------------


def solver_to_problems(solver) -> list[tuple]:
    """
    Convert a solved ``ForecastSolver`` into ``(Item, SampledDemand)`` pairs
    suitable for all portfolio plot functions.

    Call this after ``solver.solve()`` so that ``solver._demand_matrix`` is
    populated with the posterior demand samples used during optimisation.

    Example
    -------
    ::

        allocation = solver.solve("2025-01-01", "2025-12-31")
        problems   = solver_to_problems(solver)
        fig = plot_constrained_allocation(allocation, problems, solver.limits)
    """
    result = []
    for i, (item, _) in enumerate(solver.problems):
        samples = solver._demand_matrix[i]
        result.append((item, SampledDemand(samples)))
    return result


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_multi_item_allocation(
    allocation: dict[str, int],
    inventory_problems: list[tuple],
    budget: float,
):
    """
    Plot results for multi-item single constraint.
    All items are plotted on shared axes for comparison.
    """
    # Layout:
    # Row 0: Summary (Allocation & Budget)
    # Row 1: All Demand Distributions (Shared Axis)
    # Row 2: All Profit Curves (Shared Axis)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1.5])

    # --- Summary Row (Row 0) ---
    ax_qty = fig.add_subplot(gs[0, 0])
    ax_fund = fig.add_subplot(gs[0, 1])

    names = [p[0].name for p in inventory_problems]
    quantities = [allocation.get(name, 0) for name in names]
    costs = [p[0].cost_price * q for p, q in zip(inventory_problems, quantities)]

    # Use a consistent color palette for items across all plots
    palette = sns.color_palette("husl", len(inventory_problems))
    color_map = {name: color for name, color in zip(names, palette)}

    sns.barplot(
        x=names, y=quantities, ax=ax_qty, palette=palette, hue=names, legend=False
    )
    ax_qty.set_title("Allocated Units per Item")
    ax_qty.set_ylabel("Units")

    if budget:
        total_spend = sum(costs)
        ax_fund.bar(
            ["Used", "Remaining"],
            [total_spend, max(0, budget - total_spend)],
            color=["#e74c3c", "#2ecc71"],
        )
        ax_fund.set_title(f"Budget Utilization ({total_spend:.2f} / {budget:.2f})")
    else:
        sns.barplot(
            x=names, y=costs, ax=ax_fund, palette="magma", hue=names, legend=False
        )
        ax_fund.set_title("Capital Investment per Item")
        ax_fund.set_ylabel("Investment")

    # --- Row 1: Demand Distributions (Shared) ---
    ax_dem = fig.add_subplot(gs[1, :])

    for item, demand in inventory_problems:
        q_val = allocation.get(item.name, 0)
        color = color_map[item.name]
        plot_demand_distribution_helper(
            ax_dem, item, demand, q_val, color=color, label_prefix=item.name
        )

    ax_dem.set_title("Demand Distributions Comparison")
    ax_dem.legend()

    # --- Row 2: Profit Curves (Shared) ---
    ax_prof = fig.add_subplot(gs[2, :])

    for item, demand in inventory_problems:
        q_val = allocation.get(item.name, 0)
        color = color_map[item.name]
        plot_profit_curve_helper(
            ax_prof, item, demand, q_val, color=color, label_prefix=item.name
        )

    ax_prof.set_title("Expected Profit Curves Comparison")
    ax_prof.legend()

    plt.tight_layout()
    return fig


def plot_constrained_allocation(allocation, inventory_problems, limits):
    """
    Plot multi-item multi-constraint results.

    Layout:
    Row 0: Constraint Gauges (1 col per constraint)
    Row 1: Item Quantities (Spans all cols)
    Row 2: Demand Distributions (Shared Axis)
    Row 3: Profit Curves (Shared Axis)
    """
    n_constraints = len(limits)

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(
        4, n_constraints, figure=fig, height_ratios=[0.8, 1.0, 1.5, 1.5]
    )

    # --- Row 0: Constraint Gauges ---
    for i, (limit_name, limit_val) in enumerate(limits.items()):
        ax = fig.add_subplot(gs[0, i])

        used = 0
        for item, _ in inventory_problems:
            q = allocation.get(item.name, 0)
            cost = item.constraints.get(limit_name, 0)
            used += q * cost

        remaining = max(0, limit_val - used)
        pct_used = (used / limit_val) * 100 if limit_val > 0 else 0
        color = "#e74c3c" if used > limit_val else "#2ecc71"

        ax.bar(
            ["Used", "Free"],
            [used, remaining],
            color=[color, "#ececf1"],
            edgecolor="gray",
        )
        ax.set_title(f"{limit_name.capitalize()}: {pct_used:.1f}%")
        ax.text(
            0,
            used / 2 if used > 0 else 0,
            f"{used:.1f}",
            ha="center",
            color="white",
            fontweight="bold",
        )

    # --- Row 1: Quantities ---
    ax_qty = fig.add_subplot(gs[1, :])
    names = [p[0].name for p in inventory_problems]
    quantities = [allocation.get(name, 0) for name in names]

    palette = sns.color_palette("husl", len(inventory_problems))
    item_colors = {name: color for name, color in zip(names, palette)}

    sns.barplot(
        x=names, y=quantities, ax=ax_qty, palette=palette, hue=names, legend=False
    )
    ax_qty.set_title("Optimized Order Quantities (Q*)")
    ax_qty.set_ylabel("Units")
    for i, q in enumerate(quantities):
        ax_qty.text(i, q, f"{q}", ha="center", va="bottom")

    # --- Row 2: Shared Demand Distributions ---
    ax_dem = fig.add_subplot(gs[2, :])

    for item, demand in inventory_problems:
        q_val = allocation.get(item.name, 0)
        color = item_colors[item.name]
        label = f"{item.name}"

        if hasattr(demand, "samples"):
            sns.kdeplot(
                demand.samples,
                ax=ax_dem,
                color=color,
                fill=True,
                alpha=0.1,
                label=label,
            )
        else:
            x_min = max(0, demand.mean - 4 * demand.std)
            x_max = demand.mean + 4 * demand.std
            x_range = np.linspace(x_min, x_max, 200)
            y_vals = norm.pdf(x_range, demand.mean, demand.std)
            ax_dem.plot(x_range, y_vals, color=color, lw=2, label=label)
            ax_dem.fill_between(x_range, y_vals, alpha=0.1, color=color)

        ax_dem.axvline(q_val, color=color, linestyle="--", alpha=0.8)

    ax_dem.set_title("Demand Distributions Comparison")
    ax_dem.set_xlabel("Demand")
    ax_dem.set_ylabel("Probability Density")
    ax_dem.legend()

    # --- Row 3: Shared Profit Curves (delegates to core helper) ---
    ax_prof = fig.add_subplot(gs[3, :])

    for item, demand in inventory_problems:
        q_val = allocation.get(item.name, 0)
        color = item_colors[item.name]
        plot_profit_curve_helper(
            ax_prof, item, demand, q_val, color=color, label_prefix=item.name
        )

    ax_prof.set_title("Expected Profit Curves Comparison")
    ax_prof.set_xlabel("Order Quantity")
    ax_prof.set_ylabel("Profit")
    ax_prof.legend()

    plt.tight_layout()
    return fig


def plot_optimization_summary(allocation, inventory_problems, lambdas=None):
    """
    Visualizes Waterfall chart (Potential vs Realized) and Shadow Prices.

    Works with both parametric demand objects (``NormalDemand``) and
    sample-based objects (``SampledDemand`` or any object with ``.samples``).
    """
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])

    # 1. Waterfall (Impact of Constraints)
    ax_total = fig.add_subplot(gs[0, 0])
    realized_profits = {}
    unconstrained_profits = {}

    for item, demand in inventory_problems:
        q_actual = allocation.get(item.name, 0)
        realized_profits[item.name] = calculate_expected_profit(item, demand, q_actual)

        fractile = item.critical_fractile
        # Use empirical quantile when samples are available; fall back to
        # parametric get_quantile for DemandDistribution objects.
        if hasattr(demand, "samples"):
            q_opt = max(0, int(np.ceil(np.quantile(demand.samples, fractile))))
        else:
            q_opt = max(0, int(np.ceil(demand.get_quantile(fractile))))
        unconstrained_profits[item.name] = calculate_expected_profit(
            item, demand, q_opt
        )

    total_realized = sum(realized_profits.values())
    total_potential = sum(unconstrained_profits.values())
    constraint_cost = total_potential - total_realized

    ax_total.bar(
        ["Potential", "Realized"],
        [total_potential, total_realized],
        color=["#95a5a6", "#2ecc71"],
        edgecolor="black",
        alpha=0.7,
    )
    ax_total.set_title(f"Impact of Constraints\nCost: {constraint_cost:,.2f}")

    # 2. Shadow Prices
    ax_lambdas = fig.add_subplot(gs[0, 1])
    if lambdas:
        names = list(lambdas.keys())
        values = list(lambdas.values())
        colors = ["#e74c3c" if v > 0.1 else "#2ecc71" for v in values]
        sns.barplot(
            x=values, y=names, ax=ax_lambdas, palette=colors, hue=names, legend=False
        )
        ax_lambdas.set_title("Lagrangian Multipliers (Marginal Value)")
    else:
        ax_lambdas.text(0.5, 0.5, "No Lagrangian Multipliers Available", ha="center")
        ax_lambdas.axis("off")

    # 3. Item Contribution
    ax_item = fig.add_subplot(gs[1, :])
    items = list(realized_profits.keys())
    items.sort(key=lambda x: realized_profits[x], reverse=True)
    vals = [realized_profits[k] for k in items]

    sns.barplot(x=items, y=vals, ax=ax_item, palette="Blues_d", hue=items, legend=False)
    ax_item.set_title("Profit Contribution per Item")

    plt.tight_layout()
    return fig
