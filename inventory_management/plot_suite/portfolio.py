import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from .core import (
    plot_demand_distribution_helper,
    plot_profit_curve_helper,
    calculate_expected_profit,
)


def plot_multi_item_allocation(
    allocation,
    inventory_problems,
    budget: float,
):
    """
    Plot results for multi-item single constraint.
    """
    n_items = len(inventory_problems)
    # Layout: 1 Summary Row + 1 Row per Item
    total_rows = 1 + n_items

    # Make the figure tall enough to accommodate all items
    fig = plt.figure(figsize=(16, 5 + (4 * n_items)))
    gs = gridspec.GridSpec(
        total_rows, 2, figure=fig, height_ratios=[1.2] + [1] * n_items
    )

    # --- Summary Row (Row 0) ---
    ax_qty = fig.add_subplot(gs[0, 0])
    ax_fund = fig.add_subplot(gs[0, 1])

    names = [p[0].name for p in inventory_problems]
    quantities = [allocation.get(name, 0) for name in names]
    costs = [p[0].cost_price * q for p, q in zip(inventory_problems, quantities)]

    sns.barplot(
        x=names, y=quantities, ax=ax_qty, palette="viridis", hue=names, legend=False
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

    # --- Item Rows (Row 1 to N) ---
    for idx, (item, demand) in enumerate(inventory_problems):
        row_idx = idx + 1

        # Left Col: Demand
        ax_dem = fig.add_subplot(gs[row_idx, 0])
        q_val = allocation.get(item.name, 0)
        plot_demand_distribution_helper(ax_dem, item, demand, q_val)

        # Right Col: Profit Curve
        ax_prof = fig.add_subplot(gs[row_idx, 1])
        plot_profit_curve_helper(ax_prof, item, demand, q_val)

    plt.tight_layout()

    return fig


def plot_constrained_allocation(allocation, inventory_problems, limits):
    """
    Plot multi-item multi-constraint results (Gauges + Item Details).
    """
    n_items = len(inventory_problems)
    n_constraints = len(limits)
    total_rows = 2 + n_items

    fig = plt.figure(figsize=(16, 4 + 3 + (4 * n_items)))
    gs = gridspec.GridSpec(
        total_rows, 2, figure=fig, height_ratios=[0.8, 1.0] + [1] * n_items
    )

    # --- Row 0: Constraint Gauges ---
    for i, (limit_name, limit_val) in enumerate(limits.items()):
        ax = fig.add_subplot(total_rows, n_constraints, i + 1)

        used = 0
        for item, _ in inventory_problems:
            q = allocation.get(item.name, 0)
            cost = item.constraints.get(limit_name, 0)
            used += q * cost

        remaining = max(0, limit_val - used)
        pct_used = (used / limit_val) * 100
        color = "#e74c3c" if used > limit_val else "#2ecc71"

        ax.bar(
            ["Used", "Free"],
            [used, remaining],
            color=[color, "#ececf1"],
            edgecolor="gray",
        )
        ax.set_title(f"{limit_name.capitalize()}: {pct_used:.1f}%")
        ax.text(
            0, used / 2, f"{used:.1f}", ha="center", color="white", fontweight="bold"
        )

    # --- Row 1: Quantities ---
    ax_qty = fig.add_subplot(gs[1, :])
    names = [p[0].name for p in inventory_problems]
    quantities = [allocation.get(name, 0) for name in names]

    sns.barplot(
        x=names, y=quantities, ax=ax_qty, palette="viridis", hue=names, legend=False
    )
    ax_qty.set_title("Optimized Order Quantities (Q*)")
    ax_qty.set_ylabel("Units")
    for i, q in enumerate(quantities):
        ax_qty.text(i, q, f"{q}", ha="center", va="bottom")

    # --- Row 2+: Item Details ---
    for idx, (item, demand) in enumerate(inventory_problems):
        row_idx = idx + 2
        q_val = allocation.get(item.name, 0)

        ax_dem = fig.add_subplot(gs[row_idx, 0])
        plot_demand_distribution_helper(ax_dem, item, demand, q_val)

        ax_prof = fig.add_subplot(gs[row_idx, 1])
        plot_profit_curve_helper(ax_prof, item, demand, q_val)

    plt.tight_layout()
    return fig


def plot_optimization_summary(allocation, inventory_problems, lambdas=None):
    """
    Visualizes Waterfall chart (Potential vs Realized) and Shadow Prices.
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
        q_opt = max(0, np.ceil(demand.get_quantile(fractile)))
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
        ax_lambdas.set_title("Shadow Prices (Marginal Value)")
    else:
        ax_lambdas.text(0.5, 0.5, "No Shadow Prices Available", ha="center")
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
