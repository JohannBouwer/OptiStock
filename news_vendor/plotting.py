import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Dict, Tuple
from .items import Item
from .demand_distribution import DemandDistribution


class NewsvendorVisualizer:
    """
    Visualizes the results of Newsvendor analysis.
    Focuses on the trade-off between risk and reward.
    """

    def __init__(self, style: str = "darkgrid"):
        sns.set_style(style)

    def plot_single_item_analysis(
        self, item: Item, demand: DemandDistribution, quantity: int
    ):
        """
        Generates a dashboard with two plots:
        Demand Distribution with the Order Quantity marked.
        Expected Profit Curve (showing sensitivity to order quantity).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- Plot Demand Distribution & Service Level ---
        self._plot_demand_distribution(ax1, item, demand, quantity)

        # --- Plot  Expected Profit Curve ---
        self._plot_profit_curve(ax2, item, demand, quantity)

        plt.tight_layout()
        plt.show()

    def plot_multi_item_allocation(
        self,
        allocation: Dict[str, int],
        inventory_problems: List[Tuple[Item, DemandDistribution]],
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
            self._plot_demand_distribution(ax_dem, item, demand, q_val)

            # Right Col: Profit Curve
            ax_prof = fig.add_subplot(gs[row_idx, 1])
            self._plot_profit_curve(ax_prof, item, demand, q_val)

        plt.tight_layout()
        plt.show()

    def plot_constrained_allocation(
        self,
        allocation: Dict[str, int],
        inventory_problems: List[Tuple[Item, DemandDistribution]],
        limits: Dict[str, float],
    ):
        """
        Plot multi-item multi-constraint results.

        Layout:
        Row 0: Constraint Gauges (1 col per constraint)
        Row 1: Item Quantities (Spans all cols)
        Row 2...N: Item Details (2 cols: Demand | Profit)
        """
        n_items = len(inventory_problems)
        n_constraints = len(limits)

        total_rows = 2 + n_items
        fig = plt.figure(figsize=(16, 4 + 3 + (4 * n_items)))

        gs = gridspec.GridSpec(
            total_rows, 2, figure=fig, height_ratios=[0.8, 1.0] + [1] * n_items
        )

        # Constraint Usage (Row 0)
        # We manually place these axes to handle dynamic N constraints
        for i, (limit_name, limit_val) in enumerate(limits.items()):
            # Create a subplot in the top slice, dividing width by n_constraints
            ax = fig.add_subplot(gs[0, :])  # Placeholder to get geometry
            # Actually, easier to just use geometry directly:
            ax.remove()
            ax = fig.add_subplot(total_rows, n_constraints, i + 1)

            # Calculate Usage
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
            ax.set_title(f"{limit_name.capitalize()}: {pct_used:.1f}% @ ")

            # Add text label inside bar
            ax.text(
                0,
                used / 2,
                f"{used:.1f}",
                ha="center",
                color="white",
                fontweight="bold",
            )
            ax.text(
                1,
                remaining / 2 if remaining > 0 else 0,
                f"{remaining:.1f}",
                ha="center",
                color="black",
            )

        # Quantities (Row 1)
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

        # Item Details (Row 2+)
        for idx, (item, demand) in enumerate(inventory_problems):
            row_idx = idx + 2
            q_val = allocation.get(item.name, 0)

            # Demand (Left)
            ax_dem = fig.add_subplot(gs[row_idx, 0])
            self._plot_demand_distribution(ax_dem, item, demand, q_val)

            # Profit (Right)
            ax_prof = fig.add_subplot(gs[row_idx, 1])
            self._plot_profit_curve(ax_prof, item, demand, q_val)

        plt.tight_layout()
        plt.show()

    def plot_optimization_summary(
        self,
        allocation: Dict[str, int],
        inventory_problems: List[Tuple[Item, DemandDistribution]],
        lambdas: Dict[str, float] = None,
    ):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])

        # Profit Analysis (Top Left)
        ax_total = fig.add_subplot(gs[0, 0])

        # Calculate Realized vs Unconstrained Profit
        realized_profits = {}
        unconstrained_profits = {}

        for item, demand in inventory_problems:
            q_actual = allocation.get(item.name, 0)
            realized_profits[item.name] = self._calculate_expected_profit(
                item, demand, q_actual
            )

            # Calculate unconstrained optimal Q* for comparison
            # CF = (p-c)/(p-s)
            fractile = item.critical_fractile
            q_opt = max(0, np.ceil(demand.get_quantile(fractile)))
            unconstrained_profits[item.name] = self._calculate_expected_profit(
                item, demand, q_opt
            )

        total_realized = sum(realized_profits.values())
        total_potential = sum(unconstrained_profits.values())
        constraint_cost = total_potential - total_realized

        # Waterfall-style breakdown
        ax_total.bar(
            ["Potential", "Realized"],
            [total_potential, total_realized],
            color=["#95a5a6", "#2ecc71"],
            edgecolor="black",
            alpha=0.7,
        )
        ax_total.set_title(f"Impact of Constraints\nCost: {constraint_cost:,.2f}")
        ax_total.set_ylabel("Expected Profit")

        # Add value labels
        for i, v in enumerate([total_potential, total_realized]):
            ax_total.text(
                i, v, f"{v:,.0f}", ha="center", va="bottom", fontweight="bold"
            )

        # Lagrangian Multipliers (Top Right)
        ax_shadow = fig.add_subplot(gs[0, 1])
        if lambdas:
            names = list(lambdas.keys())
            values = list(lambdas.values())
            # Color code: High shadow price = Bottleneck (Red), Low = Non-binding (Green)
            colors = ["#e74c3c" if v > 0.1 else "#2ecc71" for v in values]

            sns.barplot(
                x=values, y=names, ax=ax_shadow, palette=colors, hue=names, legend=False
            )
            ax_shadow.set_title("Shadow Prices (Marginal Value of Relaxing Constraint)")
            ax_shadow.set_xlabel("Additional Profit per unit of Constraint")
        else:
            ax_shadow.text(
                0.5,
                0.5,
                "No Shadow Prices Available\n(Use Lagrangian or Scipy Solver)",
                ha="center",
                va="center",
                color="gray",
            )
            ax_shadow.axis("off")

        # Per-Item Profit Contribution (Bottom)
        ax_item = fig.add_subplot(gs[1, :])

        items = list(realized_profits.keys())
        # Sort by contribution
        items.sort(key=lambda x: realized_profits[x], reverse=True)
        vals = [realized_profits[k] for k in items]

        sns.barplot(
            x=items, y=vals, ax=ax_item, palette="Blues_d", hue=items, legend=False
        )
        ax_item.set_title("Profit Contribution per Item")
        ax_item.set_ylabel("Expected Profit")

        plt.tight_layout()
        plt.show()

    def _calculate_expected_profit(self, item, demand, q):
        # Helper to calculate profit for a specific Q
        # Uses simulation approximation for robustness
        n_sims = 2000
        if hasattr(demand, "samples"):
            sim_demand = np.random.choice(demand.samples, n_sims)
        else:
            sim_demand = np.random.normal(demand.mean, demand.std_dev, n_sims)
            sim_demand = np.maximum(0, sim_demand)

        sold = np.minimum(q, sim_demand)
        unsold = np.maximum(0, q - sim_demand)

        rev = sold * item.selling_price
        salvage = unsold * item.salvage_value
        cost = q * item.cost_price

        return np.mean(rev + salvage - cost)

    def _plot_demand_distribution(self, ax, item, demand, q_star):
        # Generate range for plotting x-axis (approx +/- 4 std devs or full sample range)
        if hasattr(demand, "samples"):
            # For sampled demand, use actual data range
            x_min, x_max = demand.samples.min(), demand.samples.max()
            x_range = np.linspace(x_min, x_max, 200)
            # Plot Histogram/KDE for sampled data
            sns.histplot(
                demand.samples, kde=True, ax=ax, stat="density", alpha=0.3, color="blue"
            )
        else:
            # For Normal, use mean +/- 4 std
            x_min = max(0, demand.mean - 4 * demand.std_dev)
            x_max = demand.mean + 4 * demand.std_dev
            x_range = np.linspace(x_min, x_max, 200)
            # Plot PDF
            from scipy.stats import norm

            y_vals = norm.pdf(x_range, demand.mean, demand.std_dev)
            ax.plot(x_range, y_vals, color="blue", lw=2)
            ax.fill_between(x_range, y_vals, alpha=0.2, color="blue")

        # Add Vertical line for Order Quantity
        ax.axvline(
            q_star,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Order Qty (Q*) = {q_star}",
        )

        # Annotation
        ax.set_title(f"Demand vs. Inventory: {item.name}")
        ax.set_xlabel("Demand")
        ax.set_ylabel("Probability Density")
        ax.legend()

    def _plot_profit_curve(self, ax, item, demand, q_star):
        # Determine meaningful range for the plot
        # If constrained q_star is low, we still want to see the potential peak (near mean)
        if hasattr(demand, "mean"):
            central_val = demand.mean
            spread = demand.std_dev * 3
        else:
            central_val = np.mean(demand.samples)
            spread = (np.max(demand.samples) - np.min(demand.samples)) / 2

        start_q = 0
        end_q = int(max(q_star * 1.5, central_val + spread))

        # Create range, ensure at least 50 points
        q_range = np.unique(np.linspace(start_q, end_q, 75).astype(int))

        profits = []
        n_sims = 2000

        # Pre-generate demand scenarios
        if hasattr(demand, "samples"):
            sim_demand = np.random.choice(demand.samples, n_sims)
        else:
            sim_demand = np.random.normal(demand.mean, demand.std_dev, n_sims)
            sim_demand = np.maximum(0, sim_demand)

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

        # Mark the peak (Unconstrained Optimal) for comparison
        max_profit_idx = np.argmax(profits)
        peak_q = q_range[max_profit_idx]
        ax.scatter(
            [peak_q], [profits[max_profit_idx]], color="green", s=30, label="Ideal Peak"
        )

        ax.set_title(f"Expected Profit Curve: {item.name}")
        ax.set_xlabel("Order Quantity")
        ax.set_ylabel("Profit")
        ax.legend(fontsize="small")
