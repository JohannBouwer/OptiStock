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
    
    def __init__(self, style: str = 'darkgrid'):
        sns.set_style(style)

    def plot_single_item_analysis(self, item: Item, demand: DemandDistribution, optimal_q: int):
        """
        Generates a dashboard with two plots:
        1. Demand Distribution with the Order Quantity marked.
        2. Expected Profit Curve (showing sensitivity to order quantity).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Plot Demand Distribution & Service Level ---
        self._plot_demand_distribution(ax1, item, demand, optimal_q)
        
        # --- Plot  Expected Profit Curve ---
        self._plot_profit_curve(ax2, item, demand, optimal_q)
        
        plt.tight_layout()
        plt.show()

    def plot_multi_item_allocation(self, inventory_problems: List[Tuple[Item, DemandDistribution]], allocation: Dict[str, int],  budget: float = None):
        n_items = len(inventory_problems)
        # Calculate rows: 1 summary row + enough rows for items (2 items per row)
        item_rows = int(np.ceil(n_items / 2))
        total_rows = 1 + item_rows
        
        fig = plt.figure(figsize=(16, 5 * total_rows))
        gs = gridspec.GridSpec(total_rows, 2, figure=fig, height_ratios=[1.5] + [1]*item_rows)

        # --- Summary Row (Row 0) ---
        ax_qty = fig.add_subplot(gs[0, 0])
        ax_fund = fig.add_subplot(gs[0, 1])
        
        names = [p[0].name for p in inventory_problems]
        quantities = [allocation.get(name, 0) for name in names]
        costs = [p[0].cost_price * q for p, q, in zip(inventory_problems, quantities)]
        
        # Plot 1: Quantities
        sns.barplot(x=names, y=quantities, ax=ax_qty, palette="viridis", hue=names, legend=False)
        ax_qty.set_title("Allocated Units per Item")
        ax_qty.set_ylabel("Units")

        # Plot 2: Budget/Investment
        if budget:
            total_spend = sum(costs)
            ax_fund.bar(["Used", "Remaining"], [total_spend, max(0, budget - total_spend)], color=["#e74c3c", "#2ecc71"])
            ax_fund.set_title(f"Budget Utilization (${total_spend:.2f} / ${budget:.2f})")
        else:
            sns.barplot(x=names, y=costs, ax=ax_fund, palette="magma", hue=names, legend=False)
            ax_fund.set_title("Capital Investment per Item")
            ax_fund.set_ylabel("Investment ($)")

        # --- Item Demand Rows (Row 1+) ---
        for idx, (item, demand) in enumerate(inventory_problems):
            row_idx = 1 + (idx // 2)
            col_idx = idx % 2
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            q_val = allocation.get(item.name, 0)
            self._plot_demand_distribution(ax, item, demand, q_val)

        plt.tight_layout()
        plt.show()

    def _plot_demand_distribution(self, ax, item, demand, q_star):
        # Generate range for plotting x-axis (approx +/- 4 std devs or full sample range)
        if hasattr(demand, 'samples'):
            # For sampled demand, use actual data range
            x_min, x_max = demand.samples.min(), demand.samples.max()
            x_range = np.linspace(x_min, x_max, 200)
            # Plot Histogram/KDE for sampled data
            sns.histplot(demand.samples, kde=True, ax=ax, stat="density", alpha=0.3, color="blue")
        else:
            # For Normal, use mean +/- 4 std
            x_min = max(0, demand.mean - 4 * demand.std_dev)
            x_max = demand.mean + 4 * demand.std_dev
            x_range = np.linspace(x_min, x_max, 200)
            # Plot PDF
            from scipy.stats import norm
            y_vals = norm.pdf(x_range, demand.mean, demand.std_dev)
            ax.plot(x_range, y_vals, color='blue', lw=2)
            ax.fill_between(x_range, y_vals, alpha=0.2, color='blue')

        # Add Vertical line for Order Quantity
        ax.axvline(q_star, color='red', linestyle='--', linewidth=2, label=f'Order Qty (Q*) = {q_star}')
        
        # Annotation
        ax.set_title(f"Demand vs. Inventory: {item.name}")
        ax.set_xlabel("Demand")
        ax.set_ylabel("Probability Density")
        ax.legend()
        
    def _plot_profit_curve(self, ax, item, demand, q_star):
        # Create a range of hypothetical order quantities to test sensitivity
        # We look at 50% to 150% of the optimal quantity
        q_range = np.unique(np.linspace(max(1, q_star * 0.5), q_star * 1.5, 50).astype(int))
        profits = []

        # Use simulation for robustness (works for both Normal and Sampled)
        # We simulate 5000 demand scenarios per quantity step
        if hasattr(demand, 'samples'):
            sim_demand = np.random.choice(demand.samples, 5000)
        else:
            sim_demand = np.random.normal(demand.mean, demand.std_dev, 5000)
            sim_demand = np.maximum(0, sim_demand) # Demand can't be negative

        for q in q_range:
            # Vectorized profit calc:
            sold = np.minimum(q, sim_demand)
            unsold = np.maximum(0, q - sim_demand)
            revenue = sold * item.selling_price
            salvage = unsold * item.salvage_value
            cost = q * item.cost_price
            
            avg_profit = np.mean(revenue + salvage - cost)
            profits.append(avg_profit)

        # Plot
        ax.plot(q_range, profits, color='green', lw=2)
        ax.axvline(q_star, color='red', linestyle='--', alpha=0.5)
        ax.scatter([q_star], [max(profits)], color='red', zorder=5)
        
        ax.set_title(f"Expected Profit Curve: {item.name}")
        ax.set_xlabel("Order Quantity")
        ax.set_ylabel("Expected Profit ($)")