import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Dict, Tuple
from .solvers import Solver

class NewsvendorVisualizer:
    """
    Visualizes the results of Newsvendor analysis.
    Focuses on the trade-off between risk and reward.
    """
    
    def __init__(self, style: str = 'darkgrid'):
        sns.set_style(style)

    def plot_single_item_analysis(self, solver: Solver):
        """
        Generates a dashboard with two plots:
        1. Demand Distribution with the Order Quantity marked.
        2. Expected Profit Curve (showing sensitivity to order quantity).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Plot Demand Distribution & Service Level ---
        self._plot_demand_distribution(ax1, solver.item, solver.demand_distribution, solver.quantity)
        
        # --- Plot  Expected Profit Curve ---
        self._plot_profit_curve(ax2, solver.item, solver.demand_distribution, solver.quantity)
        
        plt.tight_layout()
        plt.show()

    def plot_multi_item_allocation(self, solver: Solver):
        n_items = len(solver.problems)
        # Layout: 1 Summary Row + 1 Row per Item
        total_rows = 1 + n_items
        
        # Make the figure tall enough to accommodate all items
        fig = plt.figure(figsize=(16, 5 + (4 * n_items)))
        gs = gridspec.GridSpec(total_rows, 2, figure=fig, height_ratios=[1.2] + [1]*n_items)

        # --- Summary Row (Row 0) ---
        ax_qty = fig.add_subplot(gs[0, 0])
        ax_fund = fig.add_subplot(gs[0, 1])
        
        names = [p[0].name for p in solver.problems]
        quantities = [solver.allocation.get(name, 0) for name in names]
        costs = [p[0].cost_price * q for p, q, in zip(solver.problems, quantities)]
        
        sns.barplot(x=names, y=quantities, ax=ax_qty, palette="viridis", hue=names, legend=False)
        ax_qty.set_title("Allocated Units per Item")
        ax_qty.set_ylabel("Units")

        if solver.budget:
            total_spend = sum(costs)
            ax_fund.bar(["Used", "Remaining"], [total_spend, max(0, solver.budget - total_spend)], color=["#e74c3c", "#2ecc71"])
            ax_fund.set_title(f"Budget Utilization ({total_spend:.2f} / {solver.budget:.2f})")
        else:
            sns.barplot(x=names, y=costs, ax=ax_fund, palette="magma", hue=names, legend=False)
            ax_fund.set_title("Capital Investment per Item")
            ax_fund.set_ylabel("Investment")

        # --- Item Rows (Row 1 to N) ---
        for idx, (item, demand) in enumerate(solver.problems):
            row_idx = idx + 1
            
            # Left Col: Demand
            ax_dem = fig.add_subplot(gs[row_idx, 0])
            q_val = solver.allocation.get(item.name, 0)
            self._plot_demand_distribution(ax_dem, item, demand, q_val)
            
            # Right Col: Profit Curve
            ax_prof = fig.add_subplot(gs[row_idx, 1])
            self._plot_profit_curve(ax_prof, item, demand, q_val)

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
        # Determine meaningful range for the plot
        # If constrained q_star is low, we still want to see the potential peak (near mean)
        if hasattr(demand, 'mean'):
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
        if hasattr(demand, 'samples'):
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

        ax.plot(q_range, profits, color='green', lw=2)
        ax.axvline(q_star, color='black', linestyle='--', alpha=0.5, label=f'Allocated: {q_star}')
        
        # Mark the peak (Unconstrained Optimal) for comparison
        max_profit_idx = np.argmax(profits)
        peak_q = q_range[max_profit_idx]
        ax.scatter([peak_q], [profits[max_profit_idx]], color='green', s=30, label='Ideal Peak')

        ax.set_title(f"Expected Profit Curve: {item.name}")
        ax.set_xlabel("Order Quantity")
        ax.set_ylabel("Profit")
        ax.legend(fontsize='small')