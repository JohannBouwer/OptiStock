import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_risk_comparison(allocations, problems, n_sims=10000):
    """
    Overlays profit distributions for multiple allocation strategies.
    allocations: Dict[Label, AllocationDict]
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("viridis", len(allocations))
    summary_data = []

    def get_profit_dist(alloc):
        total_profits = np.zeros(n_sims)
        for item, demand_dist in problems:
            q = alloc.get(item.name, 0)

            if hasattr(demand_dist, "samples"):
                d_samp = np.random.choice(demand_dist.samples, n_sims)
            else:
                d_samp = np.random.normal(demand_dist.mean, demand_dist.std, n_sims)
                d_samp = np.maximum(0, d_samp)

            if hasattr(item, "yield_distribution") and item.yield_distribution:
                y_samp = item.yield_distribution.sample(n_sims)
            else:
                y_samp = np.ones(n_sims)

            q_eff = q * y_samp
            sales = np.minimum(q_eff, d_samp)
            leftover = q_eff - sales
            profit = (
                (sales * item.selling_price)
                + (leftover * item.salvage_value)
                - (q * item.cost_price)
            )
            total_profits += profit
        return total_profits

    for idx, (label, allocation) in enumerate(allocations.items()):
        dist = get_profit_dist(allocation)
        color = palette[idx]

        sns.kdeplot(
            dist, fill=True, color=color, label=label, ax=ax, alpha=0.15, linewidth=2
        )

        mean_val = np.mean(dist)
        cvar_5 = np.percentile(dist, 5)

        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.6)
        ax.axvline(cvar_5, color=color, linestyle=":", alpha=0.6)
        summary_data.append([label, f"{mean_val:,.0f}", f"{cvar_5:,.0f}"])

    ax.set_title("Portfolio Profit Distribution: Risk Profile Comparison")
    ax.legend()

    ax.table(
        cellText=summary_data,
        colLabels=["Profile", "Mean Profit", "Worst 5% (CVaR)"],
        loc="bottom",
        cellLoc="center",
        bbox=[0.0, -0.3, 1.0, 0.15 + (0.02 * len(allocations))],
    )

    plt.subplots_adjust(left=0.1, bottom=0.3)
    return fig
