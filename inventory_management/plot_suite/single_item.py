import matplotlib.pyplot as plt
from .core import plot_demand_distribution_helper, plot_profit_curve_helper


def plot_single_item_analysis(item, demand, quantity):
    """
    Generates dashboard: Demand Dist + Profit Curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plot_demand_distribution_helper(ax1, item, demand, quantity)
    plot_profit_curve_helper(ax2, item, demand, quantity)

    plt.tight_layout()
    return fig
