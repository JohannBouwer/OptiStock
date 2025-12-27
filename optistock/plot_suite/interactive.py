import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

from optistock.items import Item
from optistock.distributions.demand_distributions import NormalDemand
from optistock.distributions.yield_distributions import (
    BetaYield,
    PerfectYield,
)
from optistock.plot_suite.single_item import plot_single_item_analysis


def create_item_dashboard(initial_item, initial_demand):
    """
    Creates an interactive Jupyter widget dashboard initialized from an Item and Demand object.

    Args:
        initial_item (Item): The item object to set default costs/prices.
        initial_demand (DemandDistribution): The demand object to set default mean/std.

    Returns:
        widgets.VBox: The dashboard widget. Use display() to render it.
    """

    # --- 1. Extract Defaults from Objects ---
    default_price = initial_item.selling_price
    default_cost = initial_item.cost_price
    default_salvage = initial_item.salvage_value

    # Extract Mean/Std from Demand (Handle Sampled vs Normal)
    if hasattr(initial_demand, "mean"):
        default_mean = initial_demand.mean
    else:
        default_mean = 100.0

    if hasattr(initial_demand, "std_dev"):
        default_std = initial_demand.std_dev
    elif hasattr(initial_demand, "std"):
        default_std = initial_demand.std
    elif hasattr(initial_demand, "samples"):
        default_std = float(np.std(initial_demand.samples))
    else:
        default_std = 20.0

    # Extract Yield Defaults
    default_enable_yield = False
    default_alpha = 95.0
    default_beta = 5.0

    # Check if item has a specific BetaYield distribution attached
    if hasattr(initial_item, "yield_distribution") and isinstance(
        initial_item.yield_distribution, BetaYield
    ):
        default_enable_yield = True
        default_alpha = initial_item.yield_distribution.alpha
        default_beta = initial_item.yield_distribution.beta

    # --- 2. Define Widgets ---
    style = {"description_width": "initial"}
    layout = widgets.Layout(width="95%")

    # Item Economics
    w_price = widgets.FloatText(
        value=default_price, description="Selling Price:", style=style
    )

    # Dynamic slider ranges to accommodate expensive items
    max_cost_slider = max(100.0, default_price * 1.5)
    w_cost = widgets.FloatSlider(
        value=default_cost,
        min=1.0,
        max=max_cost_slider,
        step=1.0,
        description="Cost Price:",
        style=style,
        layout=layout,
    )

    max_salvage_slider = max(50.0, default_cost * 1.5)
    w_salvage = widgets.FloatSlider(
        value=default_salvage,
        min=0.0,
        max=max_salvage_slider,
        step=1.0,
        description="Salvage Value:",
        style=style,
        layout=layout,
    )

    # Demand
    w_mean_demand = widgets.FloatText(
        value=default_mean, description="Mean Demand:", style=style
    )
    w_std_demand = widgets.FloatSlider(
        value=default_std,
        min=0.0,
        max=default_mean,
        step=1.0,
        description="Demand Std Dev:",
        style=style,
        layout=layout,
    )

    # Yield / Supply
    w_enable_yield = widgets.Checkbox(
        value=default_enable_yield, description="Enable Random Yield?"
    )
    w_yield_alpha = widgets.FloatSlider(
        value=default_alpha,
        min=1.0,
        max=100.0,
        description="Yield Alpha:",
        disabled=not default_enable_yield,
        style=style,
        layout=layout,
    )
    w_yield_beta = widgets.FloatSlider(
        value=default_beta,
        min=1.0,
        max=100.0,
        description="Yield Beta:",
        disabled=not default_enable_yield,
        style=style,
        layout=layout,
    )

    # Output Area for Plot
    out = widgets.Output()

    # --- 3. Logic & Update Function ---

    def update_ui_state(change):
        """Enable/Disable yield sliders based on checkbox & update ranges."""
        w_yield_alpha.disabled = not w_enable_yield.value
        w_yield_beta.disabled = not w_enable_yield.value

        # Enforce logical constraints on sliders
        if w_price.value > 1:
            w_cost.max = w_price.value - 0.1
        if w_cost.value > 0:
            w_salvage.max = w_cost.value - 0.1

    w_enable_yield.observe(update_ui_state, names="value")
    w_price.observe(update_ui_state, names="value")
    w_cost.observe(update_ui_state, names="value")

    def render_dashboard(
        price, cost, salvage, d_mean, d_std, enable_yield, y_alpha, y_beta
    ):
        # 1. Reconstruct Objects with Current Slider Values
        if enable_yield:
            yield_dist = BetaYield(y_alpha, y_beta)
            mean_yield = y_alpha / (y_alpha + y_beta)
        else:
            yield_dist = PerfectYield()
            mean_yield = 1.0

        # Ensure logical cost constraints (UI safety)
        if cost >= price:
            cost = price - 0.01
        if salvage >= cost:
            salvage = cost - 0.01

        # Create temporary objects for calculation/plotting
        temp_item = Item(
            initial_item.name, cost, price, salvage, yield_distribution=yield_dist
        )

        # Use NormalDemand for interactive 'What-If' exploration (sliders manipulate parameters)
        temp_demand = NormalDemand(d_mean, d_std)

        # 2. Heuristic Solver (Fast Analytic Approximation for UI)
        fractile = temp_item.critical_fractile

        # Base Q* (Demand only)
        if d_std > 0:
            q_star_demand = norm.ppf(fractile, loc=d_mean, scale=d_std)
        else:
            q_star_demand = d_mean

        # Yield Safety Buffer Heuristic: Q = Demand / Yield
        # (Faster than running full Monte Carlo for every slider drag)
        q_star = int(math.ceil(q_star_demand / max(mean_yield, 0.01)))

        # 3. Visualization
        with out:
            clear_output(wait=True)

            # Metrics
            print(f"Optimal Order (Q*): {q_star} units")
            print(f"Critical Fractile:  {fractile:.2%}")
            print(f"Margin (Underage):  {temp_item.underage_cost:.2f}")
            print(f"Risk (Overage):     {temp_item.overage_cost:.2f}")
            if enable_yield:
                print(f"Avg Yield Rate:     {mean_yield:.1%}")

            # Plot
            fig = plot_single_item_analysis(temp_item, temp_demand, q_star)

            if fig:
                plt.show(fig)
                plt.close(fig)  # Prevent memory leaks in notebook

    # --- 4. Assemble Layout ---

    ui = widgets.VBox(
        [
            widgets.HTML(f"Analysis: {initial_item.name}</h3>"),
            widgets.HBox([w_price, w_cost, w_salvage]),
            widgets.HBox([w_mean_demand, w_std_demand]),
            widgets.HBox([w_enable_yield, w_yield_alpha, w_yield_beta]),
            out,
        ]
    )

    # Link widgets to function (keep reference to avoid garbage collection)
    _interactive_plot = widgets.interactive_output(
        render_dashboard,
        {
            "price": w_price,
            "cost": w_cost,
            "salvage": w_salvage,
            "d_mean": w_mean_demand,
            "d_std": w_std_demand,
            "enable_yield": w_enable_yield,
            "y_alpha": w_yield_alpha,
            "y_beta": w_yield_beta,
        },
    )

    # Trigger initial update
    update_ui_state(None)

    return ui
