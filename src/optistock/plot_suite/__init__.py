from .core import setup_style, SampledDemand
from .single_item import plot_single_item_analysis, plot_forecast_with_allocation
from .portfolio import (
    plot_multi_item_allocation,
    plot_constrained_allocation,
    plot_optimization_summary,
    solver_to_problems,
)
from .risk import plot_risk_comparison
from .interactive import create_item_dashboard

setup_style()

__all__ = [
    # Adapter
    "SampledDemand",
    # Single-item
    "plot_single_item_analysis",
    "plot_forecast_with_allocation",
    # Portfolio
    "plot_multi_item_allocation",
    "plot_constrained_allocation",
    "plot_optimization_summary",
    "solver_to_problems",
    # Risk
    "plot_risk_comparison",
    # Interactive
    "create_item_dashboard",
]
