import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .core import (
    SampledDemand,
    plot_demand_distribution_helper,
    plot_profit_curve_helper,
)


def plot_single_item_analysis(item, demand, quantity):
    """
    Generates dashboard: Demand Dist + Profit Curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plot_demand_distribution_helper(ax1, item, demand, quantity)
    plot_profit_curve_helper(ax2, item, demand, quantity)

    plt.tight_layout()
    return fig


def plot_forecast_with_allocation(item, forecaster, quantity, start_date, end_date):
    """
    3-panel dashboard combining the forecaster's time-series output with
    the newsvendor decision for the planning horizon.

    Panels
    ------
    Left   : Forecast plot (calls ``forecaster.plot_forecast()``)
    Centre : Demand distribution over [start_date, end_date] with Q* line
    Right  : Expected profit curve with Q* marked

    Parameters
    ----------
    item : Item
        The inventory item (cost/price/salvage).
    forecaster : BaseForecaster
        A fitted forecaster that has already had ``.forecast()`` called.
    quantity : int
        The optimal order quantity Q* (e.g. from ``ForecastSolver.solve()``).
    start_date, end_date : str
        Planning horizon used to aggregate the posterior demand distribution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    demand = SampledDemand.from_forecaster(forecaster, start_date, end_date)

    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- Panel 1: Time-series forecast ---
    # forecaster.plot_forecast() returns (fig, ax); we copy the content onto
    # our axis by re-rendering via the forecaster's internal idata.
    ax_forecast = fig.add_subplot(gs[0, 0])
    try:
        import arviz as az
        import numpy as np

        fc_idata = forecaster.forecast_idata
        # Try standard predictions group first, fall back to other names
        if hasattr(fc_idata, "predictions"):
            y_samples = fc_idata.predictions["y"]
        elif hasattr(fc_idata, "posterior_predictive"):
            y_samples = fc_idata.posterior_predictive["y"]
        else:
            raise AttributeError("Cannot locate forecast samples in forecast_idata.")

        # Flatten chain/draw dims
        flat = y_samples.values.reshape(-1, y_samples.shape[-1])
        mean_fc = flat.mean(axis=0)
        hdi = az.hdi(y_samples, hdi_prob=0.94)["y"].values  # (time, 2)

        time_idx = range(len(mean_fc))
        ax_forecast.plot(time_idx, mean_fc, color="#2980b9", lw=2, label="Forecast mean")
        ax_forecast.fill_between(
            time_idx,
            hdi[:, 0],
            hdi[:, 1],
            alpha=0.25,
            color="#2980b9",
            label="94% HDI",
        )
        ax_forecast.axvspan(
            0, len(mean_fc) - 1, alpha=0.05, color="orange", label="Planning horizon"
        )
        ax_forecast.set_title(f"Forecast: {item.name}\n({start_date} → {end_date})")
        ax_forecast.set_xlabel("Period")
        ax_forecast.set_ylabel("Demand")
        ax_forecast.legend(fontsize="small")
    except Exception:
        # Graceful fallback: call the forecaster's own plot method and show a note
        ax_forecast.text(
            0.5,
            0.5,
            "Call forecaster.plot_forecast()\nfor the time-series view.",
            ha="center",
            va="center",
            transform=ax_forecast.transAxes,
            fontsize=10,
        )
        ax_forecast.set_title(f"Forecast: {item.name}")
        ax_forecast.axis("off")

    # --- Panel 2: Demand distribution over horizon ---
    ax_dist = fig.add_subplot(gs[0, 1])
    plot_demand_distribution_helper(ax_dist, item, demand, quantity)

    # --- Panel 3: Profit curve ---
    ax_profit = fig.add_subplot(gs[0, 2])
    plot_profit_curve_helper(ax_profit, item, demand, quantity)

    return fig
