from __future__ import annotations

"""
Inventory planning orchestrators: train models → forecast demand → optimise supply.

Each class represents a specific replenishment policy:

    PeriodicOrderUpTo (R, S)
        Profit-optimising periodic review.  Order enough to cover
        lead_time + review_period days of demand.

    PeriodicBaseStock (R, S) with service target
        Same horizon as PeriodicOrderUpTo, but with per-item cycle-service-level
        floors.  Items listed in ``service_targets`` are constrained to meet
        the given CSL; others are profit-optimised.

    ContinuousFixedQuantity (s, Q)
        Continuous review.  Triggers an order of fixed quantity Q whenever
        the inventory position falls to or below reorder point s.  Returns a
        distribution of days-to-first-reorder across posterior scenarios.

    ContinuousOrderUpTo (s, S)
        Continuous review.  Like ContinuousFixedQuantity but the order
        quantity is S − inventory_position rather than fixed Q.

    StockKeep (deprecated)
        Alias for PeriodicOrderUpTo; emits DeprecationWarning.  Retained
        for one release to allow gradual migration.
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .forecasting.base import BaseForecaster, ErrorEstimations
from .forecasting.linear_regressors import BayesTimeSeries
from .forecasting.state_space import UnivariateSSM
from .forecasting.mix_media_models import MediaMixModel
from .items import Item
from .solvers import ForecastSolver
from .distributions.yield_distributions import YieldDistribution, PerfectYield
from .plot_suite.single_item import plot_single_item_analysis
from .plot_suite.portfolio import (
    plot_multi_item_allocation,
    plot_constrained_allocation,
    plot_optimization_summary as _plot_optimization_summary,
)
from .plot_suite.risk import plot_risk_comparison


class _DemandSamples:
    """Thin adapter that wraps a 1-D numpy array of demand samples into the
    interface expected by the plot-suite helpers (.samples, .mean, .std,
    .get_quantile)."""

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=float)

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    @property
    def std(self) -> float:
        return float(self.samples.std())

    def get_quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))


# ---------------------------------------------------------------------------
# Base class (internal — not exported)
# ---------------------------------------------------------------------------

class BaseStockKeep:
    """
    Internal engine shared by all policy subclasses.

    Not intended for direct instantiation.  Subclasses implement the three
    policy hooks:

    * ``_planning_horizon(item)``     — how many days to forecast per item
    * ``_solver_lower_bound(item, demand_samples)`` — minimum order floor
    * ``_get_service_targets()``      — per-item CSL targets for the solver
    """

    # Core columns that should *not* be treated as constraint coefficients
    _CORE_COLS = {"name", "cost_price", "selling_price", "salvage_value", "lead_time"}

    def __init__(
        self,
        histories: pd.DataFrame,
        item_configs: pd.DataFrame,
        forecaster_class: type[BaseForecaster] = BayesTimeSeries,
        forecaster_kwargs: dict[str, Any] | None = None,
        yield_profiles: dict[str, YieldDistribution] | None = None,
        target: str = "sales",
        date_col: str = "date",
        item_col: str = "item",
    ):
        self.histories = histories.copy()
        self._item_df = item_configs.copy()
        self.forecaster_class = forecaster_class
        self.forecaster_kwargs = forecaster_kwargs or {}
        self.target = target
        self.date_col = date_col
        self.item_col = item_col

        self.items: list[Item] = self._create_items(yield_profiles or {})

        self.trained_forecasters: dict[str, BaseForecaster] = {}
        self.holdout_data: dict[str, tuple[pd.DataFrame, str]] = {}
        self.allocation: dict[str, int] | None = None
        self.net_allocation: dict[str, int] | None = None
        self.solver: ForecastSolver | None = None
        self._run_start: str | None = None
        self._run_end: str | None = None
        self._mode: str | None = None
        self._current_days: int = 0  # set at the start of each _run_pipeline call

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _planning_horizon(self, item: Item) -> int:
        """Days to forecast for *item* (lead_time + review period, for example)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _planning_horizon(item)."
        )

    def _solver_lower_bound(self, item: Item, demand_samples: np.ndarray) -> float:
        """Minimum order quantity floor for *item* (0 = unconstrained)."""
        return 0.0

    def _get_service_targets(self) -> dict[str, float]:
        """Per-item service-level targets forwarded to ForecastSolver."""
        return {}

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------

    def run_holdout(
        self,
        holdout_days: int,
        inventory_state: dict[str, dict[str, int]] | None = None,
        events: dict[str, dict[str, list]] | None = None,
        objective: str = "SAA",
        limits: dict[str, float] | None = None,
        cvar_alpha: float = 0.10,
        cvar_lambda: float = 0.50,
        risk_aversion: float = 0.0,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Validation / holdout mode.

        Splits each item's history into train (all but the last ``holdout_days``
        rows) and holdout, fits the forecaster on the training portion, forecasts
        the holdout window, solves for optimal stock quantities, and evaluates
        the decision against actual holdout demand.

        Parameters
        ----------
        holdout_days : int
            Number of calendar days to hold out from the end of each item's history.
        inventory_state : dict, optional
            ``{item_name: {"on_hand": int, "on_order": int}}``.  Missing items
            default to zero for both.
        events : dict, optional
            ``{item_name: {event_name: [date_str, ...]}}``.
        objective : {'SAA', 'CVaR', 'Utility'}
        limits : dict, optional
            Shared resource constraints, e.g. ``{"budget": 50_000}``.
        cvar_alpha, cvar_lambda, risk_aversion
            Objective-specific parameters; see ForecastSolver docs.
        fit_kwargs : dict, optional
            Extra keyword arguments forwarded to ``forecaster.fit()``.

        Returns
        -------
        dict
            ``allocation``, ``net_allocation``, ``metrics``,
            ``solver_summary``, ``period``, ``mode``.
        """
        return self._run_pipeline(
            mode="holdout",
            days=holdout_days,
            inventory_state=inventory_state,
            events=events or {},
            objective=objective,
            limits=limits,
            cvar_alpha=cvar_alpha,
            cvar_lambda=cvar_lambda,
            risk_aversion=risk_aversion,
            fit_kwargs=fit_kwargs or {},
        )

    def run(
        self,
        forecast_days: int,
        inventory_state: dict[str, dict[str, int]] | None = None,
        events: dict[str, dict[str, list]] | None = None,
        objective: str = "SAA",
        limits: dict[str, float] | None = None,
        cvar_alpha: float = 0.10,
        cvar_lambda: float = 0.50,
        risk_aversion: float = 0.0,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Production mode.

        Fits on the full history for each item and forecasts the planning
        horizon determined by ``_planning_horizon(item)`` for each item.
        The ``forecast_days`` parameter is stored and used as a fallback by
        deprecated subclasses; for all standard subclasses the horizon is
        fully determined by the policy parameters.

        Parameters
        ----------
        forecast_days : int
            Kept for API compatibility.  Periodic subclasses ignore this
            in favour of ``lead_time + review_period``; pass any positive int.
        inventory_state : dict, optional
            ``{item_name: {"on_hand": int, "on_order": int}}``.
        events, objective, limits, cvar_alpha, cvar_lambda, risk_aversion, fit_kwargs
            Same as :meth:`run_holdout`.

        Returns
        -------
        dict
            ``allocation``, ``net_allocation``, ``solver_summary``,
            ``period``, ``mode``.
        """
        return self._run_pipeline(
            mode="production",
            days=forecast_days,
            inventory_state=inventory_state,
            events=events or {},
            objective=objective,
            limits=limits,
            cvar_alpha=cvar_alpha,
            cvar_lambda=cvar_lambda,
            risk_aversion=risk_aversion,
            fit_kwargs=fit_kwargs or {},
        )

    # ------------------------------------------------------------------
    # Public accessors / plots
    # ------------------------------------------------------------------

    def plot_forecast(self, item_name: str) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the forecast for a single item, overlaid with holdout actuals
        when in holdout mode.  Must be called after ``run_holdout`` or ``run``.
        """
        if not self.trained_forecasters:
            raise RuntimeError("Call run_holdout() or run() before plot_forecast().")
        if item_name not in self.trained_forecasters:
            raise ValueError(
                f"No trained forecaster for '{item_name}'. "
                f"Available: {list(self.trained_forecasters)}"
            )

        forecaster = self.trained_forecasters[item_name]
        fig, ax = forecaster.plot_forecast()

        if self._mode == "holdout" and item_name in self.holdout_data:
            holdout_df, target = self.holdout_data[item_name]
            actual = holdout_df[target]
            dates = (
                pd.to_datetime(holdout_df[self.date_col])
                if self.date_col in holdout_df.columns
                else holdout_df.index
            )
            ax.scatter(
                dates,
                actual.values,
                color="black",
                marker="x",
                s=25,
                zorder=5,
                label="Actual (holdout)",
            )
            ax.set_title(f"Forecast Validation — {item_name}")
            ax.legend(loc="upper left")

        return fig, ax

    def summary(self) -> dict:
        """Return ForecastSolver's diagnostic summary for the last run."""
        if self.solver is None:
            raise RuntimeError("Call run_holdout() or run() first.")
        return self.solver.summary()

    def plot_allocation(self) -> plt.Figure:
        """Visualise the current allocation against the posterior demand distributions."""
        self._check_solved()
        plot_problems = self._build_plot_problems()
        if self.solver.limits:
            return plot_constrained_allocation(
                self.allocation, plot_problems, self.solver.limits
            )
        budget = next(iter(self.solver.limits.values()), None) if self.solver.limits else None
        return plot_multi_item_allocation(self.allocation, plot_problems, budget)

    def plot_item(self, item_name: str) -> plt.Figure:
        """Single-item demand distribution and profit-curve dashboard."""
        self._check_solved()
        plot_problems = self._build_plot_problems()
        match = [(item, dem) for item, dem in plot_problems if item.name == item_name]
        if not match:
            raise ValueError(
                f"No solved data for '{item_name}'. "
                f"Available: {[it.name for it, _ in plot_problems]}"
            )
        item, demand = match[0]
        qty = self.allocation[item_name]
        return plot_single_item_analysis(item, demand, qty)

    def plot_risk(
        self, allocations: dict[str, dict[str, int]] | None = None
    ) -> plt.Figure:
        """Overlay profit-distribution curves for one or more allocation strategies."""
        self._check_solved()
        if allocations is None:
            allocations = {"Current": self.allocation}
        plot_problems = self._build_plot_problems()
        return plot_risk_comparison(allocations, plot_problems)

    def plot_optimization_summary(self) -> plt.Figure:
        """Waterfall chart (potential vs realised profit) and shadow-price bar chart."""
        self._check_solved()
        plot_problems = self._build_plot_problems()
        shadow = self.solver.shadow_prices if self.solver.shadow_prices else None
        return _plot_optimization_summary(self.allocation, plot_problems, lambdas=shadow)

    # ------------------------------------------------------------------
    # Private pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        mode: str,
        days: int,
        inventory_state: dict | None,
        events: dict,
        objective: str,
        limits: dict | None,
        cvar_alpha: float,
        cvar_lambda: float,
        risk_aversion: float,
        fit_kwargs: dict,
    ) -> dict[str, Any]:
        """Shared pipeline for run_holdout and run (periodic policies)."""
        self._current_days = days  # available to subclasses via _planning_horizon

        problems: list[tuple[Item, BaseForecaster]] = []
        holdout_actuals: list[tuple[Item, float]] = []
        daily_means: dict[str, np.ndarray] = {}

        start_dts: list[pd.Timestamp] = []
        end_dts: list[pd.Timestamp] = []
        item_horizon_ends: dict[str, pd.Timestamp] = {}

        for item in self.items:
            df_item = self.histories[self.histories[self.item_col] == item.name].copy()

            if df_item.empty:
                warnings.warn(
                    f"No history found for '{item.name}' — skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            df_item[self.date_col] = pd.to_datetime(df_item[self.date_col])

            horizon = self._planning_horizon(item)

            if mode == "holdout":
                train_df, holdout_df, split_date, max_date = self._date_prep(df_item, days)
                start_dt = split_date
                opt_end = start_dt + pd.Timedelta(days=horizon - 1)
                end_dt = min(opt_end, max_date)
            else:  # production
                train_df = df_item.copy()
                holdout_df = None
                max_date = df_item[self.date_col].max()
                start_dt = max_date + pd.Timedelta(days=1)
                end_dt = start_dt + pd.Timedelta(days=horizon - 1)

            start_dts.append(start_dt)
            end_dts.append(end_dt)
            item_horizon_ends[item.name] = start_dt + pd.Timedelta(days=horizon - 1)

            forecaster = self._build_forecaster(train_df)
            self._fit_and_forecast(
                forecaster,
                train_df,
                start_dt,
                end_dt,
                events.get(item.name),
                fit_kwargs,
            )
            self.trained_forecasters[item.name] = forecaster

            if mode == "holdout":
                self.holdout_data[item.name] = (holdout_df, self.target)
                holdout_actuals.append((item, holdout_df[self.target].sum()))
                dm = self._extract_daily_forecast_mean(forecaster)
                if dm is not None:
                    daily_means[item.name] = dm

            problems.append((item, forecaster))

        if not problems:
            raise RuntimeError(
                "No items had sufficient history to fit. Check your 'histories' DataFrame "
                f"and 'item_col' setting (currently '{self.item_col}')."
            )

        global_start = max(start_dts)
        global_end = min(end_dts)
        start_str = str(global_start.date())
        end_str = str(global_end.date())

        # Warn when mixed lead times cause horizon clamping
        clamped = [
            name for name, end in item_horizon_ends.items() if end > global_end
        ]
        if clamped:
            warnings.warn(
                f"Effective horizon was clamped to {global_end.date()} for items: "
                f"{clamped}. Consider running items with different lead times separately.",
                UserWarning,
                stacklevel=2,
            )

        solver = ForecastSolver(
            problems,
            objective=objective,
            limits=limits,
            cvar_alpha=cvar_alpha,
            cvar_lambda=cvar_lambda,
            risk_aversion=risk_aversion,
        )
        demand_by_item = solver.pull_demand(start_str, end_str)

        lower_bounds = {
            item.name: self._solver_lower_bound(item, demand_by_item[item.name])
            for item in self.items
            if item.name in demand_by_item
        }
        solver.lower_bounds = lower_bounds
        solver.service_targets = self._get_service_targets()
        allocation = solver.optimize()

        net_allocation = {
            name: self._net_order(self._item_by_name(name), qty, inventory_state)
            for name, qty in allocation.items()
        }

        self.solver = solver
        self.allocation = allocation
        self.net_allocation = net_allocation
        self._run_start = start_str
        self._run_end = end_str
        self._mode = mode

        result: dict[str, Any] = {
            "allocation": allocation,
            "net_allocation": net_allocation,
            "solver_summary": solver.summary(),
            "period": (start_str, end_str),
            "mode": mode,
        }

        if mode == "holdout":
            result["metrics"] = self._calculate_metrics(
                allocation, holdout_actuals, daily_means or None
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _net_order(
        self, item: Item, qty: int, inventory_state: dict | None
    ) -> int:
        """Gross quantity minus current inventory position (clamped to ≥ 0)."""
        state = (inventory_state or {}).get(item.name, {})
        on_hand = int(state.get("on_hand", 0))
        on_order = int(state.get("on_order", 0))
        return max(0, qty - on_hand - on_order)

    def _item_by_name(self, name: str) -> Item:
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(f"Item '{name}' not found.")

    def _check_solved(self) -> None:
        if self.solver is None or self.allocation is None:
            raise RuntimeError("Call run_holdout() or run() before plotting.")

    def _build_plot_problems(self) -> list[tuple[Item, "_DemandSamples"]]:
        assert self.solver is not None and self.solver._demand_matrix is not None
        return [
            (item, _DemandSamples(self.solver._demand_matrix[i]))
            for i, (item, _) in enumerate(self.solver.problems)
        ]

    def _create_items(self, yield_profiles: dict[str, YieldDistribution]) -> list[Item]:
        items = []
        for _, row in self._item_df.iterrows():
            name = row["name"]
            cp = float(row["cost_price"])
            sp = float(row["selling_price"])
            sv = float(row["salvage_value"]) if "salvage_value" in self._item_df.columns else 0.0
            lt = int(row["lead_time"]) if "lead_time" in self._item_df.columns else 0
            yd = yield_profiles.get(name, PerfectYield())

            constraints = {
                col: float(row[col])
                for col in self._item_df.columns
                if col not in self._CORE_COLS and pd.notnull(row[col])
            }

            items.append(Item(
                name=name,
                cost_price=cp,
                selling_price=sp,
                salvage_value=sv,
                Lead_time=lt,
                constraints=constraints,
                yield_distribution=yd,
            ))

        return items

    def _build_forecaster(self, train_df: pd.DataFrame) -> BaseForecaster:
        if issubclass(self.forecaster_class, UnivariateSSM):
            df = train_df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.set_index(self.date_col)
            return self.forecaster_class(df, **self.forecaster_kwargs)
        return self.forecaster_class(train_df, **self.forecaster_kwargs)

    def _fit_and_forecast(
        self,
        forecaster: BaseForecaster,
        train_df: pd.DataFrame,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        events_for_item: dict | None,
        fit_kwargs: dict,
    ) -> None:
        if events_for_item and hasattr(forecaster, "create_events"):
            forecaster.create_events(events_for_item, date_col=self.date_col)

        if isinstance(forecaster, UnivariateSSM):
            fk = dict(fit_kwargs)
            build_kwargs = fk.pop("build_model_kwargs", {})
            if forecaster.model is None:
                forecaster.build_model(**build_kwargs)
            forecaster.fit(**fk)
        else:
            forecaster.fit(target=self.target, date_col=self.date_col, **fit_kwargs)

        n_days = (end_dt - start_dt).days + 1
        if isinstance(forecaster, UnivariateSSM):
            forecaster.forecast(periods=n_days)
        elif isinstance(forecaster, MediaMixModel):
            df_future = pd.DataFrame(
                {self.date_col: pd.date_range(start=start_dt, end=end_dt)}
            )
            forecaster.forecast(df_future=df_future)
        else:
            df_future = pd.DataFrame(
                {self.date_col: pd.date_range(start=start_dt, end=end_dt)}
            )
            forecaster.forecast(
                scenario={"df_future": df_future, "date_col": self.date_col}
            )

    def _extract_daily_forecast_mean(
        self, forecaster: BaseForecaster
    ) -> np.ndarray | None:
        """Per-period posterior mean in original (unscaled) units, or None."""
        try:
            if isinstance(forecaster, UnivariateSSM):
                obs = forecaster.forecast_idata["forecast_observed"].isel(observed_state=0)
                return obs.stack(sample=["chain", "draw"]).mean(dim="sample").values

            if isinstance(forecaster, MediaMixModel):
                if forecaster.predictions is not None:
                    return forecaster.predictions.mean(dim="sample").values

            if (
                hasattr(forecaster, "forecast_idata")
                and forecaster.forecast_idata is not None
                and "y" in forecaster.forecast_idata.predictions
            ):
                scale = getattr(forecaster, "max_scaler", 1.0) or 1.0
                return (
                    forecaster.forecast_idata.predictions["y"]
                    .mean(dim=["chain", "draw"])
                    .values
                    * scale
                )
        except Exception:
            pass
        return None

    def _extract_daily_forecast_samples(
        self, forecaster: BaseForecaster
    ) -> np.ndarray | None:
        """
        Per-period posterior samples in original (unscaled) units.

        Returns
        -------
        np.ndarray of shape (n_periods, n_draws) or None when the forecaster
        type is unrecognised or the attribute is missing.
        """
        try:
            if isinstance(forecaster, UnivariateSSM):
                obs = forecaster.forecast_idata["forecast_observed"].isel(observed_state=0)
                scale = getattr(forecaster, "max_scaler", 1.0) or 1.0
                stacked = obs.stack(sample=["chain", "draw"])
                # dims after stack: (time, sample) → transpose to (n_periods, n_draws)
                return stacked.values.T * scale

            if isinstance(forecaster, MediaMixModel):
                if forecaster.predictions is not None:
                    # dims: (sample, date) → transpose to (n_periods, n_draws)
                    return forecaster.predictions.values.T

            if (
                hasattr(forecaster, "forecast_idata")
                and forecaster.forecast_idata is not None
                and "y" in forecaster.forecast_idata.predictions
            ):
                scale = getattr(forecaster, "max_scaler", 1.0) or 1.0
                samples = forecaster.forecast_idata.predictions["y"]
                # dims: (chain, draw, time) → stack → (time, sample) → transpose
                stacked = samples.stack(sample=["chain", "draw"])
                return stacked.values.T * scale
        except Exception:
            pass
        return None

    def _calculate_metrics(
        self,
        allocation: dict[str, int],
        actuals: list[tuple[Item, float]],
        daily_forecast_means: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {}
        total_profit = 0.0

        for item, actual_demand in actuals:
            qty = allocation.get(item.name, 0)

            units_sold = min(qty, actual_demand)
            units_leftover = max(0, qty - actual_demand)
            units_short = max(0, actual_demand - qty)

            revenue = units_sold * item.selling_price
            salvage = units_leftover * item.salvage_value
            cost = qty * item.cost_price
            profit = revenue + salvage - cost

            entry: dict[str, Any] = {
                "profit": round(profit, 2),
                "service_level": round(units_sold / actual_demand, 3) if actual_demand > 0 else 1.0,
                "stockout": units_short > 0,
                "leftover_units": int(units_leftover),
            }

            if daily_forecast_means is not None and item.name in daily_forecast_means:
                holdout_df, target = self.holdout_data[item.name]
                actual_series = holdout_df[target].values
                forecast_series = daily_forecast_means[item.name]
                n = min(len(actual_series), len(forecast_series))
                if n > 0:
                    entry["smape_pct"] = round(
                        ErrorEstimations.calculate_smape(
                            actual_series[:n], forecast_series[:n]
                        ),
                        2,
                    )

            report[item.name] = entry
            total_profit += profit

        report["portfolio_total_profit"] = round(total_profit, 2)
        return report

    def _date_prep(
        self, df: pd.DataFrame, days: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        max_date = df[self.date_col].max()
        split_date = max_date - pd.Timedelta(days=days - 1)

        train_df = df[df[self.date_col] < split_date].copy()
        holdout_df = df[df[self.date_col] >= split_date].copy()
        return train_df, holdout_df, split_date, max_date


# ---------------------------------------------------------------------------
# Periodic review policies
# ---------------------------------------------------------------------------

class PeriodicOrderUpTo(BaseStockKeep):
    """
    Periodic-review order-up-to policy (R, S) — profit optimising.

    On each review (every ``review_period`` days), order enough to cover
    the protection interval (lead_time + review_period) days of expected demand.
    The solver maximises expected profit (or CVaR / Utility) with no mandatory
    service-level floor.

    Parameters
    ----------
    histories, item_configs, forecaster_class, forecaster_kwargs,
    yield_profiles, target, date_col, item_col
        See :class:`BaseStockKeep`.
    review_period : int
        Number of days between replenishment reviews (≥ 1).
    """

    def __init__(
        self,
        histories: pd.DataFrame,
        item_configs: pd.DataFrame,
        review_period: int,
        forecaster_class: type[BaseForecaster] = BayesTimeSeries,
        forecaster_kwargs: dict[str, Any] | None = None,
        yield_profiles: dict[str, YieldDistribution] | None = None,
        target: str = "sales",
        date_col: str = "date",
        item_col: str = "item",
    ):
        if review_period < 1:
            raise ValueError("review_period must be ≥ 1.")
        super().__init__(
            histories=histories,
            item_configs=item_configs,
            forecaster_class=forecaster_class,
            forecaster_kwargs=forecaster_kwargs,
            yield_profiles=yield_profiles,
            target=target,
            date_col=date_col,
            item_col=item_col,
        )
        self.review_period = review_period

    def _planning_horizon(self, item: Item) -> int:
        return item.Lead_time + self.review_period


class PeriodicBaseStock(PeriodicOrderUpTo):
    """
    Periodic-review order-up-to policy (R, S) with cycle-service-level targets.

    Same horizon as :class:`PeriodicOrderUpTo`.  Items listed in
    ``service_targets`` have their order quantity floored to the given CSL
    quantile of the planning-horizon demand distribution; the solver further
    optimises within those bounds.  Items not in the dict fall back to pure
    profit optimisation.

    When resource constraints (``limits``) are active, the solver still runs
    for all items together even when some have a service target, because budget
    allocation across items must remain jointly optimal.

    Parameters
    ----------
    service_targets : dict[str, float], optional
        ``{item_name: csl_target}`` where ``csl_target`` is in ``(0, 1)``.
        Items not in the dict get no service floor.
    """

    def __init__(
        self,
        histories: pd.DataFrame,
        item_configs: pd.DataFrame,
        review_period: int,
        service_targets: dict[str, float] | None = None,
        forecaster_class: type[BaseForecaster] = BayesTimeSeries,
        forecaster_kwargs: dict[str, Any] | None = None,
        yield_profiles: dict[str, YieldDistribution] | None = None,
        target: str = "sales",
        date_col: str = "date",
        item_col: str = "item",
    ):
        super().__init__(
            histories=histories,
            item_configs=item_configs,
            review_period=review_period,
            forecaster_class=forecaster_class,
            forecaster_kwargs=forecaster_kwargs,
            yield_profiles=yield_profiles,
            target=target,
            date_col=date_col,
            item_col=item_col,
        )
        self.service_targets: dict[str, float] = service_targets or {}

    def _solver_lower_bound(self, item: Item, demand_samples: np.ndarray) -> float:
        sl = self.service_targets.get(item.name, 0.0)
        if sl <= 0.0 or demand_samples.size == 0:
            return 0.0
        return float(np.quantile(demand_samples, sl))

    def _get_service_targets(self) -> dict[str, float]:
        return self.service_targets


# ---------------------------------------------------------------------------
# Continuous review policies (simulation-based)
# ---------------------------------------------------------------------------

class ContinuousFixedQuantity(BaseStockKeep):
    """
    Continuous-review fixed-order-quantity policy (s, Q).

    Monitors inventory continuously and places an order of fixed size ``Q``
    whenever the inventory position falls to or below reorder point ``s``.

    :meth:`run` simulates demand depletion across posterior scenarios and
    returns the distribution of the first day a stockout occurs, giving
    probabilistic insight into how long current stock will last.

    Parameters
    ----------
    Q : dict[str, int]
        Fixed order quantity per item (required, no default).
    reorder_points : dict[str, float], optional
        Reorder point ``s`` per item.  When omitted, ``s`` is computed as
        the ``service_level`` quantile of lead-time demand from the posterior.
    service_level : float
        Quantile used to compute ``s`` automatically.  Default 0.95.
    """

    def __init__(
        self,
        histories: pd.DataFrame,
        item_configs: pd.DataFrame,
        Q: dict[str, int],
        reorder_points: dict[str, float] | None = None,
        service_level: float = 0.95,
        forecaster_class: type[BaseForecaster] = BayesTimeSeries,
        forecaster_kwargs: dict[str, Any] | None = None,
        yield_profiles: dict[str, YieldDistribution] | None = None,
        target: str = "sales",
        date_col: str = "date",
        item_col: str = "item",
    ):
        super().__init__(
            histories=histories,
            item_configs=item_configs,
            forecaster_class=forecaster_class,
            forecaster_kwargs=forecaster_kwargs,
            yield_profiles=yield_profiles,
            target=target,
            date_col=date_col,
            item_col=item_col,
        )
        self.Q = Q
        self.reorder_points = reorder_points or {}
        self.service_level = service_level
        self._simulation_results: dict[str, Any] = {}

    def _planning_horizon(self, item: Item) -> int:
        # Not used by run() / run_holdout() in this subclass, but required by the base.
        return item.Lead_time

    def run(  # type: ignore[override]
        self,
        forecast_days: int,
        inventory_state: dict[str, dict[str, int]] | None = None,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fit forecasters and simulate reorder timing for each item.

        Parameters
        ----------
        forecast_days : int
            Number of days to simulate forward.
        inventory_state : dict, optional
            ``{item_name: {"on_hand": int, "on_order": int}}``.
        fit_kwargs : dict, optional
            Forwarded to ``forecaster.fit()``.

        Returns
        -------
        dict
            ``stockout_days``, ``period``, ``mode``.
        """
        return self._run_continuous(
            mode="production",
            forecast_days=forecast_days,
            inventory_state=inventory_state,
            fit_kwargs=fit_kwargs or {},
            train_split_days=None,
        )

    def run_holdout(  # type: ignore[override]
        self,
        holdout_days: int,
        inventory_state: dict[str, dict[str, int]] | None = None,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fit on the training portion and simulate over the holdout window.

        Parameters
        ----------
        holdout_days : int
            Number of days to hold out (used as both the train split and
            the simulation window length).
        inventory_state : dict, optional
            ``{item_name: {"on_hand": int, "on_order": int}}``.
        fit_kwargs : dict, optional
            Forwarded to ``forecaster.fit()``.

        Returns
        -------
        dict
            ``stockout_days``, ``period``, ``mode``.
        """
        return self._run_continuous(
            mode="holdout",
            forecast_days=holdout_days,
            inventory_state=inventory_state,
            fit_kwargs=fit_kwargs or {},
            train_split_days=holdout_days,
        )

    # ------------------------------------------------------------------
    # Public analysis API
    # ------------------------------------------------------------------

    def report(self, item_name: str, hdi_prob: float = 0.90) -> str:
        """
        Human-readable HDI statement about when a stockout will occur.

        Parameters
        ----------
        item_name : str
        hdi_prob : float
            Width of the highest-density interval.  Default 0.90.

        Returns
        -------
        str
        """
        self._require_results(item_name)
        res = self._simulation_results[item_name]
        sd = res["stockout_days"]
        horizon = res["n_periods"]
        stockout_mask = sd <= horizon
        pct = 100.0 * float(np.mean(stockout_mask))
        pct_label = f"{int(round(hdi_prob * 100))}%"

        if not stockout_mask.any():
            return (
                f"'{item_name}': No stockout in any of the {len(sd):,} posterior "
                f"scenarios within the {horizon}-day horizon."
            )

        lo, hi = self._hdi(sd[stockout_mask], hdi_prob)
        return (
            f"'{item_name}': The {pct_label} HDI says a stockout will occur between "
            f"day {lo:.0f} and day {hi:.0f}  "
            f"({pct:.1f}% of scenarios result in a stockout within {horizon} days)."
        )

    def plot_stockout_distribution(
        self, item_name: str, hdi_prob: float = 0.90
    ) -> plt.Figure:
        """
        Histogram of the first-stockout-day distribution with HDI shading.

        Parameters
        ----------
        item_name : str
        hdi_prob : float
            Width of the HDI band drawn on the plot.  Default 0.90.
        """
        self._require_results(item_name)
        res = self._simulation_results[item_name]
        sd = res["stockout_days"]
        horizon = res["n_periods"]
        finite = sd[sd <= horizon]
        pct_so = 100.0 * float(np.mean(sd <= horizon))
        pct_label = f"{int(round(hdi_prob * 100))}%"

        fig, ax = plt.subplots(figsize=(9, 4))

        if len(finite) == 0:
            ax.text(
                0.5, 0.5, "No stockouts in any posterior scenario",
                transform=ax.transAxes, ha="center", va="center", fontsize=13,
            )
        else:
            n_bins = min(30, max(10, int(horizon / 2)))
            n_total = len(sd)
            ax.hist(
                finite, bins=n_bins, edgecolor="white", alpha=0.75,
                color="steelblue", label="Stockout scenarios",
                weights=np.ones(len(finite)) / n_total,
            )
            lo, hi = self._hdi(finite, hdi_prob)
            ax.axvspan(lo, hi, alpha=0.20, color="red",
                       label=f"{pct_label} HDI: day {lo:.0f}–{hi:.0f}")
            med = float(np.median(finite))
            ax.axvline(med, color="red", linestyle="--", linewidth=1.5,
                       label=f"Median: day {med:.0f}")
            ax.legend(fontsize=9)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y * 100:.1f}%")
            )

        ax.set_xlabel("Day of first stockout")
        ax.set_ylabel("Probability")
        ax.set_title(
            f"Stockout timing — {item_name}\n"
            f"({pct_so:.1f}% of scenarios stock out within {horizon} days)"
        )
        return fig

    def recommended_order_day(
        self, item_name: str, risk_tolerance: float = 0.10
    ) -> dict[str, Any]:
        """
        Day by which to place the next order to cap stockout risk.

        The recommended day is ``quantile(stockout_days, risk_tolerance) − lead_time``,
        so that only ``risk_tolerance`` fraction of posterior scenarios stock out
        before the replenishment arrives.

        Parameters
        ----------
        item_name : str
        risk_tolerance : float
            Acceptable probability of stocking out before delivery.  Default 0.10.

        Returns
        -------
        dict
            ``order_day``, ``order_quantity``, ``lead_time``, ``risk_tolerance``,
            ``action`` (human-readable string).
        """
        self._require_results(item_name)
        res = self._simulation_results[item_name]
        sd = res["stockout_days"]
        lead_time = self._item_by_name(item_name).Lead_time
        stockout_quantile = float(np.quantile(sd, risk_tolerance))
        order_day = max(1, int(stockout_quantile) - lead_time)
        qty = int(self.Q.get(item_name, 0))
        return {
            "order_day": order_day,
            "order_quantity": qty,
            "lead_time": lead_time,
            "risk_tolerance": risk_tolerance,
            "action": (
                f"Order {qty:,} units of '{item_name}' by day {order_day} "
                f"(lead time {lead_time}d, "
                f"{int(round(risk_tolerance * 100))}% stockout risk tolerance)."
            ),
        }

    # Solver-based plots are not applicable to continuous policies
    def plot_allocation(self) -> plt.Figure:  # type: ignore[override]
        raise NotImplementedError(
            "plot_allocation() is not applicable to continuous review policies. "
            "Use plot_stockout_distribution() instead."
        )

    def plot_item(self, item_name: str) -> plt.Figure:  # type: ignore[override]
        raise NotImplementedError(
            "plot_item() is not applicable to continuous review policies. "
            "Use plot_stockout_distribution() instead."
        )

    def plot_risk(self, allocations=None) -> plt.Figure:  # type: ignore[override]
        raise NotImplementedError(
            "plot_risk() is not applicable to continuous review policies."
        )

    def plot_optimization_summary(self) -> plt.Figure:  # type: ignore[override]
        raise NotImplementedError(
            "plot_optimization_summary() is not applicable to continuous review policies."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_results(self, item_name: str) -> None:
        if not self._simulation_results or item_name not in self._simulation_results:
            raise RuntimeError(
                f"No simulation results for '{item_name}'. Call run() first."
            )

    @staticmethod
    def _hdi(arr: np.ndarray, prob: float) -> tuple[float, float]:
        """Shortest interval containing ``prob`` fraction of the mass."""
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        interval_idx = int(np.floor(prob * n))
        if interval_idx >= n:
            return float(sorted_arr[0]), float(sorted_arr[-1])
        widths = sorted_arr[interval_idx:] - sorted_arr[: n - interval_idx]
        min_idx = int(np.argmin(widths))
        return float(sorted_arr[min_idx]), float(sorted_arr[min_idx + interval_idx])

    # ------------------------------------------------------------------
    # Internal simulation engine
    # ------------------------------------------------------------------

    def _run_continuous(
        self,
        mode: str,
        forecast_days: int,
        inventory_state: dict | None,
        fit_kwargs: dict,
        train_split_days: int | None,
    ) -> dict[str, Any]:
        """Shared simulation body for run() and run_holdout()."""
        self._simulation_results = {}
        self._mode = mode
        inv_state = inventory_state or {}

        run_start: pd.Timestamp | None = None
        run_end: pd.Timestamp | None = None

        for item in self.items:
            df_item = self.histories[self.histories[self.item_col] == item.name].copy()
            if df_item.empty:
                warnings.warn(
                    f"No history found for '{item.name}' — skipping.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            df_item[self.date_col] = pd.to_datetime(df_item[self.date_col])

            if train_split_days is not None:
                train_df, _, split_date, max_date = self._date_prep(df_item, train_split_days)
                start_dt = split_date
            else:
                train_df = df_item.copy()
                max_date = df_item[self.date_col].max()
                start_dt = max_date + pd.Timedelta(days=1)

            end_dt = start_dt + pd.Timedelta(days=forecast_days - 1)

            if run_start is None or start_dt > run_start:
                run_start = start_dt
            if run_end is None or end_dt < run_end:
                run_end = end_dt

            forecaster = self._build_forecaster(train_df)
            self._fit_and_forecast(
                forecaster, train_df, start_dt, end_dt, None, fit_kwargs
            )
            self.trained_forecasters[item.name] = forecaster

            daily_samples = self._extract_daily_forecast_samples(forecaster)
            if daily_samples is None:
                warnings.warn(
                    f"Cannot extract daily forecast samples for '{item.name}'. "
                    "Skipping simulation.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            n_periods, n_draws = daily_samples.shape

            # Reorder point s (used for the policy's automatic replenishment logic)
            s = self.reorder_points.get(item.name)
            if s is None:
                lt = item.Lead_time
                if lt > 0 and n_periods >= lt:
                    lt_demand = daily_samples[:lt, :].sum(axis=0)
                else:
                    lt_demand = daily_samples.sum(axis=0)
                s = float(np.quantile(lt_demand, self.service_level))

            q_order = self._order_quantity(item, s)

            state = inv_state.get(item.name, {})
            on_hand = int(state.get("on_hand", 0))
            on_order = int(state.get("on_order", 0))
            starting_inv = on_hand + on_order

            stockout_days = self._simulate(
                daily_samples, s, q_order, starting_inv, n_periods, n_draws
            )

            self._simulation_results[item.name] = {
                "stockout_days": stockout_days,
                "reorder_point": s,
                "n_periods": n_periods,
                "starting_inv": starting_inv,
            }

        self._run_start = str(run_start.date()) if run_start else None
        self._run_end = str(run_end.date()) if run_end else None

        return {
            "stockout_days": {
                name: res["stockout_days"]
                for name, res in self._simulation_results.items()
            },
            "period": (self._run_start, self._run_end),
            "mode": mode,
        }

    def _order_quantity(self, item: Item, s: float) -> int:
        """Order quantity when the reorder point is triggered."""
        if item.name not in self.Q:
            raise KeyError(
                f"No fixed order quantity Q defined for item '{item.name}'. "
                "Pass it in the Q dict."
            )
        return int(self.Q[item.name])

    @staticmethod
    def _simulate(
        daily_samples: np.ndarray,
        s: float,
        q_order: int,
        starting_inv: float,
        n_periods: int,
        n_draws: int,
    ) -> np.ndarray:
        """
        Simulate WITH the policy's reorder logic across all posterior draws.

        Returns ``stockout_days``: day of the first stockout per draw
        (``n_periods + 1`` sentinel when no stockout occurs in the horizon).
        """
        stockout_days = np.full(n_draws, n_periods + 1, dtype=float)

        for d in range(n_draws):
            inv = float(starting_inv)
            reorder_triggered = False
            for t in range(n_periods):
                inv -= daily_samples[t, d]
                if inv < 0 and stockout_days[d] > n_periods:
                    stockout_days[d] = float(t + 1)
                if inv <= s and not reorder_triggered:
                    reorder_triggered = True
                    inv += q_order

        return stockout_days


class ContinuousOrderUpTo(ContinuousFixedQuantity):
    """
    Continuous-review order-up-to policy (s, S).

    Like :class:`ContinuousFixedQuantity` but when the reorder point is
    triggered the order quantity is ``S − inventory_position`` rather than a
    fixed ``Q``.

    Parameters
    ----------
    S : dict[str, float]
        Order-up-to level per item (required).
    reorder_points : dict[str, float], optional
        Reorder point ``s`` per item.  Computed from the ``service_level``
        quantile of lead-time demand when omitted.
    service_level : float
        Quantile used to compute ``s`` automatically.  Default 0.95.
    """

    def __init__(
        self,
        histories: pd.DataFrame,
        item_configs: pd.DataFrame,
        S: dict[str, float],
        reorder_points: dict[str, float] | None = None,
        service_level: float = 0.95,
        forecaster_class: type[BaseForecaster] = BayesTimeSeries,
        forecaster_kwargs: dict[str, Any] | None = None,
        yield_profiles: dict[str, YieldDistribution] | None = None,
        target: str = "sales",
        date_col: str = "date",
        item_col: str = "item",
    ):
        # Pass an empty Q dict — order quantity is computed dynamically
        super().__init__(
            histories=histories,
            item_configs=item_configs,
            Q={},
            reorder_points=reorder_points,
            service_level=service_level,
            forecaster_class=forecaster_class,
            forecaster_kwargs=forecaster_kwargs,
            yield_profiles=yield_profiles,
            target=target,
            date_col=date_col,
            item_col=item_col,
        )
        self.S = S

    def _order_quantity(self, item: Item, s: float) -> int:
        # Returns the order-up-to level S; _simulate uses it as the ceiling.
        if item.name not in self.S:
            raise KeyError(
                f"No order-up-to level S defined for item '{item.name}'. "
                "Pass it in the S dict."
            )
        return int(self.S[item.name])

    @staticmethod
    def _simulate(  # type: ignore[override]
        daily_samples: np.ndarray,
        s: float,
        q_order: int,  # q_order is the S level
        starting_inv: float,
        n_periods: int,
        n_draws: int,
    ) -> np.ndarray:
        """
        Simulate WITH order-up-to logic; ``q_order`` is the target level S.

        Returns ``stockout_days`` (first stockout per draw, sentinel n_periods+1).
        """
        S_level = float(q_order)
        stockout_days = np.full(n_draws, n_periods + 1, dtype=float)

        for d in range(n_draws):
            inv = float(starting_inv)
            reorder_triggered = False
            for t in range(n_periods):
                inv -= daily_samples[t, d]
                if inv < 0 and stockout_days[d] > n_periods:
                    stockout_days[d] = float(t + 1)
                if inv <= s and not reorder_triggered:
                    reorder_triggered = True
                    inv += max(0.0, S_level - inv)

        return stockout_days

    def recommended_order_day(
        self, item_name: str, risk_tolerance: float = 0.10
    ) -> dict[str, Any]:
        """
        Day by which to place a top-up order to S to cap stockout risk.

        Parameters
        ----------
        item_name : str
        risk_tolerance : float
            Acceptable stockout probability before delivery.  Default 0.10.

        Returns
        -------
        dict
            ``order_day``, ``order_up_to``, ``lead_time``, ``risk_tolerance``,
            ``action`` (human-readable string).
        """
        self._require_results(item_name)
        res = self._simulation_results[item_name]
        sd = res["stockout_days"]
        lead_time = self._item_by_name(item_name).Lead_time
        stockout_quantile = float(np.quantile(sd, risk_tolerance))
        order_day = max(1, int(stockout_quantile) - lead_time)
        S = int(self.S.get(item_name, 0))
        return {
            "order_day": order_day,
            "order_up_to": S,
            "lead_time": lead_time,
            "risk_tolerance": risk_tolerance,
            "action": (
                f"Top up '{item_name}' to {S:,} units by day {order_day} "
                f"(lead time {lead_time}d, "
                f"{int(round(risk_tolerance * 100))}% stockout risk tolerance)."
            ),
        }


# ---------------------------------------------------------------------------
# Deprecated alias
# ---------------------------------------------------------------------------

class StockKeep(PeriodicOrderUpTo):
    """
    Deprecated.  Use :class:`PeriodicOrderUpTo`, :class:`PeriodicBaseStock`,
    :class:`ContinuousFixedQuantity`, or :class:`ContinuousOrderUpTo` instead.

    .. deprecated::
        Will be removed in v0.2.0.
    """

    def __init__(
        self,
        histories: pd.DataFrame,
        item_configs: pd.DataFrame,
        forecaster_class: type[BaseForecaster] = BayesTimeSeries,
        forecaster_kwargs: dict[str, Any] | None = None,
        yield_profiles: dict[str, YieldDistribution] | None = None,
        target: str = "sales",
        date_col: str = "date",
        item_col: str = "item",
        policies=None,
    ):
        warnings.warn(
            "StockKeep is deprecated and will be removed in v0.2.0. "
            "Use PeriodicOrderUpTo, PeriodicBaseStock, ContinuousFixedQuantity, "
            "or ContinuousOrderUpTo instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # review_period=0 is overridden by _planning_horizon below
        super().__init__(
            histories=histories,
            item_configs=item_configs,
            review_period=1,  # placeholder — _planning_horizon uses _current_days
            forecaster_class=forecaster_class,
            forecaster_kwargs=forecaster_kwargs,
            yield_profiles=yield_profiles,
            target=target,
            date_col=date_col,
            item_col=item_col,
        )
        self._legacy_policies = policies or {}

    def _planning_horizon(self, item: Item) -> int:
        policy = self._legacy_policies.get(item.name)
        if policy is not None:
            return policy.effective_horizon(item.Lead_time)
        # Fall back to the `days` param passed to run() / run_holdout()
        return max(1, self._current_days)

    def _solver_lower_bound(self, item: Item, demand_samples: np.ndarray) -> float:
        policy = self._legacy_policies.get(item.name)
        if policy is not None:
            return policy.min_quantity(demand_samples)
        return 0.0

    def _get_service_targets(self) -> dict[str, float]:
        return {
            name: getattr(policy, "service_level_target", 0.0)
            for name, policy in self._legacy_policies.items()
            if getattr(policy, "service_level_target", 0.0) > 0.0
        }

    def _net_order(self, item: Item, qty: int, inventory_state: dict | None) -> int:
        policy = self._legacy_policies.get(item.name)
        if policy is not None:
            return policy.net_order(qty)
        return super()._net_order(item, qty, inventory_state)
