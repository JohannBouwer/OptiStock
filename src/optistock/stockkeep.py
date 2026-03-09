from __future__ import annotations

"""
Top-level orchestrator that ties the different steps together:
    train models -> forecast demand distribution -> optimise supply allocation

Public interface
----------------
StockKeep(histories, item_configs, forecaster_class, ...)
    .run_holdout(holdout_days, ...)  ->  dict with allocation + validation metrics
    .run(forecast_days, ...)         ->  dict with allocation for a future horizon
    .plot_forecast(item_name)        ->  (fig, ax)
    .summary()                       ->  dict  (delegates to ForecastSolver.summary)
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


class StockKeep:
    """
    One-call inventory planning orchestrator.

    Chains four stages for every item in the portfolio:
    1. (Optionally) split history into train and hold-out.
    2. Fit the chosen forecaster on the training window.
    3. Forecast demand over the planning horizon.
    4. Solve the newsvendor optimisation via ForecastSolver.

    Parameters
    ----------
    histories : pd.DataFrame
        Long-format sales history. Must contain at minimum the columns
        ``item_col``, ``date_col``, and ``target``.
    item_configs : pd.DataFrame
        One row per item. Required columns: ``name``, ``cost_price``,
        ``selling_price``. Optional: ``salvage_value`` (defaults to 0).
        Any other columns are treated as resource-constraint coefficients
        and forwarded to ``Item.constraints``.
    forecaster_class : type[BaseForecaster]
        The forecaster class applied to *every* item.  Defaults to
        ``BayesTimeSeries``.
    forecaster_kwargs : dict, optional
        Extra keyword arguments forwarded to ``forecaster_class(data, ...)``
        at construction time.  E.g. ``seasonal_config``, ``exog``.
    yield_profiles : dict[str, YieldDistribution], optional
        Maps item name -> YieldDistribution.  Items not in the dict get
        ``PerfectYield()``.
    target : str
        Name of the demand/sales column in ``histories``.  Default ``"sales"``.
    date_col : str
        Name of the date column.  Default ``"date"``.
    item_col : str
        Name of the item identifier column.  Default ``"item"``.
    """

    # Core columns that should *not* be treated as constraint coefficients
    _CORE_COLS = {"name", "cost_price", "selling_price", "salvage_value"}

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
        self.solver: ForecastSolver | None = None
        self._run_start: str | None = None
        self._run_end: str | None = None
        self._mode: str | None = None

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------

    def run_holdout(
        self,
        holdout_days: int,
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
            Number of calendar days to hold out from the end of each item's
            history.
        events : dict, optional
            Structured as ``{item_name: {event_name: [date_str, ...]}}``.
            Items not in the dict receive no events.
        objective : {'SAA', 'CVaR', 'Utility'}
            Optimisation objective passed to ForecastSolver.
        limits : dict, optional
            Shared resource constraints, e.g. ``{"budget": 50_000}``.
        cvar_alpha, cvar_lambda, risk_aversion
            Objective-specific parameters; see ForecastSolver docs.
        fit_kwargs : dict, optional
            Extra keyword arguments forwarded to ``forecaster.fit()``, e.g.
            ``{"chain": 2, "samples": 500}`` for BayesTimeSeries.
            For UnivariateSSM, include a ``"build_model_kwargs"`` key with a
            nested dict of arguments for ``build_model()``.

        Returns
        -------
        dict
            ``allocation``    — item name -> optimal quantity
            ``metrics``       — per-item profit, service level, stockout,
                                leftover units, and SMAPE (% forecast error)
            ``solver_summary``— ForecastSolver.summary() output
            ``period``        — (start_date_str, end_date_str)
            ``mode``          — ``"holdout"``
        """
        return self._run_pipeline(
            mode="holdout",
            days=holdout_days,
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

        Fits on the full history for each item and forecasts the next
        ``forecast_days`` days beyond the last observed date.  No holdout
        evaluation is performed.

        Parameters
        ----------
        forecast_days : int
            Number of calendar days to forecast into the future.
        events : dict, optional
            ``{item_name: {event_name: [date_str, ...]}}``.
        objective, limits, cvar_alpha, cvar_lambda, risk_aversion, fit_kwargs
            Same as :meth:`run_holdout`.

        Returns
        -------
        dict
            ``allocation``    — item name -> optimal quantity
            ``solver_summary``— ForecastSolver.summary() output
            ``period``        — (start_date_str, end_date_str)
            ``mode``          — ``"production"``
        """
        return self._run_pipeline(
            mode="production",
            days=forecast_days,
            events=events or {},
            objective=objective,
            limits=limits,
            cvar_alpha=cvar_alpha,
            cvar_lambda=cvar_lambda,
            risk_aversion=risk_aversion,
            fit_kwargs=fit_kwargs or {},
        )

    def plot_forecast(self, item_name: str) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the forecast for a single item, overlaid with holdout actuals
        when in holdout mode.

        Must be called after ``run_holdout`` or ``run``.

        Parameters
        ----------
        item_name : str
            Must match a name in ``item_configs``.

        Returns
        -------
        (fig, ax)
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

        # In holdout mode, overlay the actual observed values
        if self._mode == "holdout" and item_name in self.holdout_data:
            holdout_df, target = self.holdout_data[item_name]
            actual = holdout_df[target]
            dates = pd.to_datetime(holdout_df[self.date_col]) if self.date_col in holdout_df.columns else holdout_df.index

            # Scale actuals to model's internal scale when a max_scaler exists
            scale = getattr(forecaster, "max_scaler", 1.0) or 1.0
            ax.scatter(
                dates,
                actual.values / scale,
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
        """
        Return ForecastSolver's diagnostic summary for the last run.

        Raises RuntimeError if called before any run.
        """
        if self.solver is None:
            raise RuntimeError("Call run_holdout() or run() first.")
        return self.solver.summary()

    # ------------------------------------------------------------------
    # Solver plot wrappers
    # ------------------------------------------------------------------

    def plot_allocation(self) -> plt.Figure:
        """
        Visualise the current allocation against the posterior demand distributions.

        Automatically selects the appropriate chart:
        - With resource constraints  → ``plot_constrained_allocation``
        - Without resource constraints → ``plot_multi_item_allocation``

        Must be called after ``run_holdout`` or ``run``.
        """
        self._check_solved()
        plot_problems = self._build_plot_problems()
        if self.solver.limits:
            return plot_constrained_allocation(
                self.allocation, plot_problems, self.solver.limits
            )
        budget = next(iter(self.solver.limits.values()), None) if self.solver.limits else None
        return plot_multi_item_allocation(self.allocation, plot_problems, budget)

    def plot_item(self, item_name: str) -> plt.Figure:
        """
        Single-item demand distribution and profit-curve dashboard.

        Parameters
        ----------
        item_name : str
            Must match a name in ``item_configs``.

        Must be called after ``run_holdout`` or ``run``.
        """
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
        """
        Overlay profit-distribution curves for one or more allocation strategies.

        Parameters
        ----------
        allocations : dict[str, dict[str, int]], optional
            Mapping of strategy label → allocation dict (item name → quantity).
            When omitted, the current allocation is shown under the label
            ``"Current"``.

        Must be called after ``run_holdout`` or ``run``.
        """
        self._check_solved()
        if allocations is None:
            allocations = {"Current": self.allocation}
        plot_problems = self._build_plot_problems()
        return plot_risk_comparison(allocations, plot_problems)

    def plot_optimization_summary(self) -> plt.Figure:
        """
        Waterfall chart (potential vs realised profit) and shadow-price bar chart.

        Must be called after ``run_holdout`` or ``run``.
        """
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
        events: dict,
        objective: str,
        limits: dict | None,
        cvar_alpha: float,
        cvar_lambda: float,
        risk_aversion: float,
        fit_kwargs: dict,
    ) -> dict[str, Any]:
        """Shared body for run_holdout and run."""
        problems: list[tuple[Item, BaseForecaster]] = []
        holdout_actuals: list[tuple[Item, float]] = []
        daily_means: dict[str, np.ndarray] = {}

        start_dts: list[pd.Timestamp] = []
        end_dts: list[pd.Timestamp] = []

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

            if mode == "holdout":
                train_df, holdout_df, split_date, max_date = self._date_prep(df_item, days)
                start_dt = split_date
                end_dt = max_date
            else:  # production
                train_df = df_item.copy()
                holdout_df = None
                max_date = df_item[self.date_col].max()
                start_dt = max_date + pd.Timedelta(days=1)
                end_dt = max_date + pd.Timedelta(days=days)

            start_dts.append(start_dt)
            end_dts.append(end_dt)

            # Build, fit, and forecast
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

        # Use the intersection of all per-item planning windows
        global_start = max(start_dts)
        global_end = min(end_dts)
        start_str = str(global_start.date())
        end_str = str(global_end.date())

        solver = ForecastSolver(
            problems,
            objective=objective,
            limits=limits,
            cvar_alpha=cvar_alpha,
            cvar_lambda=cvar_lambda,
            risk_aversion=risk_aversion,
        )
        allocation = solver.solve(start_str, end_str)

        self.solver = solver
        self.allocation = allocation
        self._run_start = start_str
        self._run_end = end_str
        self._mode = mode

        result: dict[str, Any] = {
            "allocation": allocation,
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

    def _check_solved(self) -> None:
        if self.solver is None or self.allocation is None:
            raise RuntimeError("Call run_holdout() or run() before plotting.")

    def _build_plot_problems(self) -> list[tuple[Item, "_DemandSamples"]]:
        """Return (Item, _DemandSamples) pairs built from the last solver run."""
        assert self.solver is not None and self.solver._demand_matrix is not None
        return [
            (item, _DemandSamples(self.solver._demand_matrix[i]))
            for i, (item, _) in enumerate(self.solver.problems)
        ]

    def _create_items(self, yield_profiles: dict[str, YieldDistribution]) -> list[Item]:
        """Build Item objects from the item_configs DataFrame."""
        items = []
        for _, row in self._item_df.iterrows():
            name = row["name"]
            cp = float(row["cost_price"])
            sp = float(row["selling_price"])
            sv = float(row["salvage_value"]) if "salvage_value" in self._item_df.columns else 0.0
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
                constraints=constraints,
                yield_distribution=yd,
            ))

        return items

    def _build_forecaster(self, train_df: pd.DataFrame) -> BaseForecaster:
        """Construct a fresh forecaster instance for one item's training data."""
        if issubclass(self.forecaster_class, UnivariateSSM):
            # SSM expects a DateTimeIndex DataFrame
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
        """Register events, fit, and generate a forecast — in place."""
        # Events (only BayesTimeSeries / BART / HSGP support this)
        if events_for_item and hasattr(forecaster, "create_events"):
            forecaster.create_events(events_for_item, date_col=self.date_col)

        # Fit
        if isinstance(forecaster, UnivariateSSM):
            # build_model must be called before fit; user passes build_model_kwargs
            # inside fit_kwargs under the key "build_model_kwargs"
            fk = dict(fit_kwargs)
            build_kwargs = fk.pop("build_model_kwargs", {})
            if forecaster.model is None:
                forecaster.build_model(**build_kwargs)
            forecaster.fit(**fk)
        else:
            forecaster.fit(target=self.target, date_col=self.date_col, **fit_kwargs)

        # Forecast
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
        """
        Return the per-period posterior mean forecast in original (unscaled) units.

        Returns None when the forecaster type is unrecognised or the relevant
        attribute is missing.
        """
        try:
            if isinstance(forecaster, UnivariateSSM):
                obs = forecaster.forecast_idata["forecast_observed"].isel(observed_state=0)
                return obs.stack(sample=["chain", "draw"]).mean(dim="sample").values

            if isinstance(forecaster, MediaMixModel):
                if forecaster.predictions is not None:
                    return forecaster.predictions.mean(dim="sample").values

            # BayesTimeSeries / BART / HSGP family
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

    def _calculate_metrics(
        self,
        allocation: dict[str, int],
        actuals: list[tuple[Item, float]],
        daily_forecast_means: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Compute per-item KPIs against the actual holdout demand.

        Parameters
        ----------
        allocation : dict
            Item name -> ordered quantity.
        actuals : list of (Item, float)
            Each tuple pairs an item with its *total* holdout demand.
        daily_forecast_means : dict, optional
            Per-item daily posterior mean forecast arrays used for SMAPE.
            When None or an item is missing from the dict, smape_pct is omitted.

        Returns
        -------
        dict
            Per-item sub-dicts plus a ``"portfolio_total_profit"`` key.
        """
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

            # SMAPE — only when daily forecast is available
            if daily_forecast_means is not None and item.name in daily_forecast_means:
                _, target = self.holdout_data[item.name]
                holdout_df, _ = self.holdout_data[item.name]
                actual_series = holdout_df[target].values
                forecast_series = daily_forecast_means[item.name]
                # Trim to the same length in case of minor misalignment
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

    # ------------------------------------------------------------------
    # Date helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _date_prep(
        self, df: pd.DataFrame, days: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        """Split data into training and hold-out sets."""
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        max_date = df[self.date_col].max()
        split_date = max_date - pd.Timedelta(days=days - 1)

        train_df = df[df[self.date_col] < split_date].copy()
        holdout_df = df[df[self.date_col] >= split_date].copy()
        return train_df, holdout_df, split_date, max_date
