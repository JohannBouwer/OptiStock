from typing import Optional, Type, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .forecasting import BaseForecaster, BayesTimeSeries
from .items import Item
from .distributions.demand_distributions import SampledDemand
from .solvers import Solver, SingleItemSolver
from .distributions.yield_distributions import YieldDistribution, PerfectYield

"""
Orchestrator class that ties the different steps together:
    train models -> forecast demand distribution -> optimize supply allocation
"""

class StockKeep:
    def __init__(
        self,
        histories: pd.DataFrame,
        item_configs: pd.DataFrame,
        forecaster_class: Type[BaseForecaster] = BayesTimeSeries,
        yield_profiles: Dict[str, YieldDistribution] | None = None,
        date_col: str = "date"
    ):
        self.raw_histories = histories
        self.item_configs = item_configs
        self.forecaster_class = forecaster_class
        self.date_col = date_col
        self.results = {}
        
        self.items = self._create_items(yield_profiles)
        
        self.trained_models = {}
        self.holdout_data = {}
      
        
    def _create_items(self, yields) -> List[Item]:
        items = []
        # Define the set of core attributes for the Item class
        core_cols = {'name', 'cost_price', 'selling_price', 'salvage_value', 'yield_distribution'}
        
        for _, row in self.item_configs.iterrows():
            # Extract core fields using .get() for optional parameters with defaults
            name = row['name']
            cp = row['cost_price']
            sp = row['selling_price']
            sv = row.get('salvage_value', 0.0)
            yd = yields.get(name, PerfectYield())
            
            # Map any remaining columns (e.g., 'storage', 'weight') to the constraints dict
            constraints = {
                col: row[col] for col in self.item_configs.columns 
                if col not in core_cols and pd.notnull(row[col])
            }
            
            items.append(Item(
                name=name,
                cost_price=cp,
                selling_price=sp,
                salvage_value=sv,
                constraints=constraints,
                yield_distribution=yd
            ))
            
        return items
        
    def _infer_dates(self, df: pd.DataFrame, days: int) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """Calculates split, start, and end dates based on the history and day count."""
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        max_date = df[self.date_col].max()
        
        # The hold-out period starts 'days' before the end of the data
        split_date = max_date - pd.Timedelta(days=days - 1)
        start_date = split_date
        end_date = max_date
        
        return split_date, start_date, end_date
    
    def _date_prep(self, df: pd.DataFrame, days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        """Split data into training and hold-out sets."""
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        max_date = df[self.date_col].max()
        split_date = max_date - pd.Timedelta(days=days - 1)
        
        train_df = df[df[self.date_col] < split_date].copy()
        holdout_df = df[df[self.date_col] >= split_date].copy()
        return train_df, holdout_df, split_date, max_date
    
    def _model_train(self, train_df: pd.DataFrame, target: str, events: Optional[Dict] = None) -> BaseForecaster:
        """Initialize and fit the Bayesian forecaster."""
        forecaster = self.forecaster_class(train_df, target_col=target)
        if events:
            forecaster.create_events(events, date_col=self.date_col) # type: ignore
        forecaster.fit()
        return forecaster
    
    def _forecast(self, forecaster: BaseForecaster, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> SampledDemand:
        """Generate future samples and aggregate into a demand distribution and initialise SampledDemand class."""
        future_dates = pd.date_range(start=start_dt, end=end_dt)
        df_future = pd.DataFrame({self.date_col: future_dates})
        forecaster.predict(df_future, date_col=self.date_col)
        
        samples = forecaster.get_demand_distribution(str(start_dt.date()), str(end_dt.date()))
        data_values = samples.y.values if hasattr(samples, 'y') else samples.values
        return SampledDemand(data_values)
    
    def _supply_opt(self, problems: List[Tuple[Item, SampledDemand]], solver_class: Type[Solver], params: Dict) -> Dict[str, int]:
        """Solve the stochastic optimization problem across the portfolio."""
        solver = solver_class(problems, **params)
        return solver.solve()
    
    def _calculate_metrics(self, allocation: Dict[str, int], actuals: List[Tuple[Item, float]]) -> Dict[str, Any]:
        """Calculates financial and operational KPIs based on actual hold-out demand."""
        report = {}
        total_profit = 0.0

        for item, actual_demand in actuals:
            qty = allocation.get(item.name, 0)
            
            # Simulation logic
            units_sold = min(qty, actual_demand)
            units_leftover = max(0, qty - actual_demand)
            units_short = max(0, actual_demand - qty)
            
            # Financial results
            revenue = units_sold * item.selling_price
            salvage = units_leftover * item.salvage_value
            total_cost = qty * item.cost_price
            profit = revenue + salvage - total_cost
            
            report[item.name] = {
                "profit": round(profit, 2),
                "service_level": round(units_sold / actual_demand, 3) if actual_demand > 0 else 1.0,
                "stockout": units_short > 0,
                "leftover_units": int(units_leftover)
            }
            total_profit += profit

        report["portfolio_total_profit"] = round(total_profit, 2)
        return report

    def run_simulation(
        self, 
        forecast_days: int, 
        target: str = "sales",
        solver_class: Type[Solver] = SingleItemSolver,
        solver_params: Dict[str, Any] = {},
        events_list: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        portfolio_problems = []
        holdout_actuals = []

        for i, item_obj in enumerate(self.items):
            df_item_total = self.raw_histories[self.raw_histories['item'] == item_obj.name].copy()
        
            if df_item_total.empty:
                print(f"Warning: No history found for {item_obj.name}")
                continue

            # Execute Pipeline Stages
            train, holdout, start_dt, end_dt = self._date_prep(df_item_total, forecast_days)
            model = self._model_train(train, target, events_list[i] if events_list else None)
            demand_dist = self._forecast(model, start_dt, end_dt)
            
            self.trained_models[item_obj.name] = model
            self.holdout_data[item_obj.name] = (holdout, target)
            
            portfolio_problems.append((item_obj, demand_dist))
            holdout_actuals.append((item_obj, holdout[target].sum()))

        self.allocation = self._supply_opt(portfolio_problems, solver_class, solver_params)
        self.performance = self._calculate_metrics(self.allocation, holdout_actuals)
        
        return {"allocation": self.allocation, "metrics": self.performance, "period": (start_dt, end_dt)}
    
    def plot_forecast(self, item_name: str) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualizes the forecast HDI alongside actual sales for the holdout period.
        """
        if item_name not in self.trained_models:
            raise ValueError(f"No trained model found for {item_name}. Run simulation first.")

        model = self.trained_models[item_name]
        holdout, target = self.holdout_data[item_name]
        
        fig, ax = model.plot_forecast()
        # Scale actual sales to match the model's scaled units
        actual_scaled = holdout[target] / model.max_scaler
        # Overlay actuals
        ax.scatter(
            holdout[self.date_col], 
            actual_scaled, 
            color="black", 
            marker="x", 
            label="Actual Sales (Holdout)", 
            s=25, 
            zorder=5
        )

        ax.set_title(f"Forecast Validation: {item_name}")
        ax.legend(loc="upper left")
        return fig, ax