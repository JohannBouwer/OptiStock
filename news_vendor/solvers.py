from .demand_distribution import DemandDistribution
from .items import Item
from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from collections import defaultdict


class Solver(ABC):
    """
    Abstract Base Class for all Newsvendor problem solvers.
    """

    @abstractmethod
    def solve(self) -> int | dict:
        """
        Executes the solver's logic to find the optimal solution.
        """
        pass


class SingleItemNewsvendorSolver(Solver):
    """
    Solves the classic (single-item) Newsvendor problem.

    Use Case: One item problem with no contraints
    """

    def __init__(self, item: Item, demand_distribution: DemandDistribution):
        """
        Initializes the single-item solver.

        Args:
            item (Item): The item to be stocked.
            demand_distribution (DemandDistribution): The demand distribution for the item.
        """
        self.item = item
        self.demand_distribution = demand_distribution

    def solve(self) -> int:
        """
        Calculates the optimal order quantity (Q*).

        The logic is:
        1. Calculate the critical fractile (CF) from the item's costs.
        2. Find the quantity Q such that P(Demand <= Q) = CF.
           This is done by finding the quantile of the demand distribution
           for the probability CF.
        3. Since order quantities are discrete, we round up to the nearest integer.

        Returns:
            int: The optimal order quantity.
        """
        # Get critical fractile
        fractile = self.item.critical_fractile

        # Get quantile from demand distribution
        optimal_q_float = self.demand_distribution.get_quantile(fractile)

        # Round up to the nearest integer
        # We round up because the cost function is convex and we want the
        # smallest Q such that P(D <= Q) >= CF.
        optimal_q_int = math.ceil(optimal_q_float)

        # Ensure quantity is non-negative
        self.quantity = max(0, optimal_q_int)
        return self.quantity


class MultiItemConstrainedSolver(Solver):
    """
    Solves the case for multiple items with a budget contraint using a "greedy" approach

    Use Case: Multiple Items with one budget contraint. Best for small order quatities for few number of items.

    - It calculates the Expected Marginal Profit of buying one more unit for every item.

    - It divides that profit by the Cost of the item to get a "Return on Investment" (ROI) ratio.

    - It selects the item with the highest ROI, "buys" it, updates the remaining budget, and repeats.
    """

    def __init__(
        self, inventory_problems: list[tuple[Item, DemandDistribution]], budget: float
    ):
        """
        inventory_problems (list[tuple[Item, DemandDistribution]]): list of (item, demand distribution) tuples
        budget (float): Total inventory budget
        """
        self.problems = inventory_problems
        self.budget = budget

    def solve(self) -> dict[str, int]:
        current_quantities = [0] * len(self.problems)
        current_spend = 0.0

        # Indices of items that are still viable candidates for purchase
        active_indices = set(range(len(self.problems)))

        while active_indices:
            best_item_idx = -1
            best_metric = -float("inf")
            to_remove = set()

            for i in active_indices:
                item, demand = self.problems[i]
                current_q = current_quantities[i]

                if current_spend + item.cost_price > self.budget:
                    to_remove.add(i)
                    continue

                # Probability of selling the (current_q + 1)th unit
                prob_sell_next = 1.0 - demand.get_cdf(current_q)
                prob_unsold_next = 1.0 - prob_sell_next

                # Expected marginal profit of the next unit
                mep = (prob_sell_next * item.underage_cost) - (
                    prob_unsold_next * item.overage_cost
                )

                if mep <= 0:
                    to_remove.add(i)
                    continue

                # Bang for buck: Marginal Profit per unit invested
                metric = mep / item.cost_price

                # Find the item with the best mep
                if metric > best_metric:
                    best_metric = metric
                    best_item_idx = i

            # remove items that are no longer porfitible
            active_indices -= to_remove

            # update item and current spend
            if best_item_idx != -1:
                current_quantities[best_item_idx] += 1
                current_spend += self.problems[best_item_idx][0].cost_price
            else:
                break

        self.allocation = {
            prob[0].name: q for prob, q in zip(self.problems, current_quantities)
        }
        return self.allocation


class ScipyOptimizationSolver(Solver):
    def __init__(
        self,
        problems: list[tuple[Item, DemandDistribution]],
        limits: dict[str, float],
    ):
        self.problems = problems
        self.limits = limits
        self.shadow_prices = {}

    def solve(self) -> dict[str, int]:
        n_items = len(self.problems)
        constraint_names = list(self.limits.keys())

        # Build Constraint Matrix A (Rows=Constraints, Cols=Items)
        A = np.zeros((len(constraint_names), n_items))
        # upper bound of constraints
        upper_bounds = np.array(list(self.limits.values()))
        # lower bound of constraints
        lower_bounds = np.array(
            [-np.inf] * len(constraint_names)
        )  # One-sided constraints <= limit

        # set up constraints
        for row_idx, name in enumerate(constraint_names):
            for col_idx, (item, _) in enumerate(self.problems):
                val = item.constraints.get(name, 0.0)
                A[row_idx, col_idx] = val

        linear_cons = LinearConstraint(A, lower_bounds, upper_bounds)

        # Initial Guess & Bounds
        x0 = np.array(
            [
                d.mean if hasattr(d, "mean") else np.mean(d.samples)
                for _, d in self.problems
            ]
        )
        bounds = [(0, np.inf) for _ in range(n_items)]

        # 3. Solve
        result = minimize(
            fun=self._objective_function,
            x0=x0,
            method="trust-constr",
            bounds=bounds,
            constraints=[linear_cons],
        )

        # Extract Lambdas
        if result.v:
            self.lambdas = {
                name: float(abs(m)) for name, m in zip(constraint_names, result.v[0])
            }

        final_quantities = np.floor(result.x).astype(int)
        self.allocation = {
            self.problems[i][0].name: q for i, q in enumerate(final_quantities)
        }
        return self.allocation

    def _objective_function(self, quantities):
        total_profit = 0.0
        for i, q in enumerate(quantities):
            item, demand = self.problems[i]

            # Smooth approximation for gradient descent
            if hasattr(demand, "mean"):
                mu, sigma = demand.mean, demand.std_dev
            else:
                mu = np.mean(demand.samples)
                sigma = np.std(demand.samples)

            # Analytical Normal Loss Function for smooth gradients
            if sigma > 0:
                # standard deviation q is from mean of demand
                z = (q - mu) / sigma

                # likelihood that demand will be Order Quantity.
                pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)

                # probability that Demand is less than Order Quantity
                cdf = 0.5 * (1 + math.erf(z / np.sqrt(2)))

                # The expected amount of demand that will not be met (shortage normalised)
                L_z = pdf - z * (1 - cdf)

                exp_shortage = sigma * L_z  # expected shortage
                exp_sales = mu - exp_shortage  # actual sales (demand - shortage)
                exp_leftover = q - exp_sales  # actual sales (ordered - left over)
            else:
                exp_sales = min(q, mu)
                exp_leftover = max(0, q - mu)

            profit = (
                (exp_sales * item.selling_price)
                + (exp_leftover * item.salvage_value)
                - (q * item.cost_price)
            )
            total_profit += profit

        return -total_profit
