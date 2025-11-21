from .demand_distribution import DemandDistribution
from .items import Item
from abc import ABC, abstractmethod
import math
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


class LagrangianConstraintSolver(Solver):
    """
    This solver finds the Lagrange multiplier, $\lambda$, for each constraint as well as the optimum order quantities.

    Use Case: Use the Lagrangian solver if you are dealing with low-volume items with multiple constraints ( < 10).

    Initialization: Starts with all shadow prices at 0.

    Sub-problem: It solves the single-item Newsvendor problem for every item,
    but treats the Effective Cost as:
    $$Cost_{effective} = Cost_{real} + \sum (\lambda_k \times \text{ConstraintUsage}_k)$$

    Update:If we use too much of a resource (e.g., over budget), we increase $\lambda$ for that resource. This makes items using that resource "expensive,"
    reducing their order quantity in the next iteration.If we use too little, we decrease $\lambda$.

    Iteration: This repeats until the shadow prices stabilize or we hit a limit.
    """

    def __init__(
        self,
        problems: list[tuple[Item, DemandDistribution]],
        limits: dict[str, float],
        max_iter: int = 200,
        learning_rate: float = 2.0,
    ):
        self.problems = problems
        self.limits = limits
        self.max_iter = max_iter
        self.initial_learning_rate = learning_rate

    def solve(self) -> dict[str, int]:
        # Initialize multipliers (lambdas) to 0 for each constraint
        lambdas = {k: 0.0 for k in self.limits.keys()}

        best_feasible_q = {}
        best_feasible_profit = -float("inf")

        # Subgradient Optimization Loop
        for k in range(self.max_iter):
            # Solve relaxed problem with current shadow prices
            current_q, current_usage = self._solve_relaxed_problem(lambdas)

            # Check feasibility and update best solution
            is_feasible = all(
                current_usage[res] <= limit for res, limit in self.limits.items()
            )
            if is_feasible:
                profit = self._calculate_total_expected_profit(current_q)
                if profit > best_feasible_profit:
                    best_feasible_profit = profit
                    best_feasible_q = current_q.copy()

            # Calculate gradients (Usage - Limit)
            gradients = {
                res: current_usage[res] - limit for res, limit in self.limits.items()
            }
            gradients_scale = sum(
                g**2 for g in gradients.values()
            )  # normalize gradients

            # Check for convergence (all constraints satisfied or close enough)
            # Note: In discrete problems, gradients might not hit 0 exactly.
            if all(g <= 0 for g in gradients.values()) and sum(lambdas.values()) == 0:
                break

            # Update multipliers (Subgradient Descent)
            # Step size decays over time (Polyak-like heuristic)
            step_size = self.initial_learning_rate / (
                math.sqrt(k + 1) * math.sqrt(gradients_scale)
            )

            for res in lambdas:
                # lambda = max(0, lambda + step * gradient)
                lambdas[res] = max(0.0, lambdas[res] + step_size * gradients[res])

        if not best_feasible_q:
            print(
                "Warning: No feasible solution found. Returning last iteration (likely infeasible)."
            )

            self.allocation = current_q
            return self.allocation
        self.allocation = best_feasible_q
        self.lambdas = lambdas
        return best_feasible_q

    def _solve_relaxed_problem(
        self, lambdas: dict[str, float]
    ) -> tuple[dict[str, int], dict[str, float]]:
        quantities = {}
        usage = defaultdict(float)

        for item, demand in self.problems:
            # Calculate total shadow cost for this item
            shadow_cost = sum(
                lambdas.get(res, 0) * item.constraints.get(res, 0) for res in lambdas
            )

            effective_cost = item.cost_price + shadow_cost

            # If effective cost exceeds selling price, order 0 (not profitable)
            if effective_cost >= item.selling_price:
                q = 0
            else:
                # New Critical Fractile with effective cost
                # CF = (Price - Eff_Cost) / (Price - Salvage)
                # Note: Denominator (Price - Salvage) is unchanged by shadow costs in standard derivation
                numerator = item.selling_price - effective_cost
                denominator = item.selling_price - item.salvage_value
                fractile = numerator / denominator

                q = max(0, math.ceil(demand.get_quantile(fractile)))

            quantities[item.name] = q

            # Tally usage
            for res, val in item.constraints.items():
                if res in self.limits:
                    usage[res] += q * val

        return quantities, usage

    def _calculate_total_expected_profit(self, quantities: dict[str, int]) -> float:
        total_profit = 0.0
        # Simple expected profit calculation for verification
        for item, demand in self.problems:
            q = quantities.get(item.name, 0)
            if q == 0:
                continue

            total_profit += q * (
                item.selling_price - item.cost_price
            )  # Simplified proxy
        return total_profit
