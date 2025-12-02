from .demand_distribution import DemandDistribution
from .items import Item
from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint


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


class SingleItemSolver(Solver):
    """
    Solves the classic (single-item) Newsvendor problem.

    Use Case: One item problem with no constraints
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
    Solves the case for multiple items with a budget constraint using a "greedy" approach

    Use Case: Multiple Items with one budget constraint. Best for small order quantities for few number of items.

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

                #  Marginal Profit per unit invested
                metric = mep / item.cost_price

                # Find the item with the best mep
                if metric > best_metric:
                    best_metric = metric
                    best_item_idx = i

            # remove items that are no longer profitable
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
    """
    A trust-region solver that returns the lagrangian multipliers for each constraint.

    TODO: At the moment only handles normal demand distributions.
    """

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
        x0 = np.array([d.mean for _, d in self.problems])
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
                mu, sigma = demand.mean, demand.std
            else:
                mu = np.mean(demand.samples)
                sigma = np.std(demand.samples)

            # Analytical Normal Loss Function for smooth gradients
            if sigma > 0:
                # standard deviation q is from mean of demand
                z = (q - mu) / sigma

                # likelihood that demand will be Order Quantity.
                pdf = demand.get_pdf(q)

                # probability that Demand is less than Order Quantity
                cdf = demand.get_cdf(q)

                # The expected amount of demand that will not be met (shortage normalised)
                L_z = pdf - z * (1 - cdf)

                exp_shortage = sigma * L_z  # expected shortage
                exp_sales = mu - exp_shortage  # actual sales (demand - shortage)
                exp_leftover = q - exp_sales  # actual left over (ordered - left over)
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


class StochasticMonteCarloSolver(Solver):
    """
    Stochastic solver that takes into account the full yield and demand distributions.

    It asses risk using the CVaR (Conditional Value at Risk) by punishing profit with the distance to the cvar value.
    This value is punished using a risk_aversion metric (0 - 1), with 1 being extremely risk averse, and 0 not at all.

    obj = (1 - self.risk_aversion) * mean_profit + self.risk_aversion * cvar_val

    Default is 0 risk aversion at the 5% percentile.
    """

    def __init__(
        self,
        problems: list[tuple[Item, DemandDistribution]],
        limits: dict[str, float],
        n_samples: int = 10000,
    ):
        self.problems = problems
        self.limits = limits
        self.n_samples = n_samples
        self.lambdas = {}

        # Pre-generate samples for Common Random Numbers (CRN)
        self._demand_samples_matrix = []
        self._yield_samples_matrix = []

        for item, demand_dist in self.problems:
            # 1. Prepare Demand Samples
            if hasattr(demand_dist, "samples"):
                # Resample or tile to match n_samples if needed
                d_samp = np.random.choice(demand_dist.samples, n_samples)
            else:
                # Generate from analytical distribution (Normal)
                d_samp = np.random.normal(demand_dist.mean, demand_dist.std, n_samples)
                d_samp = np.maximum(0, d_samp)  # Clamp negative demand
            self._demand_samples_matrix.append(d_samp)

            # 2. Prepare Yield Samples
            # Item.yield_distribution is now populated (defaults to PerfectYield)
            y_samp = item.yield_distribution.sample(n_samples)
            self._yield_samples_matrix.append(y_samp)

        self._demand_samples_matrix = np.array(self._demand_samples_matrix)
        self._yield_samples_matrix = np.array(self._yield_samples_matrix)

    def solve(
        self, method="Utility", risk_aversion: float = 0.0, cvar: float = 0.05
    ) -> dict[str, int]:
        self.risk_aversion = risk_aversion
        self.cvar = cvar

        n_items = len(self.problems)
        names = list(self.limits.keys())

        # Constraint Matrix
        A = np.zeros((len(names), n_items))
        upper_bound = np.array(list(self.limits.values()))
        lower_bound = np.array([-np.inf] * len(names))

        for r, name in enumerate(names):
            for c, (item, _) in enumerate(self.problems):
                val = item.constraints.get(name, 0.0)
                A[r, c] = val

        linear_cons = LinearConstraint(A, lower_bound, upper_bound)

        # Initial Guess
        x0 = []
        for i, (_, dem) in enumerate(self.problems):
            d_mean = self._demand_samples_matrix[i].mean()
            y_mean = self._yield_samples_matrix[i].mean()
            x0.append(d_mean / max(y_mean, 0.01))

        x0 = np.array(x0)
        bounds = [(0, np.inf) for _ in range(n_items)]

        match method:
            case "Utility":
                obj_func = self._utility_function
            case "CVAR":
                obj_func = self._CVAR_function

        # Optimize
        res = minimize(
            fun=obj_func,
            x0=x0,
            method="trust-constr",
            constraints=[linear_cons],
            bounds=bounds,
        )

        if res.v:
            self.lambdas = {n: float(abs(x)) for n, x in zip(names, res.v[0])}

        # Return integer quantities
        allocation = np.floor(res.x).astype(int)
        self.allocation = {
            self.problems[i][0].name: q for i, q in enumerate(allocation)
        }
        return self.allocation

    def _CVAR_function(self, quantities):
        Q = quantities.reshape(-1, 1)

        Q_eff = Q * self._yield_samples_matrix
        Sales = np.minimum(Q_eff, self._demand_samples_matrix)
        Leftover = Q_eff - Sales

        prices = np.array([p[0].selling_price for p in self.problems]).reshape(-1, 1)
        costs = np.array([p[0].cost_price for p in self.problems]).reshape(-1, 1)
        salvages = np.array([p[0].salvage_value for p in self.problems]).reshape(-1, 1)

        Revenue = Sales * prices
        Salvage = Leftover * salvages
        Cost = Q * costs

        # Sum profit across items to get Portfolio Profit per scenario
        portfolio_profit = np.sum(Revenue + Salvage - Cost, axis=0)

        mean_profit = np.mean(portfolio_profit)

        if self.risk_aversion == 0:
            return -mean_profit

        # CVaR Calculation: Average of the worst alpha% outcomes
        k = int(self.n_samples * self.cvar)
        # Partition puts the smallest k elements at the front (unsorted)
        partitioned = np.partition(portfolio_profit, k)
        cvar_val = np.mean(partitioned[:k])

        # Weighted Objective
        obj = (1 - self.risk_aversion) * mean_profit + self.risk_aversion * cvar_val

        return -obj

    def _utility_function(self, quantities):
        Q = quantities.reshape(-1, 1)
        Q_eff = Q * self._yield_samples_matrix
        Sales = np.minimum(Q_eff, self._demand_samples_matrix)
        Leftover = Q_eff - Sales

        prices = np.array([p[0].selling_price for p in self.problems]).reshape(-1, 1)
        costs = np.array([p[0].cost_price for p in self.problems]).reshape(-1, 1)
        salvages = np.array([p[0].salvage_value for p in self.problems]).reshape(-1, 1)

        Revenue = Sales * prices
        Salvage = Leftover * salvages
        Cost = Q * costs

        portfolio_profit = np.sum(Revenue + Salvage - Cost, axis=0)

        if self.risk_aversion == 0:
            return -np.mean(portfolio_profit)

        # Use the dynamically scaled lambda
        # We normalize the exponential input to avoid overflow/underflow
        # U(x) = -exp(-lam * x)

        est_revenue = 0.0
        for i, (item, _) in enumerate(self.problems):
            # Mean demand * Price
            est_revenue += self._demand_samples_matrix[i].mean() * item.selling_price

        profit_scale = max(1.0, est_revenue)  # Avoid div/0

        # Lambda = RRA / Wealth
        self._current_lambda = self.risk_aversion / profit_scale

        return np.mean(np.exp(-self._current_lambda * portfolio_profit))
