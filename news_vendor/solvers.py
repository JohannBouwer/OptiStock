
from .demand_distribution import DemandDistribution
from .items import Item
from abc import ABC, abstractmethod
import math
class Solver(ABC):
    """
    Abstract Base Class for all Newsvendor problem solvers.
    """
    
    @abstractmethod
    def solve(self):
        """
        Executes the solver's logic to find the optimal solution.
        """
        pass

class SingleItemNewsvendorSolver(Solver):
    """
    Solves the classic (single-item) Newsvendor problem.
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
        #Get critical fractile
        fractile = self.item.critical_fractile
        
        #Get quantile from demand distribution
        optimal_q_float = self.demand_distribution.get_quantile(fractile)
        
        # Round up to the nearest integer
        # We round up because the cost function is convex and we want the
        # smallest Q such that P(D <= Q) >= CF.
        optimal_q_int = math.ceil(optimal_q_float)
        
        # Ensure quantity is non-negative
        self.quantity = max(0, optimal_q_int)
        return self.quantity
    
class MultiItemConstrainedSolver:
    """
    Solves the case for multiple items with a budget contraint using a "greedy" approach
    
    - It calculates the Expected Marginal Profit of buying one more unit for every item.

    - It divides that profit by the Cost of the item to get a "Return on Investment" (ROI) ratio.

    - It selects the item with the highest ROI, "buys" it, updates the remaining budget, and repeats.
    """
    def __init__(self, inventory_problems: list[tuple[Item, DemandDistribution]], budget: float):
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
            best_metric = -float('inf')
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
                mep = (prob_sell_next * item.underage_cost) - (prob_unsold_next * item.overage_cost)

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
        
        self.allocation = {prob[0].name: q for prob, q in zip(self.problems, current_quantities)}       
        return self.allocation