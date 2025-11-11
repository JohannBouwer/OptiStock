from abc import ABC
from demand_distribution import DemandDistribution
from items import Item
class Solver(ABC):
    """
    Abstract Base Class for all Newsvendor problem solvers.
    """
    
    @abstractSmethod
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
        # 1. Get critical fractile
        fractile = self.item.critical_fractile
        
        # 2. Get quantile from demand distribution
        optimal_q_float = self.demand_distribution.get_quantile(fractile)
        
        # 3. Round up to the nearest integer
        # We round up because the cost function is convex and we want the
        # smallest Q such that P(D <= Q) >= CF.
        optimal_q_int = math.ceil(optimal_q_float)
        
        # Ensure quantity is non-negative
        return max(0, optimal_q_int)