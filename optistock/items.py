from .distributions.yield_distributions import PerfectYield, YieldDistribution


class Item:
    """
    Represents a single item, encapsulating its cost structure.
    """

    def __init__(
        self,
        name: str,
        cost_price: float,
        selling_price: float,
        salvage_value: float = 0.0,
        constraints: dict[str, float] = {},
        yield_distribution: None | YieldDistribution = None,
    ):
        """
        Initializes the Item.

        Args:
            cost_price (float): The cost to procure one unit of the item.
            selling_price (float): The price the item is sold at.
            salvage_value (float): The value (e.g., discount price, scrap)
                                     of an unsold item. Defaults to 0.0.
            constraint (dict): Dictionary of values to use for constraints. I.e {"storage" : 20}
        """
        if not (selling_price > cost_price > salvage_value >= 0):
            raise ValueError(
                "Prices must follow: selling_price > cost_price > salvage_value >= 0"
            )

        self.name = name
        self.cost_price = cost_price
        self.selling_price = selling_price
        self.salvage_value = salvage_value
        self.constraints = constraints
        self.yield_distribution = (
            PerfectYield() if yield_distribution is None else yield_distribution
        )

    @property
    def underage_cost(self) -> float:
        """
        Calculates the cost of understocking by one unit (lost profit).
        Cu = Selling Price - Cost Price
        """
        return self.selling_price - self.cost_price

    @property
    def overage_cost(self) -> float:
        """
        Calculates the cost of overstocking by one unit (loss on disposal).
        Co = Cost Price - Salvage Value
        """
        return self.cost_price - self.salvage_value

    @property
    def critical_fractile(self) -> float:
        """
        Calculates the critical fractile (or critical ratio).
        This is the probability threshold for the optimal order quantity.
        CF = Cu / (Cu + Co)
        """
        cu = self.underage_cost
        co = self.overage_cost
        return cu / (cu + co)
