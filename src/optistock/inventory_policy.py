"""
Inventory policy layer between demand forecasters and the newsvendor solver.

An ``InventoryPolicy`` translates operational parameters (review cadence,
service targets, current stock levels) into bounds and horizon adjustments
consumed by ``ForecastSolver``.

Subclasses implement policy-specific logic for the planning horizon and
service-level floor; universal inventory accounting (on-hand, on-order,
net order calculation) lives on the ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class InventoryPolicy(ABC):
    """
    Abstract base for inventory replenishment policies.

    Handles universal inventory state (on-hand, on-order) and net-order
    accounting.  Subclasses define how the planning horizon and minimum
    order quantity are computed.

    Parameters
    ----------
    on_hand : int
        Units currently in stock.
    on_order : int
        Units ordered but not yet received.
    """

    def __init__(self, on_hand: int = 0, on_order: int = 0):
        if on_hand < 0 or on_order < 0:
            raise ValueError("on_hand and on_order must be non-negative.")
        self.on_hand = on_hand
        self.on_order = on_order

    @property
    def inventory_position(self) -> int:
        return self.on_hand + self.on_order

    def net_order(self, gross_quantity: int) -> int:
        return max(0, gross_quantity - self.inventory_position)

    @abstractmethod
    def effective_horizon(self, lead_time: int) -> int:
        """
        Number of days the demand distribution must cover.

        Parameters
        ----------
        lead_time : int
            Item-level lead time in days (from ``Item.Lead_time``).

        Returns
        -------
        int
        """

    @abstractmethod
    def min_quantity(self, demand_samples: np.ndarray) -> float:
        """
        Minimum gross order quantity (service-level floor).

        Parameters
        ----------
        demand_samples : np.ndarray
            1-D array of posterior total-demand samples over the
            effective horizon.

        Returns
        -------
        float
            Lower bound on the gross order quantity.  Return 0.0 when
            no service constraint applies.
        """


class ReviewPolicy(InventoryPolicy):
    """
    Periodic-review replenishment policy.

    Sets the planning horizon to ``lead_time + review_period`` (the
    classical *protection interval*) and optionally enforces a
    cycle-service-level (CSL) floor on the gross order quantity.

    Parameters
    ----------
    review_period : int
        Days between replenishment reviews (must be >= 1).
    service_level_target : float
        Target cycle service level in ``[0, 1)``.  ``0.0`` (default)
        means no service constraint — the solver optimizes purely for
        profit.  Values > 0 add a ``NonlinearConstraint`` to the
        trust-region solver that enforces
        ``P(Q_eff >= demand) >= service_level_target``.
    on_hand : int
        Units currently in stock.  Default ``0``.
    on_order : int
        Units ordered but not yet received.  Default ``0``.
    """

    def __init__(
        self,
        review_period: int,
        service_level_target: float = 0.0,
        on_hand: int = 0,
        on_order: int = 0,
    ):
        super().__init__(on_hand=on_hand, on_order=on_order)
        if review_period < 1:
            raise ValueError("review_period must be >= 1.")
        if not 0.0 <= service_level_target < 1.0:
            raise ValueError("service_level_target must be in [0, 1).")
        self.review_period = review_period
        self.service_level_target = service_level_target

    def effective_horizon(self, lead_time: int) -> int:
        """Planning horizon = lead time + review period (protection interval)."""
        return lead_time + self.review_period

    def min_quantity(self, demand_samples: np.ndarray) -> float:
        """
        CSL quantile of the aggregate demand distribution.

        Returns the ``service_level_target`` quantile of *demand_samples*,
        which is the analytically-exact gross order floor when yield = 1.
        The solver uses this as a variable lower bound and warm-start
        floor; the exact probabilistic constraint (including yield
        uncertainty) is enforced separately via ``NonlinearConstraint``.

        Returns ``0.0`` when ``service_level_target == 0`` or the sample
        array is empty.
        """
        if self.service_level_target <= 0.0 or demand_samples.size == 0:
            return 0.0
        return float(np.quantile(demand_samples, self.service_level_target))
