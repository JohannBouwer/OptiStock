"""
Stochastic inventory solvers backed by Bayesian demand forecasters.

The ``ForecastSolver`` class accepts any fitted ``BaseForecaster`` and calls
``get_demand_distribution(start_date, end_date)`` to obtain posterior predictive
demand samples.  Stock levels are then optimised using one of three objectives:

* **SAA** (Sample Average Approximation) — maximise ``E[profit]``.
  Every posterior draw is treated as an equally-likely scenario.
  Equivalent to the critical-fractile rule for a single item.

* **CVaR** (Conditional Value at Risk) — maximise a weighted combination of
  ``E[profit]`` and the mean profit in the worst-``α`` fraction of scenarios.
  Use when downside protection is more important than the mean.

* **Utility** (Exponential / CARA) — maximise ``E[U(profit)]`` where
  ``U(x) = −exp(−x / ρ)``.  The risk-tolerance ``ρ`` controls curvature;
  ``ρ → ∞`` recovers SAA and ``ρ → 0`` becomes worst-case averse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import warnings

from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, minimize_scalar

from .forecasting.base import BaseForecaster
from .inventory_policy import InventoryPolicy
from .items import Item


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Solver(ABC):
    """Abstract base class for all newsvendor solvers."""

    @abstractmethod
    def solve(self, start_date: str, end_date: str) -> dict[str, int]:
        """
        Find optimal stock quantities for the planning horizon.

        Parameters
        ----------
        start_date, end_date : str
            Inclusive horizon passed to ``forecaster.get_demand_distribution``.

        Returns
        -------
        dict[str, int]
            Mapping of item name → optimal order quantity.
        """

    @abstractmethod
    def get_profit(self, allocation: dict[str, int] | None = None) -> float:
        """
        Expected profit for *allocation* evaluated against stored demand samples.

        Parameters
        ----------
        allocation : dict[str, int], optional
            Item name → quantity.  Defaults to the solution found by ``solve``.
        """


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


class ForecastSolver(Solver):
    """
    Newsvendor solver that draws demand samples from one or more
    ``BaseForecaster`` instances.

    Handles both single-item and multi-item problems.  For multi-item problems,
    each item is paired with its own (univariate) forecaster.  A single shared
    forecaster can be passed for the future multivariate case — see the
    ``demand_key`` parameter.

    Parameters
    ----------
    problems : tuple[Item, BaseForecaster] or list[tuple[Item, BaseForecaster]]
        A single ``(item, forecaster)`` pair or a list of such pairs.
        Each forecaster must already be in a state where
        ``get_demand_distribution`` can be called (i.e. ``smooth_and_filter``
        or ``forecast`` has been run).
    objective : {'SAA', 'CVaR', 'Utility'}
        Objective function used during optimisation. Default ``'SAA'``.
    limits : dict[str, float], optional
        Shared resource constraints across all items, e.g.
        ``{"budget": 50_000, "storage_m3": 200}``.
        Each ``Item.constraints`` dict must contain the corresponding keys.
    cvar_alpha : float
        Tail fraction for CVaR — e.g. ``0.10`` captures the worst 10 % of
        scenarios.  Only used when ``objective='CVaR'``.  Default ``0.10``.
    cvar_lambda : float
        Weight on the CVaR term: ``0`` → pure SAA, ``1`` → pure tail
        minimisation.  Default ``0.50``.
    risk_aversion : float
        Dimensionless risk-aversion level in ``[0, 1)`` for exponential
        utility.  ``0`` (default) is risk-neutral and recovers SAA; values
        approaching ``1`` are maximally risk-averse.  Internally mapped to
        the scale parameter ``ρ = σ_profit · (1 − r) / r``.
    """

    def __init__(
        self,
        problems: list[tuple[Item, BaseForecaster]] | tuple[Item, BaseForecaster],
        objective: Literal["SAA", "CVaR", "Utility"] = "SAA",
        limits: dict[str, float] | None = None,
        cvar_alpha: float = 0.10,
        cvar_lambda: float = 0.50,
        risk_aversion: float = 0.0,
        policies: dict[str, InventoryPolicy] | None = None,
    ):
        if not isinstance(problems, list):
            problems = [problems]
        self.problems = problems
        self.objective = objective
        self.limits = limits or {}
        self.cvar_alpha = cvar_alpha
        self.cvar_lambda = cvar_lambda
        self.risk_aversion = risk_aversion
        self.policies: dict[str, InventoryPolicy] = policies or {}

        self.allocation: dict[str, int] | None = None
        self.shadow_prices: dict[str, float] = {}

        # Populated by solve()
        self._demand_matrix: np.ndarray | None = None  # (n_items, n_samples)
        self._yield_matrix: np.ndarray | None = None  # (n_items, n_samples)

    def solve(self, start_date: str, end_date: str) -> dict[str, int]:
        """
        Pull demand samples from each forecaster then optimise stock quantities.

        The horizon ``[start_date, end_date]`` is forwarded directly to
        ``forecaster.get_demand_distribution``, which should return the
        *total* demand over that window per posterior draw.

        Parameters
        ----------
        start_date, end_date : str
            Planning horizon (any format accepted by pandas).

        Returns
        -------
        dict[str, int]
            Optimal quantities keyed by item name.
        """
        self._demand_matrix = self._pull_demand(start_date, end_date)
        n_samples = self._demand_matrix.shape[1]
        n_items = len(self.problems)

        self._yield_matrix = np.array(
            [item.yield_distribution.sample(n_samples) for item, _ in self.problems]
        )

        if n_items == 1 and not self.limits:
            quantities = self._solve_scalar()
        else:
            quantities = self._solve_vector()

        self.allocation = {
            self.problems[i][0].name: int(max(0, quantities[i])) for i in range(n_items)
        }
        return self.allocation

    def get_profit(self, allocation: dict[str, int] | None = None) -> float:
        """
        Expected profit across all stored demand scenarios.

        Parameters
        ----------
        allocation : dict[str, int], optional
            If ``None``, uses the allocation found by the last ``solve`` call.

        Returns
        -------
        float
            Mean profit over all posterior scenarios.
        """
        if self._demand_matrix is None:
            raise RuntimeError("Call solve() before get_profit().")
        allocation = allocation or self.allocation
        if allocation is None:
            raise RuntimeError("No allocation available.  Call solve() first.")

        q = np.array(
            [allocation.get(item.name, 0) for item, _ in self.problems],
            dtype=float,
        )
        return float(np.mean(self._portfolio_profits(q)))

    def summary(self) -> dict:
        """
        Diagnostic summary of the solution.

        Returns
        -------
        dict with keys:

        * ``allocation`` — optimal quantities
        * ``expected_profit`` — mean profit across scenarios
        * ``profit_std`` — standard deviation of profit
        * ``cvar_{cvar_alpha}pct`` — CVaR at ``cvar_alpha``
        * ``service_level`` — fraction of scenarios where all demand is met
        * ``certainty_equivalent`` / ``risk_premium`` — only when
          ``objective='Utility'`` with finite ``risk_aversion``
        * ``shadow_prices`` — constraint duals (multi-item only)
        """
        if self.allocation is None or self._demand_matrix is None:
            raise RuntimeError("Call solve() before summary().")

        q = np.array(
            [self.allocation[item.name] for item, _ in self.problems], dtype=float
        )
        profits = self._portfolio_profits(q)

        n_tail = max(1, int(np.ceil(self.cvar_alpha * len(profits))))
        cvar_val = float(np.mean(np.sort(profits)[:n_tail]))

        Q_eff = q.reshape(-1, 1) * self._yield_matrix
        service_level = float(np.mean(np.all(Q_eff >= self._demand_matrix, axis=0)))

        result: dict = {
            "allocation": self.allocation,
            "expected_profit": float(np.mean(profits)),
            "profit_std": float(np.std(profits)),
            f"cvar_{int(self.cvar_alpha * 100)}pct": cvar_val,
            "service_level": service_level,
        }

        if not np.isinf(self.risk_aversion):
            eu = float(np.mean(np.exp(-profits / self.risk_aversion)))
            ce = float(-self.risk_aversion * np.log(max(eu, 1e-300)))
            result["certainty_equivalent"] = ce
            result["risk_premium"] = float(np.mean(profits)) - ce

        if self.shadow_prices:
            result["shadow_prices"] = self.shadow_prices

        return result

    def _pull_demand(self, start_date: str, end_date: str) -> np.ndarray:
        """
        Call each forecaster's ``get_demand_distribution`` and return a
        ``(n_items, n_samples)`` demand matrix.

        When multiple forecasters return different numbers of samples,
        the matrix is trimmed to the shortest sample vector.
        """
        rows = []
        for _, forecaster in self.problems:
            dataset = forecaster.get_demand_distribution(start_date, end_date)
            samples = np.asarray(dataset["demand"].values, dtype=float).ravel()
            rows.append(samples)

        n_min = min(len(r) for r in rows)
        return np.array([r[:n_min] for r in rows])

    def _portfolio_profits(self, quantities: np.ndarray) -> np.ndarray:
        """
        Compute profit for each demand scenario.

        Parameters
        ----------
        quantities : ndarray, shape (n_items,)

        Returns
        -------
        ndarray, shape (n_samples,)
            Total portfolio profit per scenario.
        """
        assert self._demand_matrix is not None and self._yield_matrix is not None
        Q = quantities.reshape(-1, 1)
        Q_eff = Q * self._yield_matrix  # effective supply
        sales = np.minimum(Q_eff, self._demand_matrix)
        leftover = Q_eff - sales

        prices = np.array([i.selling_price for i, _ in self.problems]).reshape(-1, 1)
        costs = np.array([i.cost_price for i, _ in self.problems]).reshape(-1, 1)
        salvages = np.array([i.salvage_value for i, _ in self.problems]).reshape(-1, 1)

        revenue = sales * prices
        salvage_income = leftover * salvages
        procurement = Q * costs

        return np.sum(revenue + salvage_income - procurement, axis=0)

    def _objective_fn(self, quantities: np.ndarray) -> float:
        match self.objective:
            case "SAA":
                return self._saa(quantities)
            case "CVaR":
                return self._cvar(quantities)
            case "Utility":
                return self._utility(quantities)
            case _:
                raise ValueError(
                    f"Unknown objective {self.objective!r}. "
                    "Choose from 'SAA', 'CVaR', 'Utility'."
                )

    def _saa(self, quantities: np.ndarray) -> float:
        """Negative expected profit (SAA minimises this)."""
        return -float(np.mean(self._portfolio_profits(quantities)))

    def _cvar(self, quantities: np.ndarray) -> float:
        """
        Negative weighted combination of expected profit and CVaR.

        Objective = -[(1 - λ) · E[profit] + λ · CVaR_alpha[profit]]
        """
        profits = self._portfolio_profits(quantities)
        expected = float(np.mean(profits))
        n_tail = max(1, int(np.ceil(self.cvar_alpha * len(profits))))
        cvar_val = float(np.mean(np.sort(profits)[:n_tail]))
        return -((1 - self.cvar_lambda) * expected + self.cvar_lambda * cvar_val)

    def _utility(self, quantities: np.ndarray) -> float:
        """
        Expected exponential disutility E[exp(−profit / ρ)].

        Minimising this maximises E[U(profit)] where U(x) = −exp(−x / ρ).
        ρ is derived from the dimensionless ``risk_aversion`` parameter r via
        ρ = σ_profit · (1 − r) / r, so r = 0 recovers SAA and r → 1 is
        maximally risk-averse.
        """
        profits = self._portfolio_profits(quantities)
        r = self.risk_aversion
        if r <= 0.0:
            return -float(np.mean(profits))
        sigma = float(np.std(profits))
        if sigma < 1e-10:
            return -float(np.mean(profits))
        rho = sigma * (1.0 - r) / r
        return float(np.mean(np.exp(-profits / rho)))

    def _solve_scalar(self) -> np.ndarray:
        """Single-item, unconstrained optimization via minimize_scalar."""
        assert self._demand_matrix is not None
        demand = self._demand_matrix[0]
        upper = float(np.percentile(demand, 99.9)) * 2 + 1

        item = self.problems[0][0]
        policy = self.policies.get(item.name)
        lower = policy.min_quantity(demand) if policy is not None else 0.0

        result = minimize_scalar(
            lambda q: self._objective_fn(np.array([q])),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 0.5},
        )
        return np.array([max(lower, round(float(result.x)))])

    def _solve_vector(self) -> np.ndarray:
        """Multi-item (or constrained) optimization via trust-region method."""
        assert self._demand_matrix is not None and self._yield_matrix is not None
        n_items = len(self.problems)

        x0 = np.array(
            [
                np.median(self._demand_matrix[i])
                / max(float(self._yield_matrix[i].mean()), 1e-6)
                for i in range(n_items)
            ]
        )

        # Policy-aware bounds and warm-start floor
        bounds = []
        for i, (item, _) in enumerate(self.problems):
            policy = self.policies.get(item.name)
            lb = policy.min_quantity(self._demand_matrix[i]) if policy is not None else 0.0
            bounds.append((lb, None))
            x0[i] = max(x0[i], lb)

        constraints: list[LinearConstraint | NonlinearConstraint] = (
            [self._build_constraint()] if self.limits else []
        )
        constraints.extend(self._build_service_constraints())

        result = minimize(
            fun=self._objective_fn,
            x0=x0,
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            hess=lambda _: np.zeros((n_items, n_items)),
        )

        if getattr(result, "constr_violation", 0.0) > 1e-4:
            warnings.warn(
                f"Service-level constraint may not be fully satisfied "
                f"(violation={result.constr_violation:.4f}). "
                "Try relaxing service_level_target or increasing MCMC samples.",
                UserWarning,
                stacklevel=3,
            )

        if getattr(result, "v", None):
            names = list(self.limits.keys())
            self.shadow_prices = {
                name: float(abs(v)) for name, v in zip(names, result.v[0])
            }

        return np.floor(result.x).astype(int)

    def _build_constraint(self) -> LinearConstraint:
        """Build a LinearConstraint from ``item.constraints`` and ``self.limits``."""
        names = list(self.limits.keys())
        n = len(self.problems)
        A = np.zeros((len(names), n))
        for r, name in enumerate(names):
            for c, (item, _) in enumerate(self.problems):
                A[r, c] = item.constraints.get(name, 0.0)

        upper = np.array(list(self.limits.values()), dtype=float)
        lower = np.full(len(names), -np.inf)
        return LinearConstraint(A, lower, upper)  # type: ignore[arg-type]

    def _build_service_constraints(self) -> list[NonlinearConstraint]:
        """
        Build one ``NonlinearConstraint`` per item whose policy has a
        ``service_level_target > 0``.

        Constraint for item *i*::

            g_i(Q) = mean(Q_i * yield_i >= demand_i) - sl_target >= 0

        The closure captures ``idx`` and ``target`` by value via the inner
        ``_make`` helper to avoid the classic loop-variable aliasing bug.
        No analytic Jacobian is provided — scipy uses finite differences.
        """
        assert self._demand_matrix is not None and self._yield_matrix is not None
        constraints: list[NonlinearConstraint] = []
        for i, (item, _) in enumerate(self.problems):
            policy = self.policies.get(item.name)
            sl_target: float = getattr(policy, "service_level_target", 0.0)
            if policy is None or sl_target <= 0.0:
                continue

            def _make(idx: int, target: float) -> NonlinearConstraint:
                def g(Q: np.ndarray) -> float:
                    q_eff = Q[idx] * self._yield_matrix[idx]  # type: ignore[index]
                    return float(np.mean(q_eff >= self._demand_matrix[idx])) - target  # type: ignore[index]

                return NonlinearConstraint(g, lb=0.0, ub=np.inf)

            constraints.append(_make(i, sl_target))
        return constraints
