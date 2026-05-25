"""
Causal experiment constraints on forecaster coefficients.

Pattern: a measured per-active-day lift (mean and uncertainty) from a causal
experiment is fed back into the forecaster's ``pm.Model`` block as an extra
observed-Normal likelihood term on the corresponding ``beta_event`` coefficient.
This mirrors ``pymc_marketing``'s lift-test integration.

All values are stored in **raw (unscaled) units**; the forecaster divides by its
internal ``max_scaler`` when wiring the constraint into the model, so the user
never has to think about scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .synthetic_control import CausalEffect, SyntheticControl


@dataclass
class LiftConstraint:
    """
    A soft prior on a single ``beta_event`` coefficient sourced from a causal
    experiment.

    Parameters
    ----------
    event_name
        Must match an entry in ``forecaster.event_names`` (i.e. a key from the
        dict passed to ``create_events``).
    mean_abs_lift
        Posterior mean of the **per-active-day** absolute lift in raw units
        (same units as the forecaster's ``target_col``).
    sigma_abs_lift
        Posterior standard deviation of the per-active-day absolute lift, raw
        units. Smaller values pull ``beta_event`` more tightly toward
        ``mean_abs_lift``.
    item
        Required only for :class:`HierarchicalBayesTimeSeries` — names the
        treated item whose per-item coefficient is being constrained.
    """

    event_name: str
    mean_abs_lift: float
    sigma_abs_lift: float
    item: str | None = None

    def __post_init__(self) -> None:
        if self.sigma_abs_lift <= 0:
            raise ValueError(
                f"sigma_abs_lift must be > 0, got {self.sigma_abs_lift}"
            )

    @classmethod
    def from_causal_effect(
        cls,
        effect: "CausalEffect",
        event_name: str,
        *,
        item: str | None = None,
    ) -> "LiftConstraint":
        """
        Build a constraint from a :class:`CausalEffect` returned by
        :meth:`SyntheticControl.summary`.

        Uses ``effect.avg_abs_lift`` (per-active-day) and
        ``effect.avg_abs_lift_sd``, not the cumulative ``mean_abs_lift``.
        """
        return cls(
            event_name=event_name,
            mean_abs_lift=float(effect.avg_abs_lift),
            sigma_abs_lift=float(effect.avg_abs_lift_sd),
            item=item if item is not None else effect.treated_item,
        )

    @classmethod
    def from_synthetic_control(
        cls,
        sc: "SyntheticControl",
        event_name: str,
        *,
        item: str | None = None,
    ) -> "LiftConstraint":
        """
        Build a constraint by reading the posterior of a fitted
        :class:`SyntheticControl` directly.

        Equivalent to ``from_causal_effect(sc.summary(), ...)`` but skips the
        intermediate ``CausalEffect`` allocation.
        """
        mean, sd = sc._posterior_avg_impact()
        return cls(
            event_name=event_name,
            mean_abs_lift=mean,
            sigma_abs_lift=sd,
            item=item if item is not None else sc.treated_item,
        )
