"""
Bayesian Synthetic Control for evaluating per-item interventions.

Wraps :class:`causalpy.SyntheticControl` with the project's wide-format panel
data convention (one ``date`` column plus one column per item) and the
project's ``Prior`` / ``BasePriors`` configuration pattern.
"""

from dataclasses import dataclass
from typing import Any

import arviz as az
import causalpy as cp
import numpy as np
import pandas as pd
from causalpy.pymc_models import Prior as CPPrior
from causalpy.pymc_models import WeightedSumFitter

from .base import BaseCausalEstimator
from .priors import SyntheticControlPriors


@dataclass
class CausalEffect:
    """
    One-line answer to "did the intervention change sales, and by how much?".

    The ``__repr__`` renders the headline the project uses everywhere:
    ``"<intervention> changed sales of <item> by +X.X% [94% HDI: +L.L%, +H.H%]"``.

    ``mean_abs_lift`` is the **cumulative** post-period absolute effect (raw
    units). ``avg_abs_lift`` is the **per-period** absolute effect with its
    posterior standard deviation in ``avg_abs_lift_sd`` — the pair used to
    build :class:`LiftConstraint` for the linear forecasters, whose
    ``beta_event`` coefficient is itself a per-active-day additive effect.
    """

    treated_item: str
    intervention_name: str
    mean_lift_pct: float
    hdi_low_pct: float
    hdi_high_pct: float
    mean_abs_lift: float
    avg_abs_lift: float
    avg_abs_lift_sd: float
    hdi_prob: float = 0.94

    def __repr__(self) -> str:
        return (
            f"{self.intervention_name} changed sales of {self.treated_item} by "
            f"{self.mean_lift_pct:+.1f}% "
            f"[{int(round(self.hdi_prob * 100))}% HDI: "
            f"{self.hdi_low_pct:+.1f}%, {self.hdi_high_pct:+.1f}%]"
        )


class SyntheticControl(BaseCausalEstimator):
    """
    Bayesian synthetic control built on top of CausalPy's ``WeightedSumFitter``.

    Donor items are linearly combined (non-negative weights summing to one) to
    reconstruct the pre-intervention behaviour of the treated item; the
    post-intervention gap is the causal effect.

    Parameters
    ----------
    data
        Wide-format panel: one ``date_col`` and one column per item.
    treated_item
        Column name of the unit that received the intervention.
    donor_items
        Column names used as the control pool. Must not include ``treated_item``.
    treatment_date
        Date the intervention starts. The first row whose date is ``>=`` this
        value is treated as the first post-period observation.
    date_col
        Name of the date column in ``data``. Defaults to ``"date"``.
    intervention_name
        Human-readable label for the intervention; used in ``CausalEffect.__repr__``.
    priors
        Optional :class:`SyntheticControlPriors`. Defaults are CausalPy's:
        Dirichlet(1, ..., 1) on donor weights and HalfNormal(1) on sigma.
    sampler
        PyMC NUTS sampler backend; ``"numpyro"`` (default) or ``"nutpie"`` to
        match the existing forecasting models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treated_item: str,
        donor_items: list[str],
        treatment_date: str | pd.Timestamp,
        date_col: str = "date",
        intervention_name: str = "intervention",
        priors: SyntheticControlPriors | None = None,
        sampler: str = "numpyro",
    ) -> None:
        if date_col not in data.columns:
            raise KeyError(f"date_col {date_col!r} not in data columns")
        if treated_item not in data.columns:
            raise KeyError(f"treated_item {treated_item!r} not in data columns")
        missing = [d for d in donor_items if d not in data.columns]
        if missing:
            raise KeyError(f"donor_items not in data columns: {missing}")
        if treated_item in donor_items:
            raise ValueError(f"treated_item {treated_item!r} cannot also be a donor")
        if not donor_items:
            raise ValueError("donor_items must be a non-empty list")

        df = data.sort_values(date_col).reset_index(drop=True).copy()
        df[date_col] = pd.to_datetime(df[date_col])
        treatment_ts = pd.Timestamp(treatment_date)

        post_mask = df[date_col] >= treatment_ts
        if not post_mask.any():
            raise ValueError(
                f"treatment_date {treatment_ts.date()} is after the last "
                f"observation {df[date_col].iloc[-1].date()}"
            )
        if post_mask.all():
            raise ValueError(
                f"treatment_date {treatment_ts.date()} is at or before the "
                f"first observation; need pre-period data to fit donor weights"
            )

        self.data = df
        self.treated_item = treated_item
        self.donor_items = list(donor_items)
        self.treatment_date = treatment_ts
        self.date_col = date_col
        self.intervention_name = intervention_name
        self.priors = priors or SyntheticControlPriors()
        self.sampler = sampler

        cols = [date_col, treated_item] + self.donor_items
        raw_panel = df[cols].set_index(date_col)
        self._scale = float(max(raw_panel.values.max(), 1e-12))
        self._cp_data = raw_panel / self._scale
        self._treatment_time = self._cp_data.index[
            self._cp_data.index.get_indexer([treatment_ts], method="bfill")[0]
        ]

        self._cp_result: cp.SyntheticControl | None = None
        self.idata: az.InferenceData | None = None

    def _build_causalpy_priors(self) -> dict[str, CPPrior]:
        """Translate project ``Prior`` objects into CausalPy/pymc_extras priors."""
        n_donors = len(self.donor_items)

        dw = self.priors.donor_weights
        if dw.distribution != "Dirichlet":
            raise ValueError(
                f"donor_weights must be Dirichlet, got {dw.distribution!r}"
            )
        alpha_scalar = float(dw.params.get("a", 1.0))
        beta_prior = CPPrior(
            "Dirichlet",
            a=np.full(n_donors, alpha_scalar),
            dims=["treated_units", "coeffs"],
        )

        sg = self.priors.sigma
        if sg.distribution != "HalfNormal":
            raise ValueError(f"sigma must be HalfNormal, got {sg.distribution!r}")
        sigma_scale = float(sg.params.get("sigma", 1.0))
        y_hat_prior = CPPrior(
            "Normal",
            sigma=CPPrior("HalfNormal", sigma=sigma_scale, dims=["treated_units"]),
            dims=["obs_ind", "treated_units"],
        )

        return {"beta": beta_prior, "y_hat": y_hat_prior}

    def fit(
        self,
        samples: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        random_seed: int | None = None,
        progressbar: bool = True,
        min_donor_correlation: float = 0.0,
    ) -> az.InferenceData:
        """
        Sample the posterior of the synthetic-control model.

        ``samples`` / ``tune`` / ``chains`` / ``target_accept`` mirror the
        forecasting models. CausalPy uses the PyMC name ``draws`` internally;
        we translate.
        """
        sample_kwargs: dict[str, Any] = {
            "draws": samples,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "progressbar": progressbar,
            "nuts_sampler": self.sampler,
        }
        if random_seed is not None:
            sample_kwargs["random_seed"] = random_seed

        model = WeightedSumFitter(
            sample_kwargs=sample_kwargs,
            priors=self._build_causalpy_priors(),
        )

        self._cp_result = cp.SyntheticControl(
            self._cp_data,
            treatment_time=self._treatment_time,
            control_units=self.donor_items,
            treated_units=[self.treated_item],
            model=model,
            min_donor_correlation=min_donor_correlation,
        )
        self.idata = self._cp_result.idata
        return self.idata

    def _require_fit(self) -> cp.SyntheticControl:
        if self._cp_result is None:
            raise RuntimeError("Call fit() before requesting summary or plots")
        return self._cp_result

    def summary(self, hdi_prob: float = 0.94) -> CausalEffect:
        """
        Extract the headline causal effect.

        ``hdi_prob`` defaults to the project standard 94%. CausalPy's
        ``effect_summary`` is parameterized on ``alpha = 1 - hdi_prob``.
        """
        result = self._require_fit()
        es = result.effect_summary(
            treated_unit=self.treated_item,
            alpha=1.0 - hdi_prob,
            direction="two-sided",
        )
        row = es.table.loc["average"]
        cumulative_mean = float(es.table.loc["cumulative", "mean"])
        avg_abs_lift, avg_abs_lift_sd = self._posterior_avg_impact()
        return CausalEffect(
            treated_item=self.treated_item,
            intervention_name=self.intervention_name,
            mean_lift_pct=float(row["relative_mean"]),
            hdi_low_pct=float(row["relative_hdi_lower"]),
            hdi_high_pct=float(row["relative_hdi_upper"]),
            mean_abs_lift=cumulative_mean * self._scale,
            avg_abs_lift=avg_abs_lift,
            avg_abs_lift_sd=avg_abs_lift_sd,
            hdi_prob=hdi_prob,
        )

    def _posterior_avg_impact(self) -> tuple[float, float]:
        """
        Posterior mean and sd of the per-period average post-period impact,
        in raw (unscaled) units. Computed directly from ``post_impact`` because
        CausalPy's ``effect_summary`` table only carries HDI bounds, not sd.
        """
        result = self._require_fit()
        post_impact = result.post_impact
        if "treated_units" in post_impact.dims:
            post_impact = post_impact.sel(treated_units=self.treated_item)
        time_dim = "obs_ind" if "obs_ind" in post_impact.dims else "time"
        avg_per_draw = post_impact.mean(dim=time_dim)
        mean = float(avg_per_draw.mean(dim=("chain", "draw")).values) * self._scale
        sd = float(avg_per_draw.std(dim=("chain", "draw")).values) * self._scale
        return mean, sd

    def plot(self) -> tuple:
        """Plot observed series, synthetic counterfactual, and impact band.

        The underlying CausalPy plot is drawn in scaled units; y-axis tick
        labels are rewritten to the original data scale for interpretation.
        """
        from matplotlib.ticker import FuncFormatter

        fig, axes = self._require_fit().plot(show=False)
        scale = self._scale
        formatter = FuncFormatter(lambda y, _pos: f"{y * scale:,.2f}")
        ax_iter = np.atleast_1d(axes).ravel()
        for ax in ax_iter:
            ax.yaxis.set_major_formatter(formatter)
        return fig, axes

    def plot_weights(self) -> tuple:
        """
        Posterior of the per-donor weights.

        Sparse weights are expected — donors with low pre-period similarity
        should land near zero.
        """
        import matplotlib.pyplot as plt

        result = self._require_fit()
        beta = result.idata.posterior["beta"].mean(dim=("chain", "draw"))
        beta_vals = beta.values.squeeze()

        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(self.donor_items))))
        order = np.argsort(beta_vals)[::-1]
        ax.barh(
            np.array(self.donor_items)[order],
            beta_vals[order],
            color="steelblue",
        )
        ax.set_xlabel("Posterior mean weight")
        ax.set_title(f"Donor weights for {self.treated_item}")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig, ax
