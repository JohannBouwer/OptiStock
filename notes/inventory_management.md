# Basic Inventory Management

## Censored Demand & Stockouts

A stockout occurs when true demand exceeds available inventory. The key accounting identity is:

$$\text{Realised demand} = \min(\text{True demand},\; \text{Stock on hand})$$
$$\text{Lost sales} = \max(\text{True demand} - \text{Stock on hand},\; 0)$$

The problem is that on a stockout day your sales record shows `stock on hand`, not true demand. This is demand censoring, the true demand is unobservable, you only see an upper bound on it.

If a forecaster trains on raw sales (realised demand) it learns the censored signal and systematically underestimates future demand. This perpetuates the problem: lower forecast → lower order → more stockouts → even more censoring.

**Two ways to correct for censoring:**

- **NaN-masking** (SSM): mask stockout days as `NaN`. The Kalman filter skips the update step on those observations and interpolates through the gap instead of learning from the artificially capped values.
- **Censored likelihood** (linear models): replace the Normal likelihood on stockout days with `pm.Censored(Normal)`, which uses the survival function $P(\text{true demand} \geq \text{observed})$ rather than treating the cap as an exact observation. The model integrates over all demand levels above the cap.

Both approaches tell the model "this value is a lower bound, not the truth." The NaN approach is simpler; the censored likelihood is more statistically precise because it uses the censoring information rather than discarding it.

## The 4 Inventory Policy Types

All classical inventory policies are defined by two decisions: **when to order** and **how much to order**. The four standard types combine two review modes (continuous vs periodic) with two quantity rules (fixed quantity vs order-up-to level).

### (s, Q) — Continuous Review, Fixed Order Quantity
Order a fixed quantity $Q$ whenever inventory position falls below reorder point $s$.

- **When**: any time inventory position hits $s$
- **How much**: always $Q$ (fixed)
- Intuition: the supermarket shelf model — as soon as a shelf hits the trigger mark, restock a fixed case
- Works well for stable, high-volume items where $Q$ can be optimised once

### (s, S) — Continuous Review, Order-Up-To
Order enough to bring inventory position up to $S$ whenever it falls below $s$.

- **When**: any time inventory position < $s$
- **How much**: $S - \text{current inventory position}$ (variable)
- Generalises (s, Q): if demand comes in lumps, $Q$ would sometimes overshoot; ordering up to $S$ avoids that
- The gap $(S - s)$ is the safety stock cushion

### (R, S) — Periodic Review, Order-Up-To (Base Stock)
Every $R$ periods, order enough to bring inventory position up to $S$.

- **When**: fixed review cycle (e.g. every 7 days)
- **How much**: $S - \text{current inventory position}$ (variable)
- Planning horizon = $R + L$ where $L$ is lead time — you must cover demand until the *next* order arrives
- This is `PeriodicOrderUpTo` in OptiStock. The profit-optimal $S$ is the critical-fractile quantile of demand over $R + L$ days

### (R, s, S) — Periodic Review, Min-Max
Every $R$ periods, check inventory. If position is below $s$, order up to $S$. Otherwise do nothing.

- **When**: fixed review cycle, but only order if inventory is low enough to warrant it
- **How much**: $S - \text{current inventory position}$ (if triggered)
- Avoids placing trivially small orders: if only 2 units were sold in the review period, don't bother ordering
- Higher coordination cost than (R, S) but reduces order frequency

---

### Service-Level Constraints

Any of the above policies can have a **cycle service level (CSL)** floor imposed on top of the quantity decision. Instead of asking "what $S$ maximises expected profit?", the constraint asks "what is the smallest $S$ such that $P(\text{demand} \leq S) \geq \text{target}$?"

In OptiStock, `PeriodicBaseStock` adds this as a `NonlinearConstraint` on top of the (R, S) profit optimisation:

$$Q^* = \arg\max_{Q} \; E[\text{Profit}] \quad \text{subject to} \quad P(Q \geq \text{demand}) \geq \alpha$$

The cost of the guarantee is the gap between the profit at the constrained $Q^*$ and the unconstrained profit-optimal quantity. Higher CSL targets → more stock held → lower expected profit. This trade-off is item-specific: a high-margin item with brand-damage risk from stockouts justifies a higher CSL target than a commodity item.

**Effective horizon** = $L + R$ (lead time + review period). This is the demand window the solver sees — you are only responsible for covering demand until the next replenishment can arrive; ordering beyond that ties up capital that a future cycle will handle.

**Net order** = $\max(0,\; Q^* - \text{inventory position})$ where inventory position = on-hand + on-order. The solver outputs a gross quantity; what you actually place today accounts for what you already have.
