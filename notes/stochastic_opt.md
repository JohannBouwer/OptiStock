# Basic Stochastic Optimisation

## Jensen's Inequality 
For a convex function $f$ and random variable $\xi$:

$$E[f(\xi)] \geq f(E[\xi])$$

Equality holds only when $f$ is linear or $\xi$ is deterministic. If your objective involves a convex cost function $f(x, \xi)$ (e.g. inventory holding + stockout cost), replacing the random demand $\xi$ with its mean $\mu$ gives you $f(x, \mu)$ but the true expected cost is $E[f(x, \xi)] \geq f(x, \mu)$. You are systematically underestimating the cost, so the "optimal" order quantity you find will be wrong. Stochastic problems require optimising over the distribution, not at the mean.

## Sample Average Approximation (SAA)

The true stochastic program:

$$\min_x \; E_\xi[f(x, \xi)]$$

is usually unknowable. SAA approximates the expectation with a finite sample $\{\xi_1, \dots, \xi_N\}$:

$$\min_x \; \frac{1}{N} \sum_{i=1}^{N} f(x, \xi_i)$$
 
SAA weights every scenario equally, it doesn't care whether a bad outcome is a small miss or a catastrophic loss, only about the average. This is the risk-neutral stance: maximise/minimise expected value.

## Conditional Value at Risk (CVaR)

CVaR extends SAA by asking: what is the expected profit in the worst $(1-\alpha)$% of scenarios? (e.g. at $\alpha = 0.95$, the worst 5%) While SAA optimizes $E[\text{Profit}]$, the Mean-CVaR objective optimizes $(1-\lambda)·E[\text{Profit}] + \lambda·\text{CVaR}_\alpha[\text{Profit}]$. This allows us to consider the width of the distribution we are optimising, and therefore the "risk". The function punishes wide distributions ( high risk ) and prefers narrow distribution ( low risk, more certainty).


## Exponential Utility & the Certainty Equivalent

Exponential utility function:

$$U(w) = -e^{-\lambda w}, \quad \lambda > 0$$

$\lambda$ is the risk-aversion coefficient. We map our distribution to a domain that "feels" more natural. So early gains are weighed more than later gains. Basically, going from 10 $\rightarrow$ 20 in profit "feels" better than going from 50 $\rightarrow$ 60. We then take the average of $U$ as our objective, and $\lambda$ is a measure of the risk we are willing to accept.

### Certainty Equivalent (CE):
The CE is the guaranteed payoff that makes you indifferent to a random payoff. It tells us if we are offered a guaranteed amount of x, instead of a random chance of y, we should accept the offer.

$$U(\text{CE}) = E[U(W)] \implies \text{CE} = -\frac{1}{\lambda} \ln E\!\left[e^{-\lambda W}\right]$$

This is the negative log of the moment-generating function of $W$ evaluated at $\lambda$.

**Effect of $\lambda$:**

| $\lambda$ | Behaviour |
|---|---|
| $\lambda \to 0$ | Risk-neutral: $\text{CE} \approx E[W]$ |
| $\lambda$ moderate | CE < E[W]; you'd accept less than the mean for certainty |
| $\lambda$ large | Strongly risk-averse; CE << E[W] |


