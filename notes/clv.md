# Customer Lifetime Value

Notes on the "buy till you die" family of models. The setting is **non contractual**, meaning customers never tell you they have left, they just stop showing up. Everything below is about separating "gone quiet because they churned" from "gone quiet because they were always slow".

## Every customer gets their own rate

The core idea in all of these models. We do not fit one purchase rate for the whole business, each customer $i$ carries their own latent rate $\lambda_i$, and those rates are drawn from a population distribution:

$$\lambda_i \sim \text{Gamma}(r, \alpha)$$

Purchases then arrive as a Poisson process at that rate, for as long as the customer is still alive. The Gamma is the "population of customers", and $\lambda_i$ is where one particular customer sits inside it.

Two things fall out of this:

- Customers are allowed to be genuinely different. A weekly buyer and a once a year buyer are both perfectly normal draws from the same Gamma, not outliers to be cleaned away.
- Someone with a very short history gets pulled toward the population, because the data has not yet said much about their own $\lambda_i$. Same shrinkage logic as the hierarchical forecasters, see [hierarchical_models.md](hierarchical_models.md).

Dropout works the same way. Each customer also gets their own dropout rate $\mu_i \sim \text{Gamma}(s, \beta)$, so some people are naturally flighty and some are loyal, and we never observe which is which.

Practical note, $r$ and $\alpha$ trade off against each other, as do $s$ and $\beta$. A shape and a scale that grow together describe a similar distribution, so do not read too much into any one of them on its own. The quantity that is actually pinned down is the rate they imply, $r/\alpha$ and $s/\beta$.

## Pareto/NBD vs BG/NBD, when are you allowed to die

This is the one real assumption difference between the two, and it is worth knowing because it decides which one suits your business.

**Pareto/NBD** lets a customer drop out at **any moment in continuous time**. Their lifetime is $\tau_i \sim \text{Exponential}(\mu_i)$, and that clock runs independently of whether they are buying. Someone can quietly drift away in the middle of a long gap, with no transaction marking the exit.

**BG/NBD** only lets a customer drop out **immediately after a purchase**. After every transaction there is a coin flip, with probability $p$ they are finished forever, otherwise they carry on. Death is welded to the purchase event, so a customer sitting in a six month silence has not died in the model's eyes, they are simply overdue for the next transaction that would trigger the flip.

| | Pareto/NBD | BG/NBD |
|---|---|---|
| Dropout timing | any time, continuous | only right after a purchase |
| Lifetime | $\text{Exponential}(\mu_i)$ | geometric coin flip per transaction |
| Cost | heavier, hypergeometric terms | cheaper, closed form |
| Covariates | supported | not in the standard form |

The customer who bought once and vanished is the clearest illustration. Under BG/NBD they can only have died at that single purchase, so the whole story rests on the coin flip. Under Pareto/NBD the exponential clock has simply been running ever since, which usually feels closer to how people actually stop shopping somewhere.

BG/NBD is the cheaper and more common default, Pareto/NBD is the better fit when churn is a slow drift rather than a decision made at checkout, and it is the one to reach for if you want covariates.

Both are only the frequency half of the problem though, neither of them says anything about money.

## Gamma-Gamma, the money half

This one models **average spend per transaction**, not how often, and it assumes spend is independent of frequency. Worth checking that on real data before trusting it, and check the correlation on repeat buyers only, because customers with zero repeat purchases have a monetary value of zero and will manufacture a correlation that was never really there.

Same structure as before, each customer gets their own spend scale $\nu_i \sim \text{Gamma}(q, v)$, and each individual basket is $\text{Gamma}(p, \nu_i)$.

### The shrinking mechanic

This is the useful bit. The naive estimate of what a customer spends is just their observed average, $m_x$. For someone with a single purchase that number is almost pure noise, one unusually big basket and they look like a whale.

The model's estimate is instead a weighted average of that customer's own mean and the population mean:

$$E[M \mid x, m_x] = \underbrace{\frac{p x}{p x + q - 1}}_{\text{weight on the customer}} \, m_x \;+\; \underbrace{\frac{q - 1}{p x + q - 1}}_{\text{weight on the population}} \, \frac{p v}{q - 1}$$

where $x$ is the customer's number of transactions and $pv/(q-1)$ is the population mean spend. Read the two weights:

- $x$ small, nearly all the weight sits on the population mean, the customer's own average is barely trusted at all.
- $x$ large, the weight slides across onto $m_x$, and the customer speaks for themselves.

So as more purchases come in, that customer's spend distribution tightens around their own behaviour and drifts off the population. Early on it is wide and centred on everyone else, later it is narrow and centred on them. A customer with one 200 basket gets hauled back toward the average, a customer with thirty 200 baskets is left exactly where they are.

Fit this on **repeat buyers only**. A customer with zero repeat purchases carries no information about how their spend varies, so they have nothing to contribute.

## Putting them together

CLV is just the two halves multiplied, then discounted:

$$\text{CLV} \approx \sum_{t} \frac{E[\text{purchases in } t] \times E[\text{spend per purchase}]}{(1 + d)^{t}}$$

Expected future purchases come from the frequency model, expected basket value comes from Gamma-Gamma, and the discount rate $d$ handles present value.

Gotcha worth writing down, in `pymc-marketing` the CLV horizon is in **months** regardless of what time unit the RFM frame uses, while the frequency model's own horizon is in the data's units. Very easy to mix up, and the symptom is a leaderboard full of implausibly large numbers.

### Other combinations for other settings

Pareto/NBD plus Gamma-Gamma is only one pairing. The frequency half swaps out depending on how the business actually works:

| Setting | Frequency model |
|---|---|
| Continuous time, customers drift away | Pareto/NBD |
| Continuous time, cheaper, no covariates | BG/NBD |
| Customers with a single purchase handled better | MBG/NBD |
| Discrete or periodic opportunities, e.g. yearly renewals | BG/BB |
| Contractual, churn is actually observed | shifted beta geometric |

The money half stays Gamma-Gamma in most of these. If churn is directly observed, for example a subscription with a cancellation date, then none of this really applies, that is a survival problem and a time to event model fits it properly.

## Where covariates actually help

Covariates on Pareto/NBD shift the **prior** on a customer's rate. Once you have watched someone buy a few times their own history dominates and the covariate barely moves the prediction, and note that even a history of "bought nothing in 250 days" counts as strong information here, not missing information.

So covariates earn their keep at **cold start**, on customers with no history at all, which is exactly where plain RFM has nothing to say. Do not expect one to re-rank a customer base you have already been watching.
