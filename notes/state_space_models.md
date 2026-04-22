# State Space Models

## How does it differ from regression

Regression solves for parameters that best fit an equation to data, the coefficients are fixed and the model says nothing about how the system evolves through time. A state space model (SSM) instead defines a process that describes how a variable changes from one step to the next. The parameters themselves can drift with time.

The classic motivation is the Apollo missions. They had two sources of information: the physics of orbital mechanics (precise but never perfectly initialised) and radar observations (noisy but real). An SSM fuses both, it uses the physics to predict where the spacecraft should be, and the observation to correct that prediction.

## The Two Equations

Every SSM is defined by two equations.

**State-Transition (The Physics)** — how the hidden system evolves:

$$\alpha_{t+1} = T_t \, \alpha_t + R_t \, \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q_t)$$

**Observation (The Measurement)** — how we observe the hidden state:

$$y_t = Z_t \, \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H_t)$$

- $\alpha_t$: the **state vector** — holds all the system components (level, slope, seasonal coefficients, regression betas, etc.)
- $T_t$: the **transition matrix** — defines the physics, e.g. *new level = old level + old slope*
- $R_t$: the **selection matrix** — maps which states receive random shocks
- $Z_t$: the **observation matrix** — selects and combines states to produce the observed output $y_t$
- $H_t$: observation noise covariance — the "daily jitter" that doesn't represent a permanent change

## Hidden vs Observable States

Not every state in $\alpha_t$ is directly observed. Some are *hidden* (latent) — the model infers them from the data rather than measuring them directly.

For example, in a sales model the trend level and slope are hidden (you never observe them, you observe sales), while the regression coefficient on advertising spend is also hidden but is *driven by* an observable input. $Z_t$ is what pulls the hidden states together into the one thing you do observe: $y_t$.

## Innovations

The innovations $\eta_t$ are the random shocks that allow the states to drift. Without them, the model would be deterministic — the "physics" would lock everything to the initial conditions and nothing could ever change.

$Q_t$ controls the size of the drift. Large $Q_t$ means the state wanders freely; small $Q_t$ means the state is almost fixed. In practice you place a prior on $\sigma$ (the diagonal of $Q_t$) and let the data decide how much time-variation is warranted.

- **Innovations on trend**: allows the slope to accelerate or decelerate over time
- **Innovations on regression beta**: allows the effect of spend to change — e.g. saturation or seasonality in ad effectiveness
- **No innovations**: the parameter is fixed for all time (pooled across the full series)

## The Kalman Filter (Predict and Update)

The Kalman filter is the algorithm that runs the SSM forward in time. At each step it does two things:

**1. Predict** — use the transition equation to project the state forward:
$$\alpha_{t|t-1} = T_t \, \alpha_{t-1|t-1}$$

**2. Update** — when $y_t$ arrives, correct the prediction using the gap between what was expected and what was observed:
$$\alpha_{t|t} = \alpha_{t|t-1} + K_t \underbrace{(y_t - Z_t \, \alpha_{t|t-1})}_{\text{innovation residual}}$$

$K_t$ is the **Kalman gain** — it decides how much to trust the new observation vs. the prior prediction. High observation noise $H_t$ → small $K_t$ → trust the model more. High process noise $Q_t$ → large $K_t$ → trust the data more.

This is exactly Bayesian updating. The predicted state is the prior, the observation is the likelihood, and the updated state is the posterior. The Kalman filter is just the closed-form solution when both are Gaussian.

## Filtering vs Smoothing

- **Filtered** estimate at time $t$: uses only data up to and including $t$. This is the "real-time" estimate — what you would have known at the moment.
- **Smoothed** estimate at time $t$: uses all data, including future observations. It runs the filter forwards then backwards.

Smoothing answers: *"How much did my prediction for tomorrow change once I actually saw tomorrow's data — and how much of that should I apply back to today's estimate?"* The result is a cleaner decomposition of components (trend, seasonality, etc.) because future data resolves ambiguity about what was happening in the past.

For forecasting you use filtering; for decomposition and diagnosis you use smoothing.

## Structural Components

A structural SSM is built by composing modular components, each of which adds states to $\alpha_t$ and rows to $T_t$, $R_t$, $Z_t$:

| Component | What it models | Innovations? |
|---|---|---|
| `LevelTrend(order=2)` | Slowly drifting level + slope | Usually on slope only |
| `FrequencySeasonality` | Periodic pattern (e.g. weekly) | Optional — fixed vs. evolving |
| `Regression(innovations=True)` | Time-varying coefficient on an exogenous variable | Yes — beta drifts |
| `Regression(innovations=False)` | Fixed coefficient on an exogenous variable | No — beta is constant |
| `MeasurementError` | $H_t$ — observation noise floor | N/A |

The full state vector is just the concatenation of all component states. $T_t$ is block-diagonal across components.

## Priors and Parameters

As The model is bayesian it doesn't learn fixed coefficients but rather distributions over how the states are initialised and how much they vary. The key parameters are:

- `initial_*` — where does each state start? (mean of the prior at $t=0$)
- `sigma_*` — how much does each state drift per time step? The most important design choice.
  - Tight `sigma` → the component is nearly static (close to a fixed regression coefficient)
  - Wide `sigma` → the component evolves freely
- `P0` — the initial state covariance. Usually set to a diffuse prior (large diagonal) to let early data speak.

When fitting with MCMC (e.g. PyMC), the posteriors on `sigma_*` tell you how much time-variation the data actually supports. If the posterior is pushed towards zero, the component doesn't need to evolve.
