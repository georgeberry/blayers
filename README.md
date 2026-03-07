[![Coverage Status](https://coveralls.io/repos/github/georgeberry/blayers/badge.svg?branch=main)](https://coveralls.io/github/georgeberry/blayers?branch=main) [![License](https://img.shields.io/github/license/georgeberry/blayers)](LICENSE) [![PyPI](https://img.shields.io/pypi/v/blayers)](https://pypi.org/project/blayers/) [![Read - Docs](https://img.shields.io/badge/Read-Docs-2ea44f)](https://georgeberry.github.io/blayers/) [![View - GitHub](https://img.shields.io/badge/View-GitHub-89CFF0)](https://github.com/georgeberry/blayers) [![PyPI Downloads](https://static.pepy.tech/badge/blayers)](https://pepy.tech/projects/blayers)



# BLayers

The missing layers package for Bayesian inference.

**BLayers is in beta, errors are possible! We invite you to contribute on [GitHub](https://github.com/georgeberry/blayers).**

## Install

```
pip install blayers
```

deps: `numpyro`, `jax`, `optax`.

## Concept

Easily build Bayesian models from parts, abstract away the boilerplate, and
tweak priors as you wish.

Inspired by Keras and TensorFlow Probability, but made specifically for NumPyro + JAX.

BLayers provides:

- **Layers** that encapsulate generative model structures with sensible priors
- **Links** that handle likelihoods and output distributions
- **`fit()`** — a high-level helper to train models via VI, MCMC, or SVGD with one call
- **Spline utilities** for non-linear feature transformations
- Pure NumPyro under the hood — integrates with all of NumPyro's tools

---

## Quick start

```python
from blayers.fit import fit
from blayers.layers import AdaptiveLayer, InterceptLayer
from blayers.links import gaussian_link
from blayers.sampling import autoreshape

@autoreshape
def model(x, y=None):
    mu = AdaptiveLayer()("beta", x)
    intercept = InterceptLayer()("intercept")
    return gaussian_link(mu + intercept, y)

result = fit(model, y=y, num_steps=1000, batch_size=256, lr=0.01, x=X)

preds = result.predict(x=X, num_samples=500)   # Predictions(mean, std, samples)
summary = result.summary(x=X)                  # posterior stats per latent variable
```

---

## The `fit()` API

`fit()` handles the guide, ELBO, batching, and LR schedule. Swap `method=` to change
inference — the model doesn't change.

```python
# Variational Inference (default)
result = fit(model, y=y, num_steps=1000, batch_size=256, lr=0.01, x=X)

# MCMC (NUTS)
result = fit(model, y=y, method="mcmc", num_mcmc_samples=1000, num_warmup=500, x=X)

# SVGD
result = fit(model, y=y, method="svgd", num_steps=1000, num_particles=20, x=X)
```

Keyword arguments that are JAX arrays are treated as **data** (batched during training).
Non-array kwargs (ints, floats, strings) are bound to the model as **constants**.

---

## Layers

The full set of layers:

| Layer | Description |
|---|---|
| `AdaptiveLayer` | Hierarchical prior: `λ ~ HalfNormal(1)`, `β ~ Normal(0, λ)` |
| `FixedPriorLayer` | Fixed prior over coefficients (e.g. Normal or Laplace) |
| `InterceptLayer` | Bias/intercept term |
| `EmbeddingLayer` | Bayesian embeddings for sparse categorical features |
| `RandomEffectsLayer` | Classical random effects (embedding with dim=1) |
| `FMLayer` | Factorization Machine (order 2) |
| `FM3Layer` | Factorization Machine (order 3) |
| `LowRankInteractionLayer` | Low-rank UV interaction between two feature sets |
| `BilinearLayer` | Full bilinear interaction: `x^T W z` |
| `LowRankBilinearLayer` | Low-rank bilinear interaction |
| `InteractionLayer` | All pairwise interactions between two feature sets |
| `RandomWalkLayer` | Gaussian random walk prior (for time/ordered indices) |
| `HorseshoeLayer` | Horseshoe prior for sparse regression |
| `AttentionLayer` | Single-head self-attention over the feature dimension |

All layers accept custom prior distributions and kwargs:

```python
from numpyro import distributions

mu = AdaptiveLayer(
    lmbda_dist=distributions.Exponential,
    coef_dist=distributions.Laplace,
    lmbda_kwargs={"rate": 1.0},
    coef_kwargs={"loc": 0.0},
)("mu", x)
```

Bad kwargs raise `TypeError` at **construction time**, not call time.

---

## Links

Link functions connect model predictions to observations.

| Link | Description |
|---|---|
| `gaussian_link` | Gaussian with learned, fixed, or per-observation scale (see below) |
| `gaussian_link_exp` | Gaussian with `sigma ~ Exp(1)` |
| `lognormal_link_exp` | LogNormal with `sigma ~ Exp(1)` |
| `logit_link` | Bernoulli (logistic regression) |
| `poission_link` | Poisson |
| `negative_binomial_link` | Negative Binomial (overdispersed counts) |
| `ordinal_link` | Cumulative logit / proportional odds for ordinal outcomes |
| `zip_link` | Zero-inflated Poisson |
| `beta_link` | Beta regression for proportions in (0, 1) |

### `gaussian_link` — three scale modes

```python
# Default: sigma ~ Exp(1) learned from data
gaussian_link(mu, y)

# Fixed known scale (e.g. from XGBoost quantile regression)
gaussian_link(mu, y, scale=pred_std)

# Learned scale from a layer — softplus applied internally for stable gradients
raw = AdaptiveLayer()("log_sigma", x)
gaussian_link(mu, y, untransformed_scale=raw)
```

---

## Splines

Non-linear feature transformations via B-splines. Compute the basis once, then
pass it to any layer.

```python
from blayers.layers import make_knots, bspline_basis, AdaptiveLayer

knots = make_knots(x_train, num_knots=10)   # clamped knot vector from data quantiles
B = bspline_basis(x, knots)                 # (n, num_basis) design matrix

def model(x, y=None):
    B = bspline_basis(x, knots)
    f = AdaptiveLayer()("f", B)             # adaptive prior over spline coefficients
    return gaussian_link(f, y)
```

Additive models are straightforward:

```python
def model(x1, x2, y=None):
    f1 = AdaptiveLayer()("f1", bspline_basis(x1, knots1))
    f2 = AdaptiveLayer()("f2", bspline_basis(x2, knots2))
    return gaussian_link(f1 + f2, y)
```

---

## The adaptive prior in detail

The fundamental building block is the adaptive prior:

```
λ ~ HalfNormal(1)
β ~ Normal(0, λ)
y ~ Normal(β · x, σ)
```

`AdaptiveLayer` encapsulates this. The following are equivalent:

```python
# With BLayers
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

def model(x, y=None):
    mu = AdaptiveLayer()("mu", x)
    return gaussian_link(mu, y)
```

```python
# Raw NumPyro
from numpyro import distributions, sample

def model(x, y=None):
    lmbda = sample("lmbda", distributions.HalfNormal(1.0))
    beta = sample("beta", distributions.Normal(0.0, lmbda).expand([x.shape[1], 1]))
    mu = x @ beta
    sigma = sample("sigma", distributions.Exponential(1.0))
    return sample("obs", distributions.Normal(mu, sigma), obs=y)
```

---

## Batched ELBO (low-level)

If you prefer wiring up SVI yourself, `Batched_Trace_ELBO` handles mini-batching
without requiring `numpyro.plate`.

```python
from blayers.infer import Batched_Trace_ELBO, svi_run_batched

svi = SVI(model, guide, optax.adam(lr), loss=Batched_Trace_ELBO(num_obs=n, batch_size=256))

result = svi_run_batched(svi, rng_key, num_epochs=10, batch_size=256, x=X, y=y)
```

**⚠️ `numpyro.plate` and `Batched_Trace_ELBO` do not mix.** Use one or the other.

---

## Reparameterisation for MCMC

`@autoreparam` automatically reparameterises `LocScale` distributions for better
NUTS performance — no manual `numpyro.handlers.reparam` needed.

```python
from blayers.sampling import autoreparam
from numpyro.infer import MCMC, NUTS

@autoreparam
def model(x, y=None):
    mu = AdaptiveLayer()("mu", x)
    return gaussian_link(mu, y)

mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000)
mcmc.run(rng_key, x=X, y=y)
```

Or just use `fit(model, method="mcmc", ...)` which handles this automatically.
