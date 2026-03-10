[![Coverage Status](https://coveralls.io/repos/github/georgeberry/blayers/badge.svg?branch=main)](https://coveralls.io/github/georgeberry/blayers?branch=main) [![License](https://img.shields.io/github/license/georgeberry/blayers)](LICENSE) [![PyPI](https://img.shields.io/pypi/v/blayers)](https://pypi.org/project/blayers/) [![Read - Docs](https://img.shields.io/badge/Read-Docs-2ea44f)](https://georgeberry.github.io/blayers/) [![View - GitHub](https://img.shields.io/badge/View-GitHub-89CFF0)](https://github.com/georgeberry/blayers) [![PyPI Downloads](https://static.pepy.tech/badge/blayers)](https://pepy.tech/projects/blayers)



# BLayers

The missing layers package for Bayesian inference.

**BLayers is in beta, errors are possible! We invite you to contribute on [GitHub](https://github.com/georgeberry/blayers).**

## Write code immediately

```
pip install blayers
```

deps are: `numpyro`, `jax`, and `optax`.

## Concept

<img width="646" height="258" alt="image" src="https://github.com/user-attachments/assets/21608d4a-fe83-4ebd-a8eb-a67774ea115f" />


Easily build Bayesian models from parts, abstract away the boilerplate, and
tweak priors as you wish.

Inspiration from Keras and Tensorflow Probability, but made specifically for Numpyro + Jax.

BLayers provides tools to

- Quickly build Bayesian models from layers which encapsulate useful model parts
- Fit models either using Variational Inference (VI) or your sampling method of
choice without having to rewrite models
- Write pure Numpyro to integrate with all of Numpyro's super powerful tools
- Add more complex layers (model parts) as you wish
- Fit models in a greater variety of ways with less code

## The starting point

The simplest non-trivial (and most important!) Bayesian regression model form is
the adaptive prior,

```
scale ~ HalfNormal(1)
beta  ~ Normal(0, scale)
y     ~ Normal(beta * x, 1)
```

BLayers encapsulates a generative model structure like this in a `BLayer`. The
fundamental building block is the `AdaptiveLayer`.

```python
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link_exp
def model(x, y):
    mu = AdaptiveLayer()('mu', x)
    return gaussian_link_exp(mu, y)
```

All `AdaptiveLayer` is doing is writing Numpyro for you under the hood. This
model is exacatly equivalent to writing the following, just using way less code.

```python
from numpyro import distributions, sample

def model(x, y):
    # Adaptive layer does all of this
    input_shape = x.shape[1]
    # adaptive prior
    scale = sample(
        name="scale",
        fn=distributions.HalfNormal(1.),
    )
    # beta coefficients for regression
    beta = sample(
        name="beta",
        fn=distributions.Normal(loc=0., scale=scale),
        sample_shape=(input_shape,),
    )
    mu = jnp.einsum('ij,j->i', x, beta)

    # the link function does this
    sigma = sample(name='sigma', fn=distributions.Exponential(1.))
    return sample('obs', distributions.Normal(mu, sigma), obs=y)
```

### Mixing it up

The `AdaptiveLayer` is also fully parameterizable via arguments to the class, so let's say you wanted to change the model from

```
scale ~ HalfNormal(1)
beta  ~ Normal(0, scale)
y     ~ Normal(beta * x, 1)
```

to

```
scale ~ Exponential(1.)
beta  ~ LogNormal(0, scale)
y     ~ Normal(beta * x, 1)
```

you can just do this directly via arguments

```python
from numpyro import distributions,
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link_exp
def model(x, y):
    mu = AdaptiveLayer(
        scale_dist=distributions.Exponential,
        prior_dist=distributions.LogNormal,
        scale_kwargs={'rate': 1.},
        prior_kwargs={'loc': 0.}
    )('mu', x)
    return gaussian_link_exp(mu, y)
```

### "Factories"

Since Numpyro traces `sample` sites and doesn't record any paramters on the class, you can re-use with a particular generative model structure freely.

```python
from numpyro import distributions
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link_exp

my_lognormal_layer = AdaptiveLayer(
    scale_dist=distributions.Exponential,
    prior_dist=distributions.LogNormal,
    scale_kwargs={'rate': 1.},
    prior_kwargs={'loc': 0.}
)

def model(x, y):
    mu = my_lognormal_layer('mu1', x) + my_lognormal_layer('mu2', x**2)
    return gaussian_link_exp(mu, y)
```

## Layers

The full set of layers included with BLayers:

- `AdaptiveLayer` — Adaptive prior layer.
- `FixedPriorLayer` — Fixed prior over coefficients (e.g., Normal or Laplace).
- `InterceptLayer` — Intercept-only layer (bias term).
- `EmbeddingLayer` — Bayesian embeddings for sparse categorical features.
- `RandomEffectsLayer` — Classical random-effects.
- `FMLayer` — Factorization Machine (order 2).
- `FM3Layer` — Factorization Machine (order 3).
- `LowRankInteractionLayer` — Low-rank interaction between two feature sets.
- `RandomWalkLayer` — Random walk prior over coefficients (e.g., Gaussian walk).
- `InteractionLayer` — All pairwise interactions between two feature sets.
- `BilinearLayer` — Bilinear interaction: `x^T W z`.
- `LowRankBilinearLayer` — Low-rank bilinear interaction.
- `HorseshoeLayer` — Horseshoe prior for sparse regression.
- `AttentionLayer` — Multi-head self-attention over the feature dimension with FT-Transformer tokenisation ([Gorishniy et al. 2021](https://arxiv.org/abs/2106.11959)). `head_dim` is per-head so total embedding dim is `head_dim * num_heads` — adding heads increases capacity.

All layer prior kwargs are validated at construction time — bad kwargs raise `TypeError` immediately.

## Links

We provide link helpers in `links.py` to reduce Numpyro boilerplate. Available links:

- `gaussian_link` — Gaussian link with flexible scale: learned (default), fixed, or from a layer (see below).
- `gaussian_link_exp` — Gaussian link with `Exp` distributed homoskedastic `sigma`.
- `lognormal_link_exp` — LogNormal link with `Exp` distributed homoskedastic `sigma`
- `logit_link` — Bernoulli link for logistic regression.
- `poisson_link` — Poisson link with rate `y_hat`.
- `negative_binomial_link` — Uses `sigma ~ Exponential(rate)` and `y ~ NegativeBinomial2(mean=y_hat, concentration=sigma)`.
- `ordinal_link` — Cumulative logit / proportional odds for ordinal outcomes.
- `zip_link` — Zero-inflated Poisson for count data with excess zeros.
- `beta_link` — Beta regression for proportions strictly in (0, 1).

### `gaussian_link` scale modes

```python
# Default: sigma ~ Exp(1) learned from data
gaussian_link(mu, y)

# Fixed known scale (e.g. from XGBoost quantile regression)
gaussian_link(mu, y, scale=pred_std)

# Learned scale from a layer — softplus applied internally for stable gradients
raw = AdaptiveLayer()("log_sigma", x)
gaussian_link(mu, y, untransformed_scale=raw)
```

## Splines

Non-linear transformations via B-splines. Compute the basis matrix once with `make_knots` + `bspline_basis`, then pass it to any layer.

```python
from blayers.splines import make_knots, bspline_basis
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

knots = make_knots(x_train, num_knots=10)   # clamped knot vector from data quantiles

def model(x, y=None):
    B = bspline_basis(x, knots)             # (n, num_basis) design matrix
    f = AdaptiveLayer()("f", B)
    return gaussian_link(f, y)
```

Additive models are straightforward:

```python
def model(x1, x2, y=None):
    f1 = AdaptiveLayer()("f1", bspline_basis(x1, knots1))
    f2 = AdaptiveLayer()("f2", bspline_basis(x2, knots2))
    return gaussian_link(f1 + f2, y)
```

## fit() helpers

`fit()` handles the guide, ELBO, batching, and LR schedule. The same model runs unchanged under VI, MCMC, or SVGD.

```python
from blayers.fit import fit
from blayers.sampling import autoreshape

@autoreshape
def model(x, y=None):
    mu = AdaptiveLayer()("beta", x)
    intercept = InterceptLayer()("intercept")
    return gaussian_link(mu + intercept, y)

# Variational Inference (default)
result = fit(model, y=y, num_steps=1000, batch_size=256, lr=0.01, x=X)

# MCMC
result = fit(model, y=y, method="mcmc", num_mcmc_samples=1000, num_warmup=500, x=X)

# SVGD
result = fit(model, y=y, method="svgd", num_steps=1000, num_particles=20, x=X)
```

`result.predict()` returns a `Predictions` object with `.mean`, `.std`, and `.samples`. `result.summary()` returns posterior stats per latent variable.

```python
preds = result.predict(x=X, num_samples=500)
summary = result.summary(x=X)
```

Keyword arguments that are JAX arrays are treated as **data** (batched during training). Non-array kwargs are bound as **constants**.

## Batched loss

The default Numpyro way to fit batched VI models is to use `plate`, which confuses
me a lot. Instead, BLayers provides `Batched_Trace_ELBO` which does not require
you to use `plate` to batch in VI. Just drop your model in.

```python
from blayers.infer import Batched_Trace_ELBO, svi_run_batched

svi = SVI(model_fn, guide, optax.adam(schedule), loss=loss_instance)

svi_result = svi_run_batched(
    svi,
    rng_key,
    num_steps,
    batch_size=1000,
    **model_data,
)
```

**⚠️⚠️⚠️ `numpyro.plate` + `Batched_Trace_ELBO` do not mix. ⚠️⚠️⚠️**

`Batched_Trace_ELBO` is known to have issues when your model uses `numpyro.plate`. If your model needs plates, either:
1. Batch via `plate` and use the standard `Trace_ELBO`, or
1. Remove plates and use `Batched_Trace_ELBO` + `svi_run_batched`.

`Batched_Trace_ELBO` will warn if you if your model has plates.


### Reparameterizing

To fit MCMC models well it is crucial to [reparamterize](https://num.pyro.ai/en/latest/reparam.html). BLayers helps you do this, automatically reparameterizing the following distributions which Numpyro refers to as `LocScale` distributions.

```python
LocScaleDist = (
    dist.Normal
    | dist.LogNormal
    | dist.StudentT
    | dist.Cauchy
    | dist.Laplace
    | dist.Gumbel
)
```

Then, reparam these distributions automatically and fit with Numpyro's built in MCMC methods.

```python
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link_exp
from blayers.sampling import autoreparam

data = {...}

@autoreparam
def model(x, y):
    mu = AdaptiveLayer()('mu', x)
    return gaussian_link_exp(mu, y)

kernel = NUTS(model)
mcmc = MCMC(
    kernel,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    progress_bar=True,
)
    mcmc.run(
        rng_key,
        **data,
    )
```
