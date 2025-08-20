[![Coverage Status](https://coveralls.io/repos/github/georgeberry/blayers/badge.svg?branch=gb-fix-coverage-2)](https://coveralls.io/github/georgeberry/blayers?branch=gb-fix-coverage-2) [![License](https://img.shields.io/github/license/georgeberry/blayers)](LICENSE) [![PyPI](https://img.shields.io/pypi/v/blayers)](https://pypi.org/project/blayers/) [![Read - Docs](https://img.shields.io/badge/Read-Docs-2ea44f)](https://georgeberry.github.io/blayers/) [![View - GitHub](https://img.shields.io/badge/View-GitHub-89CFF0)](https://github.com/georgeberry/blayers)



# BLayers

The missing layers package for Bayesian inference.

**BLayers is in beta, errors are possible! We invite you to contribute on [GitHub](https://github.com/georgeberry/blayers).**

## Write code immediately

```
pip install blayers
```

deps are: `numpyro`, `jax`, and `optax`.

## Concept

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
lmbda ~ HalfNormal(1)
beta  ~ Normal(0, lmbda)
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
    lmbda = sample(
        name="lmbda",
        fn=distributions.HalfNormal(1.),
    )
    # beta coefficients for regression
    beta = sample(
        name="beta",
        fn=distributions.Normal(loc=0., scale=lmbda),
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
lmbda ~ HalfNormal(1)
beta  ~ Normal(0, lmbda)
y     ~ Normal(beta * x, 1)
```

to

```
lmbda ~ Exponential(1.)
beta  ~ LogNormal(0, lmbda)
y     ~ Normal(beta * x, 1)
```

you can just do this directly via arguments

```python
from numpyro import distributions,
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link_exp
def model(x, y):
    mu = AdaptiveLayer(
        lmbda_dist=distributions.Exponential,
        prior_dist=distributions.LogNormal,
        lmbda_kwargs={'rate': 1.},
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
    lmbda_dist=distributions.Exponential,
    prior_dist=distributions.LogNormal,
    lmbda_kwargs={'rate': 1.},
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

## Links

We provide link helpers in `links.py` to reduce Numpyro boilerplate. Available links:

- `logit_link` — Bernoulli link for logistic regression.
- `poission_link` — Poisson link with rate `y_hat`.
- `gaussian_link_exp` — Gaussian link with `Exp` distributed homoskedastic `sigma`.
- `lognormal_link_exp` — LogNormal link with `Exp` distributed homoskedastic `sigma`
- `negative_binomial_link` — Uses `sigma ~ Exponential(rate)` and `y ~ NegativeBinomial2(mean=y_hat, concentration=sigma)`.

## Batched loss

**⚠️⚠️⚠️ Plates + `Batched_Trace_ELBO` do not mix. ⚠️⚠️⚠️**

`Batched_Trace_ELBO` is known to have issues when your model uses `plate`. If your model needs plates, either:
1. Batch via `plate` and use the standard `Trace_ELBO`, or
1. Remove plates and use `Batched_Trace_ELBO` + `svi_run_batched`.

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
