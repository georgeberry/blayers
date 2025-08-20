[![Coverage Status](https://coveralls.io/repos/github/georgeberry/blayers/badge.svg?branch=gb-fix-coverage-2)](https://coveralls.io/github/georgeberry/blayers?branch=gb-fix-coverage-2) [![License](https://img.shields.io/github/license/georgeberry/blayers)](LICENSE) [![PyPI](https://img.shields.io/pypi/v/blayers)](https://pypi.org/project/blayers/)


# BLayers

The missing layers package for Bayesian inference.

**BLayers is in beta, errors are possible!**

## Write code immediately

```
pip install blayers
```

deps are: `numpyro`, `jax`, and `optax`.

## Concept

Easily build Bayesian models from parts, abstract away the boilerplate, and
tweak priors as you wish. Inspiration from Keras and Tensorflow Probability, but made specifically for Numpyro + Jax.

Fit models either using Variational Inference (VI) or your sampling method of
choice. Use BLayer's ELBO implementation to do either batched VI or sampling
without having to rewrite models.

BLayers helps you write pure Numpyro, so you can integrate it with any Numpyro
code to build models of arbitrary complexity. It also gives you a recipe to
build more complex layers as you wish.

## The starting point

The simplest non-trivial (and most important!) Bayesian regression model form is
the adaptive prior,

```
lmbda ~ HalfNormal(1)
beta  ~ Normal(0, lmbda)
y     ~ Normal(beta * x, 1)
```

BLayers takes this as its starting point and most fundamental building block,
providing the flexible `AdaptiveLayer`.

```python
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link_exp
def model(x, y):
    mu = AdaptiveLayer()('mu', x)
    return gaussian_link_exp(mu, y)
```

### Pure numpyro

All BLayers is doing is writing Numpyro for you under the hood. This model is exacatly equivalent to writing the following, just using way less code.

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

The full set of layers shipped in `layers.py`:

- `BLayer` — Abstract base class for Bayesian layers; defines the interface.
- `AdaptiveLayer` — Adaptive prior layer: hp ~ HalfNormal(1), beta ~ Normal(0, hp).
- `FixedPriorLayer` — Fixed prior over coefficients (e.g., Normal or Laplace).
- `InterceptLayer` — Intercept-only layer (bias term).
- `EmbeddingLayer` — Bayesian embeddings for sparse categorical features (set embedding_dim).
- `RandomEffectsLayer` — Classical random-effects as embeddings with embedding_dim=1.
- `FMLayer` — Factorization Machine (order 2) with adaptive priors.
- `FM3Layer` — Factorization Machine (order 3).
- `LowRankInteractionLayer` — Learns a low-rank interaction matrix between two feature sets.
- `RandomWalkLayer` — Random walk prior over coefficients (e.g., Gaussian walk).
- `InteractionLayer` — All pairwise interactions between two feature sets.
## Links

We provide link helpers in `links.py` to reduce Numpyro boilerplate. Available links:

- `negative_binomial_link` — Uses `sigma ~ Exponential(rate)` and `y ~ NegativeBinomial2(mean=y_hat, concentration=sigma)`.
- `logit_link` — Bernoulli link (pass the linear predictor `y_hat`).
- `poission_link` — Poisson link with rate `y_hat`.
- `gaussian_link_exp` — Gaussian link with `sigma ~ Exponential(1)`.
- `lognormal_link_exp` — LogNormal link with `sigma ~ Exponential(1)`.
## Batched loss

> **⚠️ Plates + `Batched_Trace_ELBO` do not mix.**
>
> `Batched_Trace_ELBO` is known to have issues when your model uses `plate`. If your model needs plates, either:
> 1) batch via `plate` and use the standard `Trace_ELBO`, or
> 2) remove plates and use `Batched_Trace_ELBO` + `svi_run_batched`.

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
