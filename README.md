[![codecov](https://codecov.io/gh/georgeberry/blayers/graph/badge.svg?token=ZDGT0C39QM)](https://codecov.io/gh/georgeberry/blayers) [![License](https://img.shields.io/github/license/georgeberry/blayers)](LICENSE) ![PyPI](https://img.shields.io/pypi/v/blayers)


# BLayers

**NOTE: BLayers is in alpha. Expect changes. Feedback welcome.**

## write code immediately

```
pip install blayers
```

deps are: `numpy`, `numpyro` and `jax`. `optax` is recommended.

## concept

The missing layers package for Bayesian inference. Inspiration from Keras and
Tensorflow Probability, but made specifically for Numpyro + Jax.

Easily build Bayesian models from parts, abstract away the boilerplate, and
tweak priors as you wish.

Fit models either using Variational Inference (VI) or your sampling method of
choice. Use BLayer's ELBO implementation to do either batched VI or sampling
without having to rewrite models.

BLayers helps you write pure Numpyro, so you can integrate it with any Numpyro
code to build models of arbitrary complexity. It also gives you a recipe to
build more complex layers as you wish.

## the starting point

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
from blayers import AdaptiveLayer, gaussian_link_exp
def model(x, y):
    mu = AdaptiveLayer()('mu', x)
    return gaussian_link_exp(mu, y)
```

### pure numpyro

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

### mixing it up

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
from blayers import AdaptiveLayer, gaussian_link_exp
def model(x, y):
    mu = AdaptiveLayer(
        lmbda_dist=distributions.Exponential,
        prior_dist=distributions.LogNormal,
        lmbda_kwargs={'rate': 1.},
        prior_kwargs={'loc': 0.}
    )('mu', x)
    return gaussian_link_exp(mu, y)
```

### "factories"

Since Numpyro traces `sample` sites and doesn't record any paramters on the class, you can re-use with a particular generative model structure freely.

```python
from numpyro import distributions,
from blayers import AdaptiveLayer, gaussian_link_exp

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

## additional layers

### fixed prior layers

For you purists out there, we also provide a `FixedPriorLayer` for standard
L1/L2 regression.

```python
from blayers import FixedPriorLayer, gaussian_link_exp
def model(x, y):
    mu = FixedPriorLayer()('mu', x)
    return gaussian_link_exp(mu, y)
```

Very useful when you have an informative prior.

### factorization machines

Developed in [Rendle 2010](https://jame-zhang.github.io/assets/algo/Factorization-Machines-Rendle2010.pdf) and [Rendle 2011](https://www.ismll.uni-hildesheim.de/pub/pdfs/FreudenthalerRendle_BayesianFactorizationMachines.pdf), FMs provide a low-rank approximation to the `x`-by-`x` interaction matrix. For those familiar with R syntax, it is an approximation to `y ~ x:x`, excluding the x^2 terms.

To fit the equivalent of an r model like `y ~ x*x` (all main effects, x^2 terms, and one-way interaction effects), you'd do

```python
from blayers import FMLayer, gaussian_link_exp
def model(x, y):
    mu = (
        AdaptiveLayer('x', x) +
        AdaptiveLayer('x2', x**2) +
        FMLayer(low_rank_dim=3)('xx', x)
    )
    return gaussian_link_exp(mu, y)
```

### uv decomp

We also provide a standard UV deccomp for low rank interaction terms

```python
from blayers import LowRankInteractionLayer, gaussian_link_exp
def model(x, z, y):
    mu = (
        AdaptiveLayer('x', x) +
        AdaptiveLayer('z', z) +
        LowRankInteractionLayer(low_rank_dim=3)('xz', x, z)
    )
    return gaussian_link_exp(mu, y)
```

### bayesian embeddings

## links

We provide link functions as a convenience to abstract away a bit more Numpyro
boilerplate.

We currently provide

* `gaussian_link_exp`

## batched loss

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

## roadmap

1. Fit helpers for models with categorical variables
2. Multioutput models
3. Examples
4. More code re-use in `layers.py` (this will only become clear after more code is written)
5. More link functions