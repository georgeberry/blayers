[![codecov](https://codecov.io/gh/georgeberry/blayers/graph/badge.svg?token=ZDGT0C39QM)](https://codecov.io/gh/georgeberry/blayers) [![License](https://img.shields.io/github/license/georgeberry/blayers)](LICENSE) ![version](https://img.shields.io/badge/version-0.1.0a1-blue)

# BLayers

**NOTE: BLayers is in alpha. Expect changes. Feedback welcome.**

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

For you purists out there, we also provide a `FixedPriorLayer` for standard
L1/L2 regression.

```python
from blayers import FixedPriorLayer, gaussian_link_exp
def model(x, y):
    mu = FixedPriorLayer()('mu', x)
    return gaussian_link_exp(mu, y)
```

## additional layers

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
