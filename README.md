[![Coverage Status](https://coveralls.io/repos/github/georgeberry/blayers/badge.svg?branch=main)](https://coveralls.io/github/georgeberry/blayers?branch=main) [![License](https://img.shields.io/github/license/georgeberry/blayers)](https://github.com/georgeberry/blayers/blob/main/LICENSE) [![PyPI](https://img.shields.io/pypi/v/blayers)](https://pypi.org/project/blayers/) [![Read - Docs](https://img.shields.io/badge/Read-Docs-2ea44f)](https://georgeberry.github.io/blayers/) [![View - GitHub](https://img.shields.io/badge/View-GitHub-89CFF0)](https://github.com/georgeberry/blayers) [![PyPI Downloads](https://static.pepy.tech/badge/blayers)](https://pepy.tech/projects/blayers)



# BLayers

The missing layers package for Bayesian regression.

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

**Scope.** BLayers works best for *structured* Bayesian regression — GLMs, hierarchical /
mixed-effects models, factorization machines, splines, and sparse priors. Layers
are meant to be **added together into a linear predictor** (`mu = layer1(...) +
layer2(...) + ...`). You can stack them into a deep net, but better tools exist for this:
[`numpyro.contrib.module`](https://num.pyro.ai/en/stable/primitives.html#module)'s
`random_flax_module` / `random_haiku_module` instead — they drop a full Flax or
Haiku net into a NumPyro model with priors on the weights.

BLayers provides tools to

- Quickly build Bayesian models from layers which encapsulate useful model parts
- Fit models either using Variational Inference (VI) or your sampling method of
choice without having to rewrite models
- Write pure Numpyro to integrate with all of Numpyro's super powerful tools
- Add more complex layers (model parts) as you wish
- Fit models in a greater variety of ways with less code

## The starting point

The simplest non-trivial (and most important!) Bayesian regression model form is
the adaptive prior (note I'm using a `.` to denote the regression sigma and any
prior on it just to save space),

```
scale ~ HalfNormal(1)
beta  ~ Normal(0, scale)
y     ~ Normal(beta * x, .)
```

BLayers encapsulates a generative model structure like this in a `BLayer`. The
fundamental building block is the `AdaptiveLayer`.

```python
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

def model(x, y):
    mu = AdaptiveLayer()('mu', x)
    return gaussian_link(mu, y)
```

All `AdaptiveLayer` is doing is writing Numpyro for you under the hood. This
model is exactly equivalent to writing the following, just using way less code.

```python
import jax.numpy as jnp
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
y     ~ Normal(beta * x, .)
```

to

```
scale ~ Exponential(1.)
beta  ~ LogNormal(0, scale)
y     ~ Normal(beta * x, .)
```

you can just do this directly via arguments

```python
from numpyro import distributions
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

def model(x, y):
    mu = AdaptiveLayer(
        scale_dist=distributions.Exponential,
        coef_dist=distributions.LogNormal,
        scale_kwargs={'rate': 1.},
        coef_kwargs={'loc': 0.}
    )('mu', x)
    return gaussian_link(mu, y)
```

### "Factories"

Since Numpyro traces `sample` sites and doesn't record any parameters on the class, you can re-use with a particular generative model structure freely.

```python
from numpyro import distributions
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

my_lognormal_layer = AdaptiveLayer(
    scale_dist=distributions.Exponential,
    coef_dist=distributions.LogNormal,
    scale_kwargs={'rate': 1.},
    coef_kwargs={'loc': 0.}
)

def model(x, y):
    mu = my_lognormal_layer('mu1', x) + my_lognormal_layer('mu2', x**2)
    return gaussian_link(mu, y)
```

## Layers

The full set of layers included with BLayers:

- `AdaptiveLayer` — Adaptive prior layer: `scale ~ HalfNormal(1)`, `beta ~ Normal(0, scale)`.
- `FixedPriorLayer` — Fixed prior over coefficients (e.g., Normal or Laplace), no hierarchical scale.
- `InterceptLayer` — Intercept-only layer (bias term).
- `EmbeddingLayer` — Bayesian embeddings for sparse categorical features.
- `RandomEffectsLayer` — Classical random-effects (embedding with output dim 1); learned variance component → partial pooling.
- `FixedEffectsLayer` — Per-category coefficients with a fixed, user-specified prior; the no-pooling counterpart of `RandomEffectsLayer`.
- `FMLayer` — Factorization Machine (order 2) for pairwise interaction terms.
- `FM3Layer` — Factorization Machine (order 3).
- `LowRankInteractionLayer` — Low-rank interaction between two feature sets.
- `InteractionLayer` — All pairwise interactions between two feature sets.
- `HorseshoeInteractionLayer` — Pairwise interactions with a per-pair horseshoe prior; the layer to reach for to *identify* sparse interactions (most pairs shrink to zero, the real ones stand out). Omit `z` for unique within-feature pairs `i<j`; pass `z` for the full cross-set grid.
- `BilinearLayer` — Bilinear interaction: `x^T W z`.
- `LowRankBilinearLayer` — Low-rank bilinear interaction.
- `RandomWalkLayer` — Gaussian random walk prior over an ordered index (e.g., time).
- `HorseshoeLayer` — Horseshoe prior for sparse regression; global-local shrinkage via HalfCauchy.
- `SpikeAndSlabLayer` — Exact spike at zero plus a configurable continuous slab, with binary inclusion indicators and fixed or learned inclusion probabilities. Fit with MCMC.
- `MixtureLayer` — Finite mixture-of-priors on coefficients (default Normal + Laplace) with a logistic-normal (or fixed) weight; the component indicator is marginalised so it works under VI, MCMC, *and* SVGD. Good for robustness / elastic-net-style priors.
- `HSGPLayer` — Hilbert-space approximate Gaussian process (1-D, squared-exponential; [Riutort-Mayol et al. 2021](https://arxiv.org/abs/2004.11408)). A GP smoother that learns its own lengthscale; use `hsgp_L(x_train)` to pick the domain boundary.

All layer prior kwargs are validated at construction time — bad kwargs raise `TypeError` immediately.

## Links

We provide link helpers in `links.py` to reduce Numpyro boilerplate. Available links:

- `gaussian_link` — Gaussian likelihood with configurable sigma prior (see below).
- `lognormal_link` — LogNormal likelihood with configurable sigma prior.
- `student_t_link` — StudentT likelihood for robust regression (default `df=4`).
- `gamma_link` — Gamma likelihood (log link) for positive continuous data; learned shape.
- `exponential_link` — Exponential likelihood (log link) for positive / survival data.
- `logit_link` — Bernoulli link for binary logistic regression.
- `categorical_link` — Categorical / softmax link for multiclass classification (`units = num_classes`).
- `poisson_link` — Poisson link with log-rate input.
- `negative_binomial_link` — NegativeBinomial2 for overdispersed counts; learned concentration via `Exponential`.
- `ordinal_link` — Cumulative logit / proportional odds for ordinal outcomes.
- `zip_link` — Zero-inflated Poisson for count data with excess zeros.
- `zinb_link` — Zero-inflated NegativeBinomial2 for overdispersed, zero-heavy counts.
- `beta_link` — Beta regression for proportions strictly in (0, 1).

For a single response, likelihood helpers accept targets shaped either `(n,)`
or `(n, 1)`, including models decorated with `@autoreshape`. They align the
singleton output dimension before evaluating the likelihood, so each row
contributes one log probability. Multi-output targets must match the predictor's
output dimensions; incompatible shapes raise `ValueError`. Predictive output
shapes retain the likelihood's existing convention.

For location-scale likelihoods, a vector `scale` or `untransformed_scale` is
interpreted as one scale per row. For multi-output predictions, use `(1, units)`
for per-output scales or `(n, units)` for a separate scale per row and output.

### `gaussian_link`, `lognormal_link`, and `student_t_link`

All three share a common location-scale base and support three scale modes:

```python
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

# Default: sigma ~ Exp(1) learned from data
gaussian_link(mu, y)

# Fixed known scale (e.g. from XGBoost quantile regression)
gaussian_link(mu, y, scale=pred_std)

# Learned scale from a layer — softplus applied internally for stable gradients
raw = AdaptiveLayer()("log_sigma", x)
gaussian_link(mu, y, untransformed_scale=raw)
```

Swap the sigma prior via `functools.partial`:

```python
from functools import partial
import numpyro.distributions as dists
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

# HalfNormal prior instead of Exponential
hn_gaussian = partial(gaussian_link, sigma_dist=dists.HalfNormal, sigma_kwargs={"scale": 1.0})

def model(x, y=None):
    mu = AdaptiveLayer()("mu", x)
    return hn_gaussian(mu, y)
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
knots1 = make_knots(x1_train, num_knots=10)
knots2 = make_knots(x2_train, num_knots=10)

def model(x1, x2, y=None):
    f1 = AdaptiveLayer()("f1", bspline_basis(x1, knots1))
    f2 = AdaptiveLayer()("f2", bspline_basis(x2, knots2))
    return gaussian_link(f1 + f2, y)
```

## Gaussian processes (HSGP)

`HSGPLayer` is a Hilbert-space approximate GP ([Riutort-Mayol et al. 2021](https://arxiv.org/abs/2004.11408)) — a smoother like splines, but it learns its own lengthscale and carries a proper GP interpretation. Pick the domain boundary `L` once on the training inputs with `hsgp_L` and reuse it at predict time; `m` is the number of basis functions (~20–50).

```python
from blayers.layers import HSGPLayer, hsgp_L
from blayers.links import gaussian_link
from blayers.decorators import autoreshape

L = hsgp_L(x_train)     # domain boundary = 1.5 * max(|x|); fixed across fit/predict

@autoreshape
def model(x, y=None, L=L, m=30):
    f = HSGPLayer()("f", x, L=L, m=m)
    return gaussian_link(f, y)
```

Like splines, HSGP terms add for a GAM-style additive model (`f1(x1) + f2(x2) + ...`). Center/scale each input so it lies within `[-L, L]`.

## Mixture priors

`MixtureLayer` draws each coefficient from a finite mixture of priors (default Normal + Laplace) — useful for robustness (a heavy-tailed component absorbs outlier coefficients) or elastic-net-flavoured priors. The mixing weights get a logistic-normal prior by default (softmax of `Normal(0, weight_scale)` logits — an unconstrained parameterisation that fits under VI, MCMC, *and* SVGD), or pass fixed `weights=`. The component indicator is marginalised internally, so it works under **VI, MCMC, and SVGD**.

```python
from blayers.layers import MixtureLayer
from blayers.links import gaussian_link

def model(x, y=None):
    mu = MixtureLayer()("beta", x)                 # Normal + Laplace, logistic-normal weights
    return gaussian_link(mu, y)
```

For pure sparsity prefer `HorseshoeLayer`; for explicit variable selection prefer `SpikeAndSlabLayer`.

## Exact spike-and-slab variable selection

`SpikeAndSlabLayer` places actual probability mass at zero:

```text
pi ~ Beta(alpha, beta)       # shared across features, separately per output
z_j ~ Bernoulli(pi)
slab_j ~ Normal(0, 1)        # configurable continuous slab
beta_j = z_j * slab_j
```

The default `alpha=beta=0.5` learns the inclusion rate. Alternatively, set
`inclusion_prob=0.1` to fix the prior probability of including each coefficient
to 10%. Choose the slab scale for your covariate and outcome scales.

```python
from blayers import SpikeAndSlabLayer, gaussian_link, fit

selection = SpikeAndSlabLayer(inclusion_prob=0.1)

def model(x, y=None):
    mu = selection("effects", x)
    return gaussian_link(mu, y)

result = fit(model, x=X, y=y, method="mcmc", num_chains=2)
stats = result.summary()
pip = stats["SpikeAndSlabLayer_effects_z"]["mean"]       # P(included | data)
coef = stats["SpikeAndSlabLayer_effects_beta"]["mean"]   # averages over inclusion
predictions = result.predict(x=X_new)
```

`fit(method="mcmc")` automatically uses NumPyro's
[`DiscreteHMCGibbs`](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc_gibbs.DiscreteHMCGibbs)
around NUTS for unenumerated finite discrete latents. Binary indicators get
Gibbs updates and continuous parameters get NUTS updates. For direct NumPyro
usage, construct `MCMC(DiscreteHMCGibbs(NUTS(model)), ...)`. The current `fit()`
VI and SVGD paths reject these discrete indicators; continuous shrinkage via
`HorseshoeLayer` remains available for those workflows.

The `*_z` posterior mean is an inclusion probability. The deterministic
`*_beta` site is the effective coefficient and includes exact zeros. The
`*_slab` site is an auxiliary coefficient, which follows its prior when the
feature is excluded. Inspect indicator mixing and effective sample sizes as
well as NUTS diagnostics, especially with correlated predictors. Gibbs sweeps
update each feature/output indicator, so cost grows with the number of candidate
coefficients. For `units > 1`, each feature/output pair is selected separately.

**Migration:** this changes the prior from the earlier continuous Beta gate.
Previously `*_z` was continuous and `*_beta` was the ungated slab. Refit models
using this layer; old posterior samples are not compatible with the new sites.

## fit() helpers

`fit()` handles the guide, ELBO, batching, and LR schedule. Models with continuous latents run unchanged under VI, MCMC, or SVGD. Exact spike-and-slab models use MCMC.

VI and MCMC automatically non-center supported latent distributions by default
(`autoreparam_model=True`). For VI, the default diagonal-normal guide is built
in these transformed coordinates, allowing hierarchical coefficient uncertainty
to vary with its learned prior scale. No `@autoreparam` decorator is needed.
Set `autoreparam_model=False` to keep the model's supplied parameterization;
centering can work better for strongly informed coefficients. SVGD is unaffected.
Guide classes work with automatic reparameterization. Prebuilt guide instances
and custom guide functions require `autoreparam_model=False` and must match the
supplied model; to non-center them, wrap the model with `autoreparam` before
constructing the guide and pass that same model to `fit()`.

```python
from blayers.fit import fit
from blayers.decorators import autoreshape
from blayers.layers import AdaptiveLayer, InterceptLayer
from blayers.links import gaussian_link

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

### Diagnostics & model comparison (ArviZ)

`result.to_arviz()` hands the fit to [ArviZ](https://python.arviz.org) for R-hat,
ESS, divergences, PSIS-LOO, and the full plotting suite — reusing NumPyro's own
ArviZ bridge rather than reinventing diagnostics. Install with `pip install
blayers[arviz]` (arviz ≥ 1.0, Python ≥ 3.12).

```python
import arviz as az

# MCMC: divergences, R-hat, ESS, and log-likelihood come through automatically
idata = fit(model, y=y, method="mcmc", num_chains=2, x=X).to_arviz()
az.summary(idata)          # R-hat / ESS per latent
az.loo(idata)              # PSIS-LOO

# VI: pass the observed y (and inputs) so the log_likelihood group can be built
idata_vi = fit(model, y=y, num_steps=2000, x=X).to_arviz(y=y, x=X)

# Compare models on out-of-sample predictive fit
az.compare({"mcmc": idata, "vi": idata_vi})
```

SVGD is not supported by `to_arviz()` (too few particles to be a meaningful
sample for LOO); fit with `method="mcmc"` or `method="vi"` for comparison.

## Model to LaTeX

`model_to_latex()` traces a model once and prints its generative form — every
prior plus the likelihood — as a block of sampling statements, the "methods
section" version you'd otherwise hand-transcribe. Pass the model and its inputs
the same way you would to `fit()`, but **without** `y`.

```python
from blayers import AdaptiveLayer, InterceptLayer, RandomEffectsLayer, gaussian_link
from blayers.latex import model_to_latex

def model(x, g, y=None):
    mu = (
        InterceptLayer()("intercept")
        + AdaptiveLayer()("mu", x)
        + RandomEffectsLayer()("grp", g, num_categories=n_groups)
    )
    return gaussian_link(mu, y)

print(model_to_latex(model, x=X, g=G))   # raw LaTeX
model_to_latex(model, x=X, g=G)           # renders inline in a notebook
```

```latex
\begin{align}
\beta_{\mathrm{intercept}} &\sim \mathrm{Normal}(0, 1) \\
\lambda_{\mathrm{mu}} &\sim \mathrm{HalfNormal}(1) \\
\beta_{\mathrm{mu}} &\sim \mathrm{Normal}(0, \lambda_{\mathrm{mu}}) \\
\lambda_{\mathrm{grp}} &\sim \mathrm{HalfNormal}(1) \\
\theta_{\mathrm{grp}} &\sim \mathrm{Normal}(0, \lambda_{\mathrm{grp}}) \\
\sigma &\sim \mathrm{Exponential}(1) \\
y_i &\sim \mathrm{Normal}(\eta_i,\; \sigma),\quad i = 1, \dots, n
\end{align}
```

The hierarchy is recovered automatically: a coefficient whose scale is a sampled
site prints `Normal(0, λ)`, not a number. Priors and the likelihood are exact
(read straight from the trace); how the layers *combine* into the linear
predictor lives in plain Python, so it's denoted `η_i` rather than reconstructed.
The return value is a `str` (so `print()` gives raw LaTeX) that also renders as
typeset math in Jupyter.

## Batched loss

The default Numpyro way to fit batched VI models is to use `plate`, which confuses
me a lot. Instead, BLayers provides `Batched_Trace_ELBO` which does not require
you to use `plate` to batch in VI. Just drop your model in.

```python
from numpyro.infer import SVI
from numpyro.infer.autoguide import AutoDiagonalNormal
import optax
from blayers.vi_infer import Batched_Trace_ELBO, svi_run_batched

loss = Batched_Trace_ELBO(num_obs=len(y), batch_size=1000)
guide = AutoDiagonalNormal(model_fn)
svi = SVI(model_fn, guide, optax.adam(0.01), loss=loss)

svi_result = svi_run_batched(
    svi,
    rng_key,
    batch_size=1000,
    num_steps=500,
    **model_data,
)
```

The likelihood is scaled by `N / B`, where `B` is the **actual** number of
rows in each batch, including a short final batch. Array inputs must be aligned
by row; bind static arrays such as spline knots into the model with a closure
or `functools.partial`. The constructor's `batch_size` is a fallback for loss
calls without array inputs.

NumPyro `scale` and `mask` handlers are preserved for both model and guide
sites. All observed sites, including `numpyro.factor` terms, are treated as
row-wise likelihood contributions and scaled by `N / B`; global latent priors
and guide densities are not batch-scaled. Models with global factors or
per-observation latent variables need a standard NumPyro ELBO instead.

**⚠️⚠️⚠️ `numpyro.plate` + `Batched_Trace_ELBO` do not mix. ⚠️⚠️⚠️**

`Batched_Trace_ELBO` does not support `numpyro.plate`: its `N / batch_size` log-likelihood rescaling double-counts plate-subsampled sites and yields an incorrect ELBO. If your model needs plates, either:
1. Batch via `plate` and use the standard `Trace_ELBO`, or
1. Remove plates and use `Batched_Trace_ELBO` + `svi_run_batched`.

`Batched_Trace_ELBO` **raises `ValueError`** if your model contains a plate.


### Reparameterizing

To fit MCMC models well it is crucial to [reparameterize](https://num.pyro.ai/en/latest/reparam.html). BLayers helps you do this via `@autoreparam`, which automatically applies `LocScaleReparam` to all `LocScale` distributions in your model (Normal, LogNormal, StudentT, Cauchy, Laplace, Gumbel).

> **Note:** `fit(method="vi")` and `fit(method="mcmc")` already apply `@autoreparam` for you (controlled by `autoreparam_model=True`, on by default). Apply the decorator yourself when driving SVI or NUTS / HMC manually, as shown below for MCMC.

```python
from numpyro.infer import MCMC, NUTS
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link
from blayers.decorators import autoreparam

data = {...}

@autoreparam
def model(x, y):
    mu = AdaptiveLayer()('mu', x)
    return gaussian_link(mu, y)

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
