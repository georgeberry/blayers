"""
Link functions connect model predictions to likelihood distributions,
abstracting away NumPyro boilerplate for common output types.

Usage::

    from blayers.layers import AdaptiveLayer
    from blayers.links import gaussian_link

    def model(x, y=None):
        mu = AdaptiveLayer()("mu", x)
        return gaussian_link(mu, y)

    # HalfNormal sigma instead of Exponential
    from functools import partial
    import numpyro.distributions as dists
    hn_gaussian = partial(gaussian_link, sigma_dist=dists.HalfNormal, sigma_kwargs={"scale": 1.0})

Available links:

* ``gaussian_link``          — Normal likelihood, configurable sigma prior
* ``lognormal_link``         — LogNormal likelihood, configurable sigma prior
* ``student_t_link``         — StudentT likelihood for robust regression (default df=4)
* ``logit_link``             — Bernoulli likelihood
* ``poisson_link``           — Poisson likelihood
* ``negative_binomial_link`` — NegativeBinomial2 likelihood, learned concentration
* ``ordinal_link``           — Ordinal (cumulative logit / proportional odds)
* ``zip_link``               — Zero-inflated Poisson
* ``beta_link``              — Beta regression for proportions in (0, 1)
"""

from functools import partial

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpyro.distributions as dists
from numpyro import sample



def _loc_scale_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
    obs_dist=dists.Normal,
    sigma_dist=dists.Exponential,
    sigma_kwargs: dict | None = None,
    scale: float | jax.Array | None = None,
    untransformed_scale: jax.Array | None = None,
) -> jax.Array:
    """Base link for location-scale likelihoods.

    Exactly one of ``scale``, ``untransformed_scale``, or neither should be
    supplied.

    * **Default** (neither): ``sigma`` is drawn from ``sigma_dist(**sigma_kwargs)``.
    * **``scale``**: a known positive std passed directly.
    * **``untransformed_scale``**: unbounded linear predictor transformed via
      ``softplus`` internally.

    Args:
        y_hat: Predicted mean/location.
        y: Observed values, or ``None`` for prior predictive / inference.
        obs_dist: Likelihood distribution class (must accept ``loc`` and ``scale``).
        sigma_dist: Prior distribution class for sigma. Default ``Exponential``.
        sigma_kwargs: Kwargs for ``sigma_dist``. Default ``{"rate": 1.0}``.
        scale: Known positive standard deviation.
        untransformed_scale: Unbounded array transformed via ``softplus`` internally.

    Returns:
        Sample site ``"obs"``.
    """
    if sigma_kwargs is None:
        sigma_kwargs = {"rate": 1.0}

    if untransformed_scale is not None:
        sigma = jax.nn.softplus(untransformed_scale)
    elif scale is not None:
        sigma = scale
    else:
        sigma = sample("sigma", sigma_dist(**sigma_kwargs))
    return sample("obs", obs_dist(loc=y_hat, scale=sigma), obs=y)


gaussian_link = partial(_loc_scale_link, obs_dist=dists.Normal)
gaussian_link.__doc__ = """Gaussian likelihood with configurable sigma prior.

Default: ``sigma ~ Exponential(rate=1.0)``.  Override via ``sigma_dist`` /
``sigma_kwargs``.  Pass a known ``scale`` or a raw ``untransformed_scale``
(transformed via softplus internally) to skip the sigma sample site.

Args:
    y_hat: Predicted mean, shape ``(n, 1)`` or ``(n,)``.
    y: Observed values, or ``None`` for prior predictive / inference.
    sigma_dist: Prior distribution class for sigma. Default ``Exponential``.
    sigma_kwargs: Kwargs for ``sigma_dist``. Default ``{"rate": 1.0}``.
    scale: Known positive standard deviation. Scalar or broadcastable array.
    untransformed_scale: Unbounded array transformed via ``softplus`` internally.

Returns:
    Sample site ``"obs"``.

Example::

    # Default: Exponential(1) prior on sigma
    gaussian_link(mu, y)

    # HalfNormal prior instead
    from functools import partial
    hn_link = partial(gaussian_link, sigma_dist=dists.HalfNormal, sigma_kwargs={"scale": 1.0})

    # Known sigma (e.g. from XGBoost quantile regression)
    gaussian_link(mu, y, scale=pred_std)

    # Learned scale from a layer — softplus applied internally
    raw = AdaptiveLayer()("log_scale", x)
    gaussian_link(mu, y, untransformed_scale=raw)
"""

lognormal_link = partial(_loc_scale_link, obs_dist=dists.LogNormal)
lognormal_link.__doc__ = """LogNormal likelihood with configurable sigma prior.

Default: ``sigma ~ Exponential(rate=1.0)``.

Args:
    y_hat: Log-scale predicted mean, shape ``(n, 1)`` or ``(n,)``.
    y: Observed positive values, or ``None``.
    sigma_dist: Prior distribution class for sigma. Default ``Exponential``.
    sigma_kwargs: Kwargs for ``sigma_dist``. Default ``{"rate": 1.0}``.
    scale: Known positive standard deviation.
    untransformed_scale: Unbounded array transformed via ``softplus`` internally.

Returns:
    Sample site ``"obs"``.
"""


student_t_link = partial(_loc_scale_link, obs_dist=partial(dists.StudentT, df=4.0))
student_t_link.__doc__ = """StudentT likelihood for robust regression.

Heavier tails than Gaussian — large residuals are down-weighted rather than
driving the fit.  Default ``df=4`` gives moderate robustness.  Customise via
``functools.partial``::

    from functools import partial
    cauchy_link = partial(student_t_link, obs_dist=partial(dists.StudentT, df=1.0))

Args:
    y_hat: Predicted location, shape ``(n, 1)`` or ``(n,)``.
    y: Observed values, or ``None``.
    sigma_dist: Prior for scale. Default ``Exponential(rate=1.0)``.
    sigma_kwargs: Kwargs for ``sigma_dist``.
    scale: Known positive scale.
    untransformed_scale: Unbounded scale transformed via softplus internally.

Returns:
    Sample site ``"obs"``.
"""


def logit_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
) -> jax.Array:
    """Bernoulli likelihood for binary classification.

    Args:
        y_hat: Log-odds (logits), shape ``(n, 1)`` or ``(n,)``.
        y: Binary observations in {0, 1}, or ``None``.

    Returns:
        Sample site ``"obs"``.
    """
    return sample("obs", dists.Bernoulli(logits=y_hat), obs=y)


def poisson_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
) -> jax.Array:
    """Poisson likelihood for count data.

    Args:
        y_hat: Log rate, shape ``(n, 1)`` or ``(n,)``.
        y: Non-negative integer observations, or ``None``.

    Returns:
        Sample site ``"obs"``.
    """
    return sample("obs", dists.Poisson(rate=jnp.exp(y_hat)), obs=y)


def negative_binomial_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
    rate: float = 1.0,
) -> jax.Array:
    """NegativeBinomial2 likelihood for overdispersed count data.

    Args:
        y_hat: Predicted mean, shape ``(n, 1)`` or ``(n,)``.
        y: Non-negative integer observations, or ``None``.
        rate: Rate parameter for the ``Exponential`` prior on concentration.

    Returns:
        Sample site ``"obs"``.
    """
    concentration = sample("sigma", dists.Exponential(rate=rate))
    return sample(
        "obs",
        dists.NegativeBinomial2(mean=y_hat, concentration=concentration),
        obs=y,
    )


def ordinal_link(
    mu: jax.Array,
    y: jax.Array | None = None,
    num_classes: int = None,
) -> jax.Array:
    """Cumulative logit (proportional odds) link for ordinal outcomes.

    Models P(Y = k | μ) via:

    .. math::
        P(Y \\leq k \\mid \\mu) = \\sigma(c_k - \\mu)

    Cutpoints are sampled with an ordered parameterisation: the first is
    free (``Normal(0, 2)``), subsequent ones add ``Exponential`` increments.

    Args:
        mu: Linear predictor, shape ``(n, 1)`` or ``(n,)``.
        y: Integer observations in ``{0, 1, ..., num_classes - 1}``, or
            ``None`` for prior predictive / inference.
        num_classes: Number of ordinal categories (required).

    Returns:
        Sample site ``"obs"`` with integer values in ``{0, …, num_classes-1}``.
    """
    mu_flat = mu.squeeze()
    K = num_classes

    c0 = sample("ordinal_c0", dists.Normal(0.0, 2.0))
    if K > 2:
        gaps = sample("ordinal_gaps", dists.Exponential(1.0).expand([K - 2]))
        cutpoints = jnp.concatenate([c0[None], c0 + jnp.cumsum(gaps)])
    else:
        cutpoints = c0[None]

    cum_probs = jnn.sigmoid(cutpoints - mu_flat[:, None])

    probs_parts = [cum_probs[:, :1]]
    if K > 2:
        probs_parts.append(jnp.diff(cum_probs, axis=1))
    probs_parts.append(1.0 - cum_probs[:, -1:])
    probs = jnp.clip(jnp.concatenate(probs_parts, axis=1), 1e-8, 1.0)

    return sample("obs", dists.Categorical(probs=probs), obs=y)


def zip_link(
    mu: jax.Array,
    y: jax.Array | None = None,
) -> jax.Array:
    """Zero-inflated Poisson link for count data with excess zeros.

    Models a mixture: with probability π the outcome is exactly 0; with
    probability 1 - π the outcome follows Poisson(exp(μ)).  π is a global
    scalar learned from data.

    Args:
        mu: Log Poisson rate, shape ``(n, 1)`` or ``(n,)``.
        y: Non-negative integer observations, or ``None``.

    Returns:
        Sample site ``"obs"``.
    """
    rate = jnp.exp(mu.squeeze())
    gate = sample("zip_gate", dists.Beta(1.0, 10.0))
    return sample("obs", dists.ZeroInflatedPoisson(gate=gate, rate=rate), obs=y)


def beta_link(
    mu: jax.Array,
    y: jax.Array | None = None,
) -> jax.Array:
    """Beta likelihood for proportional outcomes strictly in (0, 1).

    Maps the linear predictor to a mean via sigmoid, then uses a learned
    global precision φ:

    .. math::
        \\bar{\\mu} = \\sigma(\\mu), \\quad
        y \\sim Beta(\\bar{\\mu}\\,\\phi,\\; (1 - \\bar{\\mu})\\,\\phi)

    Args:
        mu: Logit of the mean proportion, shape ``(n, 1)`` or ``(n,)``.
        y: Observed proportions in (0, 1), or ``None``.

    Returns:
        Sample site ``"obs"``.
    """
    mean = jnn.sigmoid(mu.squeeze())
    phi = sample("beta_phi", dists.Exponential(1.0))
    return sample("obs", dists.Beta(mean * phi, (1.0 - mean) * phi), obs=y)
