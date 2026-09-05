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
* ``gamma_link``             — Gamma likelihood (log link) for positive continuous data
* ``exponential_link``       — Exponential likelihood (log link) for positive / survival data
* ``logit_link``             — Bernoulli likelihood (binary)
* ``categorical_link``       — Categorical / softmax likelihood (multiclass)
* ``poisson_link``           — Poisson likelihood
* ``negative_binomial_link`` — NegativeBinomial2 likelihood, learned concentration
* ``ordinal_link``           — Ordinal (cumulative logit / proportional odds)
* ``zip_link``               — Zero-inflated Poisson
* ``zinb_link``              — Zero-inflated NegativeBinomial2 (overdispersed counts)
* ``beta_link``              — Beta regression for proportions in (0, 1)
"""

from functools import partial
from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpyro.distributions as dists
from numpyro import sample


def _observe(
    name: str, fn: dists.Distribution, y: jax.Array | None
) -> jax.Array:
    """Align scalar-response columns/vectors without broadcasting across rows.

    Multi-output observations must match the predictor's output dimensions.
    A constant predictor may broadcast over observations, but targets may not
    broadcast over rows or outputs. Preserve the likelihood's predictive shape.
    """
    if y is not None:
        y = jnp.asarray(y)
        shape = fn.batch_shape
        if len(shape) <= 1 and y.ndim == 2 and y.shape[-1] == 1:
            y = y[:, 0]
        elif len(shape) == 2 and shape[-1] == 1 and y.ndim == 1:
            y = y[:, None]
        if shape:
            if (
                y.ndim != len(shape)
                or y.shape[1:] != shape[1:]
                or shape[0] not in (1, y.shape[0])
            ):
                raise ValueError(
                    f"Observation shape {y.shape} does not match likelihood "
                    f"shape {shape}; expected one target per row and output."
                )
        elif y.ndim > 1:
            raise ValueError(
                "A scalar likelihood expects a scalar or vector target."
            )
    return jnp.asarray(sample(name, fn, obs=y))


def _scalar_predictor(value: jax.Array) -> jax.Array:
    """Normalize a scalar response predictor, retaining the single-row axis."""
    value = jnp.asarray(value)
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError(
            "Expected a scalar-response predictor of shape (n,) or (n, 1)."
        )
    return value


def _align_scale(value: float | jax.Array, predictor: jax.Array) -> jax.Array:
    """Interpret vector scales as per-row scales, including multiple outputs."""
    value = jnp.asarray(value)
    if predictor.ndim == 2 and value.ndim == 1:
        value = value[:, None]
    elif predictor.ndim == 1 and value.ndim == 2 and value.shape[-1] == 1:
        value = value[:, 0]
    if jnp.broadcast_shapes(value.shape, predictor.shape) != predictor.shape:
        raise ValueError(
            "Scale must broadcast to the predictor shape without adding rows or outputs."
        )
    return value


def _loc_scale_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
    obs_dist: Any = dists.Normal,
    sigma_dist: Any = dists.Exponential,
    sigma_kwargs: dict[str, Any] | None = None,
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

    y_hat = jnp.asarray(y_hat)
    sigma: float | jax.Array
    if untransformed_scale is not None:
        sigma = jax.nn.softplus(untransformed_scale)
    elif scale is not None:
        sigma = scale
    else:
        sigma = sample("sigma", sigma_dist(**sigma_kwargs))
    sigma = _align_scale(sigma, y_hat)
    return _observe("obs", obs_dist(loc=y_hat, scale=sigma), y)


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


student_t_link = partial(
    _loc_scale_link, obs_dist=partial(dists.StudentT, df=4.0)
)
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


def gamma_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
    rate: float = 1.0,
) -> jax.Array:
    """Gamma likelihood (log link) for positive continuous data.

    Uses a mean parameterisation with a learned shape ``k``:

    .. math::
        \\mu = \\exp(\\hat{y}), \\quad
        k \\sim \\mathrm{Exponential}(\\text{rate}), \\quad
        y \\sim \\mathrm{Gamma}(k,\\; k / \\mu)

    so ``E[y] = mu`` and ``Var[y] = mu^2 / k``.

    Args:
        y_hat: Log mean, shape ``(n, 1)`` or ``(n,)``.
        y: Observed positive values, or ``None``.
        rate: Rate of the ``Exponential`` prior on the shape ``k``.

    Returns:
        Sample site ``"obs"``.
    """
    mean = jnp.exp(_scalar_predictor(y_hat))
    k = sample("gamma_shape", dists.Exponential(rate=rate))
    return jnp.asarray(
        _observe("obs", dists.Gamma(concentration=k, rate=k / mean), y)
    )


def exponential_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
) -> jax.Array:
    """Exponential likelihood (log link) for positive continuous / survival data.

    .. math::
        \\mu = \\exp(\\hat{y}), \\quad y \\sim \\mathrm{Exponential}(1 / \\mu)

    A single-parameter special case of :func:`gamma_link` (shape fixed at 1);
    the mean fully determines the variance (``Var[y] = mu^2``).

    Args:
        y_hat: Log mean, shape ``(n, 1)`` or ``(n,)``.
        y: Observed positive values, or ``None``.

    Returns:
        Sample site ``"obs"``.
    """
    rate = jnp.exp(-_scalar_predictor(y_hat))
    return jnp.asarray(_observe("obs", dists.Exponential(rate=rate), y))


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
    return jnp.asarray(_observe("obs", dists.Bernoulli(logits=y_hat), y))


def categorical_link(
    logits: jax.Array,
    y: jax.Array | None = None,
) -> jax.Array:
    """Categorical (softmax) likelihood for multiclass classification.

    The multiclass generalisation of :func:`logit_link`.  Produce one logit per
    class with a layer's ``units`` argument (``units = num_classes``); the
    number of classes is read from the trailing dimension of ``logits``.

    .. math::
        P(Y = k \\mid \\text{logits}) = \\mathrm{softmax}(\\text{logits})_k

    Args:
        logits: Unnormalised class scores of shape ``(n, num_classes)`` — e.g.
            ``AdaptiveLayer()("beta", x, units=K)``.  A trailing singleton
            (``(n, num_classes, 1)``) is squeezed automatically.
        y: Integer class labels in ``{0, ..., num_classes - 1}``, or ``None``
            for prior predictive / inference.

    Returns:
        Sample site ``"obs"`` with integer values in ``{0, …, num_classes-1}``.

    Example::

        from blayers.layers import AdaptiveLayer
        from blayers.links import categorical_link

        def model(x, y=None):
            logits = AdaptiveLayer()("beta", x, units=4)   # 4 classes
            return categorical_link(logits, y)
    """
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    return jnp.asarray(_observe("obs", dists.Categorical(logits=logits), y))


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
    return jnp.asarray(_observe("obs", dists.Poisson(rate=jnp.exp(y_hat)), y))


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
    return jnp.asarray(
        _observe(
            "obs",
            dists.NegativeBinomial2(mean=y_hat, concentration=concentration),
            y,
        )
    )


def ordinal_link(
    mu: jax.Array,
    y: jax.Array | None = None,
    *,
    num_classes: int,
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
    mu_flat = _scalar_predictor(mu)
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

    return jnp.asarray(_observe("obs", dists.Categorical(probs=probs), y))


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
    rate = jnp.exp(_scalar_predictor(mu))
    gate = sample("zip_gate", dists.Beta(1.0, 10.0))
    return jnp.asarray(
        _observe("obs", dists.ZeroInflatedPoisson(gate=gate, rate=rate), y)
    )


def zinb_link(
    mu: jax.Array,
    y: jax.Array | None = None,
    rate: float = 1.0,
) -> jax.Array:
    """Zero-inflated NegativeBinomial2 link for overdispersed counts with excess zeros.

    The overdispersed counterpart of :func:`zip_link`: a mixture that emits an
    exact 0 with probability π, and otherwise a NegativeBinomial2 count with
    ``mean = exp(μ)`` and a learned concentration (so the non-zero part can be
    more dispersed than Poisson).

    .. math::
        \\text{gate} \\sim \\mathrm{Beta}(1, 10), \\quad
        \\phi \\sim \\mathrm{Exponential}(\\text{rate}), \\quad
        y \\sim \\mathrm{ZINB}(\\exp(\\mu),\\; \\phi,\\; \\text{gate})

    Args:
        mu: Log mean, shape ``(n, 1)`` or ``(n,)``.
        y: Non-negative integer observations, or ``None``.
        rate: Rate of the ``Exponential`` prior on the concentration.

    Returns:
        Sample site ``"obs"``.
    """
    mean = jnp.exp(_scalar_predictor(mu))
    concentration = sample("zinb_concentration", dists.Exponential(rate=rate))
    gate = sample("zinb_gate", dists.Beta(1.0, 10.0))
    base = dists.NegativeBinomial2(mean=mean, concentration=concentration)
    return jnp.asarray(
        _observe("obs", dists.ZeroInflatedDistribution(base, gate=gate), y)
    )


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
    mean = jnn.sigmoid(_scalar_predictor(mu))
    phi = sample("beta_phi", dists.Exponential(1.0))
    return jnp.asarray(
        _observe("obs", dists.Beta(mean * phi, (1.0 - mean) * phi), y)
    )
