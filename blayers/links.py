"""
We provide link functions as a convenience to abstract away a bit more Numpyro
boilerplate. Link functions take model predictions as inputs to a distribution.

The simplest example is the Gaussian link

.. code-block:: python

    mu = ...
    sigma ~ Exp(1)
    y     ~ Normal(mu, sigma)

We currently provide

* ``negative_binomial_link``
* ``logit_link``
* ``poission_link``
* ``gaussian_link_exp``
* ``lognormal_link_exp``

Link functions include trainable scale parameters when needed, as in the case
of Gaussians. We also provide classes for eaisly making additional links via
the ``LocScaleLink`` and ``SingleParamLink`` classes.

For instance, the Poisson link is created like this:

.. code-block:: python

    poission_link = SingleParamLink(obs_dist=dists.Poisson)


And implements

.. code-block:: python

    rate = ...
    y    ~ Poisson(rate)


In a Numpyro model, you use a link like

.. code-block:: python

    from blayers.layers import AdaptiveLayer
    from blayers.links import poisson_link
    def model(x, y):
        rate = AdaptiveLayer()('rate', x)
        return poisson_link(rate, y)

"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpyro.distributions as dists
from numpyro import sample


class Link(ABC):
    @abstractmethod
    def __init__(self, *args: Any) -> None:
        """Initialize link parameters."""

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        """
        Execute the link function.
        """


class LocScaleLink(Link):
    def __init__(
        self,
        sigma_dist: dists.Distribution = dists.Exponential,
        sigma_kwargs: dict[str, float] = {"rate": 1.0},
        obs_dist: dists.Distribution = dists.Normal,
        obs_kwargs: dict[str, float] = {},
    ) -> None:
        self.sigma_dist = sigma_dist
        self.sigma_kwargs = sigma_kwargs
        self.obs_dist = obs_dist
        self.obs_kwargs = obs_kwargs

    def __call__(
        self,
        y_hat: jax.Array,
        y: jax.Array | None = None,
        dependent_outputs: bool = False,
    ) -> jax.Array:
        sigma = sample("sigma", self.sigma_dist(**self.sigma_kwargs))

        if dependent_outputs:
            dist = self.obs_dist(
                loc=y_hat, scale=sigma, **self.obs_kwargs
            ).to_event(1)
        dist = self.obs_dist(loc=y_hat, scale=sigma, **self.obs_kwargs)

        return sample(
            "obs",
            dist,
            obs=y,
        )


class SingleParamLink(Link):
    def __init__(
        self,
        obs_dist: dists.Distribution = dists.Bernoulli,
    ) -> None:
        self.obs_dist = obs_dist

    def __call__(
        self,
        y_hat: jax.Array,
        y: jax.Array | None = None,
        dependent_outputs: bool = False,
    ) -> jax.Array:
        if dependent_outputs:
            dist = self.obs_dist(y_hat).to_event(1)
        dist = self.obs_dist(y_hat)

        return sample(
            "obs",
            dist,
            obs=y,
        )


# Exports


def negative_binomial_link(
    y_hat: jax.Array,
    y: jax.Array | None = None,
    dependent_outputs: bool = False,
    rate: float = 1.0,
) -> jax.Array:
    sigma = sample("sigma", dists.Exponential(rate=rate))

    if dependent_outputs:
        dist = dists.NegativeBinomial2(
            mean=y_hat, concentration=sigma
        ).to_event(1)
    dist = dists.NegativeBinomial2(mean=y_hat, concentration=sigma)

    return sample(
        "obs",
        dist,
        obs=y,
    )


logit_link = SingleParamLink()
"""Logit link function."""

poission_link = SingleParamLink(obs_dist=dists.Poisson)
"""Poisson link function."""

gaussian_link_exp = LocScaleLink()
"""Gaussian link function with exponentially distributed sigma."""

lognormal_link_exp = LocScaleLink(obs_dist=dists.LogNormal)
"""Lognormal link function with exponentially distributed sigma."""


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
    mu_flat = mu.squeeze()  # (n,)
    K = num_classes

    # Ordered cutpoints: anchor + positive increments
    c0 = sample("ordinal_c0", dists.Normal(0.0, 2.0))
    if K > 2:
        gaps = sample("ordinal_gaps", dists.Exponential(1.0).expand([K - 2]))
        cutpoints = jnp.concatenate([c0[None], c0 + jnp.cumsum(gaps)])
    else:
        cutpoints = c0[None]  # (1,) — single cutpoint for binary ordinal

    # Cumulative probs: (n, K-1)
    cum_probs = jnn.sigmoid(cutpoints - mu_flat[:, None])

    # Class probs: (n, K)
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

    Models a mixture: with probability π the outcome is exactly 0 (the
    "inflated" component); with probability 1 - π the outcome follows
    Poisson(λ) where λ = exp(μ).  π is a global scalar learned from data.

    Args:
        mu: Log Poisson rate, shape ``(n, 1)`` or ``(n,)``.
        y: Non-negative integer observations, or ``None``.

    Returns:
        Sample site ``"obs"``.
    """
    rate = jnp.exp(mu.squeeze())  # (n,)
    gate = sample("zip_gate", dists.Beta(1.0, 10.0))
    return sample("obs", dists.ZeroInflatedPoisson(gate=gate, rate=rate), obs=y)


def beta_link(
    mu: jax.Array,
    y: jax.Array | None = None,
) -> jax.Array:
    """Beta link for proportional outcomes strictly in (0, 1).

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
    mean = jnn.sigmoid(mu.squeeze())  # (n,) in (0, 1)
    phi = sample("beta_phi", dists.Exponential(1.0))
    return sample("obs", dists.Beta(mean * phi, (1.0 - mean) * phi), obs=y)
