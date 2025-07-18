from abc import ABC, abstractmethod
from typing import Any

import jax
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
        dependent_outputs: bool = False,
    ) -> None:
        self.sigma_dist = sigma_dist
        self.sigma_kwargs = sigma_kwargs
        self.obs_dist = obs_dist
        self.obs_kwargs = obs_kwargs
        self.dependent_outputs = dependent_outputs

    def __call__(
        self, y_hat: jax.Array, y: jax.Array | None = None
    ) -> jax.Array:
        sigma = sample("sigma", self.sigma_dist(**self.sigma_kwargs))

        if self.dependent_outputs:
            dist = self.obs_dist(
                loc=y_hat, scale=sigma, **self.obs_kwargs
            ).to_event(1)
        dist = self.obs_dist(loc=y_hat, scale=sigma, **self.obs_kwargs)

        return sample(
            "obs",
            dist,
            obs=y,
        )


gaussian_link_exp = LocScaleLink()
lognormal_link_exp = LocScaleLink(obs_dist=dists.LogNormal)


class SingleParamLink(Link):
    def __init__(
        self,
        obs_dist: dists.Distribution = dists.Bernoulli,
        dependent_outputs: bool = False,
    ) -> None:
        self.obs_dist = obs_dist
        self.dependent_outputs = dependent_outputs

    def __call__(
        self, y_hat: jax.Array, y: jax.Array | None = None
    ) -> jax.Array:
        if self.dependent_outputs:
            dist = self.obs_dist(y_hat).to_event(1)
        dist = self.obs_dist(y_hat)

        return sample(
            "obs",
            dist,
            obs=y,
        )


logit_link = SingleParamLink()
poission_link = SingleParamLink(obs_dist=dists.Poisson)


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
