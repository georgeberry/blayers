from abc import ABC, abstractmethod
from typing import Any

import jax
import numpyro.distributions as dist
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
        sigma_dist: dist.Distribution = dist.Exponential,
        sigma_kwargs: dict[str, float] = {"rate": 1.0},
        obs_dist: dist.Distribution = dist.Normal,
        obs_kwargs: dict[str, float] = {},
    ) -> None:
        self.sigma_dist = sigma_dist
        self.sigma_kwargs = sigma_kwargs
        self.obs_dist = obs_dist
        self.obs_kwargs = obs_kwargs

    def __call__(
        self, y_hat: jax.Array, y: jax.Array | None = None
    ) -> jax.Array:
        sigma = sample("sigma", self.sigma_dist(**self.sigma_kwargs))
        return sample(
            "obs",
            self.obs_dist(loc=y_hat, scale=sigma, **self.obs_kwargs),
            obs=y,
        )


gaussian_link_exp = LocScaleLink()
lognormal_link_exp = LocScaleLink(obs_dist=dist.LogNormal)


class SingleParamLink(Link):
    def __init__(
        self,
        obs_dist: dist.Distribution = dist.Bernoulli,
    ) -> None:
        self.obs_dist = obs_dist

    def __call__(
        self, y_hat: jax.Array, y: jax.Array | None = None
    ) -> jax.Array:
        return sample(
            "obs",
            self.obs_dist(y_hat),
            obs=y,
        )


logit_link = SingleParamLink()
poission_link = SingleParamLink(obs_dist=dist.Poisson)
