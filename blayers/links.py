from typing import Any

import jax
import numpyro.distributions as dist
from numpyro import sample


def gaussian_link_exp(y_hat: jax.Array, y: jax.Array | None = None) -> Any:
    sigma = sample("sigma", dist.Exponential(1.0))
    return sample("obs", dist.Normal(loc=y_hat, scale=sigma), obs=y)
