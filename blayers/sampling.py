"""
We offer some help for automatically reparameterizing `LocScaleDist`s for
your MCMC models. Use like

```
@autoreparam
def my_model():
    ...
```
"""

from functools import wraps
from typing import Any

import jax.random as random
from numpyro import distributions as dist
from numpyro.handlers import reparam as numpyro_reparam
from numpyro.handlers import seed, trace
from numpyro.infer.reparam import LocScaleReparam

LocScaleDist = (
    dist.Normal
    | dist.LogNormal
    | dist.StudentT
    | dist.Cauchy
    | dist.Laplace
    | dist.Gumbel
)


def autoreparam(centered: float = 0.0) -> Any:
    def decorator(model_fn: Any) -> Any:
        @wraps(model_fn)
        def wrapped_model(*args: Any, **kwargs: Any) -> Any:
            # Use a fixed dummy seed so trace doesn't trigger global name
            # collisions
            dummy_key = random.PRNGKey(0)
            with seed(model_fn, rng_seed=dummy_key):
                with trace() as tr:
                    model_fn(*args, **kwargs)

            config = {}
            for name, site in tr.items():
                if site["type"] != "sample" or site.get("is_observed", False):
                    continue
                if isinstance(site["fn"], LocScaleDist) or (
                    hasattr(site["fn"], "base_dist")
                    and isinstance(site["fn"].base_dist, LocScaleDist)
                ):
                    config[name] = LocScaleReparam(centered=centered)

            # Wrap and return reparam'd model
            return numpyro_reparam(config=config)(model_fn)

        return wrapped_model

    return decorator
