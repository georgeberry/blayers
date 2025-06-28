"""
We want to systematically reparam models
"""

from functools import wraps
from numpyro.handlers import trace, reparam as numpyro_reparam, seed
from numpyro.infer.reparam import LocScaleReparam
from numpyro import distributions as dist
import jax.random as random

LocScaleDist = (
    dist.Normal
    | dist.LogNormal
    | dist.StudentT
    | dist.Cauchy
    | dist.Laplace
    | dist.Gumbel
)


def autoreparam(centered=0.0):
    def decorator(model_fn):
        @wraps(model_fn)
        def wrapped_model(*args, **kwargs):
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
                if isinstance(site["fn"], LocScaleDist):
                    config[name] = LocScaleReparam(centered=centered)

            # Wrap and return reparam'd model
            return numpyro_reparam(config=config)(model_fn)

        return wrapped_model
    return decorator