"""
We want to systematically reparam models
"""

from functools import wraps
from numpyro.handlers import trace, reparam as numpyro_reparam
from numpyro.infer.reparam import LocScaleReparam
from numpyro import distributions as dist

LocScaleDist = (
    dist.Normal
    | dist.LogNormal
    | dist.StudentT
    | dist.Cauchy
    | dist.Laplace
    | dist.Gumbel
)


def reparam():
    """Reparameterizes all valid sites in a model with `LocScaleReparam."""

    def decorator(model_fn):
        @wraps(model_fn)
        def wrapped_model(*args, **kwargs):
            # Trace once to find applicable sample sites
            with trace() as tr:
                model_fn(*args, **kwargs)

            # Build config for all supported loc-scale sites
            config = {}
            for name, site in tr.items():
                if site["type"] != "sample" or site["is_observed"]:
                    continue
                fn = site["fn"]
                if isinstance(fn, LocScaleDist):
                    config[name] = LocScaleReparam()

            # Apply reparam
            return numpyro_reparam(config=config)(model_fn)(*args, **kwargs)

        return wrapped_model

    return decorator
