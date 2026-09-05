"""
Model decorators for blayers.

- ``autoreshape``: Auto-reshape 1D arrays to (n, 1)
- ``autoreparam``: Auto-reparameterize LocScale distributions for VI and MCMC

Usage::

    @autoreshape
    @autoreparam
    def my_model(x, y=None):
        ...
"""

import logging
from functools import wraps
from typing import Any, Callable

import jax.numpy as jnp

logger = logging.getLogger(__name__)
from numpyro import distributions as dist
from numpyro.handlers import reparam as numpyro_reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

LocScaleDist = (
    dist.Normal
    | dist.LogNormal
    | dist.StudentT
    | dist.Cauchy
    | dist.Laplace
    | dist.Gumbel
)


def autoreshape(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that ensures all array inputs have shape (n, d), not (n,).

    blayers expects arrays with explicit trailing dimensions. This decorator
    auto-reshapes 1D arrays to (n, 1) so you don't have to do it manually.

    Usage:
        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()('mu', x)
            return gaussian_link(mu, y)
    """
    _logged = False  # Only log once per decorated function

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal _logged
        reshaped = []

        # Reshape positional args
        new_args = []
        for i, arg in enumerate(args):
            if hasattr(arg, "shape") and len(arg.shape) == 1:
                reshaped.append(f"arg[{i}]: {arg.shape} -> ({arg.shape[0]}, 1)")
                arg = jnp.reshape(arg, (-1, 1))
            new_args.append(arg)

        # Reshape keyword args
        new_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "shape") and len(v.shape) == 1:
                reshaped.append(f"{k}: {v.shape} -> ({v.shape[0]}, 1)")
                v = jnp.reshape(v, (-1, 1))
            new_kwargs[k] = v

        if reshaped and not _logged:
            logger.info(
                f"autoreshape({fn.__name__}): reshaped {', '.join(reshaped)}"
            )
            _logged = True

        return fn(*new_args, **new_kwargs)

    return wrapped


def autoreparam(
    model_fn: Callable[..., Any] | None = None, *, centered: float = 0.0
) -> Any:
    """Auto-reparameterize LocScale distributions in a model for VI and MCMC.

    Automatically applies ``LocScaleReparam`` to all LocScale distributions
    (Normal, LogNormal, StudentT, Cauchy, Laplace, Gumbel) found in the model,
    which can improve NUTS mixing and diagonal-guide VI by removing prior
    funnel geometries. Strongly informed coefficients may favor centering.

    Works with or without parentheses::

        @autoreparam
        def model(x, y): ...

        @autoreparam()
        def model(x, y): ...

        @autoreparam(centered=0.5)
        def model(x, y): ...

    Args:
        model_fn: The model function (when used without parentheses).
        centered: Degree of centering for ``LocScaleReparam``. 0.0 = fully
            non-centered (default, best for weak data); 1.0 = fully centered
            (better when data is informative).
    """

    def decorator(fn: Any) -> Any:
        def config(site: dict[str, Any]) -> Any:
            # Reparameterizers create their own sample sites.  In particular,
            # LocScaleReparam appends this suffix; selecting it again would
            # recurse indefinitely.
            if site["name"].endswith("_decentered"):
                return None
            if site.get("is_observed", False):
                return None

            site_fn = site["fn"]
            while isinstance(
                site_fn, (dist.Independent, dist.ExpandedDistribution)
            ):
                site_fn = site_fn.base_dist

            # LogNormal has non-real support, so LocScaleReparam cannot be
            # applied directly.  First expose its Normal base site; the same
            # config then non-centers that generated ``*_base`` site.
            if isinstance(site_fn, dist.LogNormal):
                return TransformReparam()
            if isinstance(site_fn, dist.StudentT):
                return LocScaleReparam(centered=centered, shape_params=("df",))
            if isinstance(site_fn, LocScaleDist):
                return LocScaleReparam(centered=centered)
            return None

        @wraps(fn)
        def wrapped_model(*args: Any, **kwargs: Any) -> Any:
            return numpyro_reparam(config=config)(fn)(*args, **kwargs)

        return wrapped_model

    # Called as @autoreparam (no parens) — model_fn is the decorated function
    if model_fn is not None:
        return decorator(model_fn)

    # Called as @autoreparam() or @autoreparam(centered=0.5)
    return decorator
