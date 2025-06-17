from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import random
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import SVI
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import SVIRunResult, SVIState

from blayers.fit_tools import get_dataset_size, yield_batches


class Batched_Trace_ELBO(ELBO):
    def __init__(self, n_obs: int, num_particles: int = 1):
        self.n_obs = n_obs
        self.num_particles = num_particles

    def loss(
        self,
        rng_key: jax.Array,
        param_map: dict[str, jax.Array],
        model: Callable[..., Any],
        guide: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        return -self.elbo_components(
            rng_key,
            param_map,
            model,
            guide,
            *args,
            **kwargs,
        )["elbo"]

    def elbo_components(
        self,
        rng_key: jax.Array,
        param_map: dict[str, jax.Array],
        model: Callable[..., Any],
        guide: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, jax.Array]:
        rng_keys = random.split(rng_key, self.num_particles)
        llhs, kls = [], []

        batch_size = kwargs[next(iter(kwargs.keys()))].shape[0]

        for key in rng_keys:
            guide_trace = trace(
                substitute(
                    seed(
                        guide,
                        key,
                    ),
                    param_map,
                )
            ).get_trace(
                *args,
                **kwargs,
            )

            # Extract latent sample values z ~ q(z)
            z_vals = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "sample"
            }

            # Evaluate model at those latent values
            model_trace = trace(
                substitute(
                    seed(
                        model,
                        key,
                    ),
                    z_vals,
                )
            ).get_trace(
                *args,
                **kwargs,
            )

            # log p(x | z)
            # upscale here by N / M where N is the nubmer of observations and M is the
            # batch size. This provides an estimator of the full dataset loss that
            # scales approriately with the KL.
            llhs.append(
                self.n_obs
                / batch_size
                * sum(
                    site["fn"].log_prob(site["value"]).sum()
                    for site in model_trace.values()
                    if site["type"] == "sample" and site["is_observed"]
                )
            )

            # KL = log q(z) - log p(z)
            log_pz = sum(
                site["fn"].log_prob(site["value"]).sum()
                for site in model_trace.values()
                if site["type"] == "sample" and not site["is_observed"]
            )
            log_qz = sum(
                site["fn"].log_prob(site["value"]).sum()
                for site in guide_trace.values()
                if site["type"] == "sample"
            )
            kls.append(log_qz - log_pz)

        # Average over particles
        llh_mean = jnp.mean(jnp.stack(llhs))
        kl_mean = jnp.mean(jnp.stack(kls))
        elbo = llh_mean - kl_mean

        return {
            "elbo": elbo,
            "llh": llh_mean,
            "kl": kl_mean,
        }


# ---------------------------------------------------------------------------- #


def svi_run_batched(
    svi: SVI,
    rng_key: jax.Array,
    num_steps: int,
    batch_size: int,
    **data: dict[str, jax.Array],
) -> SVIRunResult:
    @jax.jit
    def update(svi_state: SVIState, **kwargs: Any) -> SVIState:
        return svi.update(svi_state, **kwargs)

    n_obs = get_dataset_size(data)
    num_epochs = num_steps / (n_obs / batch_size)
    svi_state = svi.init(rng_key, **data)
    losses = []
    for batch in yield_batches(data, int(batch_size), int(num_epochs)):
        svi_state, loss = update(svi_state, **batch)
        losses.append(loss)
    return SVIRunResult(svi.get_params(svi_state), svi_state, jnp.stack(losses))
