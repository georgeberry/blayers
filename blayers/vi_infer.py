"""
Variational-inference utilities for blayers.

Provides :class:`Batched_Trace_ELBO`, a drop-in ``Trace_ELBO`` replacement
that handles minibatching without requiring the model to use ``numpyro.plate``,
and :func:`svi_run_batched`, an ``svi.run``-style helper that drives it.

Use ``Batched_Trace_ELBO`` + ``svi_run_batched`` for plate-free batched VI;
fall back to standard ``Trace_ELBO`` if your model already uses plates.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import tqdm
from jax import random
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import SVI
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import SVIRunResult, SVIState

from blayers._utils import get_steps_and_steps_per_epoch, yield_batches


def _raise_if_has_plate(model_trace: dict[str, dict[str, Any]]) -> None:
    if any(site["type"] == "plate" for site in model_trace.values()):
        raise ValueError(
            "Batched_Trace_ELBO does not support models that use "
            "numpyro.plate: the N/B log-likelihood rescaling double-counts "
            "plate-subsampled sites and produces an incorrect ELBO. Either "
            "(a) batch via plate and use the standard numpyro Trace_ELBO, or "
            "(b) remove the plate and use Batched_Trace_ELBO + "
            "svi_run_batched."
        )


class Batched_Trace_ELBO(ELBO):
    """ELBO estimator for minibatched VI without ``numpyro.plate``.

    Behaves like ``Trace_ELBO`` but rescales the per-batch log-likelihood by
    ``num_obs / batch_size`` so the gradient is an unbiased estimate of the
    full-dataset ELBO.  Drive it with :func:`svi_run_batched`.

    **Assumes all latent variables are global.**  The whole observed
    log-likelihood is scaled by ``num_obs / batch_size`` and the KL over
    latents is *not* rescaled, which is only correct when every latent is
    shared across observations (the usual case for BLayers: coefficients,
    scales, embeddings).  Models with **per-observation (local) latents** —
    e.g. a latent variable sampled once per row — are **not supported** here;
    use ``numpyro.plate`` with the standard ``Trace_ELBO`` instead.

    Args:
        num_obs: Total number of observations in the full training set.
        num_particles: Number of Monte Carlo samples per gradient step.
        batch_size: Minibatch size.  If ``None``, inferred from the leading
            dimension of the first batched kwarg at loss-evaluation time.

    Warning:
        Does not mix with ``numpyro.plate``.  A ``ValueError`` is raised if a
        plate is detected in the model trace — the ``num_obs / batch_size``
        rescaling double-counts plate-subsampled sites, so the ELBO would be
        silently wrong.  Use the standard ``Trace_ELBO`` with plates instead.
    """

    def __init__(
        self,
        num_obs: int,
        num_particles: int = 1,
        batch_size: int | None = None,
    ):
        self.num_obs = num_obs
        self.num_particles = num_particles
        self.batch_size = batch_size

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

        batch_size = self.batch_size
        if batch_size is None:
            if len(kwargs) != 0:
                batch_size = kwargs[next(iter(kwargs.keys()))].shape[0]
            else:
                raise ValueError("Cannot infer batch size from args or kwargs")

        for key in rng_keys:
            # a key thing to realize is that this does sampling, so it samples
            # z ~ q(z)
            # mechanically this means we take expectations over q(z), since one
            # random sample is the expectation (in expectation)
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

            _raise_if_has_plate(model_trace)

            # log p(x | z)
            # upscale here by N / B where N is the nubmer of observations and B
            # is the batch size. This provides an estimator of the full dataset
            # loss that scales approriately with the KL.
            llhs.append(
                self.num_obs
                / batch_size
                * sum(
                    site["fn"].log_prob(site["value"]).sum()
                    for site in model_trace.values()
                    if site["type"] == "sample" and site["is_observed"]
                )
            )

            # KL[q(z) || p(z)] = H(q, p) - H(p) => log q(z) - log p(z)
            # implication comes from the fact that we draw one sample z ~ q(z)
            # if you'd like, swap P and Q and work through the math here:
            # wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Motivation
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
    batch_size: int,
    num_steps: int | None = None,
    num_epochs: int | None = None,
    shuffle: bool = True,
    **data: jax.Array,
) -> SVIRunResult:
    """Drive batched VI.

    Args:
        shuffle: When ``True`` (default) the row order is re-permuted every
            epoch — the unbiased-gradient behaviour.  When ``False`` batches are
            contiguous slices in a fixed order, which skips the per-epoch
            permutation and per-row gather and is noticeably faster; use it when
            your rows are already in random order (or the bias is acceptable).
    """

    @jax.jit
    def update(svi_state: SVIState, **kwargs: Any) -> SVIState:
        return svi.update(svi_state, **kwargs)

    total_steps_to_run, steps_per_epoch = get_steps_and_steps_per_epoch(
        data,
        batch_size,
        num_steps,
        num_epochs,
    )

    init_key, batch_key = random.split(rng_key)
    svi_state = svi.init(init_key, **data)
    losses = []
    for batch in tqdm.tqdm(
        yield_batches(
            data,
            batch_size,
            total_steps_to_run,
            steps_per_epoch,
            rng_key=batch_key if shuffle else None,
        ),
        total=total_steps_to_run,
    ):
        svi_state, loss = update(svi_state, **batch)
        losses.append(loss)
    return SVIRunResult(svi.get_params(svi_state), svi_state, jnp.stack(losses))
