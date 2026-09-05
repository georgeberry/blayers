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
from numpyro.handlers import seed
from numpyro.infer import SVI
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import SVIRunResult, SVIState
from numpyro.infer.util import compute_log_probs

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
        batch_size: Fallback batch size for calls without array inputs.
            Otherwise the actual leading dimension of row-aligned positional
            and keyword arrays is used, including short remainder batches.
            Bind non-row arrays (e.g. knots) into the model with a closure.

    NumPyro scale and mask handlers are honored for model and guide sites.
    All observed sites, including ``numpyro.factor`` terms, are treated as
    row-wise likelihood contributions and receive N/B scaling. Global factors
    and local latent variables are unsupported in minibatched mode.

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
        if (
            num_obs <= 0
            or num_particles <= 0
            or (batch_size is not None and batch_size <= 0)
        ):
            raise ValueError(
                "num_obs, num_particles, and batch_size must be positive"
            )
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

        # Row-aligned inputs describe the *actual* batch, including a short
        # remainder or a requested batch larger than the entire dataset.
        sizes = [
            value.shape[0]
            for value in (*args, *kwargs.values())
            if hasattr(value, "shape") and len(value.shape) > 0
        ]
        if sizes and any(size != sizes[0] for size in sizes):
            raise ValueError("Batched inputs must have consistent row counts.")
        batch_size = sizes[0] if sizes else self.batch_size
        if batch_size is None or batch_size <= 0:
            raise ValueError(
                "Cannot infer a positive batch size from args or kwargs"
            )

        for key in rng_keys:
            # Delegate site densities to NumPyro so scale/mask handlers and
            # transformed-distribution intermediates retain their semantics.
            guide_log_probs, guide_trace = compute_log_probs(
                seed(guide, key), args, kwargs, param_map
            )
            z_vals = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "sample"
            }
            model_log_probs, model_trace = compute_log_probs(
                seed(model, key), args, kwargs, {**param_map, **z_vals}
            )
            _raise_if_has_plate(model_trace)
            _raise_if_has_plate(guide_trace)

            llh = sum(
                value
                for name, value in model_log_probs.items()
                if model_trace[name]["is_observed"]
            )
            log_pz = sum(
                value
                for name, value in model_log_probs.items()
                if not model_trace[name]["is_observed"]
            )
            log_qz = sum(guide_log_probs.values())
            llhs.append(self.num_obs / batch_size * llh)
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
