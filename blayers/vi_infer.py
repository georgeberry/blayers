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

from blayers._utils import get_dataset_size, get_steps_and_steps_per_epoch


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
    progress_bar: bool = False,
    **data: jax.Array,
) -> SVIRunResult:
    """Drive batched VI with compiled minibatch and epoch loops.

    Args:
        shuffle: Re-permute rows each epoch (default). False retains the
            existing contiguous batch order, including its statistical tradeoff.
        progress_bar: Report progress once per epoch. Defaults to False so the
            complete update sequence runs without Python dispatch per epoch.

    Returns one loss per update. Short final batches retain their actual row
    count and N/B likelihood scaling; no observations are padded or dropped.
    Batch order and random-key splitting match the Python batch generator.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_steps, steps_per_epoch = get_steps_and_steps_per_epoch(
        data,
        batch_size,
        num_steps,
        num_epochs,
    )
    if total_steps is None or total_steps <= 0:
        raise ValueError("num_steps or num_epochs must be positive")
    size = get_dataset_size(data)
    if size <= 0:
        raise ValueError("Batched data must contain at least one row")
    epochs, extra_steps = divmod(total_steps, steps_per_epoch)
    init_key, batch_key = random.split(rng_key)
    state = svi.init(init_key, **data)

    def epoch(
        carry: tuple[SVIState, jax.Array],
        arrays: dict[str, jax.Array],
        step_count: int,
    ) -> tuple[tuple[SVIState, jax.Array], jax.Array]:
        svi_state, key = carry
        if shuffle:
            key, subkey = random.split(key)
            permutation = random.permutation(subkey, size)
        else:
            permutation = None

        def full_batch(
            svi_state: SVIState, index: jax.Array
        ) -> tuple[SVIState, jax.Array]:
            start = index * batch_size
            if permutation is None:
                batch = {
                    k: jax.lax.dynamic_slice_in_dim(v, start, batch_size)
                    for k, v in arrays.items()
                }
            else:
                indices = jax.lax.dynamic_slice_in_dim(
                    permutation, start, batch_size
                )
                batch = {k: v[indices] for k, v in arrays.items()}
            next_state, loss = svi.update(svi_state, **batch)
            return next_state, loss

        full_steps = min(step_count, size // batch_size)
        # Do not trace a full-size slice when the dataset is smaller than a batch.
        if full_steps:
            svi_state, losses = jax.lax.scan(
                full_batch,
                svi_state,
                jnp.arange(full_steps),
            )
        if step_count > full_steps:
            start = full_steps * batch_size
            batch = (
                {k: v[start:] for k, v in arrays.items()}
                if permutation is None
                else {k: v[permutation[start:]] for k, v in arrays.items()}
            )
            svi_state, loss = svi.update(svi_state, **batch)
            losses = (
                jnp.concatenate((losses, loss[None]))
                if full_steps
                else loss[None]
            )
        return (svi_state, key), losses

    carry = (state, batch_key)
    loss_chunks = []
    if epochs:
        if progress_bar:
            compiled_epoch = jax.jit(epoch, static_argnums=2)
            for _ in tqdm.tqdm(range(epochs), unit="epoch"):
                carry, losses = compiled_epoch(carry, data, steps_per_epoch)
                # Progress reflects completed device work, not queued dispatch.
                jax.block_until_ready(losses)
                loss_chunks.append(losses)
        else:

            @jax.jit
            def run_epochs(
                carry: tuple[SVIState, jax.Array],
                arrays: dict[str, jax.Array],
            ) -> tuple[tuple[SVIState, jax.Array], jax.Array]:
                next_carry, losses = jax.lax.scan(
                    lambda c, _: epoch(c, arrays, steps_per_epoch),
                    carry,
                    None,
                    length=epochs,
                )
                return next_carry, losses

            carry, losses = run_epochs(carry, data)
            loss_chunks.append(losses.reshape(-1))
    if extra_steps:
        carry, losses = jax.jit(epoch, static_argnums=2)(
            carry, data, extra_steps
        )
        loss_chunks.append(losses)
    state, _ = carry
    return SVIRunResult(
        svi.get_params(state), state, jnp.concatenate(loss_chunks)
    )
