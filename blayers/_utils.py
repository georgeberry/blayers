from typing import Generator

import jax
import jax.numpy as jnp


def get_dataset_size(data: dict[str, jax.Array]) -> int:
    # Check consistency and get dataset size
    lens = [v.shape[0] for v in data.values()]
    if len([x for x in lens if x != lens[0]]) > 0:
        raise ValueError(f"Inconsistent data lengths: {lens}")
    return int(lens[0])


def get_steps_and_steps_per_epoch(
    data: dict[str, jax.Array],
    batch_size: int,
    num_steps: int | None = None,
    num_epochs: int | None = None,
) -> tuple[int, int]:
    assert (num_steps is None) != (
        num_epochs is None
    ), "Exactly one of num_steps and num_epochs must be specified."

    dataset_size = get_dataset_size(data)
    # Next line by ChatGPT, what a great idea
    steps_per_epoch = (
        dataset_size + batch_size - 1
    ) // batch_size  # Ceiling division
    if num_epochs:
        return steps_per_epoch * num_epochs, steps_per_epoch
    return num_steps, steps_per_epoch  # type: ignore


def yield_batches(
    data: dict[str, jax.Array],
    batch_size: int,
    num_batches: int,
    steps_per_epoch: int,
    rng_key: jax.Array | None = None,
) -> Generator[dict[str, jax.Array], None, None]:
    """Yield ``num_batches`` minibatches, cycling over the data as needed.

    Each epoch, the row order is re-permuted when ``rng_key`` is supplied so
    that minibatch VI sees i.i.d. batches rather than the same fixed slices in
    the same order every pass (which biases the ELBO gradient, especially on
    sorted data).  Pass ``rng_key=None`` for the legacy contiguous ordering,
    which slices the arrays directly (no per-row gather) and is noticeably
    faster — trading the gradient de-biasing for speed.
    """
    dataset_size = get_dataset_size(data)

    def epoch_batches(
        perm: jax.Array | None,
    ) -> Generator[dict[str, jax.Array], None, None]:
        for i in range(steps_per_epoch):
            start, stop = i * batch_size, (i + 1) * batch_size
            if perm is None:
                # Contiguous slice — a cheap view, no fancy-index gather.
                yield {k: v[start:stop] for k, v in data.items()}
            else:
                idx = perm[start:stop]
                yield {k: v[idx] for k, v in data.items()}

    key = rng_key
    emitted = 0
    while emitted < num_batches:
        if key is not None:
            key, subkey = jax.random.split(key)
            perm: jax.Array | None = jax.random.permutation(
                subkey, dataset_size
            )
        else:
            perm = None
        for batch in epoch_batches(perm):
            if emitted >= num_batches:
                break
            yield batch
            emitted += 1


# ---- Helpers --------------------------------------------------------------- #


def rmse(m: jax.Array, m_hat: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.mean((m - m_hat) ** 2))


identity = lambda x: x
outer_product_upper_tril_no_diag = lambda x: (x.squeeze() @ x.squeeze().T)[
    jnp.triu_indices(x.shape[0], k=1)
]
outer_product_upper_tril_with_diag = lambda x: (x.squeeze() @ x.squeeze().T)[
    jnp.triu_indices(x.shape[0], k=0)
]

outer_product = lambda x, z: (x.squeeze() @ z.squeeze().T)


def add_trailing_dim(x: jax.Array) -> jax.Array:
    # get shapes and reshape if necessary
    if len(x.shape) == 1:
        x = jnp.reshape(x, (-1, 1))
    return x
