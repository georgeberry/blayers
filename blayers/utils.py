from typing import Generator

import jax
import jax.numpy as jnp


def get_dataset_size(data: dict[str, jax.Array]) -> int:
    # Check consistency and get dataset size
    lens = [v.shape[0] for v in data.values()]
    if len([x for x in lens if x != lens[0]]) > 0:
        raise ValueError(f"Inconsistent data lengths: {lens}")
    return int(lens[0])


def get_steps_per_epoch(data: dict[str, jax.Array], batch_size: int) -> int:
    """
    Compute the number of update steps per epoch for a dataset and batch size.

    Args:
        data (dict): A dictionary of arrays (e.g., {'x': x_data, 'y': y_data}),
                     where all arrays must have the same leading dimension.
        batch_size (int): The batch size to use.

    Returns:
        int: The number of steps per full pass over the data (i.e. steps per epoch).
    """
    if not data:
        raise ValueError("Empty data dict provided.")

    dataset_size = get_dataset_size(data)
    # Next line by ChatGPT, what a great idea
    steps = (dataset_size + batch_size - 1) // batch_size  # Ceiling division
    return int(steps)


def yield_batches(
    data: dict[str, jax.Array],
    batch_size: int,
    n_epochs: int,
) -> Generator[dict[str, jax.Array], None, None]:
    """Yields batches from a dict of arrays."""
    steps_per_epoch = get_steps_per_epoch(data, batch_size)
    for _ in range(n_epochs):
        for i in range(steps_per_epoch):
            start = i * batch_size
            end = start + batch_size
            yield {k: v[start:end] for k, v in data.items()}


# ---- Helpers --------------------------------------------------------------- #


def rmse(m: jax.Array, m_hat: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.mean((m - m_hat) ** 2))


identity = lambda x: x
outer_product_upper_tril_no_diag = lambda x: (x @ x.T)[
    jnp.triu_indices(x.shape[0], k=1)
]
outer_product_upper_tril_with_diag = lambda x: (x @ x.T)[
    jnp.triu_indices(x.shape[0], k=0)
]

outer_product = lambda x, z: (x @ z.T)


def add_trailing_dim(x: jax.Array) -> jax.Array:
    # get shapes and reshape if necessary
    if len(x.shape) == 1:
        x = jnp.reshape(x, (-1, 1))
    return x
