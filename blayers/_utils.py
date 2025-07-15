import itertools
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
) -> Generator[dict[str, jax.Array], None, None]:
    def batch_iter() -> Generator[dict[str, jax.Array], None, None]:
        for i in range(steps_per_epoch):
            start = i * batch_size
            end = start + batch_size
            yield {k: v[start:end] for k, v in data.items()}

    for batch in itertools.islice(itertools.cycle(batch_iter()), num_batches):
        yield batch


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
