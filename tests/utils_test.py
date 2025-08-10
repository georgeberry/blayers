import jax.numpy as jnp
import pytest
import pytest_check

from blayers._utils import (
    add_trailing_dim,
    get_dataset_size,
    get_steps_and_steps_per_epoch,
)


def test_add_trailing_dim() -> None:
    x = jnp.array([1.0, 2, 3])
    x_with_trail = add_trailing_dim(x)

    with pytest_check.check:
        assert len(x.shape) == 1

    with pytest_check.check:
        assert len(x_with_trail.shape) == 2


def test_get_dataset_size() -> None:
    with pytest_check.check:
        with pytest.raises(ValueError):
            get_dataset_size(
                data={"x": jnp.array([1.0]), "z": jnp.array([1.0, 2])}
            )

    with pytest_check.check:
        size = get_dataset_size(
            data={"x": jnp.array([1.0, 3]), "z": jnp.array([1.0, 2])}
        )
        assert size == 2


def test_get_steps_per_epoch() -> None:
    with pytest_check.check:
        with pytest.raises(IndexError):
            get_steps_and_steps_per_epoch(
                data={},
                batch_size=1,
                num_steps=10,
            )

    with pytest_check.check:
        steps, steps_per_epoch = get_steps_and_steps_per_epoch(
            data={"x": jnp.array([1.0, 3]), "z": jnp.array([1.0, 2])},
            batch_size=1,
            num_steps=10,
        )
        assert steps_per_epoch == 2

    with pytest_check.check:
        steps, steps_per_epoch = get_steps_and_steps_per_epoch(
            data={"x": jnp.array([1.0, 3]), "z": jnp.array([1.0, 2])},
            batch_size=1,
            num_epochs=10,
        )
        assert steps == 20
