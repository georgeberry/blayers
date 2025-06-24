import jax.numpy as jnp
import pytest_check

from blayers.utils import add_trailing_dim


def test_add_trailing_dim() -> None:
    x = jnp.array([1.0, 2, 3])
    x_with_trail = add_trailing_dim(x)

    with pytest_check.check:
        assert len(x.shape) == 1

    with pytest_check.check:
        assert len(x_with_trail.shape) == 2
