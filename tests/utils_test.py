import jax.numpy as jnp
import jax.random as random
import pytest
import pytest_check

from blayers._utils import (
    add_trailing_dim,
    get_dataset_size,
    get_steps_and_steps_per_epoch,
    yield_batches,
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


def test_yield_batches_no_shuffle_is_contiguous() -> None:
    """rng_key=None reproduces the legacy contiguous ordering."""
    data = {"x": jnp.arange(6)}
    batches = list(
        yield_batches(data, batch_size=2, num_batches=3, steps_per_epoch=3)
    )
    assert [b["x"].tolist() for b in batches] == [[0, 1], [2, 3], [4, 5]]


def test_yield_batches_shuffle_covers_all_rows_per_epoch() -> None:
    """With a key, one epoch of batches is a permutation of every row."""
    data = {"x": jnp.arange(6)}
    key = random.PRNGKey(0)
    batches = list(
        yield_batches(
            data, batch_size=2, num_batches=3, steps_per_epoch=3, rng_key=key
        )
    )
    rows = sorted(v for b in batches for v in b["x"].tolist())
    assert rows == [0, 1, 2, 3, 4, 5]


def test_yield_batches_shuffle_reorders_across_epochs() -> None:
    """Consecutive epochs should not repeat the same fixed ordering."""
    data = {"x": jnp.arange(8)}
    key = random.PRNGKey(0)
    # two epochs of 4 batches each (batch_size 2)
    batches = list(
        yield_batches(
            data, batch_size=2, num_batches=8, steps_per_epoch=4, rng_key=key
        )
    )
    epoch1 = [v for b in batches[:4] for v in b["x"].tolist()]
    epoch2 = [v for b in batches[4:] for v in b["x"].tolist()]
    assert epoch1 != epoch2  # re-permuted each epoch
    assert sorted(epoch1) == sorted(epoch2) == list(range(8))


def test_yield_batches_paired_arrays_stay_aligned() -> None:
    """Shuffling must permute every array with the same index order."""
    data = {"x": jnp.arange(6), "y": jnp.arange(6) * 10}
    key = random.PRNGKey(1)
    batches = list(
        yield_batches(
            data, batch_size=2, num_batches=3, steps_per_epoch=3, rng_key=key
        )
    )
    for b in batches:
        assert (b["y"] == b["x"] * 10).all()
