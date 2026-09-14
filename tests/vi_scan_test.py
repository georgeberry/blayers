"""Compare the compiled runner against the original Python-loop algorithm."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import SVI
from numpyro.infer.autoguide import AutoDiagonalNormal

from blayers._utils import get_steps_and_steps_per_epoch, yield_batches
from blayers.vi_infer import Batched_Trace_ELBO, svi_run_batched


def reference(svi, key, data, batch_size, shuffle, **duration):
    count, per_epoch = get_steps_and_steps_per_epoch(
        data, batch_size, **duration
    )
    init_key, batch_key = jax.random.split(key)
    state = svi.init(init_key, **data)
    update = jax.jit(svi.update)
    losses = []
    for batch in yield_batches(
        data,
        batch_size,
        count,
        per_epoch,
        rng_key=batch_key if shuffle else None,
    ):
        state, loss = update(state, **batch)
        losses.append(loss)
    return state, jnp.stack(losses)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("progress_bar", [False, True])
@pytest.mark.parametrize(
    "size,batch,duration",
    [
        (12, 4, {"num_epochs": 3}),  # full epochs, no remainder
        (11, 4, {"num_epochs": 3}),  # real short remainder each epoch
        (3, 8, {"num_epochs": 3}),  # no full-size batches
        (11, 4, {"num_steps": 1}),  # stop inside first epoch
        (11, 4, {"num_steps": 5}),  # stop before second remainder
        (11, 4, {"num_steps": 6}),  # stop exactly at epoch boundary
        (11, 4, {"num_steps": 7}),  # include new epoch permutation
    ],
)
def test_equivalent_updates(size, batch, duration, shuffle, progress_bar):
    def model(x, y):
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Normal(x * beta, 1.0), obs=y)

    def make_svi():
        return SVI(
            model,
            AutoDiagonalNormal(model),
            numpyro.optim.Adam(0.01),
            Batched_Trace_ELBO(num_obs=size, batch_size=batch),
        )

    key = jax.random.PRNGKey(8)
    data = {"x": jnp.linspace(-1.0, 1.0, size), "y": jnp.arange(size) / size}
    old_state, old_losses = reference(
        make_svi(), key, data, batch, shuffle, **duration
    )
    new = svi_run_batched(
        make_svi(),
        key,
        batch,
        shuffle=shuffle,
        progress_bar=progress_bar,
        **duration,
        **data,
    )
    np.testing.assert_allclose(new.losses, old_losses, rtol=2e-5, atol=2e-5)
    for actual, expected in zip(
        jax.tree.leaves(new.state), jax.tree.leaves(old_state)
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0, "num_steps": 2},
        {"batch_size": 4, "num_steps": 0},
        {"batch_size": 4, "num_epochs": -1},
    ],
)
def test_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        svi_run_batched(None, jax.random.PRNGKey(0), x=jnp.ones(4), **kwargs)
