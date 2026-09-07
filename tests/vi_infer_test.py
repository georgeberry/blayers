import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
import pytest_check
from numpyro import plate, sample
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal

from blayers.vi_infer import (
    Batched_Trace_ELBO,
    _raise_if_has_plate,
    svi_run_batched,
)


def test_builtin_vs_batched_elbo_simple() -> None:
    def model() -> jax.Array:
        return numpyro.sample("z", dist.Normal(0.0, 1.0))

    rng_key = jax.random.PRNGKey(0)
    guide = AutoDiagonalNormal(model)
    optim = numpyro.optim.Adam(0.0)

    svi_builtin = SVI(
        model,
        guide,
        optim,
        loss=Trace_ELBO(num_particles=1),
    )
    state_builtin = svi_builtin.init(rng_key)
    svi_batched = SVI(
        model,
        guide,
        optim,
        loss=Batched_Trace_ELBO(
            num_particles=1,
            num_obs=1,
            batch_size=1,
        ),
    )
    state_batched = svi_batched.init(rng_key)

    elbo_builtin = svi_builtin.evaluate(state_builtin)
    elbo_batched = svi_batched.evaluate(state_batched)

    with pytest_check.check:
        assert jnp.allclose(
            elbo_builtin,
            elbo_batched,
            rtol=1e-3,
        ), "ELBO mismatch"


def test_builtin_vs_batched_elbo_regression() -> None:
    def model(x: jax.Array, y: jax.Array | None = None) -> None:
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        mu = x * beta
        numpyro.sample("obs", dist.Normal(mu, 1.0), obs=y)

    rng_key = jax.random.PRNGKey(0)

    # Simulate data
    N, D = 100, 1
    true_beta = jnp.array([2.5])
    x = jax.random.normal(rng_key, (N, D))
    y = x * true_beta + jax.random.normal(rng_key, (N, 1))

    guide = AutoDiagonalNormal(model)
    optim = numpyro.optim.Adam(0.0)

    svi_builtin = SVI(
        model,
        guide,
        optim,
        loss=Trace_ELBO(num_particles=1000),
        x=x,
        y=y,
    )
    state_builtin = svi_builtin.init(rng_key)
    svi_batched = SVI(
        model,
        guide,
        optim,
        loss=Batched_Trace_ELBO(
            num_particles=1000,
            num_obs=N,
            batch_size=N,
        ),
        x=x,
        y=y,
    )
    state_batched = svi_batched.init(rng_key)

    elbo_builtin = svi_builtin.evaluate(state_builtin)
    elbo_batched = svi_batched.evaluate(state_batched)

    with pytest_check.check:
        assert jnp.allclose(
            elbo_builtin,
            elbo_batched,
            rtol=1e-3,
        ), "ELBO mismatch"


def test_no_batch_error() -> None:
    def model() -> jax.Array:
        return numpyro.sample("z", dist.Normal(0.0, 1.0))

    rng_key = jax.random.PRNGKey(0)
    guide = AutoDiagonalNormal(model)
    optim = numpyro.optim.Adam(0.0)

    svi_batched = SVI(
        model,
        guide,
        optim,
        loss=Batched_Trace_ELBO(
            num_particles=1,
            num_obs=1,
        ),
    )
    state_batched = svi_batched.init(rng_key)
    with pytest.raises(ValueError):
        svi_batched.evaluate(state_batched)


def test_plate_raises() -> None:
    key = jax.random.PRNGKey(0)
    data = jnp.ones(10)

    def model_with_plate(data: jax.Array) -> None:
        mu = sample("mu", dist.Normal(0, 1))
        with plate("data", len(data)):
            sample("obs", dist.Normal(mu, 1), obs=data)

    model_trace = trace(substitute(seed(model_with_plate, key), {})).get_trace(
        data
    )

    with pytest.raises(ValueError, match="does not support"):
        _raise_if_has_plate(model_trace)


def test_no_plate_does_not_raise() -> None:
    key = jax.random.PRNGKey(0)
    data = jnp.ones(10)

    def model_no_plate(data: jax.Array) -> None:
        mu = sample("mu", dist.Normal(0, 1))
        sample("obs", dist.Normal(mu, 1), obs=data)

    model_trace = trace(substitute(seed(model_no_plate, key), {})).get_trace(
        data
    )

    _raise_if_has_plate(model_trace)  # should not raise


@pytest.mark.parametrize("shuffle", [True, False])
def test_svi_run_batched_shuffle_option(shuffle: bool) -> None:
    """Both shuffle settings run and return one loss per gradient step."""

    def model(x: jax.Array, y: jax.Array | None = None) -> None:
        beta = sample("beta", dist.Normal(0.0, 1.0))
        sample("obs", dist.Normal(x.squeeze() * beta, 1.0), obs=y)

    key = jax.random.PRNGKey(0)
    n = 40
    x = jax.random.normal(key, (n, 1))
    y = (x.squeeze() * 2.5 + jax.random.normal(key, (n,))).astype(x.dtype)

    guide = AutoDiagonalNormal(model)
    svi = SVI(
        model,
        guide,
        numpyro.optim.Adam(0.01),
        loss=Batched_Trace_ELBO(num_obs=n, batch_size=10),
    )

    result = svi_run_batched(
        svi,
        key,
        batch_size=10,
        num_epochs=2,
        shuffle=shuffle,
        x=x,
        y=y,
    )
    # 2 epochs * ceil(40 / 10) = 8 gradient steps.
    assert result.losses.shape[0] == 8
    assert jnp.all(jnp.isfinite(result.losses))


def test_plate_raises_through_elbo() -> None:
    """A plate model must fail when the batched ELBO is actually evaluated."""
    key = jax.random.PRNGKey(0)
    data = jnp.ones(10)

    def model_with_plate(data: jax.Array) -> None:
        mu = sample("mu", dist.Normal(0, 1))
        with plate("data", len(data)):
            sample("obs", dist.Normal(mu, 1), obs=data)

    guide = AutoDiagonalNormal(model_with_plate)
    loss = Batched_Trace_ELBO(num_obs=10, batch_size=5)

    with pytest.raises(ValueError, match="does not support"):
        loss.loss(
            key,
            {},
            model_with_plate,
            guide,
            data,
        )
