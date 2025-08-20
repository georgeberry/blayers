import warnings

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

from blayers.infer import Batched_Trace_ELBO, _warn_if_has_plate


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
    y = x * true_beta + jax.random.normal(rng_key, (N,))

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
            num_obs=1,
            batch_size=1,
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


def test_plate_warning() -> None:
    key = jax.random.PRNGKey(0)
    data = jnp.ones(10)

    def model_with_plate(data: jax.Array) -> None:
        mu = sample("mu", dist.Normal(0, 1))
        with plate("data", len(data)):
            sample("obs", dist.Normal(mu, 1), obs=data)

    model_trace = trace(substitute(seed(model_with_plate, key), {})).get_trace(
        data
    )

    with pytest.warns(UserWarning, match="Model contains plates"):
        _warn_if_has_plate(model_trace)


def test_no_plate_no_warning() -> None:
    key = jax.random.PRNGKey(0)
    data = jnp.ones(10)

    def model_no_plate(data: jax.Array) -> None:
        mu = sample("mu", dist.Normal(0, 1))
        sample("obs", dist.Normal(mu, 1), obs=data)

    model_trace = trace(substitute(seed(model_no_plate, key), {})).get_trace(
        data
    )

    with warnings.catch_warnings(record=True) as w:
        _warn_if_has_plate(model_trace)

    assert len(w) == 0
