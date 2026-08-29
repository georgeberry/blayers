"""Tests for automatic model reparameterization."""

import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro import distributions as dist
from numpyro.handlers import seed, trace

from blayers.decorators import autoreparam
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

X = random.normal(random.PRNGKey(1), (20, 2))


def _obs_present(model) -> bool:
    tr = trace(seed(model, random.PRNGKey(0))).get_trace(x=X)
    return "obs" in tr


def _trace(model):
    return trace(seed(model, random.PRNGKey(0))).get_trace()


def test_autoreparam_bare_decorator() -> None:
    """@autoreparam (no parens) returns a working, reparameterized model."""

    @autoreparam
    def model(x: jnp.ndarray, y=None):
        return gaussian_link(AdaptiveLayer()("b", x), y)

    assert _obs_present(model)


def test_autoreparam_called_decorator() -> None:
    """@autoreparam(centered=...) (with parens) is the factory style."""

    @autoreparam(centered=0.0)
    def model(x: jnp.ndarray, y=None):
        return gaussian_link(AdaptiveLayer()("b", x), y)

    assert _obs_present(model)


def test_autoreparam_transforms_latent_normal() -> None:
    @autoreparam
    def model():
        return numpyro.sample("x", dist.Normal(2.0, 3.0))

    tr = _trace(model)

    assert tr["x"]["type"] == "deterministic"
    assert isinstance(tr["x_decentered"]["fn"], dist.Normal)
    assert tr["x_decentered"]["fn"].loc == 0.0
    assert tr["x_decentered"]["fn"].scale == 1.0


def test_autoreparam_preserves_observed_and_unrelated_sites() -> None:
    @autoreparam
    def model():
        numpyro.sample("count", dist.Poisson(2.0))
        numpyro.sample("obs", dist.Normal(0.0, 1.0), obs=0.5)

    tr = _trace(model)

    assert isinstance(tr["count"]["fn"], dist.Poisson)
    assert tr["obs"]["is_observed"]
    assert "obs_decentered" not in tr


def test_autoreparam_handles_wrapped_distribution() -> None:
    @autoreparam
    def model():
        return numpyro.sample("x", dist.Normal(jnp.zeros(2), 1.0).to_event(1))

    tr = _trace(model)

    assert tr["x"]["type"] == "deterministic"
    assert tr["x_decentered"]["fn"].event_shape == (2,)


def test_autoreparam_handles_student_t_shape_parameter() -> None:
    @autoreparam
    def model():
        return numpyro.sample("x", dist.StudentT(4.0, 2.0, 3.0))

    tr = _trace(model)

    assert tr["x"]["type"] == "deterministic"
    assert isinstance(tr["x_decentered"]["fn"], dist.StudentT)
    assert tr["x_decentered"]["fn"].df == 4.0


def test_autoreparam_handles_lognormal_via_its_base_distribution() -> None:
    @autoreparam
    def model():
        return numpyro.sample("x", dist.LogNormal(2.0, 3.0))

    tr = _trace(model)

    assert tr["x"]["type"] == "deterministic"
    assert tr["x_base"]["type"] == "deterministic"
    assert isinstance(tr["x_base_decentered"]["fn"], dist.Normal)


def test_autoreparam_executes_model_once_and_returns_its_value() -> None:
    calls = 0

    @autoreparam
    def model():
        nonlocal calls
        calls += 1
        numpyro.sample("x", dist.Normal(0.0, 1.0))
        return "result"

    result = seed(model, random.PRNGKey(0))()

    assert calls == 1
    assert result == "result"
