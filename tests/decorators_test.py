"""Tests for the autoreparam decorator's two call styles."""

import jax.numpy as jnp
import jax.random as random
from numpyro.handlers import seed, trace

from blayers.decorators import autoreparam
from blayers.layers import AdaptiveLayer
from blayers.links import gaussian_link

X = random.normal(random.PRNGKey(1), (20, 2))


def _obs_present(model) -> bool:
    tr = trace(seed(model, random.PRNGKey(0))).get_trace(x=X)
    return "obs" in tr


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
