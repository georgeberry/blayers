"""Tests for FittedModel.to_arviz() — the ArviZ bridge.

Skipped entirely when arviz is not installed (optional dependency, and
arviz >= 1.0 requires Python >= 3.12).
"""

import jax.numpy as jnp
import jax.random as random
import pytest

from blayers.decorators import autoreshape
from blayers.fit import fit
from blayers.layers import AdaptiveLayer, InterceptLayer
from blayers.links import gaussian_link

az = pytest.importorskip("arviz")

NUM_OBS = 200
K = 3


@autoreshape
def _model(x, y=None):
    mu = InterceptLayer()("i") + AdaptiveLayer()("b", x)
    return gaussian_link(mu, y)


@pytest.fixture
def data() -> tuple:
    x = random.normal(random.PRNGKey(0), (NUM_OBS, K))
    beta = jnp.array([1.5, -2.0, 0.5])
    y = x @ beta + 0.3 * random.normal(random.PRNGKey(1), (NUM_OBS,))
    return x, y


# --------------------------------------------------------------------------- #
# VI
# --------------------------------------------------------------------------- #


def test_vi_to_arviz_groups(data) -> None:
    x, y = data
    result = fit(_model, y=y, num_steps=200, lr=0.05, seed=0, x=x)
    idata = result.to_arviz(y=y, x=x, num_samples=200)
    children = list(idata.children)
    assert "posterior" in children
    assert "log_likelihood" in children
    assert "observed_data" in children


def test_vi_to_arviz_loo_runs(data) -> None:
    """The log_likelihood group must be usable by az.loo (PSIS-LOO)."""
    x, y = data
    result = fit(_model, y=y, num_steps=200, lr=0.05, seed=0, x=x)
    idata = result.to_arviz(y=y, x=x, num_samples=200)
    loo = az.loo(idata)
    assert jnp.isfinite(float(loo.elpd))


def test_vi_to_arviz_requires_y(data) -> None:
    x, y = data
    result = fit(_model, y=y, num_steps=50, lr=0.05, seed=0, x=x)
    with pytest.raises(ValueError, match="needs the observed"):
        result.to_arviz(x=x)


# --------------------------------------------------------------------------- #
# MCMC
# --------------------------------------------------------------------------- #


def test_mcmc_to_arviz_groups(data) -> None:
    x, y = data
    result = fit(
        _model,
        y=y,
        method="mcmc",
        num_warmup=150,
        num_mcmc_samples=200,
        x=x,
    )
    idata = result.to_arviz()
    children = list(idata.children)
    assert "posterior" in children
    assert "sample_stats" in children  # divergences etc.
    assert "log_likelihood" in children


def test_mcmc_to_arviz_loo_and_summary(data) -> None:
    x, y = data
    result = fit(
        _model,
        y=y,
        method="mcmc",
        num_warmup=150,
        num_mcmc_samples=200,
        x=x,
    )
    idata = result.to_arviz()
    assert jnp.isfinite(float(az.loo(idata).elpd))
    summ = az.summary(idata, var_names=["AdaptiveLayer_b_beta"])
    assert "ess_bulk" in summ.columns


# --------------------------------------------------------------------------- #
# SVGD (unsupported)
# --------------------------------------------------------------------------- #


def test_svgd_to_arviz_not_implemented(data) -> None:
    x, y = data
    result = fit(
        _model,
        y=y,
        method="svgd",
        num_steps=100,
        num_particles=5,
        x=x,
    )
    with pytest.raises(NotImplementedError, match="does not support SVGD"):
        result.to_arviz(y=y, x=x)


# --------------------------------------------------------------------------- #
# Model comparison across methods
# --------------------------------------------------------------------------- #


def test_compare_vi_and_mcmc(data) -> None:
    x, y = data
    vi = fit(_model, y=y, num_steps=200, lr=0.05, seed=0, x=x).to_arviz(
        y=y, x=x, num_samples=200
    )
    mc = fit(
        _model,
        y=y,
        method="mcmc",
        num_warmup=150,
        num_mcmc_samples=200,
        x=x,
    ).to_arviz()
    cmp = az.compare({"vi": vi, "mcmc": mc})
    assert len(cmp) == 2
