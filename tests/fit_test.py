"""Tests for the high-level ``blayers.fit`` API."""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
import pytest
from numpyro import sample
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal

from blayers._utils import rmse
from blayers.fit import FittedModel, Predictions, _is_array, _split_data_and_constants, fit
from blayers.layers import AdaptiveLayer, InterceptLayer
from blayers.links import gaussian_link as gaussian_link_exp
from blayers.sampling import autoreshape

NUM_OBS = 2000
K = 3


# --------------------------------------------------------------------------- #
# Data generating process
# --------------------------------------------------------------------------- #


def dgp_simple(num_obs: int, k: int) -> dict[str, jax.Array]:
    lambda1 = sample("lambda1", dist.HalfNormal(1.0))
    beta = sample("beta", dist.Normal(0, lambda1).expand([k]))
    x = sample("x", dist.Normal(0, 1).expand([num_obs, k]))
    sigma = sample("sigma", dist.HalfNormal(1.0))
    mu = jnp.dot(x, beta)
    y = sample("y", dist.Normal(mu, sigma))
    return {"x": x, "y": y, "beta": beta, "sigma": sigma}


def _make_data() -> dict[str, jax.Array]:
    predictive = Predictive(dgp_simple, num_samples=1)
    samples = predictive(random.PRNGKey(0), num_obs=NUM_OBS, k=K)
    return {k: jnp.squeeze(v, axis=0) for k, v in samples.items()}


@pytest.fixture
def sim_data() -> dict[str, jax.Array]:
    return _make_data()


# --------------------------------------------------------------------------- #
# Model (what a user would write)
# --------------------------------------------------------------------------- #


@autoreshape
def linear_model(x, y=None):
    mu = InterceptLayer()("intercept") + AdaptiveLayer()("beta", x)
    return gaussian_link_exp(mu, y)


# --------------------------------------------------------------------------- #
# Helper unit tests
# --------------------------------------------------------------------------- #


class TestIsArray:
    def test_jax_2d(self) -> None:
        assert _is_array(jnp.ones((5, 3))) is True

    def test_jax_1d(self) -> None:
        assert _is_array(jnp.ones((5,))) is True

    def test_jax_0d(self) -> None:
        assert _is_array(jnp.array(5.0)) is False

    def test_python_int(self) -> None:
        assert _is_array(7) is False

    def test_python_float(self) -> None:
        assert _is_array(3.14) is False

    def test_python_str(self) -> None:
        assert _is_array("hello") is False


class TestSplitDataAndConstants:
    def test_mixed(self) -> None:
        data, constants = _split_data_and_constants(
            {"x": jnp.ones((10, 2)), "n_conditions": 7, "name": "test"}
        )
        assert "x" in data
        assert "n_conditions" in constants
        assert "name" in constants

    def test_all_arrays(self) -> None:
        data, constants = _split_data_and_constants(
            {"x": jnp.ones((10, 2)), "z": jnp.ones((10,))}
        )
        assert len(data) == 2
        assert len(constants) == 0

    def test_all_constants(self) -> None:
        data, constants = _split_data_and_constants({"a": 1, "b": 2.0})
        assert len(data) == 0
        assert len(constants) == 2


# --------------------------------------------------------------------------- #
# fit() with VI — batched
# --------------------------------------------------------------------------- #


def test_fit_vi_batched(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=30,
        lr=0.05,
        schedule="cosine",
        seed=0,
        x=sim_data["x"],
    )

    assert isinstance(result, FittedModel)
    assert result.method == "vi"
    assert result.params is not None
    assert result.guide is not None
    assert result.losses is not None
    assert len(result.losses) > 0


def test_fit_vi_batched_predict(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=30,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    preds = result.predict(x=sim_data["x"])

    assert isinstance(preds, Predictions)
    assert preds.mean.shape == (NUM_OBS,)
    assert preds.std.shape == (NUM_OBS,)
    assert preds.samples.ndim >= 2


# --------------------------------------------------------------------------- #
# fit() with VI — full-batch (no batching)
# --------------------------------------------------------------------------- #


def test_fit_vi_fullbatch(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        num_steps=500,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    assert result.method == "vi"
    assert result.params is not None
    assert result.losses is not None


def test_fit_vi_fullbatch_num_epochs(sim_data: dict[str, jax.Array]) -> None:
    """num_epochs without batch_size should work (one step = one epoch)."""
    result = fit(
        linear_model,
        y=sim_data["y"],
        num_epochs=500,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    assert result.method == "vi"
    assert result.params is not None


# --------------------------------------------------------------------------- #
# Constant auto-binding
# --------------------------------------------------------------------------- #


@autoreshape
def model_with_constant(x, n_features, y=None):
    """Model that takes a non-array constant."""
    # n_features is just used as a sanity check here; the real value is that
    # fit() binds it automatically so it doesn't flow through batching.
    assert isinstance(n_features, int)
    mu = AdaptiveLayer()("beta", x)
    return gaussian_link_exp(mu, y)


def test_constant_binding(sim_data: dict[str, jax.Array]) -> None:
    """Non-array kwargs should be bound via partial, not batched."""
    result = fit(
        model_with_constant,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=10,
        seed=0,
        x=sim_data["x"],
        n_features=K,  # int → should be auto-bound
    )

    assert result.method == "vi"
    assert result.params is not None

    # Predict should work without passing n_features again
    preds = result.predict(x=sim_data["x"])
    assert preds.mean.shape == (NUM_OBS,)


# --------------------------------------------------------------------------- #
# Custom guide
# --------------------------------------------------------------------------- #


def test_custom_guide_class(sim_data: dict[str, jax.Array]) -> None:
    """Passing a guide class should instantiate it on the model."""
    result = fit(
        linear_model,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=10,
        guide=AutoMultivariateNormal,
        seed=0,
        x=sim_data["x"],
    )

    assert isinstance(result.guide, AutoMultivariateNormal)


def test_custom_guide_instance(sim_data: dict[str, jax.Array]) -> None:
    """Passing a guide instance should use it directly."""
    guide_instance = AutoDiagonalNormal(linear_model)

    result = fit(
        linear_model,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=10,
        guide=guide_instance,
        seed=0,
        x=sim_data["x"],
    )

    assert result.guide is guide_instance


# --------------------------------------------------------------------------- #
# Schedule variants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("schedule", ["cosine", "warmup_cosine", "constant"])
def test_schedules(sim_data: dict[str, jax.Array], schedule: str) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=5,
        schedule=schedule,
        seed=0,
        x=sim_data["x"],
    )
    assert result.params is not None


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #


def test_both_epochs_and_steps_raises(sim_data: dict[str, jax.Array]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fit(
            linear_model,
            y=sim_data["y"],
            num_epochs=10,
            num_steps=100,
            x=sim_data["x"],
        )


def test_neither_epochs_nor_steps_raises(sim_data: dict[str, jax.Array]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fit(
            linear_model,
            y=sim_data["y"],
            x=sim_data["x"],
        )


def test_bad_method_raises(sim_data: dict[str, jax.Array]) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        fit(
            linear_model,
            y=sim_data["y"],
            method="bad",
            num_steps=10,
            x=sim_data["x"],
        )


def test_bad_schedule_raises(sim_data: dict[str, jax.Array]) -> None:
    with pytest.raises(ValueError, match="Unknown schedule"):
        fit(
            linear_model,
            y=sim_data["y"],
            num_steps=10,
            schedule="nonexistent",
            x=sim_data["x"],
        )


# --------------------------------------------------------------------------- #
# summary()
# --------------------------------------------------------------------------- #


def test_summary_vi(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=10,
        seed=0,
        x=sim_data["x"],
    )

    summary = result.summary(x=sim_data["x"])

    assert isinstance(summary, dict)
    assert len(summary) > 0

    for name, stats in summary.items():
        assert "mean" in stats
        assert "std" in stats
        assert "q025" in stats
        assert "q975" in stats
        assert "shape" in stats


# --------------------------------------------------------------------------- #
# MCMC
# --------------------------------------------------------------------------- #


def test_fit_mcmc(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="mcmc",
        num_warmup=100,
        num_mcmc_samples=200,
        num_chains=1,
        seed=0,
        x=sim_data["x"],
    )

    assert result.method == "mcmc"
    assert result.posterior_samples is not None
    assert result.params is None
    assert result.guide is None


def test_fit_mcmc_predict(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="mcmc",
        num_warmup=100,
        num_mcmc_samples=200,
        num_chains=1,
        seed=0,
        x=sim_data["x"],
    )

    preds = result.predict(x=sim_data["x"])

    assert isinstance(preds, Predictions)
    assert preds.mean.shape == (NUM_OBS,)
    assert preds.std.shape == (NUM_OBS,)


def test_fit_mcmc_summary(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="mcmc",
        num_warmup=100,
        num_mcmc_samples=200,
        num_chains=1,
        seed=0,
        x=sim_data["x"],
    )

    summary = result.summary()

    assert isinstance(summary, dict)
    assert len(summary) > 0


# --------------------------------------------------------------------------- #
# SVGD
# --------------------------------------------------------------------------- #


def test_fit_svgd(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="svgd",
        num_steps=300,
        num_particles=10,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    assert result.method == "svgd"
    assert result.params is not None
    assert result.guide is not None
    assert result.losses is not None
    assert result.num_particles == 10


def test_fit_svgd_predict(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="svgd",
        num_steps=300,
        num_particles=10,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    preds = result.predict(x=sim_data["x"], num_samples=20)

    assert isinstance(preds, Predictions)
    assert preds.mean.shape == (NUM_OBS,)
    assert preds.std.shape == (NUM_OBS,)
    # samples should be flattened: (num_samples * num_particles, n)
    assert preds.samples.shape[0] == 20 * 10


def test_fit_svgd_summary(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="svgd",
        num_steps=300,
        num_particles=10,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    summary = result.summary()

    assert isinstance(summary, dict)
    assert len(summary) > 0
    for name, stats in summary.items():
        assert "mean" in stats
        assert "std" in stats


def test_fit_svgd_with_constants(sim_data: dict[str, jax.Array]) -> None:
    """Constants should be auto-bound for SVGD just like VI."""
    result = fit(
        model_with_constant,
        y=sim_data["y"],
        method="svgd",
        num_steps=100,
        num_particles=5,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
        n_features=K,
    )

    preds = result.predict(x=sim_data["x"])
    assert preds.mean.shape == (NUM_OBS,)


def test_fit_svgd_num_epochs(sim_data: dict[str, jax.Array]) -> None:
    """num_epochs should work for SVGD (equivalent to num_steps)."""
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="svgd",
        num_epochs=100,
        num_particles=5,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )
    assert result.params is not None


# --------------------------------------------------------------------------- #
# Quality check: does fit() actually learn?
# --------------------------------------------------------------------------- #


def test_fit_learns_coefficients(sim_data: dict[str, jax.Array]) -> None:
    """Verify that fit() produces predictions that are better than chance."""
    result = fit(
        linear_model,
        y=sim_data["y"],
        batch_size=512,
        num_epochs=50,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    preds = result.predict(x=sim_data["x"], num_samples=200)
    y = sim_data["y"]

    prediction_rmse = float(rmse(preds.mean, y))
    baseline_rmse = float(rmse(jnp.zeros_like(y), y))

    # Model should do meaningfully better than predicting zero
    assert prediction_rmse < baseline_rmse * 0.5


def test_fit_svgd_learns(sim_data: dict[str, jax.Array]) -> None:
    """Verify that SVGD produces predictions better than chance."""
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="svgd",
        num_steps=2000,
        num_particles=20,
        lr=0.05,
        seed=0,
        x=sim_data["x"],
    )

    preds = result.predict(x=sim_data["x"], num_samples=50)
    y = sim_data["y"]

    prediction_rmse = float(rmse(preds.mean, y))
    baseline_rmse = float(rmse(jnp.zeros_like(y), y))

    # SVGD converges slower than VI, so use a looser threshold
    assert prediction_rmse < baseline_rmse * 0.75


# --------------------------------------------------------------------------- #
# Coverage: guard clauses and edge cases
# --------------------------------------------------------------------------- #


def test_predict_unknown_method_raises(sim_data: dict[str, jax.Array]) -> None:
    """predict() raises on an unknown method string."""
    result = fit(linear_model, y=sim_data["y"], num_steps=100, seed=0, x=sim_data["x"])
    result.method = "unknown"
    with pytest.raises(ValueError, match="Unknown method"):
        result.predict(x=sim_data["x"])


def test_summary_unknown_method_raises(sim_data: dict[str, jax.Array]) -> None:
    """summary() raises on an unknown method string."""
    result = fit(linear_model, y=sim_data["y"], num_steps=100, seed=0, x=sim_data["x"])
    result.method = "unknown"
    with pytest.raises(ValueError, match="Unknown method"):
        result.summary(x=sim_data["x"])


def test_summary_vi_missing_guide_raises(sim_data: dict[str, jax.Array]) -> None:
    result = fit(linear_model, y=sim_data["y"], num_steps=100, seed=0, x=sim_data["x"])
    result.guide = None
    with pytest.raises(RuntimeError, match="guide or params"):
        result.summary(x=sim_data["x"])


def test_summary_svgd_missing_params_raises(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="svgd",
        num_steps=200,
        num_particles=10,
        seed=0,
        x=sim_data["x"],
    )
    result.params = None
    with pytest.raises(RuntimeError, match="SVGD results missing params"):
        result.summary(x=sim_data["x"])


def test_summary_mcmc_missing_samples_raises(sim_data: dict[str, jax.Array]) -> None:
    result = fit(
        linear_model,
        y=sim_data["y"],
        method="mcmc",
        num_mcmc_samples=10,
        num_warmup=10,
        seed=0,
        x=sim_data["x"],
    )
    result.posterior_samples = None
    with pytest.raises(RuntimeError, match="MCMC results missing posterior_samples"):
        result.summary(x=sim_data["x"])


def test_fit_svgd_epochs_steps_conflict_raises(sim_data: dict[str, jax.Array]) -> None:
    """SVGD raises if both num_epochs and num_steps are given."""
    with pytest.raises(ValueError, match="exactly one"):
        fit(
            linear_model,
            y=sim_data["y"],
            method="svgd",
            num_steps=100,
            num_epochs=1,
            num_particles=10,
            seed=0,
            x=sim_data["x"],
        )


def test_fit_custom_optimizer(sim_data: dict[str, jax.Array]) -> None:
    """Passing a custom optax optimizer should bypass the default schedule."""
    import optax

    result = fit(
        linear_model,
        y=sim_data["y"],
        num_steps=100,
        optimizer=optax.adam(1e-3),
        seed=0,
        x=sim_data["x"],
    )
    assert result.params is not None


def test_autoreshape_positional_args() -> None:
    """autoreshape should reshape 1D positional array args to (n, 1)."""
    from blayers.sampling import autoreshape

    received = {}

    @autoreshape
    def model(x, y=None):
        received["x_shape"] = x.shape
        received["y_shape"] = y.shape if y is not None else None

    x = jnp.ones((10,))
    y = jnp.ones((10,))
    model(x, y)
    assert received["x_shape"] == (10, 1)
    assert received["y_shape"] == (10, 1)
