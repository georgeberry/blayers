"""Random slope algebra, prior semantics, and conjugate posterior validity."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from blayers import (
    RandomSlopesLayer,
    fit,
    gaussian_link,
    model_to_latex,
    sample_prior,
)


@pytest.mark.parametrize("units", [1, 2])
def test_group_lookup_and_predictor_contraction(units):
    x = jnp.array([[1.0, 2.0], [-1.0, 0.5], [0.3, -0.2]])
    groups = jnp.array([2, 0, 2])
    beta = jnp.arange(3 * 2 * units, dtype=float).reshape(3, 2, units) / 10
    layer = RandomSlopesLayer()
    model = handlers.substitute(
        lambda: layer("b", x, groups, 3, units=units),
        data={"RandomSlopesLayer_b_beta": beta},
    )
    actual = handlers.seed(model, 0)()
    expected = np.stack(
        [np.asarray(x[i]) @ np.asarray(beta[g]) for i, g in enumerate(groups)]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_prior_scales_are_per_predictor_and_shared_across_groups():
    layer = RandomSlopesLayer(
        scale_dist=dist.Exponential, scale_kwargs={"rate": 2.0}
    )

    def model(x, groups):
        return layer("b", x, groups, 4, units=2)

    tr = handlers.trace(handlers.seed(model, 0)).get_trace(
        x=jnp.ones((3, 2)), groups=jnp.array([0, 1, 0])
    )
    scale = tr["RandomSlopesLayer_b_scale"]
    beta = tr["RandomSlopesLayer_b_beta"]
    assert scale["value"].shape == (2, 2)
    assert beta["value"].shape == (4, 2, 2)
    assert scale["fn"].base_dist.rate == 2.0
    standardized = beta["fn"].log_prob(beta["value"])
    expected = dist.Normal(0.0, scale["value"]).log_prob(beta["value"])
    np.testing.assert_allclose(standardized, expected)


def test_single_row_single_predictor_and_column_groups():
    out = handlers.seed(RandomSlopesLayer(), 0)(
        "b", jnp.array([2.0]), jnp.array([[1]]), 3
    )
    assert out.shape == (1, 1)


def test_prior_independence_and_scale_moments():
    def model(x, groups):
        return RandomSlopesLayer()("b", x, groups, 3)

    draws = sample_prior(
        model, x=jnp.ones((2, 2)), groups=jnp.array([0, 1]), num_samples=6000
    )
    scale = draws["RandomSlopesLayer_b_scale"]
    beta = draws["RandomSlopesLayer_b_beta"]
    z = np.asarray(beta / scale[:, None, :, :]).reshape(6000, -1)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(np.cov(z, rowvar=False), np.eye(6), atol=0.07)
    np.testing.assert_allclose(
        np.asarray(scale**2).mean(axis=0), 1.0, atol=0.07
    )


@pytest.mark.parametrize(
    "groups,error",
    [
        (jnp.array([-1, 0]), ValueError),
        (jnp.array([0, 3]), ValueError),
        (jnp.array([0.0, 1.0]), TypeError),
        (jnp.array([0]), ValueError),
        (jnp.zeros((2, 2), dtype=int), ValueError),
    ],
)
def test_invalid_groups_rejected(groups, error):
    with pytest.raises(error):
        handlers.seed(RandomSlopesLayer(), 0)("b", jnp.ones((2, 1)), groups, 3)


def test_jitted_invalid_groups_do_not_wrap_or_clip():
    run = jax.jit(
        lambda g: handlers.seed(RandomSlopesLayer(), 0)(
            "b", jnp.ones((3, 1)), g, 2
        )
    )
    values = run(jnp.array([-1, 0, 2]))
    assert jnp.isnan(values[0]).all() and jnp.isnan(values[2]).all()
    assert jnp.isfinite(values[1]).all()


def test_invalid_prior_kwargs_rejected_at_construction():
    with pytest.raises(TypeError, match="Invalid distribution kwargs"):
        RandomSlopesLayer(scale_kwargs={"rate": 1.0})


def test_latex_includes_scale_and_slope_hierarchy():
    def model(x, groups, y=None):
        return gaussian_link(RandomSlopesLayer()("b", x, groups, 3), y)

    tex = model_to_latex(model, x=jnp.ones((2, 2)), groups=jnp.array([0, 1]))
    assert r"\lambda_{\mathrm{b}} &\sim \mathrm{HalfNormal}(1)" in tex
    assert (
        r"\beta_{\mathrm{b}} &\sim \mathrm{Normal}(0, \lambda_{\mathrm{b}})"
        in tex
    )


@pytest.mark.parametrize(
    "method,batch_size", [("vi", None), ("vi", 3), ("mcmc", None)]
)
def test_posterior_matches_conjugate_shrinkage_and_unseen_group_prior(
    method, batch_size
):
    # Orthogonal predictors give a diagonal conditional Gaussian posterior,
    # which the default diagonal VI guide can represent exactly.
    x = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )
    groups = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = x @ jnp.array([0.8, -0.6])
    tau = jnp.array([[0.3], [0.7]])
    layer = RandomSlopesLayer()
    fixed_scale_layer = handlers.block(
        handlers.condition(
            lambda x, groups: layer("b", x, groups, 3),
            data={"RandomSlopesLayer_b_scale": tau},
        ),
        hide=["RandomSlopesLayer_b_scale"],
    )

    def model(x, groups, y=None):
        return gaussian_link(fixed_scale_layer(x, groups), y, scale=0.5)

    result = fit(
        model,
        x=x,
        groups=groups,
        y=y,
        method=method,
        batch_size=batch_size,
        num_steps=4000,
        lr=0.02,
        num_warmup=400,
        num_mcmc_samples=1600,
        seed=4,
    )
    summary = result.summary(x=x, groups=groups, num_samples=8000)
    stats = summary["RandomSlopesLayer_b_beta"]
    expected_mean, expected_std = [], []
    for g in range(3):
        xg, yg = (
            np.asarray(x)[np.asarray(groups) == g],
            np.asarray(y)[np.asarray(groups) == g],
        )
        covariance = np.linalg.inv(
            xg.T @ xg / 0.25 + np.diag(1 / np.asarray(tau[:, 0]) ** 2)
        )
        expected_mean.append(covariance @ (xg.T @ yg / 0.25))
        expected_std.append(np.sqrt(np.diag(covariance)))
    np.testing.assert_allclose(
        np.asarray(stats["mean"])[..., 0], expected_mean, atol=0.055
    )
    np.testing.assert_allclose(
        np.asarray(stats["std"])[..., 0], expected_std, rtol=0.17, atol=0.015
    )
    # Both groups see the same standalone slopes; the weaker design pools more.
    assert abs(stats["mean"][0, 0, 0]) < abs(stats["mean"][1, 0, 0])
    assert result.predict(
        x=x[:1], groups=jnp.array([2]), num_samples=100
    ).samples.shape[-2:] == (1, 1)


@pytest.mark.parametrize(
    "method,noncentered,steps",
    [("vi", True, 20000), ("vi", False, 3500), ("mcmc", True, 3500)],
)
def test_learns_distinct_between_group_scales(method, noncentered, steps):
    # Many groups distinguish a nearly common slope from a highly varying one.
    n_groups = 40
    x = jnp.tile(
        jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        (n_groups, 1),
    )
    groups = jnp.repeat(jnp.arange(n_groups), 4)
    beta = jax.random.normal(jax.random.PRNGKey(10), (n_groups, 2)) * jnp.array(
        [0.1, 0.9]
    )
    y = (x * beta[groups]).sum(axis=1) + 0.2 * jax.random.normal(
        jax.random.PRNGKey(11), (len(x),)
    )

    def model(x, groups, y=None):
        return gaussian_link(
            RandomSlopesLayer()("b", x, groups, n_groups), y, scale=0.2
        )

    result = fit(
        model,
        x=x,
        groups=groups,
        y=y,
        method=method,
        num_steps=steps,
        autoreparam_model=noncentered,
        lr=0.02,
        num_warmup=400,
        num_mcmc_samples=800,
        seed=2,
    )
    learned = result.summary(x=x, groups=groups)["RandomSlopesLayer_b_scale"][
        "mean"
    ][:, 0]
    assert learned[0] < 0.3
    assert learned[1] > 0.5
    np.testing.assert_allclose(learned, np.asarray(beta).std(axis=0), atol=0.2)
