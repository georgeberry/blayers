"""Penalty, stationary covariance, forecasting, and Gaussian posterior validity."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import deterministic, handlers
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal

from blayers import AR1Layer, PSplineLayer, fit, gaussian_link
from blayers.splines import bspline_basis, make_knots

KNOTS = make_knots(np.linspace(-1, 1, 101), num_knots=2)


def fixed_layer(layer, values):
    return handlers.block(
        handlers.condition(layer, data=values), hide=list(values)
    )


def test_pspline_exact_difference_penalty_and_null_components():
    k = len(KNOTS) - 4
    d = np.diff(np.eye(k), n=2, axis=0)
    r = d.T @ np.linalg.inv(d @ d.T)
    delta = np.arange((k - 2) * 2).reshape(k - 2, 2) / 10
    trend = np.array([0.5, -0.3])
    beta = r @ delta + np.linspace(-1, 1, k)[:, None] * trend
    np.testing.assert_allclose(d @ beta, delta, atol=1e-12)
    np.testing.assert_allclose(r.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.linspace(-1, 1, k) @ r, 0, atol=1e-12)
    layer = fixed_layer(
        PSplineLayer(),
        {
            "PSplineLayer_f_differences": jnp.asarray(delta),
            "PSplineLayer_f_trend": jnp.asarray(trend),
        },
    )
    x = jnp.array([-2.0, -1.0, -0.4, 0.0, 0.8, 1.0, 2.0])
    actual = handlers.seed(layer, 0)("f", x, KNOTS, units=2)
    b = bspline_basis(jnp.clip(x, -1, 1), KNOTS) - bspline_basis(
        jnp.array([0.0]), KNOTS
    )
    np.testing.assert_allclose(actual, b @ beta, atol=2e-7)
    np.testing.assert_array_equal(actual[3], 0)
    np.testing.assert_array_equal(actual[0], actual[1])
    np.testing.assert_array_equal(actual[-1], actual[-2])
    # The reference and coefficients do not depend on batch composition.
    for i in range(len(x)):
        np.testing.assert_allclose(
            handlers.seed(layer, 0)("f", x[i : i + 1, None], KNOTS, units=2),
            actual[i : i + 1],
            atol=2e-7,
        )


@pytest.mark.parametrize("noncentered", [False, True])
@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.8])
def test_ar1_stationary_covariance(rho, noncentered):
    scale = 0.4
    layer = fixed_layer(
        AR1Layer(noncentered=noncentered),
        {
            "AR1Layer_f_rho": jnp.array([rho]),
            "AR1Layer_f_scale": jnp.array([scale]),
        },
    )
    model = lambda: deterministic("effect", layer("f", jnp.arange(5), 5))
    draws = np.asarray(
        Predictive(model, num_samples=15000, return_sites=["effect"])(
            jax.random.PRNGKey(1)
        )["effect"]
    )[:, :, 0]
    distance = np.abs(np.arange(5)[:, None] - np.arange(5))
    covariance = scale**2 / (1 - rho**2) * rho**distance
    np.testing.assert_allclose(draws.mean(axis=0), 0, atol=0.02)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), covariance, atol=0.013
    )


def test_ar1_recurrence_and_forecast_distribution():
    rho = np.array([0.7, -0.4])
    scale = np.array([0.3, 0.5])
    fixed = {
        "AR1Layer_f_rho": jnp.asarray(rho),
        "AR1Layer_f_scale": jnp.asarray(scale),
    }
    layer = fixed_layer(AR1Layer(noncentered=True), fixed)
    draws = Predictive(
        lambda: deterministic("effect", layer("f", jnp.arange(6), 6, units=2)),
        num_samples=15000,
        return_sites=["effect", "AR1Layer_f_z"],
    )(jax.random.PRNGKey(3))
    states = np.asarray(draws["effect"])
    z = np.asarray(draws["AR1Layer_f_z"])
    np.testing.assert_allclose(
        states[:, 0], scale * z[:, 0] / np.sqrt(1 - rho**2), atol=3e-7
    )
    np.testing.assert_allclose(
        states[:, 1:], rho * states[:, :-1] + scale * z[:, 1:], atol=3e-7
    )
    # h-step innovations retain the correct forecast variance, not just decay.
    for h in [1, 3, 5]:
        residual = states[:, h] - rho**h * states[:, 0]
        expected_var = scale**2 * (1 - rho ** (2 * h)) / (1 - rho**2)
        np.testing.assert_allclose(residual.mean(axis=0), 0, atol=0.015)
        np.testing.assert_allclose(
            residual.var(axis=0), expected_var, rtol=0.035
        )
    fixed["AR1Layer_f_z"] = jnp.asarray(z[0])
    run = handlers.seed(fixed_layer(AR1Layer(noncentered=True), fixed), 0)
    np.testing.assert_allclose(
        run("f", jnp.array([[5], [0], [5], [2]]), 6, units=2),
        states[0, [5, 0, 5, 2]],
        atol=3e-7,
    )


@pytest.mark.parametrize("scale", [0.1, 0.8])
def test_pspline_prior_covariance_and_roughness(scale):
    trend_scale = 0.6
    k = len(KNOTS) - 4
    d = np.diff(np.eye(k), n=2, axis=0)
    r = d.T @ np.linalg.inv(d @ d.T)
    t = np.linspace(-1, 1, k)
    x = jnp.linspace(-1, 1, 7)
    b = np.asarray(
        bspline_basis(x, KNOTS) - bspline_basis(jnp.array([0.0]), KNOTS)
    )
    covariance = (
        b @ (scale**2 * r @ r.T + trend_scale**2 * np.outer(t, t)) @ b.T
    )
    layer = fixed_layer(
        PSplineLayer(trend_scale=trend_scale),
        {"PSplineLayer_f_scale": jnp.array([scale])},
    )
    draws = Predictive(
        lambda: deterministic("effect", layer("f", x, KNOTS)),
        num_samples=12000,
        return_sites=["effect", "PSplineLayer_f_differences"],
    )(jax.random.PRNGKey(8))
    empirical = np.cov(np.asarray(draws["effect"])[:, :, 0], rowvar=False)
    variance = np.diag(covariance)
    standard_error = np.sqrt(
        (covariance**2 + np.outer(variance, variance)) / 11999
    )
    assert np.all(np.abs(empirical - covariance) < 5 * standard_error + 1e-6)
    # Expected sum of squared second differences is (K-2)*scale**2.
    roughness = (
        np.square(draws["PSplineLayer_f_differences"]).sum(axis=(1, 2)).mean()
    )
    np.testing.assert_allclose(roughness, (k - 2) * scale**2, rtol=0.025)


@pytest.mark.parametrize("kind", ["pspline", "ar1", "ar1_noncentered"])
@pytest.mark.parametrize(
    "method,batch_size", [("vi", None), ("vi", 3), ("mcmc", None)]
)
def test_conditional_posterior_matches_gaussian_solution(
    kind, method, batch_size
):
    noise = 0.35
    if kind == "pspline":
        x_all = jnp.array([-1.0, -0.8, -0.5, -0.1, 0.2, 0.5, 0.7, 1.0])
        k = len(KNOTS) - 4
        d = np.diff(np.eye(k), n=2, axis=0)
        r = d.T @ np.linalg.inv(d @ d.T)
        t = np.linspace(-1, 1, k)
        b = np.asarray(
            bspline_basis(x_all, KNOTS) - bspline_basis(jnp.array([0.0]), KNOTS)
        )
        covariance = b @ (0.4**2 * r @ r.T + np.outer(t, t)) @ b.T
        layer = fixed_layer(
            PSplineLayer(), {"PSplineLayer_f_scale": jnp.array([0.4])}
        )
        effect = lambda x: layer("f", x, KNOTS)
    else:
        x_all = jnp.arange(8)
        rho, scale = 0.7, 0.4
        covariance = (
            scale**2
            / (1 - rho**2)
            * rho ** np.abs(np.arange(8)[:, None] - np.arange(8))
        )
        layer = fixed_layer(
            AR1Layer(noncentered=kind == "ar1_noncentered"),
            {
                "AR1Layer_f_scale": jnp.array([scale]),
                "AR1Layer_f_rho": jnp.array([rho]),
            },
        )
        effect = lambda x: layer("f", x, num_categories=8)
    # Two reserved/unobserved positions also test posterior extrapolation.
    observed = np.array([0, 1, 2, 3, 4, 5])
    x, y = x_all[observed], jnp.array([-0.6, -0.2, 0.1, 0.3, 0.7, 0.4])
    a = covariance[np.ix_(observed, observed)] + noise**2 * np.eye(
        len(observed)
    )
    cross = covariance[:, observed]
    expected_mean = cross @ np.linalg.solve(a, y)
    expected_cov = (
        covariance
        - cross @ np.linalg.solve(a, cross.T)
        + noise**2 * np.eye(len(x_all))
    )

    def model(x, y=None):
        return gaussian_link(effect(x), y, scale=noise)

    result = fit(
        model,
        x=x,
        y=y,
        method=method,
        batch_size=batch_size,
        guide=AutoMultivariateNormal,
        num_steps=6500,
        lr=0.015,
        num_warmup=500,
        num_mcmc_samples=2500,
        seed=5,
    )
    draws = np.asarray(
        result.predict(x=x_all, num_samples=12000).samples
    ).reshape(-1, len(x_all))
    np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.055)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), expected_cov, atol=0.035, rtol=0.12
    )


@pytest.mark.parametrize(
    "x,error",
    [
        ([-1], ValueError),
        ([3], ValueError),
        ([0.5], TypeError),
        ([[0, 1]], ValueError),
    ],
)
def test_ar1_rejects_bad_indices(x, error):
    with pytest.raises(error):
        handlers.seed(AR1Layer(), 0)("f", jnp.array(x), 3)


def test_jitted_ar1_rejects_invalid_indices_with_nan():
    values = jax.jit(lambda x: handlers.seed(AR1Layer(), 0)("f", x, 3))(
        jnp.array([-1, 0, 3])
    )
    assert jnp.isnan(values[[0, 2], :]).all()
    assert jnp.isfinite(values[1]).all()


def test_single_time_slot():
    assert handlers.seed(AR1Layer(), 0)("f", jnp.array([0]), 1).shape == (1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": 0},
        {"degree": 1.5},
        {"trend_scale": 0},
        {"trend_scale": float("nan")},
    ],
)
def test_invalid_pspline_configuration(kwargs):
    with pytest.raises(ValueError):
        PSplineLayer(**kwargs)


@pytest.mark.parametrize(
    "knots",
    [
        jnp.zeros(8),
        KNOTS[::-1],
        jnp.array([-1.0] * 3 + [0.0, 0.0] + [1.0] * 4),
        KNOTS.at[4].set(jnp.nan),
    ],
)
def test_invalid_knots(knots):
    with pytest.raises(ValueError):
        handlers.seed(PSplineLayer(), 0)("f", jnp.array([0.0]), knots)


def test_pspline_invalid_reference_and_input():
    with pytest.raises(ValueError):
        handlers.seed(PSplineLayer(), 0)(
            "f", jnp.array([0.0]), KNOTS, reference=2.0
        )
    with pytest.raises(ValueError):
        handlers.seed(PSplineLayer(), 0)("f", jnp.array([jnp.nan]), KNOTS)
    out = jax.jit(
        lambda knots: handlers.seed(PSplineLayer(), 0)(
            "f", jnp.array([0.0]), knots
        )
    )(jnp.zeros_like(KNOTS))
    assert jnp.isnan(out).all()


@pytest.mark.parametrize("layer", [AR1Layer, PSplineLayer])
def test_custom_scale_prior_and_invalid_event_or_discrete_prior(layer):
    configured = layer(scale_dist=dist.Exponential, scale_kwargs={"rate": 2.0})
    assert configured.scale_dist(**configured.scale_kwargs).mean == 0.5
    with pytest.raises(ValueError):
        layer(scale_dist=dist.Bernoulli, scale_kwargs={"probs": 0.5})


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_ar1_learns_persistence_and_innovation_scale(method):
    rng = np.random.default_rng(42)
    rho = np.array([0.75, -0.65])
    sigma = np.array([0.3, 0.5])
    n = 160
    states = np.empty((n, 2))
    states[0] = rng.normal(size=2) * sigma / np.sqrt(1 - rho**2)
    for t in range(1, n):
        states[t] = rho * states[t - 1] + sigma * rng.normal(size=2)
    x = jnp.arange(n)
    y = jnp.asarray(states + 0.1 * rng.normal(size=states.shape))

    def model(x, y=None):
        return gaussian_link(AR1Layer()("f", x, n + 3, units=2), y, scale=0.1)

    result = fit(
        model,
        x=x,
        y=y,
        method=method,
        num_steps=12000,
        lr=0.015,
        num_warmup=600,
        num_mcmc_samples=1000,
        seed=6,
    )
    stats = result.summary(x=x, num_samples=4000)
    np.testing.assert_allclose(stats["AR1Layer_f_rho"]["mean"], rho, atol=0.16)
    np.testing.assert_allclose(
        stats["AR1Layer_f_scale"]["mean"], sigma, rtol=0.25
    )
    # Forecasting through reserved slots works after learning hyperparameters.
    assert jnp.isfinite(result.predict(x=jnp.arange(n, n + 3)).samples).all()


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_pspline_learns_smoothness_per_output(method):
    x = jnp.linspace(-1, 1, 160)
    knots = make_knots(x, num_knots=6)
    b = bspline_basis(x, knots) - bspline_basis(jnp.array([0.0]), knots)
    # One coefficient-linear (unpenalized) trend and one nonlinear curve.
    truth = jnp.column_stack(
        [b @ jnp.linspace(-0.5, 0.5, b.shape[1]), jnp.sin(jnp.pi * x)]
    )
    y = truth + 0.12 * jax.random.normal(jax.random.PRNGKey(14), truth.shape)

    def model(x, y=None):
        return gaussian_link(
            PSplineLayer()("f", x, knots, units=2), y, scale=0.12
        )

    result = fit(
        model,
        x=x,
        y=y,
        method=method,
        num_steps=12000,
        lr=0.015,
        num_warmup=600,
        num_mcmc_samples=1000,
        seed=6,
    )
    learned = result.summary(x=x, num_samples=4000)["PSplineLayer_f_scale"][
        "mean"
    ]
    assert learned[0] < 0.35 * learned[1]
    prediction = result.predict(x=x, num_samples=3000).mean
    assert jnp.sqrt(jnp.mean((prediction - truth) ** 2)) < 0.08


def test_ar1_direct_state_density_matches_stationary_markov_density():
    rho, scale = jnp.array([0.7, -0.4]), jnp.array([0.3, 0.5])
    states = jnp.array([[0.2, -0.1], [-0.1, 0.5], [0.3, -0.3]])
    model = handlers.condition(
        lambda: AR1Layer()("f", jnp.arange(3), 3, units=2),
        data={"AR1Layer_f_rho": rho, "AR1Layer_f_scale": scale},
    )
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    prior = tr["AR1Layer_f_theta"]["fn"]
    expected = (
        dist.Normal(0, scale / jnp.sqrt(1 - rho**2)).log_prob(states[0]).sum()
    )
    expected += dist.Normal(rho * states[:-1], scale).log_prob(states[1:]).sum()
    np.testing.assert_allclose(prior.log_prob(states), expected, atol=1e-6)
    assert prior.batch_shape == () and prior.event_shape == (3, 2)
