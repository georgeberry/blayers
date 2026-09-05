"""Exact spike-and-slab validity, including an enumerated Gaussian reference."""

from itertools import product

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, substitute
from numpyro.infer import DiscreteHMCGibbs, Predictive
from scipy.special import betaln, logsumexp
from scipy.stats import multivariate_normal

from blayers import (
    SpikeAndSlabLayer,
    fit,
    gaussian_link,
    model_to_latex,
    sample_prior,
)


@pytest.mark.parametrize("units", [1, 3])
def test_exact_spike_mass_and_configurable_slab(units):
    layer = SpikeAndSlabLayer(
        inclusion_prob=0.2,
        coef_dist=dist.StudentT,
        coef_kwargs={"df": 5.0, "loc": 2.0, "scale": 0.7},
    )

    def model(x):
        return layer("b", x, units=units)

    samples = sample_prior(model, x=jnp.eye(4), num_samples=4000)
    z, slab, beta = [
        samples[f"SpikeAndSlabLayer_b_{name}"] for name in ("z", "slab", "beta")
    ]
    assert z.shape == (4000, 4, units)
    assert np.all((z == 0) | (z == 1))
    np.testing.assert_array_equal(beta, z * slab)
    assert np.all(np.asarray(beta)[np.asarray(z) == 0] == 0)
    np.testing.assert_allclose((beta == 0).mean(axis=0), 0.8, atol=0.025)
    np.testing.assert_allclose(slab.mean(axis=0), 2.0, atol=0.065)
    assert "SpikeAndSlabLayer_b_pi" not in samples


def test_shared_beta_prior_gives_beta_binomial_model_size():
    d, alpha, beta = 5, 2.0, 6.0

    def model(x):
        return SpikeAndSlabLayer(alpha=alpha, beta=beta)("b", x, units=2)

    samples = sample_prior(model, x=jnp.eye(d), num_samples=6000)
    assert samples["SpikeAndSlabLayer_b_pi"].shape == (6000, 2)
    z = np.asarray(samples["SpikeAndSlabLayer_b_z"])
    counts = z.sum(axis=1)
    expected_mean = d * alpha / (alpha + beta)
    expected_variance = (
        d
        * alpha
        * beta
        * (alpha + beta + d)
        / ((alpha + beta) ** 2 * (alpha + beta + 1))
    )
    np.testing.assert_allclose(counts.mean(axis=0), expected_mean, atol=0.06)
    np.testing.assert_allclose(counts.var(axis=0), expected_variance, atol=0.12)


def test_conditioned_indicators_select_exact_columns():
    x = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    layer = SpikeAndSlabLayer(inclusion_prob=0.3)
    z = jnp.array([[1, 0], [0, 1], [1, 0]])
    slab = jnp.array([[2.0, 7.0], [6.0, -3.0], [0.5, 9.0]])
    model = substitute(
        lambda: layer("b", x, units=2),
        data={
            "SpikeAndSlabLayer_b_z": z,
            "SpikeAndSlabLayer_b_slab": slab,
        },
    )
    np.testing.assert_allclose(seed(model, 0)(), x @ (z * slab))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"beta": -1},
        {"alpha": float("nan")},
        {"beta": float("inf")},
        {"inclusion_prob": 0},
        {"inclusion_prob": 1},
        {"inclusion_prob": float("nan")},
        {"coef_dist": dist.Bernoulli, "coef_kwargs": {"probs": 0.5}},
    ],
)
def test_invalid_prior_rejected(kwargs):
    with pytest.raises(ValueError):
        SpikeAndSlabLayer(**kwargs)


@pytest.mark.parametrize("method", ["vi", "svgd"])
@pytest.mark.parametrize("batch_size", [None, 3])
def test_continuous_inference_helpers_reject_binary_indicators(
    method, batch_size
):
    def model(x, y=None):
        return gaussian_link(SpikeAndSlabLayer()("b", x), y)

    with pytest.raises(ValueError, match="Use method='mcmc'"):
        fit(
            model,
            x=jnp.ones((5, 2)),
            y=jnp.ones(5),
            method=method,
            batch_size=batch_size,
            num_steps=10,
        )


def test_fixed_probability_latex_is_exact():
    def model(x, y=None):
        return gaussian_link(SpikeAndSlabLayer(inclusion_prob=0.2)("b", x), y)

    tex = model_to_latex(model, x=jnp.ones((3, 2)))
    assert r"\mathrm{Bernoulli}(0.2)" in tex
    assert r"\mathrm{Beta}" not in tex
    assert r"\beta_{\mathrm{b},j} &= z_{\mathrm{b},j}\,b_{\mathrm{b},j}" in tex


def exact_gaussian_posterior(x, y, inclusion_prob, alpha=1.0, beta=3.0):
    """Enumerate every model and integrate Gaussian slab coefficients exactly."""
    n, d = x.shape
    indicators = np.asarray(list(product([0, 1], repeat=d)))
    log_weights, coefficient_means = [], []
    for z in indicators:
        # Unit Gaussian slab and unit observation noise.
        design = x * z
        covariance = np.eye(n) + design @ design.T
        k = z.sum()
        log_prior = (
            betaln(alpha + k, beta + d - k) - betaln(alpha, beta)
            if inclusion_prob is None
            else k * np.log(inclusion_prob)
            + (d - k) * np.log1p(-inclusion_prob)
        )
        log_weights.append(
            log_prior + multivariate_normal.logpdf(y, cov=covariance)
        )
        coefficient_means.append(design.T @ np.linalg.solve(covariance, y))
    weights = np.exp(log_weights - logsumexp(log_weights))
    return weights @ indicators, weights @ np.asarray(coefficient_means)


@pytest.fixture(scope="module", params=[None, 0.25])
def fitted_reference(request):
    inclusion_prob = request.param
    x = np.tile(
        np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]), (6, 1)
    )
    y = x @ np.array([0.55, 0.15])
    layer = SpikeAndSlabLayer(
        alpha=1.0, beta=3.0, inclusion_prob=inclusion_prob
    )

    def model(x, y=None):
        return gaussian_link(layer("b", x), y, scale=1.0)

    result = fit(
        model,
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        method="mcmc",
        num_warmup=500,
        num_mcmc_samples=3000,
        seed=42,
    )
    return result, x, y, exact_gaussian_posterior(x, y, inclusion_prob)


def test_mcmc_inclusion_and_coefficient_means_match_exact_posterior(
    fitted_reference,
):
    result, x, y, (pip, mean) = fitted_reference
    assert isinstance(result.mcmc.sampler, DiscreteHMCGibbs)
    summary = result.summary()
    np.testing.assert_allclose(
        summary["SpikeAndSlabLayer_b_z"]["mean"].ravel(), pip, atol=0.06
    )
    np.testing.assert_allclose(
        summary["SpikeAndSlabLayer_b_beta"]["mean"].ravel(), mean, atol=0.05
    )
    samples = result.posterior_samples
    np.testing.assert_array_equal(
        samples["SpikeAndSlabLayer_b_beta"],
        samples["SpikeAndSlabLayer_b_z"] * samples["SpikeAndSlabLayer_b_slab"],
    )


def test_predictions_use_effective_coefficients(fitted_reference):
    result, x, y, (pip, mean) = fitted_reference
    x_new = jnp.asarray(x[:3] * 0.7)
    samples = result.posterior_samples
    # The original model must retain both slab and indicator posterior draws.
    predictive = Predictive(
        result.model_fn,
        posterior_samples=samples,
        return_sites=["SpikeAndSlabLayer_b_beta"],
    )
    predicted_beta = predictive(jnp.array([0, 1], dtype=jnp.uint32), x=x_new)[
        "SpikeAndSlabLayer_b_beta"
    ]
    np.testing.assert_array_equal(
        predicted_beta, samples["SpikeAndSlabLayer_b_beta"]
    )
    np.testing.assert_allclose(
        result.predict(x=x_new).mean, x_new @ mean, atol=0.09
    )


def test_arviz_log_likelihood_matches_posterior_draws(fitted_reference):
    pytest.importorskip("arviz")
    result, x, y, _ = fitted_reference
    idata = result.to_arviz()
    assert "diverging" in idata["sample_stats"]
    assert idata["sample_stats"]["diverging"].shape == (1, 3000)
    actual = np.asarray(idata["log_likelihood"]["obs"]).reshape(3000, len(y))
    beta = np.asarray(result.posterior_samples["SpikeAndSlabLayer_b_beta"])[
        ..., 0
    ]
    residual = y[None, :] - beta @ x.T
    expected = -0.5 * (np.log(2 * np.pi) + residual**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
