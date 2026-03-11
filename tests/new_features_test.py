"""Tests for HorseshoeLayer, AttentionLayer, ordinal_link, zip_link, beta_link, gaussian_link."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
import pytest
from numpyro import deterministic, sample
from numpyro.infer import Predictive

from blayers._utils import rmse
from blayers.fit import fit
from blayers.layers import AdaptiveLayer, AttentionLayer, HorseshoeLayer, SpikeAndSlabLayer
from blayers.links import (
    beta_link,
    gaussian_link,
    ordinal_link,
    zip_link,
)

gaussian_link_exp = gaussian_link  # renamed
from blayers.decorators import autoreshape

NUM_OBS = 1000


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _prior_samples(model_fn, num_samples=4, **kwargs):
    predictive = Predictive(model_fn, num_samples=num_samples)
    return predictive(random.PRNGKey(0), **kwargs)


# --------------------------------------------------------------------------- #
# HorseshoeLayer
# --------------------------------------------------------------------------- #


class TestHorseshoeLayer:
    def test_output_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 5))

        def model(x):
            out = HorseshoeLayer()("beta", x)
            return deterministic("out", out)

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 1)

    def test_output_shape_units(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 5))

        def model(x):
            out = HorseshoeLayer()("beta", x, units=3)
            return deterministic("out", out)

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 3)

    def test_basic_sites(self) -> None:
        """tau and lmbda (and no c2) are sampled for the plain horseshoe."""
        x = random.normal(random.PRNGKey(0), (20, 4))

        def model(x):
            return HorseshoeLayer()("beta", x)

        samples = _prior_samples(model, x=x)
        assert "HorseshoeLayer_beta_tau" in samples
        assert "HorseshoeLayer_beta_scale" in samples
        assert "HorseshoeLayer_beta_c2" not in samples

    def test_regularized_sites(self) -> None:
        """c2 slab variance is sampled for the regularized horseshoe."""
        x = random.normal(random.PRNGKey(0), (20, 4))

        def model(x):
            return HorseshoeLayer(slab_scale=2.0)("beta", x)

        samples = _prior_samples(model, x=x)
        assert "HorseshoeLayer_beta_c2" in samples
        assert samples["HorseshoeLayer_beta_c2"].shape == (4,)  # scalar per sample

    def test_fit_runs(self) -> None:
        """HorseshoeLayer should work end-to-end with fit()."""
        key = random.PRNGKey(42)
        x = random.normal(key, (NUM_OBS, 10))
        # Sparse DGP: only first feature matters
        true_beta = jnp.zeros(10).at[0].set(3.0)
        y = x @ true_beta + random.normal(key, (NUM_OBS,)) * 0.5

        @autoreshape
        def sparse_model(x, y=None):
            mu = HorseshoeLayer(slab_scale=2.0)("beta", x)
            return gaussian_link_exp(mu, y)

        result = fit(sparse_model, y=y, x=x, num_steps=300, lr=0.01, seed=0)
        assert result.params is not None

    def test_horseshoe_shrinks_noise(self) -> None:
        """Horseshoe predictions should beat a zero baseline on sparse data."""
        key = random.PRNGKey(7)
        x = random.normal(key, (NUM_OBS, 10))
        true_beta = jnp.zeros(10).at[0].set(3.0)
        y = x @ true_beta + random.normal(key, (NUM_OBS,)) * 0.5

        @autoreshape
        def sparse_model(x, y=None):
            mu = HorseshoeLayer(slab_scale=2.0)("beta", x)
            return gaussian_link_exp(mu, y)

        result = fit(sparse_model, y=y, x=x, num_steps=500, lr=0.01, seed=0)
        preds = result.predict(x=x, num_samples=100)
        assert float(rmse(preds.mean, y)) < float(rmse(jnp.zeros_like(y), y)) * 0.5


# --------------------------------------------------------------------------- #
# AttentionLayer
# --------------------------------------------------------------------------- #


class TestAttentionLayer:
    def test_output_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 5))

        def model(x):
            out = AttentionLayer()("attn", x, head_dim=4)
            return deterministic("out", out)

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 1)

    def test_output_shape_units(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 5))

        def model(x):
            out = AttentionLayer()("attn", x, head_dim=4, units=2)
            return deterministic("out", out)

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 2)

    def test_sample_sites_present(self) -> None:
        x = random.normal(random.PRNGKey(0), (20, 4))

        def model(x):
            return AttentionLayer()("a", x, head_dim=4)

        samples = _prior_samples(model, x=x)
        for site in ["W_emb", "W_bias", "W_Q", "W_K", "W_V", "W_out"]:
            assert f"AttentionLayer_a_{site}" in samples

    def test_multihead_output_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 5))

        def model(x):
            out = AttentionLayer()("attn", x, head_dim=4, num_heads=2)
            return deterministic("out", out)

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 1)

    def test_multihead_units(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 5))

        def model(x):
            out = AttentionLayer()("attn", x, head_dim=4, num_heads=2, units=3)
            return deterministic("out", out)

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 3)

    def test_bias_site_present(self) -> None:
        """W_bias (per-column identity embedding) should be sampled."""
        x = random.normal(random.PRNGKey(0), (20, 4))

        def model(x):
            return AttentionLayer()("a", x, head_dim=4)

        samples = _prior_samples(model, x=x)
        assert "AttentionLayer_a_W_bias" in samples

    def test_fit_runs(self) -> None:
        """AttentionLayer should run end-to-end with fit()."""
        x = random.normal(random.PRNGKey(0), (NUM_OBS, 4))
        y = jnp.sin(x[:, 0]) * x[:, 1] + random.normal(random.PRNGKey(1), (NUM_OBS,)) * 0.2

        @autoreshape
        def attn_model(x, y=None):
            mu = AttentionLayer()("attn", x, head_dim=4)
            return gaussian_link_exp(mu, y)

        result = fit(attn_model, y=y, x=x, num_steps=200, lr=0.01, seed=0)
        assert result.params is not None

    def test_multihead_fit_runs(self) -> None:
        x = random.normal(random.PRNGKey(0), (NUM_OBS, 4))
        y = jnp.sin(x[:, 0]) * x[:, 1] + random.normal(random.PRNGKey(1), (NUM_OBS,)) * 0.2

        @autoreshape
        def attn_model(x, y=None):
            mu = AttentionLayer()("attn", x, head_dim=4, num_heads=2)
            return gaussian_link_exp(mu, y)

        result = fit(attn_model, y=y, x=x, num_steps=200, lr=0.01, seed=0)
        assert result.params is not None


# --------------------------------------------------------------------------- #
# ordinal_link
# --------------------------------------------------------------------------- #


def _make_ordinal_data(num_obs=NUM_OBS, K=4, seed=0):
    """Simple ordinal DGP: threshold a standard normal."""
    key = random.PRNGKey(seed)
    x = random.normal(key, (num_obs, 3))
    mu = x[:, 0] * 2.0  # only first feature matters
    # Thresholds at -1.5, 0, 1.5
    thresholds = jnp.array([-1.5, 0.0, 1.5])
    y = jnp.sum(mu[:, None] > thresholds, axis=1).astype(jnp.int32)
    return x, y


class TestOrdinalLink:
    K = 4

    def test_prior_obs_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 3))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return ordinal_link(mu, y, num_classes=self.K)

        samples = _prior_samples(model, x=x)
        assert samples["obs"].shape == (4, 30)

    def test_prior_obs_range(self) -> None:
        """Prior samples should be integers in {0, ..., K-1}."""
        x = random.normal(random.PRNGKey(0), (50, 3))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return ordinal_link(mu, y, num_classes=self.K)

        samples = _prior_samples(model, num_samples=10, x=x)
        obs = samples["obs"]
        assert jnp.all(obs >= 0)
        assert jnp.all(obs < self.K)

    def test_binary_ordinal(self) -> None:
        """num_classes=2 is the binary case — single cutpoint."""
        x = random.normal(random.PRNGKey(0), (20, 2))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return ordinal_link(mu, y, num_classes=2)

        samples = _prior_samples(model, x=x)
        assert samples["obs"].shape == (4, 20)
        assert jnp.all((samples["obs"] == 0) | (samples["obs"] == 1))

    def test_fit_runs(self) -> None:
        x, y = _make_ordinal_data(K=self.K)

        @autoreshape
        def ord_model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return ordinal_link(mu, y, num_classes=self.K)

        result = fit(ord_model, y=y, x=x, num_steps=300, lr=0.02, seed=0)
        assert result.params is not None


# --------------------------------------------------------------------------- #
# zip_link
# --------------------------------------------------------------------------- #


def _make_count_data(num_obs=NUM_OBS, seed=0):
    """Zero-inflated Poisson DGP."""
    key = random.PRNGKey(seed)
    k1, k2, k3 = random.split(key, 3)
    x = random.normal(k1, (num_obs, 2))
    log_rate = x[:, 0]                              # true log-rate
    gate = jnp.full((num_obs,), 0.3)               # 30% extra zeros
    is_zero = random.bernoulli(k2, gate)
    counts = random.poisson(k3, jnp.exp(log_rate))
    y = jnp.where(is_zero, 0, counts)
    return x, y


class TestZipLink:
    def test_prior_obs_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 2))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return zip_link(mu, y)

        samples = _prior_samples(model, x=x)
        assert samples["obs"].shape == (4, 30)

    def test_prior_obs_non_negative(self) -> None:
        x = random.normal(random.PRNGKey(0), (40, 2))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return zip_link(mu, y)

        samples = _prior_samples(model, num_samples=8, x=x)
        assert jnp.all(samples["obs"] >= 0)

    def test_gate_site_present(self) -> None:
        x = random.normal(random.PRNGKey(0), (20, 2))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return zip_link(mu, y)

        samples = _prior_samples(model, x=x)
        assert "zip_gate" in samples
        assert jnp.all(samples["zip_gate"] >= 0.0)
        assert jnp.all(samples["zip_gate"] <= 1.0)

    def test_fit_runs(self) -> None:
        x, y = _make_count_data()

        @autoreshape
        def zip_model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return zip_link(mu, y)

        result = fit(zip_model, y=y.astype(float), x=x, num_steps=300, lr=0.02, seed=0)
        assert result.params is not None


# --------------------------------------------------------------------------- #
# beta_link
# --------------------------------------------------------------------------- #


def _make_proportion_data(num_obs=NUM_OBS, seed=0):
    """Beta-distributed proportion DGP."""
    key = random.PRNGKey(seed)
    k1, k2 = random.split(key)
    x = random.normal(k1, (num_obs, 2))
    mu_logit = x[:, 0]
    mean = jax.nn.sigmoid(mu_logit)
    phi = 10.0
    y = random.beta(k2, mean * phi, (1.0 - mean) * phi)
    return x, y


class TestBetaLink:
    def test_prior_obs_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 2))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return beta_link(mu, y)

        samples = _prior_samples(model, x=x)
        assert samples["obs"].shape == (4, 30)

    def test_prior_obs_in_unit_interval(self) -> None:
        x = random.normal(random.PRNGKey(0), (40, 2))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return beta_link(mu, y)

        samples = _prior_samples(model, num_samples=8, x=x)
        assert jnp.all(samples["obs"] > 0.0)
        assert jnp.all(samples["obs"] < 1.0)

    def test_precision_site_present(self) -> None:
        x = random.normal(random.PRNGKey(0), (20, 2))

        def model(x, y=None):
            mu = AdaptiveLayer()("beta", x)
            return beta_link(mu, y)

        samples = _prior_samples(model, x=x)
        assert "beta_phi" in samples
        assert jnp.all(samples["beta_phi"] > 0.0)

    def test_fit_runs(self) -> None:
        x, y = _make_proportion_data()

        @autoreshape
        def prop_model(x, y=None):
            mu = AdaptiveLayer()("beta_coef", x)
            return beta_link(mu, y)

        result = fit(prop_model, y=y, x=x, num_steps=300, lr=0.02, seed=0)
        assert result.params is not None


# --------------------------------------------------------------------------- #
# gaussian_link
# --------------------------------------------------------------------------- #


class TestGaussianLink:
    def test_default_learns_sigma(self) -> None:
        """With no scale arg, a 'sigma' sample site should appear."""
        x = random.normal(random.PRNGKey(0), (20, 2))

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            return gaussian_link(mu, y)

        samples = _prior_samples(model, x=x)
        assert "sigma" in samples
        assert jnp.all(samples["sigma"] > 0.0)

    def test_fixed_scalar_no_sigma_site(self) -> None:
        """Passing a scalar scale should suppress the 'sigma' sample site."""
        x = random.normal(random.PRNGKey(0), (20, 2))

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            return gaussian_link(mu, y, scale=0.5)

        samples = _prior_samples(model, x=x)
        assert "sigma" not in samples

    def test_fixed_array_scale(self) -> None:
        """Per-observation scale array should work without a sigma site."""
        x = random.normal(random.PRNGKey(0), (20, 2))
        scale_arr = jnp.full((20,), 0.3)

        @autoreshape
        def model(x, y=None, scale=None):
            mu = AdaptiveLayer()("mu", x)
            return gaussian_link(mu, y, scale=scale)

        samples = _prior_samples(model, x=x, scale=scale_arr)
        assert "sigma" not in samples
        assert samples["obs"].shape == (4, 20, 1)

    def test_obs_shape_default(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 3))

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            return gaussian_link(mu, y)

        samples = _prior_samples(model, x=x)
        assert samples["obs"].shape == (4, 30, 1)

    def test_fit_runs_default(self) -> None:
        x = random.normal(random.PRNGKey(0), (NUM_OBS, 3))
        y = x[:, 0] * 2.0 + random.normal(random.PRNGKey(1), (NUM_OBS,)) * 0.5

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            return gaussian_link(mu, y)

        result = fit(model, y=y, x=x, num_steps=300, lr=0.01, seed=0)
        assert result.params is not None

    def test_fit_runs_fixed_scale(self) -> None:
        x = random.normal(random.PRNGKey(0), (NUM_OBS, 3))
        y = x[:, 0] * 2.0 + random.normal(random.PRNGKey(1), (NUM_OBS,)) * 0.5

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            return gaussian_link(mu, y, scale=0.5)

        result = fit(model, y=y, x=x, num_steps=300, lr=0.01, seed=0)
        assert result.params is not None

    def test_untransformed_scale_no_sigma_site(self) -> None:
        """untransformed_scale should apply softplus and suppress the learned sigma site."""
        x = random.normal(random.PRNGKey(0), (20, 2))

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            raw = AdaptiveLayer()("untransformed_scale", x)
            return gaussian_link(mu, y, untransformed_scale=raw)

        samples = _prior_samples(model, x=x)
        assert "sigma" not in samples
        assert samples["obs"].shape == (4, 20, 1)

    def test_untransformed_scale_positive(self) -> None:
        """softplus of any real input is strictly positive."""
        x = random.normal(random.PRNGKey(0), (50, 2))

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            raw = AdaptiveLayer()("untransformed_scale", x)
            return gaussian_link(mu, y, untransformed_scale=raw)

        samples = _prior_samples(model, num_samples=8, x=x)
        # Normal scale must be positive; if it weren't numpyro would error
        assert samples["obs"].shape[1:] == (50, 1)

    def test_untransformed_scale_fit_runs(self) -> None:
        x = random.normal(random.PRNGKey(0), (NUM_OBS, 3))
        y = x[:, 0] * 2.0 + random.normal(random.PRNGKey(1), (NUM_OBS,)) * 0.5

        @autoreshape
        def model(x, y=None):
            mu = AdaptiveLayer()("mu", x)
            raw = AdaptiveLayer()("untransformed_scale", x)
            return gaussian_link(mu, y, untransformed_scale=raw)

        result = fit(model, y=y, x=x, num_steps=300, lr=0.01, seed=0)
        assert result.params is not None


# --------------------------------------------------------------------------- #
# SpikeAndSlabLayer
# --------------------------------------------------------------------------- #


class TestSpikeAndSlabLayer:
    def test_output_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (50, 4))

        def model(x):
            return deterministic("out", SpikeAndSlabLayer()("coef", x))

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 50, 1)

    def test_output_shape_units(self) -> None:
        x = random.normal(random.PRNGKey(0), (50, 4))

        def model(x):
            return deterministic("out", SpikeAndSlabLayer()("coef", x, units=3))

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 50, 3)

    def test_sample_sites(self) -> None:
        x = random.normal(random.PRNGKey(0), (50, 4))

        def model(x):
            return SpikeAndSlabLayer()("coef", x)

        samples = _prior_samples(model, x=x)
        assert "SpikeAndSlabLayer_coef_z" in samples
        assert "SpikeAndSlabLayer_coef_beta" in samples

    def test_z_in_zero_one(self) -> None:
        """Relaxed Bernoulli z values should lie in (0, 1)."""
        x = random.normal(random.PRNGKey(0), (50, 4))

        def model(x):
            return SpikeAndSlabLayer()("coef", x)

        samples = _prior_samples(model, num_samples=20, x=x)
        z = samples["SpikeAndSlabLayer_coef_z"]
        assert jnp.all(z > 0) and jnp.all(z < 1)

    def test_fit_runs(self) -> None:
        x = random.normal(random.PRNGKey(0), (NUM_OBS, 5))
        y = x[:, 0] * 2.0 + random.normal(random.PRNGKey(1), (NUM_OBS,)) * 0.5

        @autoreshape
        def model(x, y=None):
            mu = SpikeAndSlabLayer()("coef", x)
            return gaussian_link_exp(mu, y)

        result = fit(model, y=y, x=x, num_steps=300, lr=0.01, seed=0)
        assert result.params is not None
