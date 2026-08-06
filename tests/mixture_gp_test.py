"""Tests for MixtureLayer and HSGPLayer."""

import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
import pytest
from numpyro import deterministic
from numpyro.infer import Predictive

from blayers._utils import rmse
from blayers.decorators import autoreshape
from blayers.fit import fit
from blayers.layers import HSGPLayer, MixtureLayer, hsgp_L
from blayers.links import gaussian_link

NUM_OBS = 1000


def _prior_samples(model_fn, num_samples=4, **kwargs):
    return Predictive(model_fn, num_samples=num_samples)(
        random.PRNGKey(0), **kwargs
    )


# --------------------------------------------------------------------------- #
# MixtureLayer
# --------------------------------------------------------------------------- #


class TestMixtureLayer:
    def test_output_shape(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 4))

        def model(x):
            return deterministic("out", MixtureLayer()("coef", x))

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 1)

    def test_output_shape_units(self) -> None:
        x = random.normal(random.PRNGKey(0), (30, 4))

        def model(x):
            return deterministic("out", MixtureLayer()("coef", x, units=3))

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 3)

    def test_sites(self) -> None:
        """Logistic-normal weight logits + mixture beta are sampled by default."""
        x = random.normal(random.PRNGKey(0), (20, 4))

        def model(x):
            return MixtureLayer()("coef", x)

        samples = _prior_samples(model, x=x)
        assert "MixtureLayer_coef_logits" in samples
        assert "MixtureLayer_coef_beta" in samples

    def test_fixed_weights_no_weight_site(self) -> None:
        x = random.normal(random.PRNGKey(0), (20, 4))

        def model(x):
            return MixtureLayer(weights=[0.5, 0.5])("coef", x)

        samples = _prior_samples(model, x=x)
        assert "MixtureLayer_coef_logits" not in samples
        assert "MixtureLayer_coef_beta" in samples

    def test_bad_kwargs_raise_at_construction(self) -> None:
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            MixtureLayer(
                component_dists=(dist.Normal, dist.Laplace),
                component_kwargs=({"loc": 0.0, "scale": 1.0}, {"bad": 1.0}),
            )

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            MixtureLayer(
                component_dists=(dist.Normal, dist.Laplace),
                component_kwargs=({"loc": 0.0, "scale": 1.0},),
            )

    def test_too_few_components_raise(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            MixtureLayer(
                component_dists=(dist.Normal,),
                component_kwargs=({"loc": 0.0, "scale": 1.0},),
            )

    def test_weights_length_mismatch_raise(self) -> None:
        with pytest.raises(ValueError, match="one entry per component"):
            MixtureLayer(weights=[1.0])  # 2 default components, 1 weight

    def test_fit_learns(self) -> None:
        key = random.PRNGKey(0)
        x = random.normal(key, (NUM_OBS, 4))
        y = x @ jnp.array([2.0, 0.0, 0.0, -1.5]) + 0.3 * random.normal(
            random.PRNGKey(1), (NUM_OBS,)
        )

        @autoreshape
        def model(x, y=None):
            return gaussian_link(MixtureLayer()("coef", x), y)

        result = fit(model, y=y, x=x, num_steps=500, lr=0.03, seed=0)
        preds = result.predict(x=x, num_samples=200)
        assert (
            float(rmse(preds.mean, y)) < float(rmse(jnp.zeros_like(y), y)) * 0.5
        )

    def test_fit_mcmc_runs(self) -> None:
        """MixtureGeneral marginalises the indicator, so NUTS works too."""
        key = random.PRNGKey(0)
        x = random.normal(key, (200, 3))
        y = x @ jnp.array([1.5, 0.0, -1.0]) + 0.3 * random.normal(
            random.PRNGKey(1), (200,)
        )

        @autoreshape
        def model(x, y=None):
            return gaussian_link(MixtureLayer()("coef", x), y)

        result = fit(
            model,
            y=y,
            method="mcmc",
            num_warmup=100,
            num_mcmc_samples=100,
            x=x,
        )
        assert result.posterior_samples is not None

    def test_fit_svgd_runs(self) -> None:
        """Logistic-normal weights keep every latent unconstrained, so SVGD's
        particle flattener works — a raw Dirichlet simplex site would not."""
        key = random.PRNGKey(0)
        x = random.normal(key, (200, 3))
        y = x @ jnp.array([1.5, 0.0, -1.0]) + 0.3 * random.normal(
            random.PRNGKey(1), (200,)
        )

        @autoreshape
        def model(x, y=None):
            return gaussian_link(MixtureLayer()("coef", x), y)

        result = fit(
            model,
            y=y,
            method="svgd",
            num_steps=100,
            num_particles=8,
            x=x,
        )
        assert result.params is not None


# --------------------------------------------------------------------------- #
# HSGPLayer
# --------------------------------------------------------------------------- #


class TestHSGPLayer:
    def test_output_shape(self) -> None:
        x = random.uniform(random.PRNGKey(0), (30,), minval=-2, maxval=2)
        L = hsgp_L(x)

        def model(x, L=L, m=20):
            return deterministic("out", HSGPLayer()("f", x, L=L, m=m))

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 1)

    def test_output_shape_units(self) -> None:
        x = random.uniform(random.PRNGKey(0), (30,), minval=-2, maxval=2)
        L = hsgp_L(x)

        def model(x, L=L, m=20):
            return deterministic("out", HSGPLayer()("f", x, L=L, m=m, units=2))

        samples = _prior_samples(model, x=x)
        assert samples["out"].shape == (4, 30, 2)

    def test_sites(self) -> None:
        x = random.uniform(random.PRNGKey(0), (20,), minval=-2, maxval=2)
        L = hsgp_L(x)

        def model(x, L=L, m=15):
            return HSGPLayer()("f", x, L=L, m=m)

        samples = _prior_samples(model, x=x)
        assert "HSGPLayer_f_lengthscale" in samples
        assert "HSGPLayer_f_sigma" in samples
        assert "HSGPLayer_f_beta" in samples
        assert samples["HSGPLayer_f_beta"].shape == (4, 15, 1)

    def test_hsgp_L(self) -> None:
        x = jnp.array([-3.0, 1.0, 2.0])
        assert hsgp_L(x, c=1.5) == pytest.approx(4.5)  # 1.5 * max(|x|)=3

    def test_bad_kwargs_raise(self) -> None:
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            HSGPLayer(sigma_kwargs={"bad": 1.0})

    def test_recovers_smooth_function(self) -> None:
        """HSGP should recover a smooth signal well below the noise level."""
        key = random.PRNGKey(0)
        x = jnp.sort(random.uniform(key, (NUM_OBS,), minval=-3, maxval=3))
        f_true = jnp.sin(1.5 * x) + 0.3 * x
        y = f_true + 0.2 * random.normal(random.PRNGKey(1), (NUM_OBS,))
        L = hsgp_L(x)
        m = 30

        @autoreshape
        def model(x, y=None, L=L, m=m):
            return gaussian_link(HSGPLayer()("f", x, L=L, m=m), y)

        result = fit(model, y=y, x=x, L=L, m=m, num_steps=1500, lr=0.02, seed=0)
        preds = result.predict(x=x, L=L, m=m, num_samples=200)
        # noise std is 0.2; a good GP fit gets well under that vs the truth
        assert float(rmse(preds.mean.reshape(-1), f_true)) < 0.15
