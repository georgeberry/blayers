"""Tests for SplineLayer, _bspline_basis, and make_knots."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
import pytest
from numpyro import sample
from numpyro.infer import Predictive

from blayers._utils import rmse
from blayers.fit import fit
from blayers.layers import SplineLayer, _bspline_basis, make_knots
from blayers.links import gaussian_link_exp
from blayers.sampling import autoreshape

NUM_OBS = 1000


# --------------------------------------------------------------------------- #
# make_knots
# --------------------------------------------------------------------------- #


class TestMakeKnots:
    def test_shape_with_interior_knots(self) -> None:
        x = jnp.linspace(0.0, 1.0, 100)
        knots = make_knots(x, num_knots=5, degree=3)
        # full length = num_knots + 2*(degree+1) = 5 + 8 = 13
        assert knots.shape == (13,)

    def test_shape_no_interior_knots(self) -> None:
        x = jnp.linspace(0.0, 1.0, 100)
        knots = make_knots(x, num_knots=0, degree=3)
        # full length = 0 + 2*(degree+1) = 8
        assert knots.shape == (8,)

    def test_num_basis_cubic(self) -> None:
        x = jnp.linspace(0.0, 1.0, 100)
        knots = make_knots(x, num_knots=5, degree=3)
        num_basis = knots.shape[0] - 3 - 1  # = 9
        assert num_basis == 9

    def test_boundary_knots(self) -> None:
        x = jnp.linspace(2.0, 5.0, 100)
        knots = make_knots(x, num_knots=3, degree=3)
        assert float(knots[0]) == pytest.approx(2.0)
        assert float(knots[-1]) == pytest.approx(5.0)

    def test_clamped_repetition(self) -> None:
        """First degree+1 and last degree+1 knots should be repeated."""
        x = jnp.linspace(0.0, 1.0, 50)
        knots = make_knots(x, num_knots=3, degree=3)
        assert jnp.all(knots[:4] == knots[0])
        assert jnp.all(knots[-4:] == knots[-1])

    def test_accepts_numpy_input(self) -> None:
        import numpy as np

        x = np.linspace(-1.0, 1.0, 200)
        knots = make_knots(x, num_knots=4)
        assert knots.shape == (12,)  # 4 + 2*4 = 12


# --------------------------------------------------------------------------- #
# _bspline_basis
# --------------------------------------------------------------------------- #


class TestBsplineBasis:
    """Unit tests for the low-level basis function."""

    def _make_knots(self, x_min=0.0, x_max=1.0, num_knots=3, degree=3) -> jax.Array:
        x = jnp.linspace(x_min, x_max, 200)
        return make_knots(x, num_knots=num_knots, degree=degree)

    def test_output_shape(self) -> None:
        knots = self._make_knots(num_knots=5, degree=3)
        x = jnp.linspace(0.0, 1.0, 50)
        B = _bspline_basis(x, knots, degree=3)
        num_basis = knots.shape[0] - 3 - 1
        assert B.shape == (50, num_basis)

    def test_partition_of_unity(self) -> None:
        """Each row of the basis matrix should sum to 1 for interior points."""
        knots = self._make_knots(num_knots=4, degree=3)
        x = jnp.linspace(0.01, 0.99, 100)
        B = _bspline_basis(x, knots, degree=3)
        row_sums = B.sum(axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_partition_of_unity_at_boundaries(self) -> None:
        knots = self._make_knots(num_knots=3, degree=3)
        x = jnp.array([0.0, 1.0])  # exact boundary points
        B = _bspline_basis(x, knots, degree=3)
        assert jnp.allclose(B.sum(axis=1), 1.0, atol=1e-5)

    def test_non_negative(self) -> None:
        knots = self._make_knots(num_knots=5, degree=3)
        x = jnp.linspace(0.0, 1.0, 100)
        B = _bspline_basis(x, knots, degree=3)
        assert jnp.all(B >= -1e-8)

    def test_linear_degree(self) -> None:
        """Degree-1 B-splines (piecewise linear) should partition unity."""
        x = jnp.linspace(0.0, 1.0, 200)
        knots = make_knots(x, num_knots=4, degree=1)
        x_eval = jnp.linspace(0.01, 0.99, 50)
        B = _bspline_basis(x_eval, knots, degree=1)
        assert jnp.allclose(B.sum(axis=1), 1.0, atol=1e-5)

    def test_constant_degree(self) -> None:
        """Degree-0 B-splines should be pure indicators summing to 1."""
        x = jnp.linspace(0.0, 1.0, 200)
        knots = make_knots(x, num_knots=3, degree=0)
        x_eval = jnp.linspace(0.01, 0.99, 50)
        B = _bspline_basis(x_eval, knots, degree=0)
        assert jnp.allclose(B.sum(axis=1), 1.0, atol=1e-5)
        # Each row should be a one-hot vector
        assert jnp.all((B == 0.0) | (B == 1.0))

    def test_no_interior_knots(self) -> None:
        """Zero interior knots → polynomial basis of degree+1 functions."""
        x = jnp.linspace(0.0, 1.0, 200)
        knots = make_knots(x, num_knots=0, degree=3)
        x_eval = jnp.linspace(0.01, 0.99, 30)
        B = _bspline_basis(x_eval, knots, degree=3)
        assert B.shape == (30, 4)
        assert jnp.allclose(B.sum(axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# SplineLayer
# --------------------------------------------------------------------------- #


class TestSplineLayerShapes:
    """Check that SplineLayer produces the expected output shapes."""

    def _run_model(self, model_fn, x, y=None, num_samples=5):
        predictive = Predictive(model_fn, num_samples=num_samples)
        return predictive(random.PRNGKey(0), x=x, y=y)

    def test_output_shape_1d(self) -> None:
        x = jnp.linspace(0.0, 1.0, 50).reshape(-1, 1)
        knots = make_knots(x, num_knots=4)

        def model(x, y=None):
            mu = SplineLayer(knots)("f", x)
            return gaussian_link_exp(mu, y)

        samples = self._run_model(model, x)
        # SplineLayer returns (n, units=1); obs shape is (num_samples, n, 1)
        assert samples["obs"].shape == (5, 50, 1)

    def test_output_shape_2d_input(self) -> None:
        """Two features → each gets its own spline basis."""
        x = random.normal(random.PRNGKey(0), (40, 2))
        knots = make_knots(x[:, 0], num_knots=3)

        def model(x, y=None):
            mu = SplineLayer(knots)("f", x)
            return gaussian_link_exp(mu, y)

        samples = self._run_model(model, x)
        assert samples["obs"].shape == (5, 40, 1)

    def test_units_gt_1(self) -> None:
        x = jnp.linspace(0.0, 1.0, 30).reshape(-1, 1)
        knots = make_knots(x, num_knots=3)

        def model(x, y=None):
            return SplineLayer(knots)("f", x, units=2)

        samples = self._run_model(model, x)
        assert samples["SplineLayer_f_beta"].shape[-1] == 2

    def test_num_basis_in_beta(self) -> None:
        """Beta coefficient shape should match d * num_basis."""
        x = jnp.ones((20, 1))
        knots = make_knots(x, num_knots=5, degree=3)
        layer = SplineLayer(knots, degree=3)
        # d=1, num_basis = 5 + 3 + 1 = 9
        assert layer.num_basis == 9

    def test_degree_stored(self) -> None:
        knots = make_knots(jnp.linspace(0, 1, 50), num_knots=3, degree=2)
        layer = SplineLayer(knots, degree=2)
        assert layer.degree == 2


# --------------------------------------------------------------------------- #
# Integration: fit() with SplineLayer
# --------------------------------------------------------------------------- #


def _nonlinear_dgp(num_obs: int) -> dict[str, jax.Array]:
    """y = sin(2π x) + noise."""
    x = sample("x", dist.Uniform(0.0, 1.0).expand([num_obs]))
    sigma = sample("sigma", dist.HalfNormal(0.1))
    mu = jnp.sin(2.0 * jnp.pi * x)
    y = sample("y", dist.Normal(mu, sigma))
    return {"x": x, "y": y}


@pytest.fixture
def nonlinear_data() -> dict[str, jax.Array]:
    predictive = Predictive(_nonlinear_dgp, num_samples=1)
    samples = predictive(random.PRNGKey(42), num_obs=NUM_OBS)
    return {k: jnp.squeeze(v, axis=0) for k, v in samples.items()}


def test_spline_fit_runs(nonlinear_data: dict[str, jax.Array]) -> None:
    """SplineLayer should work end-to-end with fit()."""
    x = nonlinear_data["x"]
    knots = make_knots(x, num_knots=6)

    @autoreshape
    def spline_model(x, y=None):
        mu = SplineLayer(knots)("f", x)
        return gaussian_link_exp(mu, y)

    result = fit(
        spline_model,
        y=nonlinear_data["y"],
        batch_size=256,
        num_epochs=20,
        lr=0.05,
        seed=0,
        x=x,
    )
    assert result.params is not None
    assert result.losses is not None


def test_spline_learns_nonlinear(nonlinear_data: dict[str, jax.Array]) -> None:
    """Spline model should capture sin(2π x) better than predicting zero."""
    x = nonlinear_data["x"]
    y = nonlinear_data["y"]
    knots = make_knots(x, num_knots=8)

    @autoreshape
    def spline_model(x, y=None):
        mu = SplineLayer(knots)("f", x)
        return gaussian_link_exp(mu, y)

    result = fit(
        spline_model,
        y=y,
        batch_size=256,
        num_epochs=100,
        lr=0.05,
        seed=0,
        x=x,
    )
    preds = result.predict(x=x, num_samples=100)
    prediction_rmse = float(rmse(preds.mean, y))
    baseline_rmse = float(rmse(jnp.zeros_like(y), y))
    # A decent spline should do meaningfully better than predicting zero
    assert prediction_rmse < baseline_rmse * 0.5
