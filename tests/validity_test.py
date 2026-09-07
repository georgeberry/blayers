"""Statistical contracts: row-wise likelihoods and correctly weighted ELBOs."""

from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import mask, scale, seed, trace
from numpyro.infer import Trace_ELBO

from blayers import links
from blayers.decorators import autoreshape
from blayers.fit import fit
from blayers.layers import AdaptiveLayer
from blayers.vi_infer import Batched_Trace_ELBO

LINKS = [
    links.gaussian_link,
    links.lognormal_link,
    links.student_t_link,
    links.gamma_link,
    links.exponential_link,
    links.logit_link,
    links.poisson_link,
    links.negative_binomial_link,
    partial(links.ordinal_link, num_classes=3),
    links.zip_link,
    links.zinb_link,
    links.beta_link,
    links.categorical_link,
]


def reference_distribution(index, mu, tr):
    """Handwritten scalar-response likelihood, independent of shape helpers."""
    value = lambda name: tr[name]["value"]
    if index == 0:
        return dist.Normal(mu, value("sigma"))
    if index == 1:
        return dist.LogNormal(mu, value("sigma"))
    if index == 2:
        return dist.StudentT(4, mu, value("sigma"))
    if index == 3:
        k = value("gamma_shape")
        return dist.Gamma(k, k / jnp.exp(mu))
    if index == 4:
        return dist.Exponential(jnp.exp(-mu))
    if index == 5:
        return dist.Bernoulli(logits=mu)
    if index == 6:
        return dist.Poisson(jnp.exp(mu))
    if index == 7:
        return dist.NegativeBinomial2(mu, value("sigma"))
    if index == 8:
        cuts = jnp.concatenate(
            [
                value("ordinal_c0")[None],
                value("ordinal_c0") + jnp.cumsum(value("ordinal_gaps")),
            ]
        )
        return dist.OrderedLogistic(mu, cuts)
    if index == 9:
        return dist.ZeroInflatedPoisson(value("zip_gate"), rate=jnp.exp(mu))
    if index == 10:
        return dist.ZeroInflatedDistribution(
            dist.NegativeBinomial2(jnp.exp(mu), value("zinb_concentration")),
            gate=value("zinb_gate"),
        )
    if index == 11:
        mean = jax.nn.sigmoid(mu)
        phi = value("beta_phi")
        return dist.Beta(mean * phi, (1 - mean) * phi)
    return dist.Categorical(logits=mu)


@pytest.mark.parametrize("index", range(len(LINKS)))
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("column_target", [False, True])
@pytest.mark.parametrize("reshape_model", [False, True])
def test_links_have_one_density_per_row(index, n, column_target, reshape_model):
    mu = jnp.linspace(0.2, 0.8, n)
    target = jnp.linspace(0.3, 0.7, n)
    if index in (5, 6, 7, 8, 9, 10, 12):
        target = jnp.arange(n) % 2
    predictor = mu[:, None]
    if index == 12:
        predictor = jnp.stack([mu, -mu, 2 * mu], axis=-1)

    def model(x, y=None):
        return LINKS[index](x, y)

    if reshape_model:
        model = autoreshape(model)
    tr = trace(seed(model, 17)).get_trace(
        x=predictor, y=target[:, None] if column_target else target
    )
    obs = tr["obs"]
    actual = obs["fn"].log_prob(obs["value"])
    expected = reference_distribution(
        index, predictor if index == 12 else mu, tr
    ).log_prob(target)
    assert actual.size == n
    np.testing.assert_allclose(
        actual.reshape(n), expected, rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("scale_column", [False, True])
@pytest.mark.parametrize("predictor_column", [False, True])
def test_gaussian_heteroscedastic_density_and_gradient(
    scale_column, predictor_column
):
    x = jnp.array([-2.0, -0.5, 1.0, 3.0])
    y = jnp.array([1.0, 2.0, -1.0, 0.5])
    sigma = jnp.array([0.5, 1.0, 1.5, 2.0])

    def actual(beta):
        mu = beta * x
        tr = trace(seed(links.gaussian_link, 0)).get_trace(
            mu[:, None] if predictor_column else mu,
            y,
            scale=sigma[:, None] if scale_column else sigma,
        )
        obs = tr["obs"]
        return obs["fn"].log_prob(obs["value"]).sum()

    expected = lambda beta: dist.Normal(beta * x, sigma).log_prob(y).sum()
    np.testing.assert_allclose(actual(0.7), expected(0.7), rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(actual)(0.7), jax.grad(expected)(0.7), rtol=1e-6
    )


def test_multioutput_likelihood_preserves_outputs_and_per_row_scale():
    mu = jnp.arange(6.0).reshape(3, 2)
    y = mu + jnp.array([[0.2, -0.5], [0.3, 0.8], [-0.4, 0.1]])
    sigma = jnp.array([0.5, 1.0, 2.0])
    tr = trace(seed(links.gaussian_link, 0)).get_trace(mu, y, scale=sigma)
    obs = tr["obs"]
    actual = obs["fn"].log_prob(obs["value"])
    assert actual.shape == (3, 2)
    np.testing.assert_allclose(
        actual, dist.Normal(mu, sigma[:, None]).log_prob(y)
    )


@pytest.mark.parametrize(
    "mu_shape,y_shape",
    [((4, 1), (3,)), ((4, 2), (4,)), ((4,), (4, 2)), ((4, 1), (1,))],
)
def test_likelihood_rejects_incompatible_targets(mu_shape, y_shape):
    with pytest.raises(ValueError, match="Observation shape"):
        seed(links.gaussian_link, 0)(jnp.zeros(mu_shape), jnp.ones(y_shape))


def test_layer_regression_matches_handwritten_log_density_gradient():
    x = jnp.array([[-2.0, 1.0], [0.5, -1.0], [1.0, 2.0]])
    y = jnp.array([1.0, -0.5, 3.0])
    beta = jnp.array([[0.4], [-0.2]])

    def actual(beta):
        def model():
            return links.gaussian_link(AdaptiveLayer()("b", x), y, scale=0.8)

        conditioned = numpyro.handlers.substitute(
            model, data={"AdaptiveLayer_b_beta": beta}
        )
        obs = trace(seed(conditioned, 0)).get_trace()["obs"]
        return obs["fn"].log_prob(obs["value"]).sum()

    def expected(beta):
        return dist.Normal((x @ beta)[:, 0], 0.8).log_prob(y).sum()

    np.testing.assert_allclose(actual(beta), expected(beta), rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(actual)(beta), jax.grad(expected)(beta), rtol=1e-6
    )


def point_guide(x, y):
    beta = numpyro.param("location", 0.3)
    numpyro.sample("beta", dist.Delta(beta))


def regression_model(x, y):
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(x * beta, 0.7), obs=y)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_expected_minibatch_elbo_and_gradient_equal_full_data(batch_size):
    x = jnp.array([-2.0, -0.5, 1.0, 3.0])
    y = jnp.array([1.0, 2.0, -1.0, 0.5])
    # Deliberately configure a larger batch; the inputs determine the real size.
    loss = Batched_Trace_ELBO(num_obs=4, batch_size=8)
    indices = list(combinations(range(4), batch_size))

    def average(beta):
        return jnp.mean(
            jnp.stack(
                [
                    loss.loss(
                        jax.random.PRNGKey(0),
                        {"location": beta},
                        regression_model,
                        point_guide,
                        x=x[jnp.array(idx)],
                        y=y[jnp.array(idx)],
                    )
                    for idx in indices
                ]
            )
        )

    def exact(beta):
        return (
            -dist.Normal(0.0, 1.0).log_prob(beta)
            - dist.Normal(beta * x, 0.7).log_prob(y).sum()
        )

    np.testing.assert_allclose(average(0.3), exact(0.3), rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(average)(0.3), jax.grad(exact)(0.3), rtol=1e-6
    )


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("shuffle", [False, True])
def test_fit_short_and_oversized_batches_have_correct_objective(
    batch_size, shuffle
):
    # Identical rows make every batch's scaled objective exactly the full one.
    x = jnp.ones(5)
    y = jnp.full(5, 2.0)
    result = fit(
        regression_model,
        y=y,
        x=x,
        guide=point_guide,
        autoreparam_model=False,
        batch_size=batch_size,
        num_epochs=2,
        lr=0.0,
        schedule="constant",
        shuffle=shuffle,
    )
    expected = -dist.Normal(0.0, 1.0).log_prob(0.3) - 5 * dist.Normal(
        0.3, 0.7
    ).log_prob(2.0)
    np.testing.assert_allclose(result.losses, expected, rtol=1e-6)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_scaled_masked_elbo_and_gradients_match_numpyro(masked, batch_size):
    x = jnp.array([-2.0, -0.5, 1.0, 3.0])[:batch_size]
    y = jnp.array([1.0, 2.0, -1.0, 0.5])[:batch_size]
    weights = jnp.arange(1.0, batch_size + 1)
    keep = (
        (jnp.arange(batch_size) % 2 == 0)
        if masked
        else jnp.ones(batch_size, dtype=bool)
    )

    def model(x, y, observation_scale=1.0):
        with scale(scale=2.0):
            beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        offset = numpyro.param("offset", -0.1)
        with scale(scale=3.0 * observation_scale), scale(scale=weights), mask(
            mask=keep
        ):
            numpyro.sample("obs", dist.Normal(x * beta + offset, 0.7), obs=y)
            numpyro.factor("extra_likelihood", -0.1 * (x * beta - y) ** 2)

    def guide(x, y):
        beta = numpyro.param("location", 0.3)
        with scale(scale=1.5):
            numpyro.sample("beta", dist.Delta(beta, log_density=beta**2))

    # Built-in ELBO with explicit N/B scaling of only observed terms.
    full_weighted_model = partial(model, observation_scale=4 / batch_size)

    loss = Batched_Trace_ELBO(num_obs=4, batch_size=4)
    params = {"location": jnp.array(0.3), "offset": jnp.array(0.8)}
    key = jax.random.PRNGKey(42)
    actual = lambda params: loss.loss(key, params, model, guide, x, y)
    expected = lambda params: Trace_ELBO().loss(
        key, params, full_weighted_model, guide, x, y
    )
    np.testing.assert_allclose(actual(params), expected(params), rtol=1e-6)
    for name in params:
        np.testing.assert_allclose(
            jax.grad(actual)(params)[name],
            jax.grad(expected)(params)[name],
            rtol=1e-6,
        )
