import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from numpyro.handlers import seed, trace

from blayers.links import gaussian_link_exp, logit_link, negative_binomial_link

"""
@pytest.mark.parametrize(
    "dependent_outputs",
    [
        True,
        False,
    ],
)
def test_links(dependent_outputs: bool) -> None:
    with pytest_check.check:
        logit_link(
            jnp.array([0.5, 0.7]),
            jnp.array([1.0, 0.0]),
            dependent_outputs,
        )
    with pytest_check.check:
        gaussian_link_exp(
            jnp.array([0.5, 0.7]),
            jnp.array([1.0, 0.0]),
            dependent_outputs,
        )
    with pytest_check.check:
        negative_binomial_link(
            jnp.array([0.5, 0.7]),
            jnp.array([1.0, 0.0]),
            dependent_outputs,
        )
"""


def test_negative_binomial_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return negative_binomial_link(
            y_hat=jnp.array([5.0, 10.0]), dependent_outputs=True
        )

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    # Check distribution type
    assert isinstance(obs_site["fn"], dist.NegativeBinomial2)

    # Check shapes
    assert obs_site["value"].shape == (2,)
    assert obs_site["fn"].mean.shape == (2,)


def test_negative_binomial_link_with_obs():
    key = random.PRNGKey(1)
    y_obs = jnp.array([3.0, 4.0])

    def model():
        return negative_binomial_link(
            y_hat=jnp.array([5.0, 10.0]), y=y_obs, dependent_outputs=True
        )

    tr = trace(seed(model, key)).get_trace()

    # Verify the observed value is set
    assert jnp.all(tr["obs"]["value"] == y_obs)

    # Likelihood log prob should be finite
    log_prob = tr["obs"]["fn"].log_prob(y_obs)
    assert jnp.isfinite(log_prob).all()


def test_negative_binomial_link_independent():
    key = random.PRNGKey(2)

    def model():
        return negative_binomial_link(
            y_hat=jnp.array([5.0, 10.0]), dependent_outputs=False
        )

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    # In this case, the dist is NOT vectorized with .to_event(1)
    assert obs_site["fn"].event_shape == ()


def test_logit_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return logit_link(y_hat=jnp.array([0.2, 0.8]), dependent_outputs=True)

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    # Check shapes
    assert obs_site["value"].shape == (2,)
    assert obs_site["fn"].probs.shape == (2,)


def test_logit_link_with_obs():
    key = random.PRNGKey(1)
    y_obs = jnp.array([0.0, 1.0])

    def model():
        return logit_link(
            y_hat=jnp.array([0.2, 0.8]), y=y_obs, dependent_outputs=True
        )

    tr = trace(seed(model, key)).get_trace()

    # Verify observed values
    assert jnp.all(tr["obs"]["value"] == y_obs)

    # Likelihood log prob should be finite
    log_prob = tr["obs"]["fn"].log_prob(y_obs)
    assert jnp.isfinite(log_prob).all()


def test_logit_link_independent():
    key = random.PRNGKey(2)

    def model():
        return logit_link(y_hat=jnp.array([0.2, 0.8]), dependent_outputs=False)

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    # In this case, the dist is NOT vectorized with .to_event(1)
    assert obs_site["fn"].event_shape == ()


def test_gaussian_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return gaussian_link_exp(
            y_hat=jnp.array([1.0, -1.0]), dependent_outputs=True
        )

    tr = trace(seed(model, key)).get_trace()

    # Check that sigma was sampled
    assert "sigma" in tr
    sigma_site = tr["sigma"]
    assert isinstance(sigma_site["fn"], dist.Exponential)
    assert sigma_site["value"].ndim == 0  # scalar

    # Check obs site
    obs_site = tr["obs"]
    assert isinstance(obs_site["fn"], dist.Normal)

    # Check shapes
    assert obs_site["value"].shape == (2,)
    assert obs_site["fn"].loc.shape == (2,)
    assert obs_site["fn"].scale.shape == (1,)


def test_gaussian_link_with_obs():
    key = random.PRNGKey(1)
    y_obs = jnp.array([0.5, -0.5])

    def model():
        return gaussian_link_exp(
            y_hat=jnp.array([1.0, -1.0]), y=y_obs, dependent_outputs=True
        )

    tr = trace(seed(model, key)).get_trace()

    # Verify observed values
    assert jnp.all(tr["obs"]["value"] == y_obs)

    # Likelihood log prob finite
    log_prob = tr["obs"]["fn"].log_prob(y_obs)
    assert jnp.isfinite(log_prob).all()


def test_gaussian_link_independent():
    key = random.PRNGKey(2)

    def model():
        return gaussian_link_exp(
            y_hat=jnp.array([1.0, -1.0]), dependent_outputs=False
        )

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    # Not vectorized across outputs
    assert obs_site["fn"].event_shape == ()
