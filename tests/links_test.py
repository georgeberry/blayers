from functools import partial

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from numpyro.handlers import seed, trace

from blayers.links import gaussian_link, logit_link, lognormal_link, negative_binomial_link, student_t_link


def test_negative_binomial_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return negative_binomial_link(
            y_hat=jnp.array([5.0, 10.0])
        )

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    assert isinstance(obs_site["fn"], dist.NegativeBinomial2)
    assert obs_site["value"].shape == (2,)
    assert obs_site["fn"].mean.shape == (2,)


def test_negative_binomial_link_with_obs():
    key = random.PRNGKey(1)
    y_obs = jnp.array([3.0, 4.0])

    def model():
        return negative_binomial_link(
            y_hat=jnp.array([5.0, 10.0]), y=y_obs
        )

    tr = trace(seed(model, key)).get_trace()

    assert jnp.all(tr["obs"]["value"] == y_obs)
    log_prob = tr["obs"]["fn"].log_prob(y_obs)
    assert jnp.isfinite(log_prob).all()


def test_negative_binomial_link_independent():
    key = random.PRNGKey(2)

    def model():
        return negative_binomial_link(
            y_hat=jnp.array([5.0, 10.0])
        )

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    assert obs_site["fn"].event_shape == ()


def test_logit_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return logit_link(y_hat=jnp.array([0.2, 0.8]))

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    assert obs_site["value"].shape == (2,)
    assert obs_site["fn"].probs.shape == (2,)


def test_logit_link_with_obs():
    key = random.PRNGKey(1)
    y_obs = jnp.array([0.0, 1.0])

    def model():
        return logit_link(
            y_hat=jnp.array([0.2, 0.8]), y=y_obs
        )

    tr = trace(seed(model, key)).get_trace()

    assert jnp.all(tr["obs"]["value"] == y_obs)
    log_prob = tr["obs"]["fn"].log_prob(y_obs)
    assert jnp.isfinite(log_prob).all()


def test_logit_link_independent():
    key = random.PRNGKey(2)

    def model():
        return logit_link(y_hat=jnp.array([0.2, 0.8]))

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    assert obs_site["fn"].event_shape == ()


def test_gaussian_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return gaussian_link(y_hat=jnp.array([1.0, -1.0]))

    tr = trace(seed(model, key)).get_trace()

    assert "sigma" in tr
    sigma_site = tr["sigma"]
    assert isinstance(sigma_site["fn"], dist.Exponential)
    assert sigma_site["value"].ndim == 0  # scalar

    obs_site = tr["obs"]
    assert isinstance(obs_site["fn"], dist.Normal)
    assert obs_site["value"].shape == (2,)
    assert obs_site["fn"].loc.shape == (2,)


def test_gaussian_link_with_obs():
    key = random.PRNGKey(1)
    y_obs = jnp.array([0.5, -0.5])

    def model():
        return gaussian_link(y_hat=jnp.array([1.0, -1.0]), y=y_obs)

    tr = trace(seed(model, key)).get_trace()

    assert jnp.all(tr["obs"]["value"] == y_obs)
    log_prob = tr["obs"]["fn"].log_prob(y_obs)
    assert jnp.isfinite(log_prob).all()


def test_gaussian_link_independent():
    key = random.PRNGKey(2)

    def model():
        return gaussian_link(y_hat=jnp.array([1.0, -1.0]))

    tr = trace(seed(model, key)).get_trace()
    obs_site = tr["obs"]

    assert obs_site["fn"].event_shape == ()


def test_gaussian_link_halfnormal_sigma():
    """sigma_dist can be swapped to HalfNormal via partial or direct kwarg."""
    key = random.PRNGKey(3)

    hn_link = partial(gaussian_link, sigma_dist=dist.HalfNormal, sigma_kwargs={"scale": 1.0})

    def model():
        return hn_link(y_hat=jnp.array([1.0, -1.0]))

    tr = trace(seed(model, key)).get_trace()

    assert "sigma" in tr
    assert isinstance(tr["sigma"]["fn"], dist.HalfNormal)
    assert isinstance(tr["obs"]["fn"], dist.Normal)


def test_gaussian_link_known_scale():
    """Passing scale= skips the sigma sample site."""
    key = random.PRNGKey(4)

    def model():
        return gaussian_link(y_hat=jnp.array([1.0, -1.0]), scale=0.5)

    tr = trace(seed(model, key)).get_trace()

    assert "sigma" not in tr
    assert isinstance(tr["obs"]["fn"], dist.Normal)


def test_lognormal_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return lognormal_link(y_hat=jnp.array([1.0, 2.0]))

    tr = trace(seed(model, key)).get_trace()

    assert "sigma" in tr
    assert isinstance(tr["sigma"]["fn"], dist.Exponential)
    assert isinstance(tr["obs"]["fn"], dist.LogNormal)
    assert tr["obs"]["value"].shape == (2,)


def test_lognormal_link_halfnormal_sigma():
    key = random.PRNGKey(1)

    hn_lognormal = partial(lognormal_link, sigma_dist=dist.HalfNormal, sigma_kwargs={"scale": 1.0})

    def model():
        return hn_lognormal(y_hat=jnp.array([1.0, 2.0]))

    tr = trace(seed(model, key)).get_trace()

    assert isinstance(tr["sigma"]["fn"], dist.HalfNormal)
    assert isinstance(tr["obs"]["fn"], dist.LogNormal)


def test_student_t_link_sample_shape():
    key = random.PRNGKey(0)

    def model():
        return student_t_link(y_hat=jnp.array([1.0, -1.0]))

    tr = trace(seed(model, key)).get_trace()

    assert "sigma" in tr
    assert isinstance(tr["obs"]["fn"], dist.StudentT)
    assert tr["obs"]["value"].shape == (2,)
    assert tr["obs"]["fn"].df == 4.0


def test_student_t_link_custom_df():
    key = random.PRNGKey(1)
    cauchy_link = partial(student_t_link, obs_dist=partial(dist.StudentT, df=1.0))

    def model():
        return cauchy_link(y_hat=jnp.array([1.0, -1.0]))

    tr = trace(seed(model, key)).get_trace()

    assert isinstance(tr["obs"]["fn"], dist.StudentT)
    assert tr["obs"]["fn"].df == 1.0


def test_student_t_link_with_obs():
    key = random.PRNGKey(2)
    y_obs = jnp.array([0.5, -0.5])

    def model():
        return student_t_link(y_hat=jnp.array([1.0, -1.0]), y=y_obs)

    tr = trace(seed(model, key)).get_trace()
    assert jnp.all(tr["obs"]["value"] == y_obs)
    assert jnp.isfinite(tr["obs"]["fn"].log_prob(y_obs)).all()
