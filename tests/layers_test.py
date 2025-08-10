from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
import optax
import pytest
import pytest_check
from _pytest.fixtures import SubRequest
from numpyro import sample
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal

from blayers._utils import (
    add_trailing_dim,
    identity,
    outer_product,
    outer_product_upper_tril_no_diag,
    rmse,
)
from blayers.infer import Batched_Trace_ELBO, svi_run_batched
from blayers.layers import (
    AdaptiveLayer,
    EmbeddingLayer,
    FixedPriorLayer,
    FMLayer,
    InteractionLayer,
    LowRankInteractionLayer,
    RandomEffectsLayer,
    RandomWalkLayer,
    _matmul_factorization_machine,
    _matmul_interaction,
    _matmul_randomwalk,
    _matmul_uv_decomp,
)
from blayers.links import gaussian_link_exp
from blayers.sampling import autoreparam

NUM_OBS = 10000
LOW_RANK_DIM = 3
EMB_DIM = 1
NUM_EMB_CATEGORIES = 10


# ---- Data Generating Processes --------------------------------------------- #


def dgp_simple(num_obs: int, k: int) -> dict[str, jax.Array]:
    lambda1 = sample("lambda1", dist.HalfNormal(1.0))
    beta = sample("beta", dist.Normal(0, lambda1).expand([k]))

    x1 = sample("x1", dist.Normal(0, 1).expand([num_obs, k]))

    sigma = sample("sigma", dist.HalfNormal(1.0))
    mu = jnp.dot(x1, beta)
    y = sample("y", dist.Normal(mu, sigma))
    return {
        "x1": x1,
        "y": y,
        "beta": beta,
        "lambda1": lambda1,
        "sigma": sigma,
    }


def dgp_fm(num_obs: int, k: int) -> dict[str, jax.Array]:
    x1 = sample("x1", dist.Normal(0, 1).expand([num_obs, k]))
    lmbda = sample("lambda", dist.HalfNormal(1.0))
    theta = sample(
        "theta", dist.Normal(0.0, lmbda).expand([k, LOW_RANK_DIM, 1])
    )

    sigma = sample("sigma", dist.HalfNormal(1.0))
    mu = _matmul_factorization_machine(x1, theta)
    y = sample("y", dist.Normal(mu, sigma))
    return {
        "x1": x1,
        "y": y,
        "theta": theta,
        "lambda": lmbda,
        "sigma": sigma,
    }


def dgp_emb(num_obs: int, k: int, num_categories: int) -> dict[str, jax.Array]:
    lmbda = sample("lambda", dist.HalfNormal(1.0))
    beta = sample("beta", dist.Normal(0, lmbda).expand([num_categories, k]))
    x1 = sample(
        "x1",
        dist.Categorical(
            probs=jnp.ones(num_categories) / num_categories
        ).expand([num_obs]),
    )

    sigma = sample("sigma", dist.HalfNormal(1.0))

    mu = jnp.sum(beta[x1], axis=1)
    y = sample("y", dist.Normal(mu, sigma))
    return {
        "x1": x1,
        "y": y,
        "beta": beta,
        "lambda": lmbda,
        "sigma": sigma,
    }


def dgp_lowrank(num_obs: int, k: int) -> dict[str, jax.Array]:
    offset = 1

    x1 = sample("x1", dist.Normal(0, 1).expand([num_obs, k]))
    x2 = sample("x2", dist.Normal(0, 1).expand([num_obs, k + offset]))

    lambda1 = sample("lambda1", dist.HalfNormal(1.0))
    theta1_lowrank = sample(
        "theta1", dist.Normal(0.0, lambda1).expand([k, LOW_RANK_DIM, 1])
    )

    lambda2 = sample("lambda2", dist.HalfNormal(1.0))
    theta2_lowrank = sample(
        "theta2",
        dist.Normal(0.0, lambda1).expand([k + offset, LOW_RANK_DIM, 1]),
    )

    sigma = sample("sigma", dist.HalfNormal(1.0))
    mu = _matmul_uv_decomp(x1, x2, theta1_lowrank, theta2_lowrank)

    y = sample("y", dist.Normal(mu, sigma))
    return {
        "x1": x1,
        "y": y,
        "theta1": theta1_lowrank,
        "theta2": theta2_lowrank,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "sigma": sigma,
    }


def dgp_interaction(num_obs: int, k: int) -> dict[str, jax.Array]:
    lambda1 = sample("lambda1", dist.HalfNormal(1.0))
    beta = add_trailing_dim(
        sample("beta1", dist.Normal(0, lambda1).expand([k * k]))
    )

    x1 = sample("x1", dist.Normal(0, 1).expand([num_obs, k]))
    x2 = sample("x2", dist.Normal(0, 1).expand([num_obs, k]))

    mu = _matmul_interaction(beta, x1, x2)

    sigma = sample("sigma", dist.HalfNormal(1.0))
    y = sample("y", dist.Normal(mu, sigma))
    return {
        "x1": x1,
        "x2": x2,
        "y": y,
        "beta1": beta,
        "lambda1": lambda1,
        "sigma": sigma,
    }


def dgp_rw(num_obs: int, k: int, num_categories: int) -> dict[str, jax.Array]:
    lmbda = sample("lambda", dist.HalfNormal(1.0))
    theta = sample("theta", dist.Normal(0, lmbda).expand([num_categories, k]))

    x1 = sample(
        "x1",
        dist.Categorical(
            probs=jnp.ones(num_categories) / num_categories
        ).expand([num_obs]),
    )

    sigma = sample("sigma", dist.HalfNormal(1.0))

    mu = _matmul_randomwalk(theta, x1)
    y = sample("y", dist.Normal(mu, sigma))
    return {
        "x1": x1,
        "y": y,
        "theta": theta,
        "lambda": lmbda,
        "sigma": sigma,
    }


# ---- Simulated data helpers ------------------------------------------------ #


def simulated_data(
    dgp: Callable[..., Any],
    **kwargs: Any,
) -> dict[str, jax.Array]:
    rng_key = random.PRNGKey(0)
    predictive = Predictive(dgp, num_samples=1)
    samples = predictive(
        rng_key,
        **kwargs,
    )
    res = {k: jnp.squeeze(v, axis=0) for k, v in samples.items()}
    return res


@pytest.fixture
def simulated_data_simple() -> dict[str, jax.Array]:
    return simulated_data(dgp_simple, num_obs=NUM_OBS, k=2)


@pytest.fixture
def simulated_data_fm() -> dict[str, jax.Array]:
    return simulated_data(dgp_fm, num_obs=NUM_OBS, k=10)


@pytest.fixture
def simulated_data_emb() -> dict[str, jax.Array]:
    return simulated_data(
        dgp_emb,
        num_obs=NUM_OBS,
        k=EMB_DIM,
        num_categories=NUM_EMB_CATEGORIES,
    )


@pytest.fixture
def simulated_data_lowrank() -> dict[str, jax.Array]:
    return simulated_data(
        dgp_lowrank,
        num_obs=NUM_OBS,
        k=10,
    )


@pytest.fixture
def simulated_data_interaction() -> dict[str, jax.Array]:
    return simulated_data(
        dgp_interaction,
        num_obs=NUM_OBS,
        k=3,
    )


@pytest.fixture
def simulated_data_rw() -> dict[str, jax.Array]:
    return simulated_data(
        dgp_rw,
        num_obs=NUM_OBS,
        k=EMB_DIM,
        num_categories=NUM_EMB_CATEGORIES,
    )


# ---- Models ---------------------------------------------------------------- #


@pytest.fixture
def linear_regression_adaptive_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        beta = AdaptiveLayer()("beta", x1)
        return gaussian_link_exp(beta, y)

    return model, [(["AdaptiveLayer_beta_beta"], identity)]


@pytest.fixture
def linear_regression_fixed_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        beta = FixedPriorLayer()("beta", x1)
        return gaussian_link_exp(beta, y)

    return model, [(["FixedPriorLayer_beta_beta"], identity)]


@pytest.fixture
def emb_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        beta = EmbeddingLayer()(
            "beta",
            x1,
            num_categories=NUM_EMB_CATEGORIES,
            embedding_dim=EMB_DIM,
        )
        return gaussian_link_exp(beta, y)

    return (
        model,
        [
            (["EmbeddingLayer_beta_beta"], identity),
        ],
    )


@pytest.fixture
def re_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        beta = RandomEffectsLayer()(
            "beta",
            x1,
            num_categories=NUM_EMB_CATEGORIES,
        )
        return gaussian_link_exp(beta, y)

    return (
        model,
        [
            (["RandomEffectsLayer_beta_beta"], identity),
        ],
    )


@pytest.fixture
def fm_regression_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        theta = FMLayer()("theta", x1, low_rank_dim=LOW_RANK_DIM)
        return gaussian_link_exp(theta, y)

    return (
        model,
        [
            (["FMLayer_theta_theta"], outer_product_upper_tril_no_diag),
        ],
    )


@pytest.fixture
def lowrank_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, x2: jax.Array, y: jax.Array | None = None) -> Any:
        beta1 = LowRankInteractionLayer()(
            "lowrank", x1, x2, low_rank_dim=LOW_RANK_DIM
        )
        return gaussian_link_exp(beta1, y)

    return (
        model,
        [
            (
                [
                    "LowRankInteractionLayer_lowrank_theta1",
                    "LowRankInteractionLayer_lowrank_theta2",
                ],
                outer_product,
            ),
        ],
    )


@pytest.fixture
def interaction_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, x2: jax.Array, y: jax.Array | None = None) -> Any:
        beta1 = InteractionLayer()(
            "beta",
            x1,
            x2,
        )
        return gaussian_link_exp(beta1, y)

    return (
        model,
        [
            (
                [
                    "InteractionLayer_beta_beta1",
                ],
                identity,
            ),
        ],
    )


@pytest.fixture
def rw_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        beta = RandomWalkLayer()(
            "beta",
            x1,
            num_categories=NUM_EMB_CATEGORIES,
            embedding_dim=EMB_DIM,
        )
        return gaussian_link_exp(beta, y)

    return (
        model,
        [
            (["RandomWalkLayer_beta_theta"], identity),
        ],
    )


# ---- Loss classes ---------------------------------------------------------- #


@pytest.fixture
def trace_elbo() -> Trace_ELBO:
    return Trace_ELBO()


@pytest.fixture
def trace_elbo_batched() -> Batched_Trace_ELBO:
    return Batched_Trace_ELBO(num_obs=NUM_OBS)


# ---- Dispatchers ----------------------------------------------------------- #

"""
These are pytest helpers that let us cycle through fixtures. This setup is a
little wonky and I'm sure we could come up with something better in the long
run, but it works for now. Just make one with the name for the thing you want
to pass to the ultimate test function.
"""


@pytest.fixture
def model_bundle(request: SubRequest) -> Any:
    return request.getfixturevalue(request.param)


@pytest.fixture
def data(request: SubRequest) -> Any:
    return request.getfixturevalue(request.param)


@pytest.fixture
def loss_instance(request: SubRequest) -> Any:
    return request.getfixturevalue(request.param)


# ---- VI -------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "loss_instance",
    [
        # "trace_elbo",
        "trace_elbo_batched",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("model_bundle", "data"),
    [
        # ("linear_regression_adaptive_model", "simulated_data_simple"),
        # ("linear_regression_fixed_model", "simulated_data_simple"),
        # ("emb_model", "simulated_data_emb"),
        # ("re_model", "simulated_data_emb"),
        # ("fm_regression_model", "simulated_data_fm"),
        # ("lowrank_model", "simulated_data_lowrank"),
        # ("interaction_model", "simulated_data_interaction"),
        ("rw_model", "simulated_data_rw"),
    ],
    indirect=True,
)
def test_models_vi(
    data: Any,
    model_bundle: Any,
    loss_instance: Any,
) -> Any:
    model_fn, coef_groups = model_bundle
    model_data = {k: v for k, v in data.items() if k in ("y", "x1", "x2")}
    model_data["y"] = jnp.reshape(model_data["y"], (-1, 1))

    guide = AutoDiagonalNormal(model_fn)

    num_steps = 20000

    schedule = optax.cosine_onecycle_schedule(
        transition_steps=num_steps,
        peak_value=5e-2,
        pct_start=0.1,
        div_factor=25,
    )

    svi = SVI(model_fn, guide, optax.adam(schedule), loss=loss_instance)

    rng_key = random.PRNGKey(2)

    if isinstance(loss_instance, Trace_ELBO):
        svi_result = svi.run(
            rng_key,
            num_steps=num_steps,
            **model_data,
        )
    if isinstance(loss_instance, Batched_Trace_ELBO):
        svi_result = svi_run_batched(
            svi,
            rng_key,
            1000,
            num_steps,
            **model_data,
        )
    guide_predicitive = Predictive(
        guide,
        params=svi_result.params,
        num_samples=1000,
    )
    guide_samples = guide_predicitive(
        random.PRNGKey(1),
        **{k: v for k, v in model_data.items() if k != "y"},
    )
    guide_means = {k: jnp.mean(v, axis=0) for k, v in guide_samples.items()}

    for coef_list, coef_fn in coef_groups:
        with pytest_check.check:
            # import ipdb; ipdb.set_trace()
            val = rmse(
                coef_fn(*[guide_means[x] for x in coef_list]).squeeze(),
                coef_fn(*[data[x.split("_")[2]] for x in coef_list]).squeeze(),
            )
            assert val < 0.15

    with pytest_check.check:
        assert (
            rmse(
                guide_means["sigma"],
                data["sigma"],
            )
            < 0.03
        )


# ---- HMC ------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model_bundle", "data"),
    [
        ("linear_regression_adaptive_model", "simulated_data_simple"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "is_reparam",
    [
        True,
        False,
    ],
)
def test_models_hmc(
    data: Any,
    model_bundle: Any,
    is_reparam: bool,
) -> Any:
    model_fn, coef_groups = model_bundle
    model_data = {k: v for k, v in data.items() if k in ("y", "x1", "x2")}
    model_data["y"] = jnp.reshape(model_data["y"], (-1, 1))

    if is_reparam:
        model_fn = autoreparam()(model_fn)

    rng_key = random.PRNGKey(2)

    kernel = NUTS(model_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        progress_bar=True,
    )
    mcmc.run(
        rng_key,
        **model_data,
    )
    samples = mcmc.get_samples()
    sample_means = {k: jnp.mean(v, axis=0) for k, v in samples.items()}

    mcmc.print_summary()

    """
    predictive = Predictive(
        model_fn,
        samples,
    )
    rng_key, rng_key_ = random.split(rng_key)
    predictions = predictive(rng_key_, **model_data)['obs']
    """

    for coef_list, coef_fn in coef_groups:
        with pytest_check.check:
            val = rmse(
                coef_fn(*[sample_means[x] for x in coef_list]).squeeze(),
                coef_fn(*[data[x.split("_")[2]] for x in coef_list]),
            )
            assert val < 0.03

    with pytest_check.check:
        assert (
            rmse(
                sample_means["sigma"],
                data["sigma"],
            )
            < 0.03
        )
