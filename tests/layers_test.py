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
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal

from blayers import (
    AdaptiveLayer,
    EmbeddingLayer,
    FixedPriorLayer,
    FMLayer,
    LowRankInteractionLayer,
    RandomEffectsLayer,
)
from blayers.infer import Batched_Trace_ELBO, svi_run_batched
from blayers.links import gaussian_link_exp
from blayers.utils import (
    identity,
    outer_product,
    outer_product_upper_tril_no_diag,
    rmse,
)

N_OBS = 10000
LOW_RANK_DIM = 3
EMB_DIM = 1
N_EMB_CATEGORIES = 10


# ---- Data Generating Processes --------------------------------------------- #


def dgp_simple(n_obs: int, k: int) -> dict[str, jax.Array]:
    lambda1 = sample("lambda1", dist.HalfNormal(1.0))
    beta = sample("beta", dist.Normal(0, lambda1).expand([k]))

    x1 = sample("x1", dist.Normal(0, 1).expand([n_obs, k]))

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


def dgp_fm(n_obs: int, k: int) -> dict[str, jax.Array]:
    x1 = sample("x1", dist.Normal(0, 1).expand([n_obs, k]))
    lmbda = sample("lambda", dist.HalfNormal(1.0))
    theta = sample("theta", dist.Normal(0.0, lmbda).expand([k, LOW_RANK_DIM]))

    sigma = sample("sigma", dist.HalfNormal(1.0))
    mu = FMLayer.matmul(theta, x1)
    y = sample("y", dist.Normal(mu, sigma))
    return {
        "x1": x1,
        "y": y,
        "theta": theta,
        "lambda": lmbda,
        "sigma": sigma,
    }


def dgp_emb(n_obs: int, k: int, n_categories: int) -> dict[str, jax.Array]:
    lmbda = sample("lambda", dist.HalfNormal(1.0))
    beta = sample("beta", dist.Normal(0, lmbda).expand([n_categories, k]))
    x1 = sample(
        "x1",
        dist.Categorical(probs=jnp.ones(n_categories) / n_categories).expand(
            [n_obs]
        ),
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


def dgp_lowrank(n_obs: int, k: int) -> dict[str, jax.Array]:
    offset = 1

    x1 = sample("x1", dist.Normal(0, 1).expand([n_obs, k]))
    x2 = sample("x2", dist.Normal(0, 1).expand([n_obs, k + offset]))

    lambda1 = sample("lambda1", dist.HalfNormal(1.0))
    theta1_lowrank = sample(
        "theta1", dist.Normal(0.0, lambda1).expand([k, LOW_RANK_DIM])
    )

    lambda2 = sample("lambda2", dist.HalfNormal(1.0))
    theta2_lowrank = sample(
        "theta2", dist.Normal(0.0, lambda1).expand([k + offset, LOW_RANK_DIM])
    )

    sigma = sample("sigma", dist.HalfNormal(1.0))
    mu = LowRankInteractionLayer.matmul(
        theta1=theta1_lowrank,
        theta2=theta2_lowrank,
        x=x1,
        z=x2,
    )

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
    return simulated_data(dgp_simple, n_obs=N_OBS, k=2)


@pytest.fixture
def simulated_data_fm() -> dict[str, jax.Array]:
    return simulated_data(dgp_fm, n_obs=N_OBS, k=10)


@pytest.fixture
def simulated_data_emb() -> dict[str, jax.Array]:
    return simulated_data(
        dgp_emb,
        n_obs=N_OBS,
        k=EMB_DIM,
        n_categories=N_EMB_CATEGORIES,
    )


@pytest.fixture
def simulated_data_lowrank() -> dict[str, jax.Array]:
    return simulated_data(
        dgp_lowrank,
        n_obs=N_OBS,
        k=10,
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
def fm_regression_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        theta = FMLayer(low_rank_dim=LOW_RANK_DIM)("theta", x1)
        return gaussian_link_exp(theta, y)

    return (
        model,
        [
            (["FMLayer_theta_theta"], outer_product_upper_tril_no_diag),
        ],
    )


@pytest.fixture
def emb_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, y: jax.Array | None = None) -> Any:
        beta = EmbeddingLayer()(
            "beta",
            x1,
            n_categories=N_EMB_CATEGORIES,
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
            n_categories=N_EMB_CATEGORIES,
        )
        return gaussian_link_exp(beta, y)

    return (
        model,
        [
            (["RandomEffectsLayer_beta_beta"], identity),
        ],
    )


@pytest.fixture
def lowrank_model() -> (
    tuple[Callable[..., Any], list[tuple[list[str], Callable[..., jax.Array]]]]
):
    def model(x1: jax.Array, x2: jax.Array, y: jax.Array | None = None) -> Any:
        beta1 = LowRankInteractionLayer(low_rank_dim=LOW_RANK_DIM)(
            "lowrank",
            x1,
            x2,
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


# ---- Loss classes ---------------------------------------------------------- #


@pytest.fixture
def trace_elbo() -> Trace_ELBO:
    return Trace_ELBO()


@pytest.fixture
def trace_elbo_batched() -> Batched_Trace_ELBO:
    return Batched_Trace_ELBO(n_obs=N_OBS)


# ---- Dispatchers ----------------------------------------------------------- #

"""
These are pytest helpers that let us cycle through fixtures. This setup is a
little wonky and I'm sure we could come up with something better in the long
run, but it works for now. Just make one with the name for the thing you want
to pass to the ultimate test function.
"""


@pytest.fixture
def model(request: SubRequest) -> Any:
    return request.getfixturevalue(request.param)


@pytest.fixture
def data(request: SubRequest) -> Any:
    return request.getfixturevalue(request.param)


@pytest.fixture
def loss_instance(request: SubRequest) -> Any:
    return request.getfixturevalue(request.param)


# ---- Test functions -------------------------------------------------------- #


@pytest.mark.parametrize(
    "loss_instance",
    [
        "trace_elbo",
        "trace_elbo_batched",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("model", "data"),
    [
        ("linear_regression_adaptive_model", "simulated_data_simple"),
        ("linear_regression_fixed_model", "simulated_data_simple"),
        ("fm_regression_model", "simulated_data_fm"),
        ("emb_model", "simulated_data_emb"),
        ("re_model", "simulated_data_emb"),
        ("lowrank_model", "simulated_data_lowrank"),
    ],
    indirect=True,
)
def test_models(
    data: Any,
    model: Any,
    loss_instance: Any,
) -> Any:
    model_fn, coef_groups = model
    model_data = {k: v for k, v in data.items() if k in ("y", "x1", "x2")}

    guide = AutoDiagonalNormal(model_fn)

    num_steps = 30000

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
            num_steps,
            batch_size=1000,
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
            val = rmse(
                coef_fn(*[guide_means[x] for x in coef_list]),
                coef_fn(*[data[x.split("_")[2]] for x in coef_list]),
            )
            assert val < 0.1

    with pytest_check.check:
        assert (
            rmse(
                guide_means["sigma"],
                data["sigma"],
            )
            < 0.03
        )
