"""Every built-in layer must work with both VI and ordinary HMC/NUTS.

These short fits test inference compatibility and finite outputs, not posterior
convergence or calibration; statistical recovery tests live elsewhere.
"""

import jax.numpy as jnp
import numpyro.infer as infer
import pytest

import blayers
from blayers import layers
from blayers.fit import fit
from blayers.links import gaussian_link

# Explicit call recipes ensure a new layer requires a compatibility test.
CALLS = {
    "AdaptiveLayer": lambda layer, x: layer("b", x),
    "FixedPriorLayer": lambda layer, x: layer("b", x),
    "InterceptLayer": lambda layer, x: layer("b"),
    "FMLayer": lambda layer, x: layer("b", x, low_rank_dim=2),
    "FM3Layer": lambda layer, x: layer("b", x, low_rank_dim=2),
    "LowRankInteractionLayer": lambda layer, x: layer(
        "b", x, x[:, :2], low_rank_dim=2
    ),
    "InteractionLayer": lambda layer, x: layer("b", x),
    "BilinearLayer": lambda layer, x: layer("b", x, x[:, :2]),
    "LowRankBilinearLayer": lambda layer, x: layer(
        "b", x, x[:, :2], low_rank_dim=2
    ),
    "EmbeddingLayer": lambda layer, x: layer(
        "b", (x[:, 0] > 0).astype(int), num_categories=2, embedding_dim=1
    ),
    "RandomEffectsLayer": lambda layer, x: layer(
        "b", (x[:, 0] > 0).astype(int), num_categories=2
    ),
    "FixedEffectsLayer": lambda layer, x: layer(
        "b", (x[:, 0] > 0).astype(int), num_categories=2
    ),
    "RandomWalkLayer": lambda layer, x: layer(
        "b", (x[:, 0] > 0).astype(int), num_categories=2, embedding_dim=1
    ),
    "HorseshoeLayer": lambda layer, x: layer("b", x),
    "HorseshoeInteractionLayer": lambda layer, x: layer("b", x),
    "MixtureLayer": lambda layer, x: layer("b", x),
    "HSGPLayer": lambda layer, x: layer("b", x[:, 0], L=2.0, m=4),
}


def test_all_builtin_layers_have_inference_coverage():
    builtin = {
        name
        for name, cls in vars(layers).items()
        if isinstance(cls, type)
        and issubclass(cls, layers.BLayer)
        and cls is not layers.BLayer
    }
    assert set(CALLS) == builtin
    assert all(
        getattr(blayers, name) is getattr(layers, name) for name in builtin
    )


@pytest.mark.parametrize("layer_name", CALLS)
@pytest.mark.parametrize(
    "method,batch_size", [("vi", None), ("vi", 3), ("mcmc", None)]
)
def test_layer_supports_vi_and_hmc(layer_name, method, batch_size):
    layer = getattr(layers, layer_name)()

    def model(x, y=None):
        mu = CALLS[layer_name](layer, x)
        mu = jnp.broadcast_to(mu, (x.shape[0], 1))
        return gaussian_link(mu, y, scale=1.0)

    x = jnp.array(
        [
            [-0.8, 0.2, 0.4],
            [0.4, -0.3, 0.1],
            [-0.2, 0.6, -0.1],
            [0.3, 0.2, 0.7],
            [-0.4, -0.1, 0.3],
            [0.7, 0.5, -0.2],
            [-0.1, 0.4, 0.2],
            [0.2, -0.6, -0.5],
        ]
    )
    result = fit(
        model,
        x=x,
        y=0.2 * x[:, 0],
        method=method,
        batch_size=batch_size,
        num_steps=12,
        num_warmup=20,
        num_mcmc_samples=20,
    )
    if method == "vi":
        assert jnp.isfinite(result.losses).all()
    else:
        assert isinstance(result.mcmc.sampler, infer.NUTS)
        assert all(
            jnp.isfinite(value).all()
            for value in result.posterior_samples.values()
        )
    predictions = result.predict(x=x[:2], num_samples=10)
    assert predictions.mean.shape == (2,)
    assert jnp.isfinite(predictions.samples).all()
