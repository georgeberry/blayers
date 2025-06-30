# type: ignore

from typing import Any

import jax
import pytest
from numpyro.infer import Predictive

from blayers.experimental.syntax import SymbolFactory, SymbolicLayer, bl
from blayers.layers import AdaptiveLayer
from tests.layers_test import (  # noqa
    data,
    linear_regression_adaptive_model,
    model_bundle,
    simulated_data_simple,
)


def test_ast() -> None:
    class AdaptiveLayerMock:
        def __call__(self, x):
            return f"{x}"

    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayerMock())

    expr = a(f.x1) + a(f.x2 + f.x1) * a(f.x3 | f.x1)

    assert (
        expr.pretty()
        == """Add(
    DeferredLayer(DeferredArray(x1)),
    Prod(
        DeferredLayer(Add(DeferredArray(x2), DeferredArray(x1))),
        DeferredLayer(Concat(DeferredArray(x3), DeferredArray(x1)))
    )
)"""
    )


@pytest.mark.parametrize(
    ("model_bundle", "data"),
    [
        ("linear_regression_adaptive_model", "simulated_data_simple"),
    ],
    indirect=True,
)
def test_syntax_model(
    data: Any,  # noqa
    model_bundle: Any,  # noqa
) -> Any:
    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayer())

    _, coef_groups = model_bundle

    bl(
        f.y == a(f.x1) + a(f.x2 + f.x1) * a(f.x3 | f.x1),
        data=data,
    )


@pytest.mark.parametrize(
    ("model_bundle", "data"),
    [
        ("linear_regression_adaptive_model", "simulated_data_simple"),
    ],
    indirect=True,
)
def test_formula(
    model_bundle: Any,  # noqa
    data: Any,  # noqa
) -> None:
    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayer())

    _, coef_groups = model_bundle

    formula = f.y % a(f.x1)  # + a(f.x2 + f.x1) * a(f.x3 | f.x1)

    def model(data):
        return formula(data)

    key = jax.random.PRNGKey(2)

    predictive = Predictive(model, num_samples=1)
    prior_samples = predictive(key, data=data)
