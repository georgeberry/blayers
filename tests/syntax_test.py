# type: ignore

from typing import Any

import jax
import pytest
import pytest_check
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDiagonalNormal

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


def test_formula_fail() -> None:
    class AdaptiveLayerMock:
        def __call__(self, x):
            return f"{x}"

    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayerMock())

    with pytest_check.check.raises(TypeError):
        f.y % f.x1 % f.x2

    with pytest_check.check.raises(TypeError):
        f.y % a(f.x1 * f.x2) % f.x2


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

    # all of the math operators get evaluted before the <= operator so
    # the <= operator will always go last
    # order is PEDM(Modulus)AS -> bitwise -> comparison
    # so we want to keep our expression to the first group, then bitwise
    # can concat arrays with |, then comparison does assignment and formula
    # building
    formula = f.y <= a(f.x1) + a(f.x1 + f.x1) * a(f.x1 | f.x1)

    def model(data):
        return formula(data)

    key = jax.random.PRNGKey(2)
    predictive = Predictive(model, num_samples=1)
    prior_samples = predictive(key, data=data)

    assert len(prior_samples) == 8 and (
        prior_samples[
            "AdaptiveLayer_AdaptiveLayer_Add(DeferredArray(x1), DeferredArray(x1))_beta"
        ]
    ).shape == (1, 2)


@pytest.mark.parametrize(
    ("model_bundle", "data"),
    [
        ("linear_regression_adaptive_model", "simulated_data_simple"),
    ],
    indirect=True,
)
def test_fit(
    model_bundle: Any,  # noqa
    data: Any,  # noqa
) -> None:
    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayer())

    _, coef_groups = model_bundle

    # all of the math operators get evaluted before the <= operator so
    # the <= operator will always go last
    # order is PEDM(Modulus)AS -> bitwise -> comparison
    # so we want to keep our expression to the first group, then bitwise
    # can concat arrays with |, then comparison does assignment and formula
    # building

    model_data = {k: v for k, v in data.items() if k in ("y", "x1")}
    formula = f.y <= a(f.x1) + a(f.x1)

    def model(data):
        return formula(data)

    guide = AutoDiagonalNormal(model)

    key = jax.random.PRNGKey(2)
    predictive = Predictive(guide, num_samples=1)
    guide_samples = predictive(key, data=model_data)

    predictive = Predictive(model, num_samples=1)
    model_samples = predictive(key, data=model_data)

    guide_samples = bl(
        formula=formula,
        data=model_data,
    )

    import ipdb

    ipdb.set_trace()
