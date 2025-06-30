# type: ignore

from typing import Any

import pytest

from blayers.experimental.syntax import SymbolFactory, SymbolicLayer, bl
from blayers.layers import AdaptiveLayer


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
    data: Any,
    model_bundle: Any,
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
    model_bundle: Any,
    data: Any,
) -> None:
    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayer())

    _, coef_groups = model_bundle

    formula = f.y == a(f.x1)  #  + a(f.x2 + f.x1) * a(f.x3 | f.x1)

    def model(data):
        return formula(data)
