from blayers.experimental.syntax import SymbolicLayer, SymbolFactory
from blayers.layers import AdaptiveLayer
from .layers_test import linear_regression_adaptive_model, simulated_data_simple
import pytest
from typing import Any

def test_ast() -> None:
    class AdaptiveLayerMock:
        def __call__(self, x):
            return f"{x}"

    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayerMock())


    expr = a(f.x1) + a(f.x2 + f.x1) * a(f.x3 | f.x1)
    
    assert expr.pretty() == """Add(
    DeferredLayer(DeferredArray(x1)),
    Prod(
        DeferredLayer(Add(DeferredArray(x2), DeferredArray(x1))),
        DeferredLayer(Concat(DeferredArray(x3), DeferredArray(x1)))
    )
)"""

@pytest.mark.parametrize(
    ("model_bundle", "data"),
    [
        ("linear_regression_adaptive_model", "simulated_data_simple"),
    ]
)
def test_syntax_model(
    data: Any,
    model_bundle: Any,
) -> Any:
    f = SymbolFactory()
    a = SymbolicLayer(AdaptiveLayer())

    bl(
        depvar=f.y,
        rhs=a(f.x1) + a(f.x2 + f.x1) * a(f.x3 | f.x1)
        data=data,
    )