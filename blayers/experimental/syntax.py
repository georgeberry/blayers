# type: ignore

"""Deferred computation for easier model building.

The key insight here is that we build up a stored comptuation graph composed
of Layers and Arrays, and then we pass data to the stored computation graph and
it knows what to do.

This means we have to define the set of operations we want to support for Layers
and Arrays in advance.

Abstractly, at the end of the day we want an Array, so everything needs to be
resolvable to arrays.

You can think of this in two parts, instance creation creates the deferred
computation graph, and the call method accepts actual data and resolves the
deferred computation to a real JAX Array.

So let's focus on a specific thing

a(f.x1 + f.x2) * a(f.x1 | f.x2)

What's going to happen here is we go from right to left so

```
Prod(
  AdaptiveLayer(
    Sum(
      f.x1,
      f.x2
    )
  ),
  AdaptiveLayer(
    Concat(
      f.x1,
      f.x2
    )
  )
```

deferred.__call__ --> now
"""

import itertools
import logging
import operator
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import optax
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal

from blayers.layers import (
    AdaptiveLayer,
    EmbeddingLayer,
    FMLayer,
    InterceptLayer,
    LowRankInteractionLayer,
    RandomEffectsLayer,
    RandomWalkLayer,
)
from blayers.links import gaussian_link_exp

logger = logging.getLogger("syntax")

_uid_counter = itertools.count()


def _next_uid():
    return next(_uid_counter)


_FOUR_SPACES = "    "


class Deferred(ABC):
    pass


def _now(x, data):
    if isinstance(x, Deferred):
        return x(data)
    return x


# ---- Deferred ops ---------------------------------------------------------- #


class DeferredBinaryOp(Deferred):
    """Defers and then calls op(left_now, right_now)"""

    def __init__(self, left_deferred, right_deferred, op, symbol):
        self.left_deferred = left_deferred
        self.right_deferred = right_deferred
        self.op = op
        self.symbol = symbol

    def __call__(self, data=None):
        # the results from left_deferred and right_deferred must be composable
        # via `op` or this fails

        left_now = self.left_deferred(data)
        right_now = self.right_deferred(data)
        return self.op(left_now, right_now)

    def _mock_call(self):
        uid = _next_uid()
        left_now = self.left_deferred._mock_call()
        right_now = self.right_deferred._mock_call()
        call_stack = f"{uid}_{self.symbol}({left_now}, {right_now})"
        return call_stack

    def __repr__(self):
        return f"{self.symbol}({self.left_deferred}, {self.right_deferred})"

    def __bool__(self):
        raise ValueError(
            "Ops cannot be used in a boolean context. Avoid chained comparisons like 'f.y <= f.x1 <= f.x2'."
        )

    def pretty(self, indent=0):
        s = _FOUR_SPACES * indent + f"{self.symbol}(\n"
        s += self.left_deferred.pretty(indent + 1) + ",\n"
        s += self.right_deferred.pretty(indent + 1) + "\n"
        s += _FOUR_SPACES * indent + ")"
        return s

    def __or__(self, other):
        return Concat(self, other)

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Prod(self, other)


class Sum(DeferredBinaryOp):
    def __init__(self, left, right):
        super().__init__(left, right, operator.add, "Add")


class Prod(DeferredBinaryOp):
    def __init__(self, left, right):
        super().__init__(left, right, operator.mul, "Prod")


class Concat(DeferredBinaryOp):
    def __init__(self, left, right):
        super().__init__(
            left,
            right,
            lambda x, y: jnp.concat([x, y], axis=1),
            "Concat",
        )


class DeferredManyOp(Deferred):
    """Defers and then calls op(left_now, right_now)"""

    def __init__(self, op, symbol, *args):
        self.deferred_args = args
        self.op = op
        self.symbol = symbol

    def __call__(self, data=None):
        return self.op([_now(x, data) for x in self.deferred_args])

    def _mock_call(self):
        uid = _next_uid()
        m = [x._mock_call() for x in self.deferred_args]
        call_stack = f"{uid}_{self.symbol}({m})"
        return call_stack

    def __repr__(self):
        return f"{self.symbol}({[x for x in self.deferred_args]})"

    def __bool__(self):
        raise ValueError(
            "Ops cannot be used in a boolean context. Avoid chained comparisons like 'f.y <= f.x1 <= f.x2'."
        )

    def __or__(self, other):
        return Concat(self, other)

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Prod(self, other)


class ConcatMany(DeferredManyOp):
    def __init__(self, *args):
        super().__init__(
            lambda arrs: jnp.concat(arrs, axis=1),
            "Concat",
            *args,
        )


# ---- Higher level deferred stuff ------------------------------------------- #


class Formula:
    def __init__(self, lhs, rhs):
        logger.warning("Formulas are in alpha. Use with care!")
        assert isinstance(
            lhs, DeferredArray
        ), f"LHS of Formula must be a DeferredArray, got {type(lhs)}."
        assert isinstance(
            rhs,
            (
                DeferredArray,
                DeferredBinaryOp,
                DeferredLayer,
            ),
        ), f"RHS of Formula must be a DeferredArray, got {type(rhs)}."

        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, data, pred_only=False):
        if pred_only:
            gaussian_link_exp(
                y=None,
                y_hat=self.rhs(data),
            )

        return gaussian_link_exp(
            y=self.lhs(data),
            y_hat=self.rhs(data),
        )

    def _mock_call(self):
        uid = _next_uid()
        lhs_mock = self.lhs._mock_call()
        rhs_mock = self.rhs._mock_call()
        print(f"{uid}_{lhs_mock}<={rhs_mock}")

    def __repr__(self):
        return f"{self.lhs} <= {self.rhs}"

    def __bool__(self):
        raise ValueError(
            "Formulas cannot be used in a boolean context. Avoid chained comparisons like 'f.y <= f.x1 <= f.x2'."
        )

    def __le__(self, other):
        if isinstance(other, Formula):
            raise ValueError(
                "Invalid chained formula: RHS is already a Formula. Use parentheses or break into separate formulas."
            )
        if isinstance(self, Formula):
            raise ValueError(
                "Invalid chained formula: LHS is already a Formula. Use parentheses or break into separate formulas."
            )
        return Formula(lhs=self, rhs=other)


class DeferredArray(Deferred):
    def __init__(self, name):
        self.name = name

    def __call__(self, data):
        return data[self.name]

    def _mock_call(self):
        uid = _next_uid()
        call_stack = f"{uid}_{self.name}"
        print(call_stack)
        return call_stack

    def __add__(self, other):
        return Sum(self, other)

    def __or__(self, other):
        return Concat(self, other)

    def __mul__(self, other):
        return Prod(self, other)

    def __le__(self, other):
        return Formula(lhs=self, rhs=other)

    def __repr__(self):
        return f"DeferredArray({self.name})"

    def __str__(self):
        return "".join(c for c in self.__repr__() if ord(c) < 128)


# ---- Layers ---------------------------------------------------------------- #


class SymbolicLayer:
    def __init__(self, layer_instance):
        # this is an instance
        self.layer_instance = layer_instance

    @abstractmethod
    def __call__(self, *args, **kwargs):
        # pure passthrough, if you pass the wrong things the layer will error
        # on call
        return DeferredLayer(self.layer_instance, *args, **kwargs)


class DeferredLayer:
    def __init__(self, layer_instance, *args, **kwargs):
        self.layer_instance = layer_instance
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data=None):
        args_now = [_now(x, data) for x in self.args]
        kwargs_now = {k: _now(v, data) for k, v in self.kwargs.items()}

        name = f"{self.layer_instance.__class__.__name__}({data})"

        return self.layer_instance(name, *args_now, **kwargs_now)

    def _mock_call(self):
        uid = _next_uid()
        call_stack = f"{uid}_{self.layer_instance.__class__.__name__}({self.deferred._mock_call()})"
        print(call_stack)
        return call_stack

    def __repr__(self):
        return f"{self.layer_instance.__class__.__name__}()"

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Prod(self, other)


# ---- Pass thru for data ---------------------------------------------------- #


class SymbolFactory:
    def __getattr__(self, name: str):
        return DeferredArray(name)


# ---- Public facing --------------------------------------------------------- #


def bl(
    formula: Formula,
    data: dict[str, jax.Array],
    num_steps=20000,
):
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=num_steps,
        peak_value=5e-2,
        pct_start=0.1,
        div_factor=25,
    )

    def model_fn(data):
        return formula(data)

    rng_key = random.PRNGKey(2)

    guide = AutoDiagonalNormal(model_fn)

    svi = SVI(model_fn, guide, optax.adam(schedule), loss=Trace_ELBO())

    rng_key = random.PRNGKey(2)

    svi_result = svi.run(
        rng_key,
        num_steps,
        data,
    )
    guide_predicitive = Predictive(
        guide,
        params=svi_result.params,
        num_samples=1000,
    )
    guide_samples = guide_predicitive(
        random.PRNGKey(1),
        {k: v for k, v in data.items() if k != "y"},
    )

    return guide_samples


# -------- Default stuff ----------------------------------------------------- #


c = SymbolicLayer(InterceptLayer())
a = SymbolicLayer(AdaptiveLayer())

re = SymbolicLayer(RandomEffectsLayer())
emb = SymbolicLayer(EmbeddingLayer())

fm = SymbolicLayer(FMLayer())
lint = SymbolicLayer(LowRankInteractionLayer())

rw = SymbolicLayer(RandomWalkLayer())

f = SymbolFactory()
cat = ConcatMany
