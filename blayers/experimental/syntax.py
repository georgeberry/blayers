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


deferred.__call__ --> now
"""

import operator

import jax
import jax.numpy as jnp
import optax
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal

from blayers.links import gaussian_link_exp

FOUR_SPACES = "    "

# ---- Deferred ops ---------------------------------------------------------- #


class DeferredBinaryOp:
    """Defers and then calls op(left_now, right_now)"""

    def __init__(self, left_deferred, right_deferred, op, symbol):
        self.left_deferred = left_deferred
        self.right_deferred = right_deferred
        self.op = op
        self.symbol = symbol

    def __call__(self, data):
        # the results from left_deferred and right_deferred must be composable
        # via `op` or this fails

        left_now = self.left_deferred(data)

        right_now = self.right_deferred(data)
        return self.op(left_now, right_now)

    def __repr__(self):
        return f"{self.symbol}({self.left_deferred}, {self.right_deferred})"

    def pretty(self, indent=0):
        s = FOUR_SPACES * indent + f"{self.symbol}(\n"
        s += self.left_deferred.pretty(indent + 1) + ",\n"
        s += self.right_deferred.pretty(indent + 1) + "\n"
        s += FOUR_SPACES * indent + ")"
        return s


class Sum(DeferredBinaryOp):
    def __init__(self, left, right):
        super().__init__(left, right, operator.add, "Add")


class Prod(DeferredBinaryOp):
    def __init__(self, left, right):
        super().__init__(left, right, operator.mul, "Prod")


class Power(DeferredBinaryOp):
    def __init__(self, left, right):
        super().__init__(left, right, operator.pow, "Pow")


class Concat(DeferredBinaryOp):
    def __init__(self, left, right):
        super().__init__(
            left,
            right,
            lambda x, y: jnp.concat([x, y], axis=1),
            "Concat",
        )


# ---- Higher level deferred stuff ------------------------------------------- #


class Formula:
    def __init__(self, lhs, rhs):
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

    def __repr__(self):
        return f"{self.lhs} ~ {self.rhs}"


class DeferredArray:
    def __init__(self, name):
        self.name = name

    def __call__(self, data):
        return data[self.name]

    def __add__(self, other):
        return Sum(self, other)

    def __or__(self, other):
        return Concat(self, other)

    def __eq__(self, other):
        return Formula(lhs=self, rhs=other)

    def __le__(self, other):
        return Formula(lhs=self, rhs=other)

    def __repr__(self):
        return f"DeferredArray({self.name})"

    def __str__(self):
        return "".join(c for c in self.__repr__() if ord(c) < 128)


class DeferredLayer:
    def __init__(self, layer, deferred):
        self.layer = layer
        self.deferred = deferred

    def __call__(self, data):
        name = f"{self.layer.__class__.__name__}_{str(self.deferred)}"
        dt = self.deferred(data)
        return self.layer(name, dt)

    def __repr__(self):
        return f"{self.layer.__class__.__name__}({self.deferred})"

    def pretty(self, indent=0):
        return FOUR_SPACES * indent + f"DeferredLayer({self.deferred})"

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Prod(self, other)


class SymbolicLayer:
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, deferred):
        return DeferredLayer(self.layer, deferred)


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
