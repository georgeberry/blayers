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
import jax.numpy as jnp


class DeferredBinaryOp:
    """Defers and then calls op(left_now, right_now)"""

    def __init__(self, left_deferred, right_deferred, op, symbol):
        self.left_deferred = left_deferred
        self.right_deferred = right_deferred
        self.op = op
        self.symbol = symbol

    def __call__(self, data, name_prefix=""):
        # the results from left_deferred and right_deferred must be composable
        # via `op` or this fails
        left_now = self.left_deferred(data, name_prefix + "l_")
        right_now = self.right_deferred(data, name_prefix + "r_")
        return self.op(left_now, right_now)

    def __repr__(self):
        return f"{self.symbol}({self.left_deferred}, {self.right_deferred})"

    def pretty(self, indent=0):
        s = "    " * indent + f"{self.symbol}(\n"
        s += self.left_deferred.pretty(indent + 1) + ",\n"
        s += self.right_deferred.pretty(indent + 1) + "\n"
        s += "    " * indent + ")"
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
        super().__init__(left, right, jnp.concat, "Concat")


class DeferredArray:
    def __init__(self, name):
        self.name = name

    def __call__(self, data):
        return data[self.name]

    def __add__(self, other):
        return Sum(self, other)

    def __or__(self, other):
        return Concat(self, other)

    def __repr__(self):
        return f"DeferredArray({self.name})"


class DeferredLayer:
    def __init__(self, layer, deferred):
        self.layer = layer
        self.deferred = deferred

    def __call__(self, data, name_prefix=""):
        return self.layer(self.deferred(data))

    def __repr__(self):
        return f"{self.layer.__class__.__name__}({self.deferred})"

    def pretty(self, indent=0):
        return "    " * indent + f"DeferredLayer({self.deferred})"

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