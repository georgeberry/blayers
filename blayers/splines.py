"""B-spline utilities for non-linear feature transformations.

Typical usage::

    from blayers.splines import make_knots, bspline_basis
    from blayers.layers import AdaptiveLayer
    from blayers.links import gaussian_link

    knots = make_knots(x_train, num_knots=10)

    def model(x, y=None):
        B = bspline_basis(x, knots)
        f = AdaptiveLayer()("f", B)
        return gaussian_link(f, y)
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def bspline_basis(x: jax.Array, knots: jax.Array, degree: int = 3) -> jax.Array:
    """Compute the B-spline design matrix via Cox–de Boor recursion (JAX-compatible).

    Args:
        x: 1D input array of shape ``(n,)``.
        knots: Full clamped knot vector of shape ``(num_basis + degree + 1,)``.
            Use ``make_knots`` to construct this.
        degree: B-spline degree (3 = cubic).

    Returns:
        jax.Array of shape ``(n, num_basis)`` where
        ``num_basis = len(knots) - degree - 1``.
    """
    x_col = x[:, None]  # (n, 1) for broadcasting against knot intervals

    # Degree-0 base case: B_{i,0}(t) = 1 if knots[i] <= t < knots[i+1].
    # We use a half-open interval [left, right) everywhere; the right boundary
    # special case (x == knots[-1]) is corrected after the recursion.
    left = knots[:-1]  # (num_knots - 1,)
    right = knots[1:]  # (num_knots - 1,)
    B = jnp.where((x_col >= left) & (x_col < right), 1.0, 0.0)  # (n, num_knots-1)

    # Cox–de Boor recursion
    for p in range(1, degree + 1):
        m = knots.shape[0] - p - 1  # number of basis functions at this level

        ti = knots[:m]  # t_i         (m,)
        ti_p = knots[p : m + p]  # t_{i+p}      (m,)
        d1 = ti_p - ti  # denominator of left term

        ti_p1 = knots[p + 1 : m + p + 1]  # t_{i+p+1}   (m,)
        ti_1 = knots[1 : m + 1]  # t_{i+1}     (m,)
        d2 = ti_p1 - ti_1  # denominator of right term

        # Avoid division by zero (degenerate knot intervals → coefficient = 0)
        alpha = jnp.where(d1 > 0, (x_col - ti) / jnp.where(d1 > 0, d1, 1.0), 0.0)
        beta_c = jnp.where(
            d2 > 0, (ti_p1 - x_col) / jnp.where(d2 > 0, d2, 1.0), 0.0
        )

        B = alpha * B[:, :m] + beta_c * B[:, 1 : m + 1]

    # For clamped splines, x == knots[-1] (the right boundary) must evaluate
    # to 1 on the last basis function.  The half-open base-case convention
    # misses this point because the rightmost repeated boundary intervals are
    # all degenerate ([t_max, t_max)), so we fix it here after the recursion.
    at_right = (x == knots[-1])  # (n,)
    last_basis = jnp.zeros_like(B).at[:, -1].set(1.0)  # (n, num_basis), 1 in last col
    B = jnp.where(at_right[:, None], last_basis, B)

    return B  # (n, num_basis)


def make_knots(x: Any, num_knots: int, degree: int = 3) -> jax.Array:
    """Compute a clamped B-spline knot vector from data.

    Interior knots are placed at evenly-spaced quantiles of ``x``.  Call
    this once at preprocessing time (outside any JAX-traced function) and
    pass the returned array to ``bspline_basis``.

    Args:
        x: Reference data (any shape).  Only used for quantile computation.
        num_knots: Number of interior knots.  The total number of basis
            functions will be ``num_knots + degree + 1``.
        degree: B-spline degree (default 3 for cubic splines).

    Returns:
        Full clamped knot vector as a ``jax.Array`` of shape
        ``(num_knots + 2 * (degree + 1),)``.

    Example::

        knots = make_knots(x_train, num_knots=5)
        B = bspline_basis(x, knots)
    """
    x_np = np.asarray(x).ravel()
    x_min, x_max = float(x_np.min()), float(x_np.max())

    if num_knots > 0:
        quantiles = np.linspace(0.0, 1.0, num_knots + 2)[1:-1]
        interior = np.quantile(x_np, quantiles)
    else:
        interior = np.array([], dtype=float)

    full_knots = np.concatenate(
        [
            np.full(degree + 1, x_min),
            interior,
            np.full(degree + 1, x_max),
        ]
    )
    return jnp.array(full_knots)
