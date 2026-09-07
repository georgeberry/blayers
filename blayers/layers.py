"""
Implements Bayesian Layers using Jax and Numpyro.

Design:
  - There are three levels of complexity here: class-level, instance-level, and
    call-level
  - The class-level handles things like choosing generic model form and how to
    multiply coefficents with data. Defined by the ``class Layer(BLayer)`` def
    itself.
  - The instance-level handles specific distributions that fit into a generic
    model and the initial parameters for those distributions. Defined by
    creating an instance of the class: ``Layer(*args, **kwargs)``.
  - The call-level handles seeing a batch of data, sampling from the
    distributions defined on the class and multiplying coefficients and data to
    produce an output, works like ``result = Layer(*args, **kwargs)(data)``

Notation:
  - ``n``: observations in a batch
  - ``c``: number of categories of things for time, random effects, etc
  - ``d``: number of coefficients
  - ``l``: low rank dimension of low rank models
  - ``m``: embedding dimension
  - ``u``: units aka output dimension
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from numpyro import distributions, sample

from blayers._utils import add_trailing_dim
from blayers.splines import bspline_basis

# ---- Matmul functions ------------------------------------------------------ #


def pairwise_interactions(x: jax.Array, z: jax.Array) -> jax.Array:
    """
    Compute all pairwise interactions between features in ``x`` and ``z``.

    Args:
        x: Input matrix of shape ``(n, d1)``.
        z: Input matrix of shape ``(n, d2)``.

    Returns:
        jax.Array of shape ``(n, d1 * d2)`` containing the flattened outer
        product ``x[:, i] * z[:, j]`` for each pair ``(i, j)``.
    """

    n, d1 = x.shape
    _, d2 = z.shape
    return jnp.reshape(x[:, :, None] * z[:, None, :], (n, d1 * d2))


def _matmul_dot_product(x: jax.Array, beta: jax.Array) -> jax.Array:
    """Standard dot product between beta and x.

    Args:
        beta: Coefficient vector of shape `(d, u)`.
        x: Input matrix of shape `(n, d)`.

    Returns:
        jax.Array: Output of shape `(n, u)`.
    """
    return jnp.einsum("nd,du->nu", x, beta)


def _matmul_factorization_machine(x: jax.Array, theta: jax.Array) -> jax.Array:
    """Apply second-order factorization machine interaction.

    Based on Rendle (2010). Computes:

    .. math::
        0.5 * sum((xV)^2 - (x^2 V^2))

    Args:
        theta: Weight matrix of shape `(d, l, u)`.
        x: Input data of shape `(n, d)`.

    Returns:
        jax.Array: Output of shape `(n, u)`.
    """
    vx2 = jnp.einsum("nd,dlu->nlu", x, theta) ** 2
    v2x2 = jnp.einsum("nd,dlu->nlu", x**2, theta**2)
    return 0.5 * jnp.einsum("nlu->nu", vx2 - v2x2)


def _matmul_fm3(x: jax.Array, theta: jax.Array) -> jax.Array:
    """Apply third-order factorization machine interaction.

    Computes all triple-product interactions via Newton's identities
    (Blondel et al. 2016).  Defining the per-rank power sums
    :math:`p_k = \\sum_i x_i^k \\theta_i^k`:

    .. math::
        \\text{output} = \\sum_l \\frac{p_1^3 - 3 p_2 p_1 + 2 p_3}{6}

    This computes all :math:`\\binom{d}{3}` triplet interactions without
    enumerating them.

    Args:
        theta: Weight matrix of shape `(d, l, u)`.
        x: Input data of shape `(n, d)`.

    Returns:
        jax.Array: Output of shape `(n, u)`.
    """
    # x: (n_features,)
    # E: (n_features, k)  embedding matrix
    linear_sum = jnp.einsum("nd,dlu->nlu", x, theta)  # jnp.dot(x, theta)
    square_sum = jnp.einsum(
        "nd,dlu->nlu", x**2, theta**2
    )  # jnp.dot(x**2, theta**2)
    cube_sum = jnp.einsum(
        "nd,dlu->nlu", x**3, theta**3
    )  # jnp.dot(x**3, theta**3)

    term = (
        linear_sum**3 - 3.0 * square_sum * linear_sum + 2.0 * cube_sum
    ) / 6.0
    return jnp.einsum("nlu->nu", term)  # scalar


def _matmul_uv_decomp(
    theta1: jax.Array,
    theta2: jax.Array,
    x: jax.Array,
    z: jax.Array,
) -> jax.Array:
    """Low-rank factorised bilinear interaction between ``x`` and ``z``.

    Projects each input into a shared ``l``-dimensional space via ``theta1``
    and ``theta2``, then computes the element-wise product summed over the
    low-rank axis.  Equivalent to a rank-``l`` approximation of the full
    bilinear form ``x^T (theta1 theta2^T) z``.

    Args:
        theta1: Weight matrix of shape `(d1, l, u)`.
        theta2: Weight matrix of shape `(d2, l, u)`.
        x: Input data of shape `(n, d1)`.
        z: Input data of shape `(n, d2)`.

    Returns:
        jax.Array: Output of shape `(n, u)`.
    """
    xb = jnp.einsum("nd,dlu->nlu", x, theta1)
    zb = jnp.einsum("nd,dlu->nlu", z, theta2)
    return jnp.einsum("nlu->nu", xb * zb)


def _matmul_randomwalk(
    theta: jax.Array,
    idx: jax.Array,
) -> jax.Array:
    """Vertical cumsum and then picks out index.

    We do a vertical cumsum of `theta` across `m` embedding dimensions, and then
    pick out the index.

    Args:
        theta: Weight matrix of shape `(c, m)`
        idx: Integer indexes of shape `(n, 1)` or `(n,)` with indexes up to `c`

    Returns:
        jax.Array: Output of shape `(n, m)`

    """
    theta_cumsum = jnp.cumsum(theta, axis=0)
    idx_flat = idx.reshape(-1).astype(jnp.int32)
    return theta_cumsum[idx_flat]


def _matmul_interaction(
    beta: jax.Array,
    x: jax.Array,
    z: jax.Array,
) -> jax.Array:
    """Full pairwise interaction between ``x`` and ``z``.

    Builds the flattened outer product of ``x`` and ``z`` and contracts it
    against a per-pair weight matrix.

    Args:
        beta: Weight matrix of shape ``(d1 * d2, u)``.
        x: Input matrix of shape ``(n, d1)``.
        z: Input matrix of shape ``(n, d2)``.

    Returns:
        jax.Array of shape ``(n, u)``.
    """
    interactions = pairwise_interactions(x, z)

    return jnp.einsum("nd,du->nu", interactions, beta)


# ---- Classes --------------------------------------------------------------- #


def _validate_prior_kwargs(
    coef_dist: type[distributions.Distribution],
    coef_kwargs: dict[str, Any],
    scale_dist: type[distributions.Distribution] | None = None,
    scale_kwargs: dict[str, Any] | None = None,
) -> None:
    """Eagerly instantiate distributions at construction time to catch bad kwargs.

    Raises ``TypeError`` immediately if the supplied kwargs are incompatible
    with the distribution, rather than waiting until the layer is called.
    """
    try:
        if scale_dist is not None:
            assert scale_kwargs is not None
            scale_dist(**scale_kwargs)
            coef_dist(scale=1.0, **coef_kwargs)
        else:
            coef_dist(**coef_kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid distribution kwargs: {e}") from e


class BLayer(ABC):
    """Abstract base class for Bayesian layers. Lays out an interface."""

    @abstractmethod
    def __init__(self, *args: Any) -> None:
        """Initialize layer parameters. This is the Bayesian model."""

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        """
        Run the layer's forward pass.

        Args:
            *args: Inputs to the layer.

        Returns:
            jax.Array: The result of the forward computation.
        """


class AdaptiveLayer(BLayer):
    """Bayesian layer with adaptive prior using hierarchical modeling.

    Generates coefficients from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        \\beta \\sim Normal(0., \\lambda)
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: NumPyro distribution class for the scale (λ) of the
                prior.
            coef_dist: NumPyro distribution class for the coefficient prior.
            coef_kwargs: Parameters for the prior distribution.
            scale_kwargs: Parameters for the scale distribution.
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Forward pass with adaptive prior on coefficients.

        Args:
            name: Variable name.
            x: Input data array of shape ``(n, d)``.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """

        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [input_shape, units]
            ),
        )

        # matmul and return
        return activation(_matmul_dot_product(x, beta))


class FixedPriorLayer(BLayer):
    """Bayesian layer with a fixed prior distribution over coefficients.

    Generates coefficients from the model

    .. math::

        \\beta \\sim Normal(0., 1.)
    """

    def __init__(
        self,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0, "scale": 1.0},
    ):
        """
        Args:
            coef_dist: NumPyro distribution class for the coefficients.
            coef_kwargs: Parameters to initialize the prior distribution.
        """
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Forward pass with fixed prior.

        Args:
            name: Variable name.
            x: Input data array of shape ``(n, d)``.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """

        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(**self.coef_kwargs).expand([input_shape, units]),
        )
        # matmul and return
        return activation(_matmul_dot_product(x, beta))


class InterceptLayer(BLayer):
    """Bayesian intercept (bias) term with a fixed prior.

    Samples a scalar bias from

    .. math::
        \\beta \\sim Normal(0., 1.)

    and broadcasts it to every observation. No input ``x`` is needed.
    """

    def __init__(
        self,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0, "scale": 1.0},
    ):
        """
        Args:
            ``coef_dist``: NumPyro distribution class for the coefficients.
            ``coef_kwargs``: Parameters to initialize the prior distribution.
        """
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs)

    def __call__(
        self,
        name: str,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Forward pass with fixed prior.

        Args:
            name: Variable name.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(1, u)``.
        """

        # sampling block
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(**self.coef_kwargs).expand([1, units]),
        )
        return activation(beta)


class FMLayer(BLayer):
    """Bayesian factorization machine layer with adaptive priors.

    Generates coefficients from the hierarchical model

    .. math::

        \\lambda \\sim HalfNormal(1.)

    .. math::

        \\beta \\sim Normal(0., \\lambda)

    The shape of ``beta`` is ``(j, l)``, where ``j`` is the number
    if input covariates and ``l`` is the low rank dim.

    Then performs matrix multiplication using the formula in `Rendle (2010) <https://jame-zhang.github.io/assets/algo/Factorization-Machines-Rendle2010.pdf>`_.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: Distribution for scaling factor λ.
            coef_dist: Prior for beta parameters.
            coef_kwargs: Arguments for prior distribution.
            scale_kwargs: Arguments for λ distribution.
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        low_rank_dim: int,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Forward pass through the factorization machine layer.

        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d)``.
            low_rank_dim: Dimensionality of low-rank approximation.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """
        # get shapes and reshape if necessary
        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [input_shape, low_rank_dim, units]
            ),
        )
        # matmul and return
        return activation(_matmul_factorization_machine(x, theta))


class FM3Layer(BLayer):
    """Bayesian order-3 factorization machine layer with adaptive prior.

    Samples low-rank factors from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        \\theta \\sim Normal(0., \\lambda), \\quad \\theta \\in \\mathbb{R}^{d \\times l}

    Then computes the 3rd-order ANOVA kernel via Newton's identity
    (`Blondel et al. 2016 <https://proceedings.neurips.cc/paper/2016/file/158fc2ddd52ec2cf54d3c161f2dd6517-Paper.pdf>`_).
    Defining power sums :math:`p_k = \\sum_i x_i^k \\theta_i^k`:

    .. math::
        \\text{output} = \\frac{p_1^3 - 3\\, p_2\\, p_1 + 2\\, p_3}{6}

    This efficiently computes all 3rd-order interaction terms without
    enumerating all :math:`\\binom{d}{3}` triples.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: Distribution for scaling factor λ.
            coef_dist: Prior for beta parameters.
            coef_kwargs: Arguments for prior distribution.
            scale_kwargs: Arguments for λ distribution.
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        low_rank_dim: int,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Forward pass through the factorization machine layer.

        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d)``.
            low_rank_dim: Dimensionality of low-rank approximation.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """
        # get shapes and reshape if necessary
        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [input_shape, low_rank_dim, units]
            ),
        )
        # matmul and return
        return activation(_matmul_fm3(x, theta))


class LowRankInteractionLayer(BLayer):
    """Bayesian low-rank bilinear interaction between two feature sets (UV decomposition).

    Samples separate low-rank projections for ``x`` and ``z`` from the
    hierarchical model

    .. math::
        \\lambda_1 \\sim HalfNormal(1.), \\quad
        \\theta_1 \\sim Normal(0., \\lambda_1), \\quad \\theta_1 \\in \\mathbb{R}^{d_1 \\times l}

    .. math::
        \\lambda_2 \\sim HalfNormal(1.), \\quad
        \\theta_2 \\sim Normal(0., \\lambda_2), \\quad \\theta_2 \\in \\mathbb{R}^{d_2 \\times l}

    and computes the element-wise product of the projections, summed over the
    low-rank dimension:

    .. math::
        \\text{output} = \\sum_{r=1}^{l} (x \\theta_1)_r \\cdot (z \\theta_2)_r
            = x^\\top (\\theta_1 \\theta_2^\\top) z

    This is equivalent to a rank-:math:`l` approximation of the full bilinear
    form :math:`x^\\top W z`.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: NumPyro distribution class for the scale (λ) of the
                prior.  Each input gets its own scale.
            coef_dist: NumPyro distribution class for the coefficient prior.
            coef_kwargs: Parameters for the prior distribution.
            scale_kwargs: Parameters for the scale distribution.
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        z: jax.Array,
        low_rank_dim: int,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Low-rank bilinear interaction ``x^T (theta1 theta2^T) z`` between X and Z.

        Projects ``x`` and ``z`` into a shared ``low_rank_dim``-dimensional
        space via independent factors ``theta1`` and ``theta2``, then
        contracts.

        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d1)``.
            z: Input matrix of shape ``(n, d2)``.
            low_rank_dim: Dimensionality of low-rank approximation.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """
        # get shapes and reshape if necessary
        x = add_trailing_dim(x)
        z = add_trailing_dim(z)
        input_shape1 = x.shape[1]
        input_shape2 = z.shape[1]

        # sampling block
        scale1 = sample(
            name=f"{self.__class__.__name__}_{name}_scale1",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        theta1 = sample(
            name=f"{self.__class__.__name__}_{name}_theta1",
            fn=self.coef_dist(scale=scale1, **self.coef_kwargs).expand(
                [input_shape1, low_rank_dim, units]
            ),
        )
        scale2 = sample(
            name=f"{self.__class__.__name__}_{name}_scale2",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        theta2 = sample(
            name=f"{self.__class__.__name__}_{name}_theta2",
            fn=self.coef_dist(scale=scale2, **self.coef_kwargs).expand(
                [input_shape2, low_rank_dim, units]
            ),
        )
        return activation(_matmul_uv_decomp(theta1, theta2, x, z))


class InteractionLayer(BLayer):
    """Bayesian pairwise interaction layer with adaptive prior.

    Samples one coefficient per pair of features from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.), \\quad \\beta \\sim Normal(0., \\lambda)

    and computes the weighted sum of the interaction design.

    Two modes:

    * **Within a single feature set** (``z`` omitted): the unique pairs
      :math:`x_i x_j` for :math:`i < j` — no squares, no duplicates —
      :math:`\\binom{d}{2}` coefficients, in lexicographic ``i < j`` order.
    * **Between two feature sets** (``z`` given): the full flattened outer product
      :math:`x \\otimes z` of shape :math:`(n, d_1 d_2)`.

    Scales as :math:`O(d^2)` parameters; prefer :class:`LowRankInteractionLayer`
    when :math:`d` is large, or :class:`HorseshoeInteractionLayer` for a sparse
    (variable-selecting) prior over the pairs.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        z: jax.Array | None = None,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Pairwise interaction design times a per-pair coefficient.

        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d1)``.
            z: Optional second feature set of shape ``(n, d2)``. If omitted, the
                interactions are the unique within-``x`` pairs ``i < j``.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """
        x = add_trailing_dim(x)
        if z is None:
            # within-set: unique pairs i < j (no squares, no duplicates)
            i, j = np.triu_indices(x.shape[1], k=1)
            x_int = x[:, i] * x[:, j]
        else:
            # cross-set: full d1 x d2 grid
            x_int = pairwise_interactions(x, add_trailing_dim(z))

        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale1",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta1",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [x_int.shape[1], units]
            ),
        )
        return activation(_matmul_dot_product(x_int, beta))


class BilinearLayer(BLayer):
    """Bayesian full bilinear layer with adaptive prior.

    Samples a full interaction matrix from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        W \\sim Normal(0., \\lambda), \\quad W \\in \\mathbb{R}^{d_1 \\times d_2}

    and computes the bilinear form:

    .. math::
        \\text{output} = x^\\top W z

    This learns a distinct weight for every pair :math:`(x_i, z_j)`, making
    it the densest two-input layer. Has :math:`O(d_1 d_2)` parameters;
    prefer :class:`LowRankBilinearLayer` when dimensions are large.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: prior on scale of coefficients
            coef_dist: distribution for coefficients
            coef_kwargs: kwargs for coef distribution
            scale_kwargs: kwargs for scale prior
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        z: jax.Array,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Full bilinear form ``x^T W z`` between feature matrices X and Z.

        Samples a dense weight tensor ``W`` of shape ``(d1, d2, units)`` and
        contracts it against ``x`` and ``z``.

        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d1)``.
            z: Input matrix of shape ``(n, d2)``.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """
        # ensure inputs are [batch, dim]
        x = add_trailing_dim(x)
        z = add_trailing_dim(z)
        input_shape1, input_shape2 = x.shape[1], z.shape[1]

        # sample coefficient scales
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        # full W: [input_shape1, input_shape2, units]
        W = sample(
            name=f"{self.__class__.__name__}_{name}_W",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [input_shape1, input_shape2, units]
            ),
        )
        # bilinear form: x^T W z for each unit
        return activation(jnp.einsum("ni,iju,nj->nu", x, W, z))


class LowRankBilinearLayer(BLayer):
    """Bayesian low-rank bilinear layer with adaptive prior.

    Samples shared-scale low-rank factors for both inputs from the
    hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        A \\sim Normal(0., \\lambda), \\quad A \\in \\mathbb{R}^{d_1 \\times l}

    .. math::
        B \\sim Normal(0., \\lambda), \\quad B \\in \\mathbb{R}^{d_2 \\times l}

    and computes the bilinear form with a rank-:math:`l` weight matrix
    :math:`W = AB^\\top`:

    .. math::
        \\text{output} = x^\\top W z = (xA) \\cdot (zB)

    Compared to :class:`LowRankInteractionLayer`, ``A`` and ``B`` share a
    single scale :math:`\\lambda`, tying the regularisation across both inputs.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: prior on scale of coefficients
            coef_dist: distribution for coefficients
            coef_kwargs: kwargs for coef distribution
            scale_kwargs: kwargs for scale prior
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        z: jax.Array,
        low_rank_dim: int,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Low-rank bilinear form ``x^T (A B^T) z``.

        Projects ``x`` and ``z`` into a shared ``low_rank_dim``-dimensional
        space via shared-scale factors ``A`` and ``B``, then contracts.

        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d1)``.
            z: Input matrix of shape ``(n, d2)``.
            low_rank_dim: Dimensionality of low-rank approximation.
            units: Number of outputs.
            activation: Activation function to apply to output.

        Returns:
            jax.Array: Output array of shape ``(n, u)``.
        """
        # ensure inputs are [batch, dim]
        x = add_trailing_dim(x)
        z = add_trailing_dim(z)
        input_shape1, input_shape2 = x.shape[1], z.shape[1]

        # sample coefficient scales
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )

        A = sample(
            name=f"{self.__class__.__name__}_{name}_A",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [input_shape1, low_rank_dim, units]
            ),
        )
        B = sample(
            name=f"{self.__class__.__name__}_{name}_B",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [input_shape2, low_rank_dim, units]
            ),
        )
        # project x and z into rank-r space, then take dot product
        x_proj = jnp.einsum("ni,ilu->nlu", x, A)  # [batch, rank, units]
        z_proj = jnp.einsum("nj,jlu->nlu", z, B)  # [batch, rank, units]
        out = jnp.sum(x_proj * z_proj, axis=1)  # [batch, units]

        return activation(out)


# ---- Embeddings ------------------------------------------------------------ #


class EmbeddingLayer(BLayer):
    """Bayesian embedding layer for sparse categorical features.

    Samples an embedding table from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        \\theta \\sim Normal(0., \\lambda), \\quad
        \\theta \\in \\mathbb{R}^{c \\times m}

    and performs a lookup for each observation:

    .. math::
        \\text{output}_i = \\theta[x_i]

    where :math:`c` is the number of categories and :math:`m` is the
    embedding dimension. For :math:`m = 1` prefer :class:`RandomEffectsLayer`.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: NumPyro distribution class for the scale (λ) of the
                prior.
            coef_dist: NumPyro distribution class for the coefficient prior.
            coef_kwargs: Parameters for the prior distribution.
            scale_kwargs: Parameters for the scale distribution.
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        num_categories: int,
        embedding_dim: int,
    ) -> jax.Array:
        """
        Forward pass through embedding lookup.

        Args:
            name: Variable name scope.
            x: Integer indices indicating embeddings to use.
            num_categories: The number of distinct things getting an embedding
            embedding_dim: The size of each embedding, e.g. 2, 4, 8, etc.

        Returns:
            jax.Array: Embedding vectors of shape ``(n, m)``.
        """

        # sampling block
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [num_categories, embedding_dim]
            ),
        )
        # matmul and return
        return jnp.asarray(theta[x.reshape(-1).astype(jnp.int32)])


class RandomEffectsLayer(BLayer):
    """Bayesian random-effects layer — a scalar embedding per category.

    Special case of :class:`EmbeddingLayer` with ``embedding_dim=1``.
    Samples one scalar random effect per category from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        \\theta \\sim Normal(0., \\lambda), \\quad \\theta \\in \\mathbb{R}^{c}

    and returns the scalar for each observation's category:

    .. math::
        \\text{output}_i = \\theta[x_i]

    Equivalent to a classical mixed-effects intercept with a learned
    variance :math:`\\lambda^2`.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            scale_dist: NumPyro distribution class for the scale (λ) of the
                prior.
            coef_dist: NumPyro distribution class for the coefficient prior.
            coef_kwargs: Parameters for the prior distribution.
            scale_kwargs: Parameters for the scale distribution.
        """
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        num_categories: int,
    ) -> jax.Array:
        """
        Forward pass through scalar random-effect lookup.

        Args:
            name: Variable name scope.
            x: Integer indices indicating which random effect to use.
            num_categories: The number of distinct random-effect groups.

        Returns:
            jax.Array: Random-effect values of shape ``(n, 1)``.
        """

        # sampling block
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [num_categories, 1]
            ),
        )
        return jnp.asarray(theta[x.reshape(-1).astype(jnp.int32)])


class RandomSlopesLayer(BLayer):
    """Independent, partially pooled group-specific slope deviations.

    For each predictor j and output u, sample a scale shared across groups::

        tau[j, u] ~ scale_dist(**scale_kwargs)
        beta[g, j, u] ~ Normal(0, tau[j, u])
        output[i, u] = sum_j x[i, j] * beta[groups[i], j, u]

    Population slopes belong in a separate layer. A column of ones in x adds
    a varying intercept. Coefficients are independent conditional on the scales;
    this layer does not estimate correlations between slopes.

    The centered generative prior follows other BLayers layers; fit() can
    non-center it through autoreparam_model=True for VI and HMC/NUTS.
    """

    def __init__(
        self,
        scale_dist: type[distributions.Distribution] = distributions.HalfNormal,
        scale_kwargs: dict[str, Any] | None = None,
    ):
        """Configure the between-group scale prior (default HalfNormal(1)).

        The scale prior must be continuous, positive, and scalar-valued;
        scalar or (d, units)-broadcastable parameters are supported.
        """
        self.scale_dist = scale_dist
        self.scale_kwargs = (
            {"scale": 1.0} if scale_kwargs is None else dict(scale_kwargs)
        )
        _validate_prior_kwargs(
            distributions.Normal, {"loc": 0.0}, scale_dist, self.scale_kwargs
        )
        prior = scale_dist(**self.scale_kwargs)
        if prior.is_discrete or prior.event_shape:
            raise ValueError(
                "scale_dist must be a continuous scalar distribution"
            )

    def __call__(
        self,
        name: str,
        x: jax.Array,
        groups: jax.Array,
        num_categories: int,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """Return slope deviations of shape (n, units).

        Args:
            name: Sample-site namespace.
            x: Predictor matrix (n, d), or vector (n,) for one slope.
            groups: Integer group IDs, shape (n,) or (n, 1), in
                [0, num_categories). IDs must retain their meaning at prediction.
            num_categories: Size of the full coefficient table. Keep fixed
                across training batches and prediction. Unobserved groups can
                occupy reserved slots, whose slopes retain their conditional
                prior; adding new slots after fitting is unsupported.
            units: Number of output dimensions, with independent slope priors.
            activation: Optional transform of the resulting predictor.

        Invalid concrete group IDs raise ValueError. Under JIT, invalid IDs
        produce NaN outputs instead of silently wrapping or clipping indices.
        """
        if num_categories <= 0 or units <= 0:
            raise ValueError("num_categories and units must be positive")
        x = add_trailing_dim(jnp.asarray(x))
        groups = jnp.asarray(groups)
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError(
                "x must have shape (n, d) with at least one predictor"
            )
        if groups.ndim == 2 and groups.shape[1] == 1:
            groups = groups[:, 0]
        if groups.ndim != 1 or groups.shape[0] != x.shape[0]:
            raise ValueError("groups must contain one group ID per row of x")
        if not jnp.issubdtype(groups.dtype, jnp.integer):
            raise TypeError("groups must contain integer group IDs")
        valid = (groups >= 0) & (groups < num_categories)
        if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
            raise ValueError("groups must be in [0, num_categories)")

        prefix = f"{self.__class__.__name__}_{name}"
        d = x.shape[1]
        scale = sample(
            f"{prefix}_scale",
            self.scale_dist(**self.scale_kwargs).expand([d, units]),
        )
        beta = sample(
            f"{prefix}_beta",
            distributions.Normal(0.0, scale).expand([num_categories, d, units]),
        )
        row_beta = jnp.where(valid[:, None, None], beta[groups], jnp.nan)
        return activation(jnp.einsum("nd,ndu->nu", x, row_beta))


class FixedEffectsLayer(BLayer):
    """Bayesian fixed-effects layer — per-category coefficients, fixed prior.

    The no-pooling counterpart of :class:`RandomEffectsLayer`: each category
    gets its own scalar coefficient drawn from a fixed, user-specified prior

    .. math::
        \\theta_c \\sim Normal(0., 1.), \\quad \\theta \\in \\mathbb{R}^{c}

    with **no learned variance component**. The learned scale is exactly what
    makes a random effect "random" (it drives the partial pooling); fixing the
    prior instead gives the Bayesian analogue of classical fixed effects —
    ridge-regularised per-category dummies, with the prior scale controlling
    the regularisation strength.

    Each observation gets the coefficient of its category:

    .. math::
        \\text{output}_i = \\theta[x_i]

    Prefer :class:`RandomEffectsLayer` when you have many small groups that
    should share information; use this layer when groups are few and
    well-observed, or when you explicitly do not want pooling.
    """

    def __init__(
        self,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0, "scale": 1.0},
    ):
        """
        Args:
            coef_dist: NumPyro distribution class for the coefficients.
            coef_kwargs: Parameters to initialize the prior distribution.
        """
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        num_categories: int,
    ) -> jax.Array:
        """
        Forward pass through scalar fixed-effect lookup.

        Args:
            name: Variable name scope.
            x: Integer indices indicating which effect to use.
            num_categories: The number of distinct groups.

        Returns:
            jax.Array: Effect values of shape ``(n, 1)``.
        """
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(**self.coef_kwargs).expand([num_categories, 1]),
        )
        return jnp.asarray(theta[x.reshape(-1).astype(jnp.int32)])


class RandomWalkLayer(BLayer):
    """Bayesian Gaussian random walk over ordered categories.

    Samples i.i.d. increments from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        \\delta_t \\sim Normal(0., \\lambda), \\quad t = 1, \\ldots, c

    and accumulates them into positions via a cumulative sum:

    .. math::
        \\theta_t = \\sum_{s=1}^{t} \\delta_s

    Each observation is then assigned the position of its category:

    .. math::
        \\text{output}_i = \\theta[x_i]

    The ``embedding_dim`` ``m`` runs ``m`` independent walks in parallel,
    producing output of shape ``(n, m)``. Typical use: a time index where
    adjacent periods share information through the walk prior.
    """

    def __init__(
        self,
        scale_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        scale_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        self.scale_dist = scale_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.scale_kwargs = scale_kwargs
        _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist, scale_kwargs)

    def __call__(
        self,
        name: str,
        x: jax.Array,
        num_categories: int,
        embedding_dim: int,
    ) -> jax.Array:
        """
        Forward pass through embedding lookup.

        Args:
            name: Variable name scope.
            x: Integer indices indicating embeddings to use.
            num_categories: The number of distinct things getting an embedding
            embedding_dim: The size of each embedding, e.g. 2, 4, 8, etc.

        Returns:
            jax.Array: Embedding vectors of shape ``(n, m)``.
        """

        # sampling block
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale",
            fn=self.scale_dist(**self.scale_kwargs),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [
                    num_categories,
                    embedding_dim,
                ]
            ),
        )
        # matmul and return
        return _matmul_randomwalk(theta, x)


class AR1Layer(BLayer):
    """Stationary, zero-mean AR(1) effects over equally spaced time slots.

    Independently for each output::

        p ~ Beta(rho_concentration, rho_concentration)
        rho = 2 * p - 1
        sigma ~ scale_dist(**scale_kwargs)
        z[t] ~ Normal(0, 1)
        theta[0] = sigma / sqrt(1 - rho**2) * z[0]
        theta[t] = rho * theta[t-1] + sigma * z[t]

    sigma is the innovation standard deviation, not the marginal standard
    deviation. The stationary initial prior gives covariance
    sigma**2 * rho**abs(t-s) / (1-rho**2). Add a separate InterceptLayer for
    the population mean. States are not sample-centered: that would change
    this stationary prior and its forecasting behavior.
    """

    def __init__(
        self,
        scale_dist: type[distributions.Distribution] = distributions.HalfNormal,
        scale_kwargs: dict[str, Any] | None = None,
        rho_concentration: float = 2.0,
        noncentered: bool = False,
    ):
        """Configure a positive innovation-scale prior and symmetric rho prior.

        rho_concentration=1 gives Uniform(-1, 1); the default 2 mildly favors
        persistence near zero over either stationarity boundary. Scale kwargs
        may broadcast to (units,). The default samples states directly. Set
        noncentered=True to sample standardized innovations instead, often
        useful when states are weakly observed. Both forms have continuous
        latents and support VI/NUTS. This explicit choice is not changed by
        fit(autoreparam_model=...).
        """
        if not np.isfinite(rho_concentration) or rho_concentration <= 0:
            raise ValueError("rho_concentration must be finite and positive")
        self.noncentered = noncentered
        self.rho_concentration = rho_concentration
        self.scale_dist = scale_dist
        self.scale_kwargs = (
            {"scale": 1.0} if scale_kwargs is None else dict(scale_kwargs)
        )
        prior = scale_dist(**self.scale_kwargs)
        if prior.is_discrete or prior.event_shape:
            raise ValueError(
                "scale_dist must be a continuous scalar distribution"
            )

    def __call__(
        self,
        name: str,
        x: jax.Array,
        num_categories: int,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """Look up states, returning (n, units).

        x contains integer time slots, shape (n,) or (n, 1), in
        [0, num_categories). Repeated and unsorted observations are supported.
        Keep the full time grid fixed across batches and prediction, including
        reserved future slots. Missing periods must retain their slots: do not
        compress gaps or treat irregularly spaced times as adjacent periods.
        Future states then follow the same recurrence with fresh innovations.
        Extending the grid after fitting is unsupported.

        Invalid concrete indices raise; invalid traced indices yield NaNs.
        """
        if num_categories <= 0 or units <= 0:
            raise ValueError("num_categories and units must be positive")
        x = jnp.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError("x must have shape (n,) or (n, 1)")
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise TypeError("x must contain integer time slots")
        valid = (x >= 0) & (x < num_categories)
        if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
            raise ValueError("x must be in [0, num_categories)")
        prefix = f"{self.__class__.__name__}_{name}"
        rho = sample(
            f"{prefix}_rho",
            distributions.TransformedDistribution(
                distributions.Beta(
                    self.rho_concentration, self.rho_concentration
                ),
                distributions.transforms.AffineTransform(-1.0, 2.0),
            ).expand([units]),
        )
        scale = sample(
            f"{prefix}_scale",
            self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        initial_scale = scale / jnp.sqrt((1.0 - rho) * (1.0 + rho))
        if self.noncentered:
            z = sample(
                f"{prefix}_z",
                distributions.Normal(0.0, 1.0).expand([num_categories, units]),
            )
            initial = initial_scale * z[0]

            def step(
                previous: jax.Array, innovation: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                current = rho * previous + scale * innovation
                return current, current

            _, remaining = jax.lax.scan(step, initial, z[1:])
            states = jnp.concatenate([initial[None, :], remaining], axis=0)
        else:
            innovation_scale = jnp.broadcast_to(scale, (num_categories, units))
            innovation_scale = innovation_scale.at[0].set(initial_scale)
            states = sample(
                f"{prefix}_theta",
                distributions.TransformedDistribution(
                    distributions.Normal(0.0, innovation_scale).to_event(2),
                    distributions.transforms.RecursiveLinearTransform(
                        jnp.diag(rho)
                    ),
                ),
            )
        return activation(jnp.where(valid[:, None], states[x], jnp.nan))


class PSplineLayer(BLayer):
    """Anchored B-spline smoother with a second-difference coefficient prior.

    If D is the second-difference matrix, use its minimum-norm right inverse R
    to construct coefficients::

        scale[u] ~ scale_dist(**scale_kwargs)
        differences[:, u] ~ Normal(0, scale[u])
        trend[u] ~ Normal(0, trend_scale)
        beta = R @ differences + linspace(-1, 1, K)[:, None] * trend
        f(x) = (B(clip(x)) - B(reference)) @ beta

    Thus D @ beta equals the sampled differences exactly. Smaller scales
    penalize roughness more strongly. The coefficient-linear null component
    has its own proper prior, independent of the smoothing scale. The constant
    null component is removed by anchoring f(reference)=0; add InterceptLayer
    separately. No batch-dependent centering is performed.

    This is a coefficient-difference P-spline, not a derivative penalty. With
    clamped/uneven knots the unpenalized coefficient trend need not be exactly
    linear in x (especially near the boundaries). It is included in this
    layer; adding a separate linear term can introduce confounding.
    """

    def __init__(
        self,
        scale_dist: type[distributions.Distribution] = distributions.HalfNormal,
        scale_kwargs: dict[str, Any] | None = None,
        trend_scale: float = 1.0,
        degree: int = 3,
    ):
        """Configure smoothing and unpenalized-trend priors.

        degree must be at least 1. Scale priors must be positive, continuous,
        scalar-valued distributions, with kwargs broadcastable to (units,).
        trend_scale is the prior SD of the coefficient trend's half-range.
        Smoothing scales depend on the number and placement of knots; choose
        the basis once, then check prior curves at that resolution.
        """
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be an integer >= 1")
        if not np.isfinite(trend_scale) or trend_scale <= 0:
            raise ValueError("trend_scale must be finite and positive")
        self.degree = degree
        self.trend_scale = trend_scale
        self.scale_dist = scale_dist
        self.scale_kwargs = (
            {"scale": 1.0} if scale_kwargs is None else dict(scale_kwargs)
        )
        prior = scale_dist(**self.scale_kwargs)
        if prior.is_discrete or prior.event_shape:
            raise ValueError(
                "scale_dist must be a continuous scalar distribution"
            )

    def __call__(
        self,
        name: str,
        x: jax.Array,
        knots: jax.Array,
        units: int = 1,
        reference: float | None = None,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """Return (n, units) smooth effects for a single predictor.

        x has shape (n,) or (n, 1). knots is a full clamped knot vector, e.g.
        from make_knots, defining at least three basis functions. Capture
        knots in your model closure, outside fit's row-wise data arguments.
        Reuse identical knots and reference for all batches and predictions.
        reference defaults to the domain midpoint and must lie in the domain.
        Outside the knot domain, the fitted curve holds its boundary value
        (constant extrapolation). Inputs and knots must be finite.
        """
        if units <= 0:
            raise ValueError("units must be positive")
        x = jnp.asarray(x)
        knots = jnp.asarray(knots)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError("x must have shape (n,) or (n, 1)")
        p = self.degree
        if knots.ndim != 1 or knots.shape[0] < 2 * (p + 1):
            raise ValueError("knots must be a full clamped knot vector")
        k = knots.shape[0] - p - 1
        if k < 3:
            raise ValueError("at least three basis functions are required")
        lo, hi = knots[0], knots[-1]
        ref = (lo + hi) / 2 if reference is None else jnp.asarray(reference)
        if jnp.ndim(ref) != 0:
            raise ValueError("reference must be scalar")
        valid_knots = (
            jnp.all(jnp.isfinite(knots))
            & jnp.all(jnp.diff(knots) >= 0)
            & (hi > lo)
            & jnp.all(knots[: p + 1] == lo)
            & jnp.all(knots[-p - 1 :] == hi)
            & jnp.all(knots[p + 1 : -p - 1] > lo)
            & jnp.all(knots[p + 1 : -p - 1] < hi)
            & (ref >= lo)
            & (ref <= hi)
        )
        if not isinstance(valid_knots, jax.core.Tracer) and not bool(
            valid_knots
        ):
            raise ValueError(
                "knots must be finite, ordered and clamped with a nonempty domain; reference must lie within it"
            )
        valid_x = jnp.isfinite(x)
        if not isinstance(valid_x, jax.core.Tracer) and not bool(
            jnp.all(valid_x)
        ):
            raise ValueError("x must be finite")
        basis = bspline_basis(jnp.clip(x, lo, hi), knots, p)
        basis = basis - bspline_basis(jnp.reshape(ref, (1,)), knots, p)
        # NumPy uses double precision for the small fixed basis decomposition;
        # only the resulting constant enters JAX's traced model.
        difference = np.diff(np.eye(k), n=2, axis=0)
        right_inverse = np.linalg.solve(difference @ difference.T, difference).T
        prefix = f"{self.__class__.__name__}_{name}"
        scale = sample(
            f"{prefix}_scale",
            self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        delta = sample(
            f"{prefix}_differences",
            distributions.Normal(0.0, scale).expand([k - 2, units]),
        )
        trend = sample(
            f"{prefix}_trend",
            distributions.Normal(0.0, self.trend_scale).expand([units]),
        )
        beta = (
            jnp.asarray(right_inverse) @ delta
            + jnp.linspace(-1.0, 1.0, k)[:, None] * trend
        )
        output = jnp.where(
            (valid_x & valid_knots)[:, None], basis @ beta, jnp.nan
        )
        return activation(output)


# ---- Sparse priors --------------------------------------------------------- #


class HorseshoeLayer(BLayer):
    """Bayesian layer with horseshoe prior for sparse regression.

    Implements the (regularized) horseshoe prior of Piironen & Vehtari (2017).

    Basic horseshoe:

    .. math::
        \\tau \\sim HalfCauchy(\\tau_0), \\quad
        \\lambda_j \\sim HalfCauchy(1), \\quad
        \\beta_j \\sim Normal(0,\\; \\tau \\lambda_j)

    Regularized horseshoe (``slab_scale`` set) — prevents large coefficients
    from escaping the slab:

    .. math::
        \\tilde{\\lambda}_j^2 = \\frac{c^2 \\lambda_j^2}{c^2 + \\tau^2 \\lambda_j^2},
        \\quad c^2 \\sim InverseGamma(s/2,\\; s/2 \\cdot scale_{slab}^2)
    """

    def __init__(
        self,
        tau0: float = 1.0,
        slab_scale: float | None = None,
        slab_df: float = 4.0,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
    ):
        """
        Args:
            tau0: Scale of the HalfCauchy prior on the GLOBAL shrinkage ``tau``.
                This is the knob that decides whether the layer selects. At
                ``p >> n`` the default of 1.0 is very loose: tau drifts to O(10)
                and, under a regularized horseshoe, the per-coefficient scale
                ``tau * sqrt(c^2 l^2 / (c^2 + tau^2 l^2))`` collapses to just
                ``c`` for any ``l >> c/tau`` — so the prior degenerates to
                ``Normal(0, slab_scale)`` and shrinks nothing. Piironen & Vehtari
                suggest ``tau0 ~ (p0 / (p - p0)) * sigma / sqrt(n)`` for an
                expected ``p0`` non-zero coefficients; in practice something like
                0.05 is a reasonable starting point for sparse selection.
            slab_scale: If set, uses the regularized horseshoe with this slab
                scale.  ``None`` gives the plain horseshoe.
            slab_df: Degrees of freedom for the slab variance prior (only
                used when ``slab_scale`` is set).
            coef_dist: Distribution for the coefficients. Must accept a
                ``scale`` keyword (derived from the horseshoe shrinkage).
                Defaults to ``Normal``.
            coef_kwargs: Extra kwargs for ``coef_dist`` (beyond ``scale``).
                Default ``{"loc": 0.0}``.
        """
        self.tau0 = tau0
        self.slab_scale = slab_scale
        self.slab_df = slab_df
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        try:
            coef_dist(scale=1.0, **coef_kwargs)
        except TypeError as e:
            raise TypeError(f"Invalid coef_dist kwargs: {e}") from e

    def __call__(
        self,
        name: str,
        x: jax.Array,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Forward pass with horseshoe prior on coefficients.

        Args:
            name: Variable name scope.
            x: Input array of shape ``(n, d)``.
            units: Number of output dimensions.
            activation: Activation function.

        Returns:
            jax.Array of shape ``(n, units)``.
        """
        x = add_trailing_dim(x)
        d = x.shape[1]
        cls = self.__class__.__name__

        # Global shrinkage: one scale per output unit
        tau = sample(
            f"{cls}_{name}_tau",
            distributions.HalfCauchy(self.tau0).expand([units]),
        )
        # Local shrinkage: one per feature per output unit
        scale = sample(
            f"{cls}_{name}_scale",
            distributions.HalfCauchy(1.0).expand([d, units]),
        )

        if self.slab_scale is not None:
            # Soft upper bound on coefficient size via a finite-variance slab
            c2 = sample(
                f"{cls}_{name}_c2",
                distributions.InverseGamma(
                    self.slab_df / 2.0,
                    self.slab_df / 2.0 * self.slab_scale**2,
                ),
            )
            scale_tilde = jnp.sqrt(
                c2 * scale**2 / (c2 + tau**2 * scale**2)
            )
            scale = tau * scale_tilde
        else:
            scale = tau * scale  # (d, units)

        # Centered coefficient: beta ~ coef_dist(0, scale).  This is Neal's
        # funnel (beta tightly coupled to the global tau).  NUTS handles it, so
        # MCMC identifies tau fine; mean-field VI fits the funnel poorly, so
        # prefer MCMC for horseshoe selection.  A manual non-centering here was
        # tried and reverted — it destabilised mean-field VI on high-dimensional
        # interaction bases (1000+ coefficients).
        beta = sample(
            f"{cls}_{name}_beta",
            self.coef_dist(scale=scale, **self.coef_kwargs),
        )
        return activation(_matmul_dot_product(x, beta))


class HorseshoeInteractionLayer(HorseshoeLayer):
    """Sparse pairwise interactions under a horseshoe prior.

    Builds an explicit interaction design and places a (regularized) horseshoe
    prior on the per-pair coefficients. Local shrinkage pulls most interactions to
    zero and leaves the few real ones standing, so this is the layer to reach for
    to **identify sparse interactions** (as opposed to :class:`InteractionLayer`'s
    single global scale, which cannot localize).

    Two modes:

    * **Within a single feature set** (``z`` omitted): the unique pairs ``x_i x_j``
      for ``i < j`` — no squares, no duplicates — ``C(d, 2)`` columns. Column ``k``
      is the ``k``-th pair in lexicographic ``i < j`` order (row-major upper
      triangle), so posterior coefficients map back to feature pairs.
    * **Between two feature sets** (``z`` given): the full ``d1 * d2`` outer product
      ``x_i z_j`` (like :class:`InteractionLayer`); column ``k`` is
      ``(i, j) = divmod(k, d2)``.

    Costs ``O(d^2)`` coefficients — for large inputs where you only need prediction,
    prefer :class:`LowRankInteractionLayer` or :class:`FMLayer`. Inherits its prior
    configuration (``slab_scale``, ``slab_df``, ``coef_dist``, ``coef_kwargs``) from
    :class:`HorseshoeLayer`.
    """

    def __call__(  # type: ignore[override]  # (x, z?) differs from HorseshoeLayer
        self,
        name: str,
        x: jax.Array,
        z: jax.Array | None = None,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d1)``.
            z: Optional second feature set of shape ``(n, d2)``. If omitted, the
                interactions are the unique within-``x`` pairs ``i < j``.
            units: Number of output dimensions.
            activation: Activation function.

        Returns:
            jax.Array of shape ``(n, units)``.
        """
        x = add_trailing_dim(x)
        if z is None:
            # within-set: unique pairs i < j (no squares, no duplicates)
            i, j = np.triu_indices(x.shape[1], k=1)
            x_int = x[:, i] * x[:, j]
        else:
            # cross-set: full d1 x d2 grid, like InteractionLayer
            x_int = pairwise_interactions(x, add_trailing_dim(z))
        return super().__call__(name, x_int, units=units, activation=activation)


# ---- Mixture priors -------------------------------------------------------- #


class MixtureLayer(BLayer):
    """Coefficients from a finite mixture-of-priors (e.g. Normal + Laplace).

    Each coefficient is drawn from a ``K``-component mixture

    .. math::
        \\beta_j \\sim \\sum_{k=1}^{K} w_k \\, p_k(\\cdot)

    where the mixing weights ``w`` are either fixed or given a ``Dirichlet``
    prior (shared across coefficients). The component indicator is marginalised
    analytically by :class:`numpyro.distributions.MixtureGeneral`, so the
    log-density is smooth and works under VI *and* MCMC.

    Useful for robustness (a heavy-tailed component absorbs a few outlier
    coefficients while the rest stay Gaussian) and elastic-net-flavoured priors
    (Normal + Laplace). For sparse shrinkage prefer :class:`HorseshoeLayer`.
    """

    def __init__(
        self,
        component_dists: tuple[type[distributions.Distribution], ...] = (
            distributions.Normal,
            distributions.Laplace,
        ),
        component_kwargs: tuple[dict[str, float], ...] = (
            {"loc": 0.0, "scale": 1.0},
            {"loc": 0.0, "scale": 1.0},
        ),
        weights: list[float] | None = None,
        weight_scale: float = 1.0,
    ):
        """
        Args:
            component_dists: NumPyro distribution classes, one per mixture
                component (>= 2). All must share the same (real) support.
            component_kwargs: Kwargs for each component distribution.
            weights: Fixed mixing weights (one per component, summing to 1). If
                ``None``, a **logistic-normal** prior is placed on the weights:
                ``softmax`` of ``Normal(0, weight_scale)`` logits. This keeps the
                weight latent in unconstrained space so the layer fits under VI,
                MCMC, *and* SVGD — a raw ``Dirichlet`` simplex site breaks SVGD's
                particle flattening (its unconstrained dimension differs from its
                constrained one).
            weight_scale: Prior standard deviation of the Normal logits used when
                ``weights`` is ``None``. Larger spreads the weights more.
        """
        if len(component_dists) != len(component_kwargs):
            raise ValueError(
                "component_dists and component_kwargs must have the same length"
            )
        if len(component_dists) < 2:
            raise ValueError("A mixture needs at least two components")
        if weights is not None and len(weights) != len(component_dists):
            raise ValueError("weights must have one entry per component")
        self.component_dists = component_dists
        self.component_kwargs = component_kwargs
        self.weights = weights
        self.weight_scale = weight_scale
        try:
            for dst, kw in zip(component_dists, component_kwargs):
                dst(**kw)
        except TypeError as e:
            raise TypeError(f"Invalid distribution kwargs: {e}") from e

    def __call__(
        self,
        name: str,
        x: jax.Array,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Args:
            name: Variable name scope.
            x: Input of shape ``(n, d)``.
            units: Number of output dimensions.
            activation: Activation function.

        Returns:
            jax.Array of shape ``(n, units)``.
        """
        x = add_trailing_dim(x)
        d = x.shape[1]
        k = len(self.component_dists)
        cls = self.__class__.__name__

        if self.weights is None:
            # Logistic-normal: softmax of unconstrained Normal logits. Avoids a
            # Dirichlet simplex site, which SVGD's particle flattener cannot
            # handle (constrained dim k != unconstrained dim k-1).
            logits = sample(
                f"{cls}_{name}_logits",
                distributions.Normal(0.0, self.weight_scale)
                .expand([k])
                .to_event(1),
            )
            w = jnn.softmax(logits)
        else:
            w = jnp.asarray(self.weights)

        probs = jnp.broadcast_to(w, (d, units, k))
        mixing = distributions.Categorical(probs=probs)
        components = [
            dst(**kw).expand([d, units])
            for dst, kw in zip(self.component_dists, self.component_kwargs)
        ]
        beta = sample(
            f"{cls}_{name}_beta",
            distributions.MixtureGeneral(mixing, components),
        )
        return activation(_matmul_dot_product(x, beta))


# ---- Gaussian processes ---------------------------------------------------- #


def hsgp_L(x: Any, c: float = 1.5) -> float:
    """Boundary ``L = c * max(|x|)`` for the Hilbert-space GP basis.

    Compute this once on the training inputs and pass the same ``L`` to
    :class:`HSGPLayer` at both fit and predict time — the eigenfunction basis
    is only valid on a fixed domain ``[-L, L]``. ``c`` in ~[1.2, 2.0]; larger
    is safer near the data edges. Center ``x`` first so it straddles 0.
    """
    return float(c * np.max(np.abs(np.asarray(x))))


def _hsgp_basis(x: jax.Array, L: float, m: int) -> tuple[jax.Array, jax.Array]:
    """Laplacian eigenfunctions/eigenvalues on ``[-L, L]`` (Cox–de Boor-free).

    Returns ``(phi, sqrt_lambda)`` where ``phi`` is ``(n, m)`` and
    ``sqrt_lambda`` is ``(m,)``.
    """
    x_flat = x.reshape(-1)
    j = jnp.arange(1, m + 1)
    sqrt_lambda = j * jnp.pi / (2.0 * L)  # (m,)
    phi = jnp.sqrt(1.0 / L) * jnp.sin(
        sqrt_lambda[None, :] * (x_flat[:, None] + L)
    )
    return phi, sqrt_lambda


def _spd_squared_exponential(
    alpha: jax.Array, ell: jax.Array, sqrt_lambda: jax.Array
) -> jax.Array:
    """Spectral density of the squared-exponential kernel at ``sqrt_lambda``."""
    return (
        alpha**2
        * jnp.sqrt(2.0 * jnp.pi)
        * ell
        * jnp.exp(-0.5 * (ell * sqrt_lambda) ** 2)
    )


class HSGPLayer(BLayer):
    """Hilbert-space approximate Gaussian process (1-D, squared-exponential).

    Low-rank GP of `Riutort-Mayol et al. (2020) <https://arxiv.org/abs/2004.11408>`_:
    a stationary GP on ``[-L, L]`` is approximated with ``m`` Laplacian
    eigenfunctions, turning the GP into a basis-function layer

    .. math::
        f(x) \\approx \\sum_{j=1}^{m} \\phi_j(x)\\, \\sqrt{S(\\sqrt{\\lambda_j})}\\, \\beta_j,
        \\quad \\beta_j \\sim \\mathrm{Normal}(0, 1)

    where ``\\phi_j`` / ``\\lambda_j`` are the eigenfunctions / eigenvalues on
    ``[-L, L]`` and ``S`` is the squared-exponential spectral density (a
    function of the sampled lengthscale ``ell`` and marginal std ``alpha``).
    Sits alongside :func:`blayers.splines.bspline_basis` and
    :class:`RandomWalkLayer` as a smoother, but learns its own lengthscale and
    carries a proper GP interpretation.

    Center / scale ``x`` so it lies within ``[-L, L]``; pick ``L`` with
    :func:`hsgp_L` on the training data and reuse it at predict time. ``m``
    trades accuracy for cost (~20–50 is typical); the approximation degrades
    for lengthscales that are very short relative to the domain.
    """

    def __init__(
        self,
        lengthscale_dist: distributions.Distribution = distributions.InverseGamma,
        lengthscale_kwargs: dict[str, float] = {
            "concentration": 5.0,
            "rate": 5.0,
        },
        sigma_dist: distributions.Distribution = distributions.HalfNormal,
        sigma_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lengthscale_dist: Prior distribution class for the GP lengthscale.
            lengthscale_kwargs: Kwargs for the lengthscale prior.
            sigma_dist: Prior distribution class for the GP marginal std.
            sigma_kwargs: Kwargs for the marginal-std prior.
        """
        self.lengthscale_dist = lengthscale_dist
        self.lengthscale_kwargs = lengthscale_kwargs
        self.sigma_dist = sigma_dist
        self.sigma_kwargs = sigma_kwargs
        try:
            lengthscale_dist(**lengthscale_kwargs)
            sigma_dist(**sigma_kwargs)
        except TypeError as e:
            raise TypeError(f"Invalid distribution kwargs: {e}") from e

    def __call__(
        self,
        name: str,
        x: jax.Array,
        L: float,
        m: int,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Args:
            name: Variable name scope.
            x: 1-D input of shape ``(n,)`` or ``(n, 1)`` within ``[-L, L]``.
            L: Domain boundary (see :func:`hsgp_L`). Fixed across fit/predict.
            m: Number of basis functions.
            units: Number of output dimensions.
            activation: Activation function.

        Returns:
            jax.Array of shape ``(n, units)``.
        """
        cls = self.__class__.__name__
        phi, sqrt_lambda = _hsgp_basis(x, L, m)  # (n, m), (m,)
        ell = sample(
            f"{cls}_{name}_lengthscale",
            self.lengthscale_dist(**self.lengthscale_kwargs),
        )
        alpha = sample(
            f"{cls}_{name}_sigma", self.sigma_dist(**self.sigma_kwargs)
        )
        spd = jnp.sqrt(
            _spd_squared_exponential(alpha, ell, sqrt_lambda)
        )  # (m,)
        beta = sample(
            f"{cls}_{name}_beta",
            distributions.Normal(0.0, 1.0).expand([m, units]),
        )
        f = jnp.einsum("nm,mu->nu", phi, spd[:, None] * beta)
        return activation(f)
