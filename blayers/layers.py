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
    """Bayesian full pairwise interaction layer with adaptive prior.

    Samples one coefficient per pair of features from the hierarchical model

    .. math::
        \\lambda \\sim HalfNormal(1.)

    .. math::
        \\beta \\sim Normal(0., \\lambda), \\quad
        \\beta \\in \\mathbb{R}^{d_1 d_2}

    and computes the weighted sum of all outer-product interactions:

    .. math::
        \\text{output} = (x \\otimes z)\\, \\beta

    where :math:`x \\otimes z` is the flattened outer product of shape
    :math:`(n, d_1 d_2)`. For large inputs this scales as
    :math:`O(d_1 d_2)` parameters; prefer :class:`LowRankInteractionLayer`
    when :math:`d_1` or :math:`d_2` is large.
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
        z: jax.Array,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Full pairwise interaction between feature matrices X and Z.

        Samples one coefficient per ``(x_i, z_j)`` pair (``d1 * d2`` total)
        and returns the weighted sum of all outer-product interactions.

        Args:
            name: Variable name scope.
            x: Input matrix of shape ``(n, d1)``.
            z: Input matrix of shape ``(n, d2)``.
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
        scale = sample(
            name=f"{self.__class__.__name__}_{name}_scale1",
            fn=self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta1",
            fn=self.coef_dist(scale=scale, **self.coef_kwargs).expand(
                [input_shape1 * input_shape2, units]
            ),
        )

        return activation(_matmul_interaction(beta, x, z))


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


# ---- Sparse priors --------------------------------------------------------- #


class HorseshoeLayer(BLayer):
    """Bayesian layer with horseshoe prior for sparse regression.

    Implements the (regularized) horseshoe prior of Piironen & Vehtari (2017).

    Basic horseshoe:

    .. math::
        \\tau \\sim HalfCauchy(1), \\quad
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
        slab_scale: float | None = None,
        slab_df: float = 4.0,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
    ):
        """
        Args:
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
            distributions.HalfCauchy(1.0).expand([units]),
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

        beta = sample(
            f"{cls}_{name}_beta",
            self.coef_dist(scale=scale, **self.coef_kwargs),
        )
        return activation(_matmul_dot_product(x, beta))


# ---- Spike and slab -------------------------------------------------------- #


class SpikeAndSlabLayer(BLayer):
    """Sparse regression via a spike-and-slab prior.

    Each coefficient has a Beta-distributed inclusion weight ``z_j`` in
    (0, 1). Included features (``z_j ≈ 1``) take the full slab coefficient;
    excluded features (``z_j ≈ 0``) are gated toward zero (the spike).

    Generative model::

        z_j ~ Beta(alpha, beta)          # inclusion weight (hardcoded Beta)
        β_j ~ coef_dist(**coef_kwargs)   # slab coefficient
        y   ~ link(z · β · x, ...)       # z gates each coefficient

    The default ``Beta(0.5, 0.5)`` (Jeffreys prior) places mass near 0 and 1,
    encouraging features to be clearly included or excluded.  The posterior
    mean of ``z_j`` approximates ``P(feature j included | data)``.

    The slab distribution defaults to ``Normal(0, 1)`` but can be swapped for
    e.g. ``StudentT`` for heavier-tailed slab behaviour.

    Args:
        alpha: First concentration parameter of the Beta prior on ``z``.
        beta: Second concentration parameter of the Beta prior on ``z``.
        coef_dist: Distribution for the slab coefficients.
        coef_kwargs: Kwargs for ``coef_dist``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0, "scale": 1.0},
    ):
        self.alpha = alpha
        self.beta = beta
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
        cls = self.__class__.__name__

        # Inclusion weight: posterior z_j ≈ P(feature j included | data)
        z = sample(
            f"{cls}_{name}_z",
            distributions.Beta(self.alpha, self.beta).expand([d, units]),
        )

        # Slab coefficients
        beta = sample(
            f"{cls}_{name}_beta",
            self.coef_dist(**self.coef_kwargs).expand([d, units]),
        )

        # Gate: z≈1 → full slab value; z≈0 → near zero (spike at 0)
        return activation(_matmul_dot_product(x, z * beta))


# ---- Mixture priors -------------------------------------------------------- #


class MixtureLayer(BLayer):
    """Coefficients from a finite mixture-of-priors (e.g. Normal + Laplace).

    Each coefficient is drawn from a ``K``-component mixture

    .. math::
        \\beta_j \\sim \\sum_{k=1}^{K} w_k \\, p_k(\\cdot)

    where the mixing weights ``w`` are either fixed or given a ``Dirichlet``
    prior (shared across coefficients). The component indicator is marginalised
    analytically by :class:`numpyro.distributions.MixtureGeneral`, so the
    log-density is smooth and works under VI *and* MCMC — unlike a discrete
    spike-and-slab indicator.

    Useful for robustness (a heavy-tailed component absorbs a few outlier
    coefficients while the rest stay Gaussian) and elastic-net-flavoured priors
    (Normal + Laplace). For pure sparsity prefer :class:`HorseshoeLayer`; for
    explicit variable selection prefer :class:`SpikeAndSlabLayer`.
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
        dirichlet_concentration: float = 1.0,
    ):
        """
        Args:
            component_dists: NumPyro distribution classes, one per mixture
                component (>= 2). All must share the same (real) support.
            component_kwargs: Kwargs for each component distribution.
            weights: Fixed mixing weights (one per component, summing to 1). If
                ``None``, a ``Dirichlet`` prior is placed on the weights.
            dirichlet_concentration: Symmetric ``Dirichlet`` concentration used
                when ``weights`` is ``None``.
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
        self.dirichlet_concentration = dirichlet_concentration
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
            w = sample(
                f"{cls}_{name}_weights",
                distributions.Dirichlet(
                    jnp.full(k, self.dirichlet_concentration)
                ),
            )
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
