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
    Compute all pairwise interactions between features in X and Y.

    Parameters:
        X: (n_samples, n_features1)
        Y: (n_samples, n_features2)

    Returns:
        interactions: (n_samples, n_features1 * n_features2)
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
    """Implements low rank multiplication.

    According to ChatGPT this is a "factorized bilinear interaction".
    Basically, you just need to project x and z down to a common number of
    low rank terms and then just multiply those terms.

    This is equivalent to a UV decomposition where you use n=low_rank_dim
    on the columns of the U/V matrices.

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
    idx_flat = idx.squeeze().astype(jnp.int32)
    return theta_cumsum[idx_flat]


def _matmul_interaction(
    beta: jax.Array,
    x: jax.Array,
    z: jax.Array,
) -> jax.Array:
    """Full interaction between `x` and `z`.

    Args:
        beta: Weight matrix for each interaction between `x` and `z`.
        x: First feature matrix.
        z: Second feature matrix.

    Returns:
        jax.Array

    """

    # thanks chat GPT
    interactions = pairwise_interactions(x, z)

    return jnp.einsum("nd,du->nu", interactions, beta)


# ---- Classes --------------------------------------------------------------- #


def _validate_prior_kwargs(coef_dist, coef_kwargs, scale_dist=None, scale_kwargs=None):
    """Eagerly instantiate distributions at construction time to catch bad kwargs.

    Raises ``TypeError`` immediately if the supplied kwargs are incompatible
    with the distribution, rather than waiting until the layer is called.
    """
    try:
        if scale_dist is not None:
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
            jax.Array: Output array of shape ``(n,)``.
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
        Interaction between feature matrices X and Z in a low rank way. UV decomp.

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
        Interaction between feature matrices X and Z in a low rank way. UV decomp.

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
        Interaction between feature matrices X and Z in a low rank way. UV decomp.

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
        Interaction between feature matrices X and Z in a low rank way. UV decomp.

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
        return theta[x.squeeze()]


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
            num_embeddings: Total number of discrete embedding entries.
            embedding_dim: Dimensionality of each embedding vector.
            coef_dist: Prior distribution for embedding weights.
            coef_kwargs: Parameters for the prior distribution.
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
        Forward pass through embedding lookup.

        Args:
            name: Variable name scope.
            x: Integer indicating embeddings to use.
            num_categories: The number of distinct things getting an embedding

        Returns:
            jax.Array: Embedding vectors of shape (n, embedding_dim).
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
        return theta[x.squeeze()]


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

        beta = sample(f"{cls}_{name}_beta", self.coef_dist(scale=scale, **self.coef_kwargs))
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


# ---- Attention ------------------------------------------------------------- #


class AttentionLayer(BLayer):
    """Multi-head Bayesian self-attention over the feature dimension.

    Treats the ``d`` input features as tokens using FT-Transformer style
    tokenisation (Gorishniy et al. 2021, https://arxiv.org/abs/2106.11959):
    each feature gets a per-column bias embedding (identity) plus a
    value-scaled embedding, so tokens are distinct even when the feature
    value is zero.

    For each observation ``x_i ∈ R^d``:

    1. Tokenise: ``H_j = x_{i,j} · W_emb_j + W_bias_j``  (``head_dim``-dim each)
    2. Per head: ``Q_m, K_m, V_m = H W_Q_m, H W_K_m, H W_V_m``
    3. ``Attn_m = softmax(Q_m K_m^T / √h_k)``
    4. Concatenate heads  →  mean-pool over features  →  project to ``units``

    Requires ``d ≥ 2`` for attention to be non-trivial.
    Total embedding dimension is ``head_dim * num_heads`` — adding heads
    increases capacity rather than splitting a fixed budget.
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
        head_dim: int = 8,
        num_heads: int = 1,
        units: int = 1,
        activation: Callable[[jax.Array], jax.Array] = jnn.identity,
    ) -> jax.Array:
        """
        Args:
            name: Variable name scope.
            x: Input of shape ``(n, d)``.  Each column is a feature token.
            head_dim: Dimension of each individual head.  Total embedding
                dimension is ``head_dim * num_heads``, so adding heads
                increases capacity.
            num_heads: Number of attention heads.
            units: Number of output dimensions.
            activation: Activation function.

        Returns:
            jax.Array of shape ``(n, units)``.
        """
        x = add_trailing_dim(x)
        n, d = x.shape[0], x.shape[1]
        h_k = head_dim        # per-head dimension
        m = num_heads
        h = head_dim * m      # total embedding dimension
        cls = self.__class__.__name__

        # FT-Transformer tokenisation: value scaling + per-column bias
        # H[i,j] = x[i,j] * W_emb[j] + W_bias[j]  → (n, d, h)
        scale_emb = sample(
            f"{cls}_{name}_scale_emb",
            self.scale_dist(**self.scale_kwargs).expand([h]),
        )
        W_emb = sample(
            f"{cls}_{name}_W_emb",
            self.coef_dist(scale=scale_emb, **self.coef_kwargs).expand([d, h]),
        )
        W_bias = sample(
            f"{cls}_{name}_W_bias",
            self.coef_dist(scale=scale_emb, **self.coef_kwargs).expand([d, h]),
        )
        H = x[:, :, None] * W_emb[None, :, :] + W_bias[None, :, :]  # (n, d, h)

        # Q, K, V projections — one set per head: (m, h, h_k)
        # scale_qkv is (m, h_k); unsqueeze to (m, 1, h_k) so it broadcasts to (m, h, h_k)
        scale_qkv = sample(
            f"{cls}_{name}_scale_qkv",
            self.scale_dist(**self.scale_kwargs).expand([m, h_k]),
        )
        scale_qkv_bc = scale_qkv[:, None, :]  # (m, 1, h_k)
        W_Q = sample(
            f"{cls}_{name}_W_Q",
            self.coef_dist(scale=scale_qkv_bc, **self.coef_kwargs).expand([m, h, h_k]),
        )
        W_K = sample(
            f"{cls}_{name}_W_K",
            self.coef_dist(scale=scale_qkv_bc, **self.coef_kwargs).expand([m, h, h_k]),
        )
        W_V = sample(
            f"{cls}_{name}_W_V",
            self.coef_dist(scale=scale_qkv_bc, **self.coef_kwargs).expand([m, h, h_k]),
        )

        # Project to per-head Q/K/V: (n, d, m, h_k)
        Q = jnp.einsum("ndh,mhk->ndmk", H, W_Q)
        K = jnp.einsum("ndh,mhk->ndmk", H, W_K)
        V = jnp.einsum("ndh,mhk->ndmk", H, W_V)

        # Scaled dot-product attention per head: (n, m, d, d)
        scores = jnp.einsum("ndmk,nqmk->nmdq", Q, K) / h_k**0.5
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("nmdq,nqmk->ndmk", weights, V)  # (n, d, m, h_k)

        # Concatenate heads, mean-pool over features: (n, h)
        pooled = out.reshape(n, d, h).mean(axis=1)

        # Output projection
        scale_out = sample(
            f"{cls}_{name}_scale_out",
            self.scale_dist(**self.scale_kwargs).expand([units]),
        )
        W_out = sample(
            f"{cls}_{name}_W_out",
            self.coef_dist(scale=scale_out, **self.coef_kwargs).expand([h, units]),
        )
        return activation(pooled @ W_out)
