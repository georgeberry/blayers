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


def _pairwise_interactions(x: jax.Array, z: jax.Array) -> jax.Array:
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
    interactions = _pairwise_interactions(x, z)

    return jnp.einsum("nd,du->nu", interactions, beta)


# ---- Classes --------------------------------------------------------------- #


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
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lmbda_dist: NumPyro distribution class for the scale (λ) of the
                prior.
            coef_dist: NumPyro distribution class for the coefficient prior.
            coef_kwargs: Parameters for the prior distribution.
            lmbda_kwargs: Parameters for the scale distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
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
            ``coef_dist``: NumPyro distribution class for the coefficients.
            ``coef_kwargs``: Parameters to initialize the prior distribution.
        """
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs

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
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lmbda_dist: Distribution for scaling factor λ.
            coef_dist: Prior for beta parameters.
            coef_kwargs: Arguments for prior distribution.
            lmbda_kwargs: Arguments for λ distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape, low_rank_dim, units]
            ),
        )
        # matmul and return
        return activation(_matmul_factorization_machine(x, theta))


class FM3Layer(BLayer):
    """Order 3 FM. See `Blondel et al 2016 <https://proceedings.neurips.cc/paper/2016/file/158fc2ddd52ec2cf54d3c161f2dd6517-Paper.pdf>`_."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lmbda_dist: Distribution for scaling factor λ.
            coef_dist: Prior for beta parameters.
            coef_kwargs: Arguments for prior distribution.
            lmbda_kwargs: Arguments for λ distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape, low_rank_dim, units]
            ),
        )
        # matmul and return
        return activation(_matmul_fm3(x, theta))


class LowRankInteractionLayer(BLayer):
    """Takes two sets of features and learns a low-rank interaction matrix."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda1 = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda1",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )
        theta1 = sample(
            name=f"{self.__class__.__name__}_{name}_theta1",
            fn=self.coef_dist(scale=lmbda1, **self.coef_kwargs).expand(
                [input_shape1, low_rank_dim, units]
            ),
        )
        lmbda2 = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda2",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )
        theta2 = sample(
            name=f"{self.__class__.__name__}_{name}_theta2",
            fn=self.coef_dist(scale=lmbda2, **self.coef_kwargs).expand(
                [input_shape2, low_rank_dim, units]
            ),
        )
        return activation(_matmul_uv_decomp(theta1, theta2, x, z))


class InteractionLayer(BLayer):
    """Computes every interaction coefficient between two sets of inputs."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda1",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta1",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape1 * input_shape2, units]
            ),
        )

        return activation(_matmul_interaction(beta, x, z))


class BilinearLayer(BLayer):
    """Bayesian bilinear interaction layer: computes x^T W z."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lmbda_dist: prior on scale of coefficients
            coef_dist: distribution for coefficients
            coef_kwargs: kwargs for coef distribution
            lmbda_kwargs: kwargs for scale prior
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )
        # full W: [input_shape1, input_shape2, units]
        W = sample(
            name=f"{self.__class__.__name__}_{name}_W",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape1, input_shape2, units]
            ),
        )
        # bilinear form: x^T W z for each unit
        return activation(jnp.einsum("ni,iju,nj->nu", x, W, z))


class LowRankBilinearLayer(BLayer):
    """Bayesian bilinear interaction layer: computes x^T W z. W low rank."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lmbda_dist: prior on scale of coefficients
            coef_dist: distribution for coefficients
            coef_kwargs: kwargs for coef distribution
            lmbda_kwargs: kwargs for scale prior
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([units]),
        )

        A = sample(
            name=f"{self.__class__.__name__}_{name}_A",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape1, low_rank_dim, units]
            ),
        )
        B = sample(
            name=f"{self.__class__.__name__}_{name}_B",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
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
    """Bayesian embedding layer for sparse categorical features."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lmbda_dist: NumPyro distribution class for the scale (λ) of the
                prior.
            coef_dist: NumPyro distribution class for the coefficient prior.
            coef_kwargs: Parameters for the prior distribution.
            lmbda_kwargs: Parameters for the scale distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [num_categories, embedding_dim]
            ),
        )
        # matmul and return
        return beta[x.squeeze()]


class RandomEffectsLayer(BLayer):
    """Exactly like the EmbeddingLayer but with ``embedding_dim=1``."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            num_embeddings: Total number of discrete embedding entries.
            embedding_dim: Dimensionality of each embedding vector.
            coef_dist: Prior distribution for embedding weights.
            coef_kwargs: Parameters for the prior distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [num_categories, 1]
            ),
        )
        return beta[x.squeeze()]


# ---- Spline utilities ------------------------------------------------------ #


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
    pass the returned array to ``SplineLayer``.

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
        layer = SplineLayer(knots)
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


class RandomWalkLayer(BLayer):
    """Random walk of embedding dim ``m``, defaults to Gaussian walk."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [
                    num_categories,
                    embedding_dim,
                ]
            ),
        )
        # matmul and return
        return _matmul_randomwalk(theta, x)
