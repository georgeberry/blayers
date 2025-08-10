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
from typing import Any

import jax
import jax.numpy as jnp
from numpyro import distributions, sample

from blayers._utils import add_trailing_dim

# ---- Matmul functions ------------------------------------------------------ #


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

    n, d1 = x.shape
    _, d2 = z.shape

    # thanks chat GPT
    interactions = jnp.reshape(x[:, :, None] * z[:, None, :], (n, d1 * d2))

    return jnp.einsum("nd,du->nu", interactions, beta)


# ---- Classes --------------------------------------------------------------- #


class BLayer(ABC):
    """Abstract base class for Bayesian layers. Lays out an interface."""

    @abstractmethod
    def __init__(self, *args: Any) -> None:
        """Initialize layer parameters."""

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        """
        Run the layer's forward pass.

        Args:
            name: Name scope for sampled variables. Note due to mypy stuff we
                  only write the `name` arg explicitly in subclass.
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
        units: int = 1,
    ):
        """
        Args:
            lmbda_dist: NumPyro distribution class for the scale (λ) of the
                prior.
            coef_dist: NumPyro distribution class for the coefficient prior.
            coef_kwargs: Parameters for the prior distribution.
            lmbda_kwargs: Parameters for the scale distribution.
            units: The number of outputs
            dependent_outputs: For multi-output models whether to treat the outputs as dependent. By deafult they are independent.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs
        self.units = units

    def __call__(
        self,
        name: str,
        x: jax.Array,
    ) -> jax.Array:
        """
        Forward pass with adaptive prior on coefficients.

        Args:
            name: Variable name scope.
            x: Input data array of shape (n, d, u).

        Returns:
            jax.Array: Output array of shape (n, u).
        """

        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([self.units]),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape, self.units]
            ),
        )

        # matmul and return
        return _matmul_dot_product(x, beta)


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
        units: int = 1,
    ):
        """
        Args:
            coef_dist: NumPyro distribution class for the coefficients.
            coef_kwargs: Parameters to initialize the prior distribution.
        """
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.units = units

    def __call__(
        self,
        name: str,
        x: jax.Array,
    ) -> jax.Array:
        """
        Forward pass with fixed prior.

        Args:
            name: Variable name prefix.
            x: Input data array of shape (n, d).

        Returns:
            jax.Array: Output array of shape (n, u).
        """

        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(**self.coef_kwargs).expand(
                [input_shape, self.units]
            ),
        )
        # matmul and return
        return _matmul_dot_product(x, beta)


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
        units: int = 1,
    ):
        """
        Args:
            ``coef_dist``: NumPyro distribution class for the coefficients.
            ``coef_kwargs``: Parameters to initialize the prior distribution.
        """
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.units = units

    def __call__(
        self,
        name: str,
    ) -> jax.Array:
        """
        Forward pass with fixed prior.

        Args:
            name: Variable name prefix.

        Returns:
            jax.Array: Output array of shape (1, u).
        """

        # sampling block
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.coef_dist(**self.coef_kwargs).expand([1, self.units]),
        )
        return beta


class EmbeddingLayer(BLayer):
    """Bayesian embedding layer for sparse categorical features."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
        units: int = 1,
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
        self.units = units

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
            x: Integer indices of shape (n,) indicating embeddings to use.
            num_categories: The number of distinct things getting an embedding
            embedding_dim: The size of each embedding, e.g. 2, 4, 8, etc.

        Returns:
            jax.Array: Embedding vectors of shape (n, m).
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
        units: int = 1,
    ):
        """
        Args:
            ``num_embeddings``: Total number of discrete embedding entries.
            ``embedding_dim``: Dimensionality of each embedding vector.
            ``coef_dist``: Prior distribution for embedding weights.
            ``coef_kwargs``: Parameters for the prior distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs
        self.units = units

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
            x: Integer indices of shape (n,) indicating embeddings to use.
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
        units: int = 1,
    ):
        """
        Args:
            lmbda_dist: Distribution for scaling factor λ.
            coef_dist: Prior for beta parameters.
            coef_kwargs: Arguments for prior distribution.
            lmbda_kwargs: Arguments for λ distribution.
            low_rank_dim: Dimensionality of low-rank approximation.
        """
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs
        self.units = units

    def __call__(
        self,
        name: str,
        x: jax.Array,
        low_rank_dim: int,
    ) -> jax.Array:
        """
        Forward pass through the factorization machine layer.

        Args:
            name: Variable name scope.
            x: Input matrix of shape (n, d).

        Returns:
            jax.Array: Output array of shape (n,).
        """
        # get shapes and reshape if necessary
        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([self.units]),
        )
        theta = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape, low_rank_dim, self.units]
            ),
        )
        # matmul and return
        return _matmul_factorization_machine(x, theta)


class LowRankInteractionLayer(BLayer):
    """Takes two sets of features and learns a low-rank interaction matrix."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
        units: int = 1,
    ):
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs
        self.units = units

    def __call__(
        self,
        name: str,
        x: jax.Array,
        z: jax.Array,
        low_rank_dim: int,
    ) -> jax.Array:
        # get shapes and reshape if necessary
        x = add_trailing_dim(x)
        z = add_trailing_dim(z)
        input_shape1 = x.shape[1]
        input_shape2 = z.shape[1]

        # sampling block
        lmbda1 = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda1",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([self.units]),
        )
        theta1 = sample(
            name=f"{self.__class__.__name__}_{name}_theta1",
            fn=self.coef_dist(scale=lmbda1, **self.coef_kwargs).expand(
                [input_shape1, low_rank_dim, self.units]
            ),
        )
        lmbda2 = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda2",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([self.units]),
        )
        theta2 = sample(
            name=f"{self.__class__.__name__}_{name}_theta2",
            fn=self.coef_dist(scale=lmbda2, **self.coef_kwargs).expand(
                [input_shape2, low_rank_dim, self.units]
            ),
        )
        return _matmul_uv_decomp(theta1, theta2, x, z)


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
        """ """

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


class InteractionLayer(BLayer):
    """Computes every interaction coefficient between two sets of inputs."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        coef_dist: distributions.Distribution = distributions.Normal,
        coef_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
        units: int = 1,
    ):
        self.lmbda_dist = lmbda_dist
        self.coef_dist = coef_dist
        self.coef_kwargs = coef_kwargs
        self.lmbda_kwargs = lmbda_kwargs
        self.units = units

    def __call__(
        self,
        name: str,
        x: jax.Array,
        z: jax.Array,
    ) -> jax.Array:
        # get shapes and reshape if necessary
        x = add_trailing_dim(x)
        z = add_trailing_dim(z)
        input_shape1 = x.shape[1]
        input_shape2 = z.shape[1]

        # sampling block
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda1",
            fn=self.lmbda_dist(**self.lmbda_kwargs).expand([self.units]),
        )
        beta = sample(
            name=f"{self.__class__.__name__}_{name}_beta1",
            fn=self.coef_dist(scale=lmbda, **self.coef_kwargs).expand(
                [input_shape1 * input_shape2, self.units]
            ),
        )

        return _matmul_interaction(beta, x, z)
