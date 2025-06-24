"""
Implements Bayesian Layers using Jax and Numpyro.

Design:
  - There are three levels of complexity here: class-level, instance-level, and
    call-level
  - The class-level handles things like choosing generic model form and how to
    multiply coefficents with data. Defined by the `class Layer(BLayer)` def
    itself.
  - The instance-level handles specific distributions that fit into a generic
    model and the initial parameters for those distributions. Defined by
    creating an instance of the class: `Layer(*args, **kwargs)`.
  - The call-level handles seeing a batch of data, sampling from the
    distributions defined on the class and multiplying coefficients and data to
    produce an output, works like `result = Layer(*args, **kwargs)(data)`

Notation:
  - `i`: observations in a batch
  - `j, k`: number of sampled coefficients
  - `l`: low rank dimension of low rank models
"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from numpyro import distributions, sample

from blayers.utils import add_trailing_dim


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
            name: Name scope for sampled variables.
            *args: Inputs to the layer.

        Returns:
            jax.Array: The result of the forward computation.
        """

    @staticmethod
    @abstractmethod
    def matmul(*args: Any) -> Any:
        """
        Abstract static method for matrix multiplication logic.

        Args:
            *args: Parameters to multiply.

        Returns:
            jax.Array: The result of the matrix multiplication.
        """


class FixedPriorLayer(BLayer):
    """Bayesian layer with a fixed prior distribution over coefficients."""

    def __init__(
        self,
        prior_dist: distributions.Distribution = distributions.Normal,
        prior_kwargs: dict[str, float] = {"loc": 0.0, "scale": 1.0},
    ):
        """
        Args:
            prior_dist: NumPyro distribution class for the coefficients.
            prior_kwargs: Parameters to initialize the prior distribution.
        """
        self.prior_dist = prior_dist
        self.prior_kwargs = prior_kwargs

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
            jax.Array: Output array of shape (n,).
        """

        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        betas = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.prior_dist(**self.prior_kwargs),
            sample_shape=(input_shape,),
        )
        # matmul and return
        return self.matmul(x, betas)

    @staticmethod
    def matmul(beta: jax.Array, x: jax.Array) -> jax.Array:
        """A dot product.

        Args:
            beta: Model coefficients of shape (j,).
            x: Input data array of shape (n, d).

        Returns:
            jax.Array: Output array of shape (n,).
        """
        return jnp.einsum("ij,j->i", beta, x)


class AdaptiveLayer(BLayer):
    """Bayesian layer with adaptive prior using hierarchical modeling."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        prior_dist: distributions.Distribution = distributions.Normal,
        prior_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            lmbda_dist: NumPyro distribution class for the scale (λ) of the
                prior.
            prior_dist: NumPyro distribution class for the coefficient prior.
            prior_kwargs: Parameters for the prior distribution.
            lmbda_kwargs: Parameters for the scale distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.prior_dist = prior_dist
        self.prior_kwargs = prior_kwargs
        self.lmbda_kwargs = lmbda_kwargs

    def __call__(
        self,
        name: str,
        x: jax.Array,
    ) -> jax.Array:
        """
        Forward pass with adaptive prior on coefficients.

        Args:
            name: Variable name scope.
            x: Input data array of shape (n, d).

        Returns:
            jax.Array: Output array of shape (n,).
        """

        x = add_trailing_dim(x)
        input_shape = x.shape[1]

        # sampling block
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        betas = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.prior_dist(scale=lmbda, **self.prior_kwargs),
            sample_shape=(input_shape,),
        )
        # matmul and return
        return self.matmul(x, betas)

    @staticmethod
    def matmul(beta: jax.Array, x: jax.Array) -> jax.Array:
        """
        Standard dot product between beta and x.

        Args:
            beta: Coefficient vector of shape (d,).
            x: Input matrix of shape (n, d).

        Returns:
            jax.Array: Output of shape (n,).
        """
        return jnp.einsum("ij,j->i", beta, x)


class FMLayer(BLayer):
    """Bayesian factorization machine layer with adaptive priors."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        prior_dist: distributions.Distribution = distributions.Normal,
        prior_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
        low_rank_dim: int = 3,
    ):
        """
        Args:
            lmbda_dist: Distribution for scaling factor λ.
            prior_dist: Prior for beta parameters.
            prior_kwargs: Arguments for prior distribution.
            lmbda_kwargs: Arguments for λ distribution.
            low_rank_dim: Dimensionality of low-rank approximation.
        """
        self.lmbda_dist = lmbda_dist
        self.prior_dist = prior_dist
        self.prior_kwargs = prior_kwargs
        self.lmbda_kwargs = lmbda_kwargs
        self.low_rank_dim = low_rank_dim

    def __call__(
        self,
        name: str,
        x: jax.Array,
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
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        thetas = sample(
            name=f"{self.__class__.__name__}_{name}_theta",
            fn=self.prior_dist(scale=lmbda, **self.prior_kwargs),
            sample_shape=(input_shape, self.low_rank_dim),
        )
        # matmul and return
        return self.matmul(thetas, x)

    @staticmethod
    def matmul(theta: jax.Array, x: jax.Array) -> jax.Array:
        """
        Apply second-order factorization machine interaction.

        Based on Rendle (2010). Computes:
        0.5 * sum((xV)^2 - (x^2 V^2))

        Args:
            theta: Weight matrix of shape (d, k).
            x: Input data of shape (n, d).

        Returns:
            jax.Array: Output of shape (n,).
        """
        vx2 = jnp.einsum("ij,jk->ik", x, theta) ** 2
        v2x2 = jnp.einsum("ij,jk->ik", x**2, theta**2)
        return 0.5 * jnp.einsum("ik->i", vx2 - v2x2)


class LowRankInteractionLayer(BLayer):
    """Takes two sets of features and learns a low-rank interaction matrix."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        prior_dist: distributions.Distribution = distributions.Normal,
        low_rank_dim: int = 3,
        prior_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        self.lmbda_dist = lmbda_dist
        self.prior_dist = prior_dist
        self.low_rank_dim = low_rank_dim
        self.prior_kwargs = prior_kwargs
        self.lmbda_kwargs = lmbda_kwargs

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
        lmbda1 = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda1",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        theta1 = sample(
            name=f"{self.__class__.__name__}_{name}_theta1",
            fn=self.prior_dist(scale=lmbda1, **self.prior_kwargs),
            sample_shape=(input_shape1, self.low_rank_dim),
        )
        lmbda2 = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda2",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        theta2 = sample(
            name=f"{self.__class__.__name__}_{name}_theta2",
            fn=self.prior_dist(scale=lmbda2, **self.prior_kwargs),
            sample_shape=(input_shape2, self.low_rank_dim),
        )
        # matmul and return
        return self.matmul(theta1, theta2, x, z)

    @staticmethod
    def matmul(
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
        """
        xb = jnp.einsum("ij,jk->ik", x, theta1)
        zb = jnp.einsum("ij,jk->ik", z, theta2)
        return jnp.einsum("ik->i", xb * zb)


class EmbeddingLayer(BLayer):
    """Bayesian embedding layer for sparse categorical features."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        prior_dist: distributions.Distribution = distributions.Normal,
        prior_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            num_embeddings: Total number of discrete embedding entries.
            embedding_dim: Dimensionality of each embedding vector.
            prior_dist: Prior distribution for embedding weights.
            prior_kwargs: Parameters for the prior distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.prior_dist = prior_dist
        self.prior_kwargs = prior_kwargs
        self.lmbda_kwargs = lmbda_kwargs

    def __call__(
        self,
        name: str,
        x: jax.Array,
        n_categories: int,
        embedding_dim: int,
    ) -> jax.Array:
        """
        Forward pass through embedding lookup.

        Args:
            name: Variable name scope.
            x: Integer indices of shape (n,) indicating embeddings to use.
            n_categories: The number of distinct things getting an embedding
            embedding_dim: The size of each embedding, e.g. 2, 4, 8, etc.

        Returns:
            jax.Array: Embedding vectors of shape (n, embedding_dim).
        """
        # sampling block
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        betas = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.prior_dist(scale=lmbda, **self.prior_kwargs),
            sample_shape=(n_categories, embedding_dim),
        )
        # matmul and return
        return self.matmul(betas, x)

    @staticmethod
    def matmul(beta: jax.Array, x: jax.Array) -> jax.Array:
        """
        Index into the embedding table using the provided indices.

        Args:
            beta: Embedding table of shape (num_embeddings, embedding_dim).
            x: Indices array of shape (n,).

        Returns:
            jax.Array: Looked-up embeddings of shape (n, embedding_dim).
        """
        return beta[x.squeeze()].squeeze()


class RandomEffectsLayer(BLayer):
    """Exactly like the EmbeddingLayer but with `embedding_dim=1`."""

    def __init__(
        self,
        lmbda_dist: distributions.Distribution = distributions.HalfNormal,
        prior_dist: distributions.Distribution = distributions.Normal,
        prior_kwargs: dict[str, float] = {"loc": 0.0},
        lmbda_kwargs: dict[str, float] = {"scale": 1.0},
    ):
        """
        Args:
            num_embeddings: Total number of discrete embedding entries.
            embedding_dim: Dimensionality of each embedding vector.
            prior_dist: Prior distribution for embedding weights.
            prior_kwargs: Parameters for the prior distribution.
        """
        self.lmbda_dist = lmbda_dist
        self.prior_dist = prior_dist
        self.prior_kwargs = prior_kwargs
        self.lmbda_kwargs = lmbda_kwargs

    def __call__(
        self,
        name: str,
        x: jax.Array,
        n_categories: int,
    ) -> jax.Array:
        """
        Forward pass through embedding lookup.

        Args:
            name: Variable name scope.
            x: Integer indices of shape (n,) indicating embeddings to use.
            n_categories: The number of distinct things getting an embedding

        Returns:
            jax.Array: Embedding vectors of shape (n, embedding_dim).
        """
        embedding_dim = 1

        # sampling block
        lmbda = sample(
            name=f"{self.__class__.__name__}_{name}_lmbda",
            fn=self.lmbda_dist(**self.lmbda_kwargs),
        )
        betas = sample(
            name=f"{self.__class__.__name__}_{name}_beta",
            fn=self.prior_dist(scale=lmbda, **self.prior_kwargs),
            sample_shape=(n_categories, embedding_dim),
        )
        # matmul and return
        return self.matmul(betas, x)

    @staticmethod
    def matmul(beta: jax.Array, x: jax.Array) -> jax.Array:
        """
        Index into the embedding table using the provided indices.

        Args:
            beta: Embedding table of shape (num_embeddings, embedding_dim).
            x: Indices array of shape (n,).

        Returns:
            jax.Array: Looked-up embeddings of shape (n, embedding_dim).
        """
        return beta[x.squeeze()].squeeze()
