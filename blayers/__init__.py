from importlib.metadata import PackageNotFoundError, version

from blayers.decorators import autoreparam, autoreshape
from blayers.fit import FittedModel, Predictions, fit, sample_prior
from blayers.latex import model_to_latex
from blayers.layers import (
    AdaptiveLayer,
    BilinearLayer,
    EmbeddingLayer,
    FixedEffectsLayer,
    FixedPriorLayer,
    FM3Layer,
    FMLayer,
    HorseshoeLayer,
    HSGPLayer,
    InteractionLayer,
    InterceptLayer,
    LowRankBilinearLayer,
    LowRankInteractionLayer,
    MixtureLayer,
    RandomEffectsLayer,
    RandomWalkLayer,
    SpikeAndSlabLayer,
    hsgp_L,
    pairwise_interactions,
)
from blayers.links import (
    beta_link,
    categorical_link,
    exponential_link,
    gamma_link,
    gaussian_link,
    logit_link,
    lognormal_link,
    negative_binomial_link,
    ordinal_link,
    poisson_link,
    student_t_link,
    zinb_link,
    zip_link,
)

try:
    __version__ = version("blayers")
except PackageNotFoundError:  # package not installed (e.g. running from source)
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Layers
    "AdaptiveLayer",
    "BilinearLayer",
    "EmbeddingLayer",
    "FixedEffectsLayer",
    "FixedPriorLayer",
    "FMLayer",
    "FM3Layer",
    "HorseshoeLayer",
    "HSGPLayer",
    "InteractionLayer",
    "InterceptLayer",
    "LowRankBilinearLayer",
    "LowRankInteractionLayer",
    "MixtureLayer",
    "hsgp_L",
    "pairwise_interactions",
    "RandomEffectsLayer",
    "RandomWalkLayer",
    "SpikeAndSlabLayer",
    # Links
    "beta_link",
    "categorical_link",
    "exponential_link",
    "gamma_link",
    "gaussian_link",
    "logit_link",
    "lognormal_link",
    "negative_binomial_link",
    "ordinal_link",
    "poisson_link",
    "student_t_link",
    "zinb_link",
    "zip_link",
    # Decorators
    "autoreparam",
    "autoreshape",
    # Fit
    "fit",
    "sample_prior",
    "model_to_latex",
    "FittedModel",
    "Predictions",
]
