from importlib.metadata import PackageNotFoundError, version

from blayers.decorators import autoreparam, autoreshape
from blayers.fit import FittedModel, Predictions, fit, sample_prior
from blayers.layers import (
    AdaptiveLayer,
    BilinearLayer,
    EmbeddingLayer,
    FixedPriorLayer,
    FM3Layer,
    FMLayer,
    HorseshoeLayer,
    InteractionLayer,
    InterceptLayer,
    LowRankBilinearLayer,
    LowRankInteractionLayer,
    RandomEffectsLayer,
    RandomWalkLayer,
    SpikeAndSlabLayer,
    pairwise_interactions,
)
from blayers.links import (
    beta_link,
    categorical_link,
    gaussian_link,
    logit_link,
    lognormal_link,
    negative_binomial_link,
    ordinal_link,
    poisson_link,
    student_t_link,
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
    "FixedPriorLayer",
    "FMLayer",
    "FM3Layer",
    "HorseshoeLayer",
    "InteractionLayer",
    "InterceptLayer",
    "LowRankBilinearLayer",
    "LowRankInteractionLayer",
    "pairwise_interactions",
    "RandomEffectsLayer",
    "RandomWalkLayer",
    "SpikeAndSlabLayer",
    # Links
    "beta_link",
    "categorical_link",
    "gaussian_link",
    "logit_link",
    "lognormal_link",
    "negative_binomial_link",
    "ordinal_link",
    "poisson_link",
    "student_t_link",
    "zip_link",
    # Decorators
    "autoreparam",
    "autoreshape",
    # Fit
    "fit",
    "sample_prior",
    "FittedModel",
    "Predictions",
]
