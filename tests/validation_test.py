"""Tests for construction-time validation of prior kwargs."""

import pytest
from numpyro import distributions

from blayers.layers import (
    AdaptiveLayer,
    AttentionLayer,
    BilinearLayer,
    EmbeddingLayer,
    FMLayer,
    FM3Layer,
    FixedPriorLayer,
    InterceptLayer,
    InteractionLayer,
    LowRankBilinearLayer,
    LowRankInteractionLayer,
    RandomEffectsLayer,
    RandomWalkLayer,
)


# --------------------------------------------------------------------------- #
# Adaptive-prior layers (lmbda + coef)
# --------------------------------------------------------------------------- #


class TestAdaptiveLayerValidation:
    def test_valid_defaults(self):
        AdaptiveLayer()  # should not raise

    def test_bad_scale_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            AdaptiveLayer(scale_kwargs={"nonexistent_arg": 1.0})

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            AdaptiveLayer(coef_kwargs={"loc": 0.0, "bad_arg": 99.0})

    def test_error_at_construction_not_call(self):
        """TypeError fires during __init__, not during __call__."""
        with pytest.raises(TypeError):
            AdaptiveLayer(scale_kwargs={"bad": 1.0})


class TestFMLayerValidation:
    def test_valid_defaults(self):
        FMLayer()

    def test_bad_scale_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            FMLayer(scale_kwargs={"bad_kwarg": 1.0})


class TestFM3LayerValidation:
    def test_valid_defaults(self):
        FM3Layer()

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            FM3Layer(coef_kwargs={"loc": 0.0, "bad_kwarg": 1.0})


class TestAttentionLayerValidation:
    def test_valid_defaults(self):
        AttentionLayer()

    def test_bad_scale_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            AttentionLayer(scale_kwargs={"oops": 2.0})


class TestEmbeddingLayerValidation:
    def test_valid_defaults(self):
        EmbeddingLayer()

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            EmbeddingLayer(coef_kwargs={"loc": 0.0, "nope": 1.0})


class TestRandomEffectsLayerValidation:
    def test_valid_defaults(self):
        RandomEffectsLayer()

    def test_bad_scale_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            RandomEffectsLayer(scale_kwargs={"bad": 1.0})


class TestRandomWalkLayerValidation:
    def test_valid_defaults(self):
        RandomWalkLayer()

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            RandomWalkLayer(coef_kwargs={"loc": 0.0, "bogus": 5.0})


class TestLowRankInteractionLayerValidation:
    def test_valid_defaults(self):
        LowRankInteractionLayer()

    def test_bad_scale_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            LowRankInteractionLayer(scale_kwargs={"bad": 0.5})


class TestInteractionLayerValidation:
    def test_valid_defaults(self):
        InteractionLayer()

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            InteractionLayer(coef_kwargs={"loc": 0.0, "bad": 1.0})


class TestBilinearLayerValidation:
    def test_valid_defaults(self):
        BilinearLayer()

    def test_bad_scale_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            BilinearLayer(scale_kwargs={"bad": 1.0})


class TestLowRankBilinearLayerValidation:
    def test_valid_defaults(self):
        LowRankBilinearLayer()

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            LowRankBilinearLayer(coef_kwargs={"loc": 0.0, "bad": 9.0})


# --------------------------------------------------------------------------- #
# Fixed-prior layers (coef only)
# --------------------------------------------------------------------------- #


class TestFixedPriorLayerValidation:
    def test_valid_defaults(self):
        FixedPriorLayer()

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            FixedPriorLayer(coef_kwargs={"loc": 0.0, "scale": 1.0, "bad": 99.0})

    def test_error_at_construction_not_call(self):
        with pytest.raises(TypeError):
            FixedPriorLayer(coef_kwargs={"bad": 1.0})


class TestInterceptLayerValidation:
    def test_valid_defaults(self):
        InterceptLayer()

    def test_bad_coef_kwargs(self):
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            InterceptLayer(coef_kwargs={"loc": 0.0, "scale": 1.0, "extra": 0.0})


# --------------------------------------------------------------------------- #
# Custom distributions
# --------------------------------------------------------------------------- #


class TestCustomDistValidation:
    def test_valid_custom_dist(self):
        """Exponential coef with correct kwarg."""
        AdaptiveLayer(
            coef_dist=distributions.Laplace,
            coef_kwargs={"loc": 0.0},
        )

    def test_invalid_custom_dist_kwargs(self):
        """Wrong kwarg for Laplace raises at construction."""
        with pytest.raises(TypeError, match="Invalid distribution kwargs"):
            AdaptiveLayer(
                coef_dist=distributions.Laplace,
                coef_kwargs={"loc": 0.0, "bad_kwarg": 1.0},
            )
