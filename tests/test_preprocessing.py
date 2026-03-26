"""
tests/test_preprocessing.py
=============================
Unit tests for ELA preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from backend.inference.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    ELA_MEAN,
    ELA_STD,
    IN_CHANNELS,
    compute_ela_rgb,
    compute_multi_quality_rgb_ela,
    preprocess_image,
)


class TestComputeElaRgb:
    """Tests for single-quality ELA computation."""

    def test_output_shape(self, sample_image: Image.Image):
        result = compute_ela_rgb(sample_image, quality=85, size=384)
        assert result.shape == (384, 384, 3)

    def test_output_dtype(self, sample_image: Image.Image):
        result = compute_ela_rgb(sample_image, quality=85, size=100)
        assert result.dtype == np.uint8

    def test_custom_size(self, sample_image: Image.Image):
        result = compute_ela_rgb(sample_image, quality=85, size=128)
        assert result.shape == (128, 128, 3)

    def test_different_qualities_differ(self, sample_image: Image.Image):
        q75 = compute_ela_rgb(sample_image, quality=75, size=100)
        q95 = compute_ela_rgb(sample_image, quality=95, size=100)
        # Different qualities should produce different ELA maps
        assert not np.array_equal(q75, q95)


class TestMultiQualityRgbEla:
    """Tests for 9-channel Multi-Q ELA stacking."""

    def test_output_shape(self, sample_image: Image.Image):
        result = compute_multi_quality_rgb_ela(sample_image, size=384)
        assert result.shape == (384, 384, 9)

    def test_custom_qualities(self, sample_image: Image.Image):
        result = compute_multi_quality_rgb_ela(
            sample_image, qualities=[50, 90], size=100
        )
        assert result.shape == (100, 100, 6)  # 2 qualities × 3 channels

    def test_output_range(self, sample_image: Image.Image):
        result = compute_multi_quality_rgb_ela(sample_image, size=100)
        assert result.min() >= 0
        assert result.max() <= 255


class TestPreprocessImage:
    """Tests for the full preprocessing pipeline."""

    def test_output_shape(self, sample_image: Image.Image):
        result = preprocess_image(sample_image)
        assert result.shape == (1, IN_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    def test_output_is_tensor(self, sample_image: Image.Image):
        result = preprocess_image(sample_image)
        assert isinstance(result, torch.Tensor)

    def test_output_dtype(self, sample_image: Image.Image):
        result = preprocess_image(sample_image)
        assert result.dtype == torch.float32

    def test_custom_size(self, sample_image: Image.Image):
        result = preprocess_image(sample_image, size=128)
        assert result.shape == (1, 9, 128, 128)

    def test_normalization_applied(self, sample_image: Image.Image):
        result = preprocess_image(sample_image)
        # After normalization with non-zero mean/std, values should
        # span both positive and negative
        assert result.min() < 0 or result.max() > 1


class TestConstants:
    """Test that preprocessing constants are consistent."""

    def test_ela_mean_shape(self):
        assert ELA_MEAN.shape == (9,)

    def test_ela_std_shape(self):
        assert ELA_STD.shape == (9,)

    def test_ela_std_positive(self):
        assert (ELA_STD > 0).all()

    def test_in_channels(self):
        assert IN_CHANNELS == 9
