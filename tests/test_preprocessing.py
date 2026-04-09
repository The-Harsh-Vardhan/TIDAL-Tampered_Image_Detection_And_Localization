"""Test vR.P.30.1 ELA preprocessing."""

import numpy as np
import torch

from backend.inference.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    ELA_MEAN,
    ELA_STD,
    IN_CHANNELS,
    compute_ela_grayscale,
    compute_multi_quality_ela,
    preprocess_image,
)


def test_ela_grayscale_shape(sample_image):
    result = compute_ela_grayscale(sample_image, 85, 384)
    assert result.shape == (384, 384)
    assert result.dtype == np.uint8


def test_multi_q_ela_shape(sample_image):
    result = compute_multi_quality_ela(sample_image, size=384)
    assert result.shape == (384, 384, 3)
    assert result.dtype == np.uint8


def test_preprocess_shape(sample_image):
    result = preprocess_image(sample_image)
    assert result.shape == (1, IN_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32


def test_constants():
    assert ELA_MEAN.shape == (3,)
    assert ELA_STD.shape == (3,)
    assert (ELA_STD > 0).all()
