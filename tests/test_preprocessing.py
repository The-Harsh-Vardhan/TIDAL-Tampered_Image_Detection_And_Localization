"""Test ELA preprocessing."""

import numpy as np
import torch

from backend.inference.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    ELA_MEAN,
    ELA_STD,
    IN_CHANNELS,
    compute_ela_rgb,
    compute_multi_quality_rgb_ela,
    preprocess_image,
)


def test_ela_rgb_shape(sample_image):
    r = compute_ela_rgb(sample_image, 85, 384)
    assert r.shape == (384, 384, 3) and r.dtype == np.uint8


def test_multi_q_ela_shape(sample_image):
    r = compute_multi_quality_rgb_ela(sample_image, size=384)
    assert r.shape == (384, 384, 9)


def test_preprocess_shape(sample_image):
    r = preprocess_image(sample_image)
    assert r.shape == (1, IN_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert isinstance(r, torch.Tensor) and r.dtype == torch.float32


def test_constants():
    assert ELA_MEAN.shape == (9,) and ELA_STD.shape == (9,) and (ELA_STD > 0).all()
