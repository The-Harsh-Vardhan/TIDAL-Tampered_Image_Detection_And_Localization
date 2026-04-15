"""Grayscale Multi-Quality ELA preprocessing for vR.P.30.1 inference."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance

ELA_MEAN = torch.tensor([0.0684, 0.0605, 0.0402], dtype=torch.float32)
ELA_STD = torch.tensor([0.0656, 0.0604, 0.0471], dtype=torch.float32)
ELA_QUALITIES = [75, 85, 95]
DEFAULT_IMAGE_SIZE = 384
IN_CHANNELS = 3


def compute_ela_grayscale(
    image: Image.Image, quality: int, size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    image_rgb = image.convert("RGB")
    buf = io.BytesIO()
    image_rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(image_rgb, resaved)
    extrema = ela.getextrema()
    max_diff = max(v[1] for v in extrema)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale).convert("L")
    ela = ela.resize((size, size), Image.BILINEAR)
    return np.array(ela, dtype=np.uint8)


def compute_multi_quality_ela(
    image: Image.Image, qualities=None, size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    if qualities is None:
        qualities = ELA_QUALITIES
    channels = [compute_ela_grayscale(image, q, size) for q in qualities]
    return np.stack(channels, axis=-1)


def preprocess_image(
    image: Image.Image, size=DEFAULT_IMAGE_SIZE, qualities=None, mean=None, std=None
) -> torch.Tensor:
    if mean is None:
        mean = ELA_MEAN
    if std is None:
        std = ELA_STD
    mqela = compute_multi_quality_ela(image, qualities, size)
    tensor = torch.from_numpy(mqela.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1)
    for c in range(tensor.shape[0]):
        tensor[c] = (tensor[c] - mean[c]) / (std[c] + 1e-8)
    return tensor.unsqueeze(0)
