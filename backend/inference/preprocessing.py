"""Multi-Quality RGB Error Level Analysis preprocessing pipeline."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance

ELA_MEAN = torch.tensor(
    [0.3021, 0.2938, 0.2847, 0.2156, 0.2089, 0.2012, 0.1043, 0.1005, 0.0963], dtype=torch.float32
)
ELA_STD = torch.tensor(
    [0.2187, 0.2134, 0.2098, 0.1876, 0.1831, 0.1799, 0.1234, 0.1198, 0.1172], dtype=torch.float32
)
ELA_QUALITIES = [75, 85, 95]
DEFAULT_IMAGE_SIZE = 384
IN_CHANNELS = 9


def compute_ela_rgb(image: Image.Image, quality: int, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    image_rgb = image.convert("RGB")
    buf = io.BytesIO()
    image_rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    ela = ImageChops.difference(image_rgb, resaved)
    extrema = ela.getextrema()
    max_diff = max(v[1] for v in extrema)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    ela = ela.resize((size, size), Image.BILINEAR)
    return np.array(ela)


def compute_multi_quality_rgb_ela(
    image: Image.Image, qualities=None, size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    if qualities is None:
        qualities = ELA_QUALITIES
    channels = [compute_ela_rgb(image, q, size) for q in qualities]
    return np.concatenate(channels, axis=-1)


def preprocess_image(
    image: Image.Image, size=DEFAULT_IMAGE_SIZE, qualities=None, mean=None, std=None
) -> torch.Tensor:
    if mean is None:
        mean = ELA_MEAN
    if std is None:
        std = ELA_STD
    mqela = compute_multi_quality_rgb_ela(image, qualities, size)
    tensor = torch.from_numpy(mqela.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1)
    for c in range(tensor.shape[0]):
        tensor[c] = (tensor[c] - mean[c]) / std[c]
    return tensor.unsqueeze(0)
