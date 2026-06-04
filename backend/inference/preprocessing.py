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
) -> tuple[np.ndarray, dict]:
    """Return (scaled_grayscale_array, energy_stats_dict)."""
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
    ela_scaled = ImageEnhance.Brightness(ela).enhance(scale).convert("L")
    ela_scaled = ela_scaled.resize((size, size), Image.BILINEAR)
    arr = np.array(ela_scaled, dtype=np.uint8)
    # Energy stats computed on the raw (pre-scale) difference image
    ela_gray_raw = np.array(ela.convert("L"), dtype=np.float32)
    stats = {
        "quality": quality,
        "mean_energy": round(float(ela_gray_raw.mean()), 4),
        "max_energy": round(float(ela_gray_raw.max()), 4),
        "scale_factor": round(float(scale), 4),
    }
    return arr, stats


def compute_multi_quality_ela(
    image: Image.Image, qualities=None, size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    if qualities is None:
        qualities = ELA_QUALITIES
    channels = [compute_ela_grayscale(image, q, size)[0] for q in qualities]
    return np.stack(channels, axis=-1)


def compute_ela_stats(
    image: Image.Image, qualities=None, size: int = DEFAULT_IMAGE_SIZE
) -> list[dict]:
    """Return per-quality ELA energy statistics (Feature 3).

    Each entry: {quality, mean_energy, max_energy, scale_factor}
    Higher mean_energy at lower quality suggests heavier re-compression
    in tampered regions.
    """
    if qualities is None:
        qualities = ELA_QUALITIES
    return [compute_ela_grayscale(image, q, size)[1] for q in qualities]


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
