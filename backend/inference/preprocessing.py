"""
backend/inference/preprocessing.py
===================================
Multi-Quality RGB Error Level Analysis (ELA) preprocessing pipeline.

Extracts the core ELA computation from the research notebook (vrp19.py)
into a clean, production-ready module with precomputed normalization
statistics from the training set.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance

# ---------------------------------------------------------------------------
# Precomputed ELA channel statistics (from 500 training samples, vrp19.py)
# 9 channels: [Q75_R, Q75_G, Q75_B, Q85_R, Q85_G, Q85_B, Q95_R, Q95_G, Q95_B]
# ---------------------------------------------------------------------------
ELA_MEAN = torch.tensor(
    [0.3021, 0.2938, 0.2847, 0.2156, 0.2089, 0.2012, 0.1043, 0.1005, 0.0963],
    dtype=torch.float32,
)
ELA_STD = torch.tensor(
    [0.2187, 0.2134, 0.2098, 0.1876, 0.1831, 0.1799, 0.1234, 0.1198, 0.1172],
    dtype=torch.float32,
)

ELA_QUALITIES = [75, 85, 95]
DEFAULT_IMAGE_SIZE = 384
IN_CHANNELS = 9


def compute_ela_rgb(
    image: Image.Image,
    quality: int,
    size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Compute ELA at a given JPEG quality level, returning RGB (H, W, 3).

    Args:
        image: PIL Image in RGB mode.
        quality: JPEG recompression quality (e.g. 75, 85, 95).
        size: Target spatial dimension (square).

    Returns:
        ELA map as uint8 array of shape (H, W, 3).
    """
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
    return np.array(ela)  # (H, W, 3) uint8


def compute_multi_quality_rgb_ela(
    image: Image.Image,
    qualities: list[int] | None = None,
    size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Stack RGB ELA at multiple quality levels → (H, W, 9).

    Args:
        image: PIL Image.
        qualities: List of JPEG quality levels. Defaults to [75, 85, 95].
        size: Target spatial dimension.

    Returns:
        Concatenated ELA array of shape (H, W, 9) as uint8.
    """
    if qualities is None:
        qualities = ELA_QUALITIES
    channels = [compute_ela_rgb(image, q, size) for q in qualities]
    return np.concatenate(channels, axis=-1)  # (H, W, 9)


def preprocess_image(
    image: Image.Image,
    size: int = DEFAULT_IMAGE_SIZE,
    qualities: list[int] | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full preprocessing pipeline: image → normalized 9-channel ELA tensor.

    Args:
        image: Input PIL Image.
        size: Target spatial dimension.
        qualities: JPEG quality levels for ELA.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.

    Returns:
        Tensor of shape (1, 9, H, W) ready for model inference.
    """
    if mean is None:
        mean = ELA_MEAN
    if std is None:
        std = ELA_STD

    # Compute 9-channel Multi-Q RGB ELA
    mqela = compute_multi_quality_rgb_ela(image, qualities, size)

    # Normalize to [0, 1] then standardize
    tensor = torch.from_numpy(mqela.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1)  # (9, H, W)

    for c in range(tensor.shape[0]):
        tensor[c] = (tensor[c] - mean[c]) / std[c]

    return tensor.unsqueeze(0)  # (1, 9, H, W)
