"""
dvc_pipeline/src/utils.py
==========================
Shared utilities for the DVC training pipeline.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_params(path: str = "params.yaml") -> dict:
    """Load DVC params from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics to JSON file (DVC-compatible)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def get_device() -> torch.device:
    """Resolve compute device."""
    device_str = os.environ.get("DEVICE", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def compute_pixel_f1(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute pixel-level precision, recall, F1, and IoU.

    Args:
        preds: Predicted probabilities, shape (N, H, W) or (N, 1, H, W).
        targets: Binary ground truth, same shape.
        threshold: Binarization threshold.

    Returns:
        Dict with precision, recall, f1, iou.
    """
    preds_bin = (preds > threshold).astype(np.float32).flatten()
    targets_flat = targets.astype(np.float32).flatten()

    tp = (preds_bin * targets_flat).sum()
    fp = (preds_bin * (1 - targets_flat)).sum()
    fn = ((1 - preds_bin) * targets_flat).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
    }
