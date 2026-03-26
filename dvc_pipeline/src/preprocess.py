"""
dvc_pipeline/src/preprocess.py
===============================
Compute ELA normalization statistics from the training set.

Outputs artifacts/ela_statistics.json with per-channel mean and std.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from dataset import CASIASegmentationDataset, collect_image_paths
from utils import load_params, save_metrics, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def preprocess(params: dict) -> None:
    """Compute and save ELA channel statistics."""
    seed_everything(params["seed"])

    data_cfg = params["data"]
    prep_cfg = params["preprocessing"]

    dataset_root = data_cfg["dataset_root"]
    extensions = set(data_cfg.get("supported_extensions", [".jpg", ".jpeg", ".png"]))

    au_paths = collect_image_paths(f"{dataset_root}/Au", extensions)
    tp_paths = collect_image_paths(f"{dataset_root}/Tp", extensions)
    all_paths = au_paths + tp_paths

    logger.info("Computing ELA statistics from %d images (sampling %d)",
                len(all_paths), prep_cfg["normalization_samples"])

    # Sample subset
    n_samples = min(prep_cfg["normalization_samples"], len(all_paths))
    indices = np.random.choice(len(all_paths), n_samples, replace=False)

    # Dummy dataset just for ELA computation
    dummy_ds = CASIASegmentationDataset(
        image_paths=[all_paths[i] for i in indices],
        mask_paths=[None] * n_samples,
        labels=[0] * n_samples,
        ela_mean=torch.zeros(9),
        ela_std=torch.ones(9),
        img_size=prep_cfg["image_size"],
        qualities=prep_cfg["ela_qualities"],
    )

    # Compute channel statistics
    all_pixels = []
    for i in range(len(dummy_ds)):
        try:
            tensor, _, _ = dummy_ds[i]
            # tensor is already normalized with zeros/ones, so just get raw values
            all_pixels.append(tensor.numpy().reshape(9, -1))
        except Exception:
            continue

    if not all_pixels:
        logger.error("No valid images found!")
        return

    pixels = np.concatenate(all_pixels, axis=1)  # (9, total_pixels)
    mean = pixels.mean(axis=1).tolist()
    std = pixels.std(axis=1).tolist()
    std = [max(s, 1e-6) for s in std]

    # Save
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"mean": mean, "std": std, "n_samples": n_samples}
    with open(output_dir / "ela_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    save_metrics(
        {"n_images": len(all_paths), "n_samples": n_samples, "au_count": len(au_paths), "tp_count": len(tp_paths)},
        "metrics/preprocess_metrics.json",
    )

    logger.info("ELA statistics saved to artifacts/ela_statistics.json")
    logger.info("  Mean: %s", [f"{v:.4f}" for v in mean])
    logger.info("  Std:  %s", [f"{v:.4f}" for v in std])


if __name__ == "__main__":
    params = load_params("params.yaml")
    preprocess(params)
