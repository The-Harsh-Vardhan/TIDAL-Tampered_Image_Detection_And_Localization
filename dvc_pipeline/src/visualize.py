"""
dvc_pipeline/src/visualize.py
==============================
Generate before/after comparison grids for tamper detection results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import CASIASegmentationDataset, collect_image_paths
from models import build_model
from utils import get_device, load_params, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def visualize(params: dict) -> None:
    """Generate visualization grids."""
    seed_everything(params["seed"])
    device = get_device()

    model_cfg = params["model"]
    vis_cfg = params["visualization"]
    prep_cfg = params["preprocessing"]
    data_cfg = params["data"]

    # Load model
    model = build_model(
        encoder=model_cfg["encoder"],
        encoder_weights=None,
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    checkpoint = torch.load("../models/best_model.pt", map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load ELA stats
    with open("artifacts/ela_statistics.json") as f:
        stats = json.load(f)
    ela_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    ela_std = torch.tensor(stats["std"], dtype=torch.float32)

    # Collect test images (tampered only for visualization)
    dataset_root = data_cfg["dataset_root"]
    extensions = set(data_cfg.get("supported_extensions", [".jpg", ".jpeg", ".png"]))
    tp_paths = collect_image_paths(f"{dataset_root}/Tp", extensions)

    n_samples = min(vis_cfg["num_samples"], len(tp_paths))
    sample_paths = tp_paths[:n_samples]

    ds = CASIASegmentationDataset(
        image_paths=sample_paths,
        mask_paths=[None] * n_samples,
        labels=[1] * n_samples,
        ela_mean=ela_mean,
        ela_std=ela_std,
        img_size=prep_cfg["image_size"],
        qualities=prep_cfg["ela_qualities"],
    )

    output_dir = Path(vis_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate grid
    cols = vis_cfg.get("grid_cols", 4)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        row = i // cols
        col_base = (i % cols) * 3

        tensor, mask, _ = ds[i]
        original = Image.open(sample_paths[i]).convert("RGB").resize(
            (prep_cfg["image_size"], prep_cfg["image_size"])
        )

        # Predict
        with torch.no_grad():
            pred = model(tensor.unsqueeze(0).to(device))
            pred_mask = torch.sigmoid(pred.float()).squeeze().cpu().numpy()

        # Original
        axes[row, col_base].imshow(np.array(original))
        axes[row, col_base].set_title("Original", fontsize=8)
        axes[row, col_base].axis("off")

        # Predicted mask
        axes[row, col_base + 1].imshow(pred_mask, cmap="hot", vmin=0, vmax=1)
        axes[row, col_base + 1].set_title("Predicted Mask", fontsize=8)
        axes[row, col_base + 1].axis("off")

        # Overlay
        orig_arr = np.array(original).astype(np.float32) / 255.0
        overlay = orig_arr.copy()
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 0] = pred_mask
        overlay = overlay * 0.6 + mask_rgb * 0.4
        axes[row, col_base + 2].imshow(np.clip(overlay, 0, 1))
        axes[row, col_base + 2].set_title("Overlay", fontsize=8)
        axes[row, col_base + 2].axis("off")

    # Hide empty axes
    for i in range(n_samples, rows * cols):
        row = i // cols
        col_base = (i % cols) * 3
        for j in range(3):
            if col_base + j < axes.shape[1]:
                axes[row, col_base + j].axis("off")

    plt.suptitle("TIDAL — Tamper Detection Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved comparison grid to %s", output_dir / "comparison_grid.png")


if __name__ == "__main__":
    params = load_params("params.yaml")
    visualize(params)
