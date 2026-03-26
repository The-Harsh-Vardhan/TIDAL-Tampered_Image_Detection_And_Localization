"""
dvc_pipeline/src/evaluate.py
==============================
Evaluate the trained TIDAL model on the test set.

Outputs metrics/eval_metrics.json and evaluation_results/ directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CASIASegmentationDataset, collect_image_paths
from models import build_model
from utils import compute_pixel_f1, get_device, load_params, save_metrics, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate(params: dict) -> None:
    """Evaluate on test set and check against quality thresholds."""
    seed_everything(params["seed"])
    device = get_device()

    model_cfg = params["model"]
    eval_cfg = params["evaluation"]
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

    # Build test set
    from sklearn.model_selection import train_test_split

    dataset_root = data_cfg["dataset_root"]
    extensions = set(data_cfg.get("supported_extensions", [".jpg", ".jpeg", ".png"]))
    au_paths = collect_image_paths(f"{dataset_root}/Au", extensions)
    tp_paths = collect_image_paths(f"{dataset_root}/Tp", extensions)
    all_paths = au_paths + tp_paths
    all_labels = [0] * len(au_paths) + [1] * len(tp_paths)

    seed = params["seed"]
    _, temp_idx = train_test_split(range(len(all_paths)), test_size=0.30, stratify=all_labels, random_state=seed)
    temp_labels = [all_labels[i] for i in temp_idx]
    _, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed)

    test_ds = CASIASegmentationDataset(
        image_paths=[all_paths[i] for i in test_idx],
        mask_paths=[None] * len(test_idx),
        labels=[all_labels[i] for i in test_idx],
        ela_mean=ela_mean,
        ela_std=ela_std,
        img_size=prep_cfg["image_size"],
        qualities=prep_cfg["ela_qualities"],
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Inference
    all_preds, all_masks = [], []
    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)
            preds = model(images)
            probs = torch.sigmoid(preds.float())
            all_preds.append(probs.cpu().numpy())
            all_masks.append(masks.numpy())

    preds_np = np.concatenate(all_preds)
    masks_np = np.concatenate(all_masks)
    metrics = compute_pixel_f1(preds_np, masks_np)

    # Quality gate
    thresholds = eval_cfg["thresholds"]
    passed = (
        metrics["pixel_f1"] >= thresholds["min_pixel_f1"]
        and metrics["pixel_iou"] >= thresholds["min_iou"]
    )
    metrics["quality_gate_passed"] = passed
    metrics["test_samples"] = len(test_idx)

    # Save
    Path("evaluation_results").mkdir(exist_ok=True)
    save_metrics(metrics, "metrics/eval_metrics.json")

    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("Quality gate: %s", "PASSED ✓" if passed else "FAILED ✗")


if __name__ == "__main__":
    params = load_params("params.yaml")
    evaluate(params)
