"""
dvc_pipeline/src/train.py
==========================
Training CLI for the TIDAL DVC pipeline.

Usage:
    python src/train.py                    # uses params.yaml defaults
    python src/train.py --params params.yaml
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import CASIASegmentationDataset, collect_image_paths
from models import build_model, get_model_info
from utils import compute_pixel_f1, get_device, load_params, save_metrics, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _build_dataloaders(params: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders from params."""
    from sklearn.model_selection import train_test_split

    data_cfg = params["data"]
    prep_cfg = params["preprocessing"]
    train_cfg = params["training"]

    # Collect image paths
    dataset_root = data_cfg["dataset_root"]
    au_dir = f"{dataset_root}/Au"
    tp_dir = f"{dataset_root}/Tp"

    extensions = set(data_cfg.get("supported_extensions", [".jpg", ".jpeg", ".png"]))
    au_paths = collect_image_paths(au_dir, extensions)
    tp_paths = collect_image_paths(tp_dir, extensions)

    all_paths = au_paths + tp_paths
    all_labels = [0] * len(au_paths) + [1] * len(tp_paths)
    all_masks = [None] * len(all_paths)  # GT masks loaded separately if available

    # Stratified split
    seed = params["seed"]
    train_idx, temp_idx = train_test_split(
        range(len(all_paths)), test_size=0.30, stratify=all_labels, random_state=seed
    )
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed
    )

    # ELA statistics (precomputed or compute on the fly)
    import json
    from pathlib import Path

    stats_path = Path("artifacts/ela_statistics.json")
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        ela_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        ela_std = torch.tensor(stats["std"], dtype=torch.float32)
    else:
        ela_mean = torch.zeros(9)
        ela_std = torch.ones(9)
        logger.warning("ELA statistics not found, using defaults")

    def _make_loader(indices: list[int], shuffle: bool) -> DataLoader:
        ds = CASIASegmentationDataset(
            image_paths=[all_paths[i] for i in indices],
            mask_paths=[all_masks[i] for i in indices],
            labels=[all_labels[i] for i in indices],
            ela_mean=ela_mean,
            ela_std=ela_std,
            img_size=prep_cfg["image_size"],
            qualities=prep_cfg["ela_qualities"],
        )
        return DataLoader(
            ds,
            batch_size=train_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=train_cfg["num_workers"],
            pin_memory=True,
            drop_last=shuffle,
        )

    return _make_loader(train_idx, True), _make_loader(val_idx, False), _make_loader(test_idx, False)


def train(params: dict) -> None:
    """Run the full training loop."""
    seed_everything(params["seed"])
    device = get_device()
    model_cfg = params["model"]
    train_cfg = params["training"]

    logger.info("Device: %s", device)

    # Build model
    model = build_model(
        encoder=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        freeze_strategy=model_cfg["freeze_strategy"],
    ).to(device)

    info = get_model_info(model)
    logger.info("Model: %s trainable / %s total (%.1f%%)",
                f"{info['trainable_params']:,}", f"{info['total_params']:,}", info['trainable_pct'])

    # Data
    train_loader, val_loader, _ = _build_dataloaders(params)
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    # Loss, optimizer, scheduler
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def criterion(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=train_cfg["scheduler_factor"],
        patience=train_cfg["scheduler_patience"],
    )
    scaler = GradScaler(enabled=train_cfg["use_amp"])

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=train_cfg["use_amp"]):
                preds = model(images)
                loss = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_masks_list = [], []
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                with autocast(enabled=train_cfg["use_amp"]):
                    preds = model(images)
                    loss = criterion(preds, masks)
                val_loss += loss.item()
                probs = torch.sigmoid(preds.float())
                all_preds.append(probs.cpu().numpy())
                all_masks_list.append(masks.cpu().numpy())
        val_loss /= len(val_loader)

        preds_np = np.concatenate(all_preds)
        masks_np = np.concatenate(all_masks_list)
        metrics = compute_pixel_f1(preds_np, masks_np)
        val_f1 = metrics["pixel_f1"]

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_f1=%.4f (%.1fs)",
            epoch, train_cfg["epochs"], train_loss, val_loss, val_f1, elapsed,
        )

        # Early stopping + checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "best_f1": best_f1},
                "../models/best_model.pt",
            )
            logger.info("  → Saved best model (F1=%.4f)", best_f1)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save training metrics
    save_metrics(
        {
            "best_pixel_f1": best_f1,
            "total_epochs": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            **{k: v for k, v in get_model_info(model).items()},
        },
        "metrics/train_metrics.json",
    )
    logger.info("Training complete — best F1=%.4f", best_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIDAL training pipeline")
    parser.add_argument("--params", default="params.yaml", help="Path to params YAML")
    args = parser.parse_args()
    params = load_params(args.params)
    train(params)
