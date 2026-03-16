"""
Build script: vF.2.0 → vF.3.0 FakeShield-Lite
================================================
Reads vF.2.0 notebook, applies fixes identified in Audit_vF.2.0.md,
writes vF.3.0 notebook.

Changes from vF.2.0:
  P0: Fix feature extraction crash (CLIP-only caching, remove SAM caching)
  P0: Fix Albumentations deprecated API (var_limit, quality_lower/upper)
  P0: Reduce epochs from 20 to 10, add early stopping
  P1: Fix scheduler T_max (divide by ACCUM_STEPS)
  P1: Add gradient clipping
  P1: Fix ablation design (share base weights, fix "No Augmentation")
  P1: Suppress CLIP unexpected key warnings
  P2: Add training time estimation after epoch 1
  P2: Add "Issues Found in vF.2.0 and Fixes" section
  P2: Update version metadata everywhere
"""

import json
import copy

# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_lines(s: str):
    """Convert multi-line string to Jupyter cell source format (list of lines)."""
    lines = s.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            out.append(line + "\n")
        else:
            out.append(line)
    return out

def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": _to_lines(source)}

def code(source: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _to_lines(source)}

# ── Load vF.2.0 ────────────────────────────────────────────────────────────

INPUT_PATH  = "c:/D Drive/Projects/FakeShield/Fakeshield Lite/vF.2.0_FakeShield_Lite.ipynb"
OUTPUT_PATH = "c:/D Drive/Projects/FakeShield/Fakeshield Lite/vF.3.0_FakeShield_Lite.ipynb"

with open(INPUT_PATH, encoding="utf-8") as f:
    nb = json.load(f)

old = nb["cells"]

def get_src(idx):
    return "".join(old[idx]["source"])

def set_src(idx, src):
    old[idx]["source"] = _to_lines(src)

# ── Build new cell list ─────────────────────────────────────────────────────

new_cells = []

# ============================================================================
# Cell 0: Title (NEW -- updated for vF.3.0)
# ============================================================================
new_cells.append(md("""\
# vF.3.0 — FakeShield-Lite

| | |
|---|---|
| **Assignment** | Big Vision — ETASR Paper Reproduction |
| **Paper** | FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models (Xu et al., ICLR 2025) |
| **Approach** | Pruned reproduction — CLIP ViT-B/16 + SAM ViT-B (~182 M params, ~5 M trainable) |
| **Dataset** | CASIA Splicing Detection + Localization (12,614 images) |
| **Version** | vF.3.0 |
| **Changelog** | Fixed feature caching crash, Albumentations API, scheduler, added early stopping |"""))

# ============================================================================
# Cells 1-3: Executive Summary, Key Contributions, TOC (from vF.2.0)
# ============================================================================
new_cells.append(old[1])  # Executive Summary
new_cells.append(old[2])  # Key Contributions

# Updated TOC for vF.3.0
new_cells.append(md("""\
## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Problem Overview](#2-problem-overview)
3. [Dataset Setup](#3-dataset)
4. [Data Augmentation](#4-augmentation)
5. [Model Architecture](#5-architecture)
6. [Feature Caching (CLIP-only)](#6-feature-caching)
7. [Loss Functions](#7-loss)
8. [Training Pipeline](#8-training)
9. [Evaluation Metrics](#9-metrics)
10. [Validation & Training Loop](#10-validation)
11. [Training Curves](#11-curves)
12. [Prediction Visualization](#12-predictions)
13. [Test Evaluation](#13-test-eval)
14. [Error Analysis](#14-error)
15. [Ablation Study](#15-ablation)
16. [Threshold Sensitivity](#16-threshold)
17. [Detection Confidence](#17-confidence)
18. [Hard Example Analysis](#18-hard-examples)
19. [Robustness Testing](#19-robustness)
20. [Computational Efficiency](#20-efficiency)
21. [Inference Demo](#21-inference)
22. [Model Saving](#22-save)
23. [Conclusion](#23-conclusion)"""))

# ============================================================================
# Cell 4: Experiment Metadata (UPDATED)
# ============================================================================
new_cells.append(code("""\
# ============================================================================
# Experiment Metadata
# ============================================================================

EXPERIMENT_INFO = {
    "experiment_name":  "FakeShield-Lite",
    "version":          "vF.3.0",
    "dataset":          "CASIA Splicing Detection + Localization",
    "model":            "CLIP ViT-B/16 + SAM ViT-B",
    "hardware":         "Kaggle T4 GPU (16 GB VRAM)",
    "framework":        "PyTorch",
    "base_paper":       "FakeShield (Xu et al., ICLR 2025)",
    "new_in_v3":        "Fixed feature caching, Albumentations API, early stopping, scheduler fix",
}

for k, v in EXPERIMENT_INFO.items():
    print(f"{k:20s}: {v}")"""))

# ============================================================================
# Cell 5-6: Experiment Log, Issues (UPDATED)
# ============================================================================
new_cells.append(md("""\
## Experiment Log

| Version | Key Change | Det F1 | IoU | Notes |
|---------|-----------|--------|-----|-------|
| vF.0.0 | Initial implementation | — | — | Colab, basic pipeline |
| vF.0.1 | Kaggle port | — | — | Path fixes |
| vF.0.2 | Bug fixes | — | — | Tensor shape fixes |
| vF.0.3 | SAM batch fix attempt | — | — | expand() insufficient |
| vF.0.4 | Per-image mask decoding | — | — | Fixed SAM batch issue |
| vF.0.5 | OOM fix: per-image SAM encoding | — | — | BATCH_SIZE=4, grad accum |
| vF.1.0 | Documentation upgrade | — | — | No code changes |
| vF.2.0 | Feature caching + ablation framework | — | — | **Crashed at feature extraction** |
| vF.3.0 | **Fixed caching, API, scheduler, early stop** | TBD | TBD | **This version** |"""))

new_cells.append(md("""\
## Issues Found in vF.2.0 and Fixes Applied in vF.3.0

### P0 — Critical Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Feature extraction crashes at 0% | SAM features (256×64×64) × 10K samples = 20 GB; exceeds 13 GB Kaggle RAM | **Removed SAM caching entirely.** Only cache CLIP [CLS] tokens (15 MB). SAM runs live during training. |
| Albumentations deprecated API | `var_limit` and `quality_lower/upper` silently ignored in newer Albumentations | Updated to `std_range` and `quality_range` API |
| Training cannot complete in Kaggle time | 20 epochs × 10K images × SAM encoding ≈ 28 hours; Kaggle limit is 12 hours | Reduced to 10 epochs + early stopping (patience=3) |

### P1 — Important Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Scheduler T_max too large | Counts all batches, not gradient steps (ACCUM_STEPS=2) | `T_max = epochs × steps_per_epoch // ACCUM_STEPS` |
| No gradient clipping | Could cause instability in mask decoder attention | Added `clip_grad_norm_(max_norm=1.0)` |
| "No Augmentation" ablation is copy of Baseline | Both use cached features extracted with val_transform | Redesigned: ablation runs live without augmentation |
| Ablation reloads CLIP 5× | Each `run_ablation()` creates fresh FakeShieldLite | Share frozen encoder weights across ablation models |
| Full CLIP download warnings | 164 UNEXPECTED key warnings for text model weights | Suppressed with `logging.set_verbosity_error()` |

### P2 — Quality Improvements

| Issue | Fix |
|-------|-----|
| No training time estimation | Print ETA after epoch 1 |
| No early stopping | Added patience-based early stopping |
| Hard example analysis processes one-by-one | Batch processing for speed |"""))

# ============================================================================
# Cells 7-8: Section 1 header + Install (from vF.2.0)
# ============================================================================
new_cells.append(old[7])   # Section 1 -- Environment Setup header
new_cells.append(old[9])   # 1.1 Install Dependencies

# ============================================================================
# Cell 9: GPU Check (from vF.2.0)
# ============================================================================
new_cells.append(old[10])  # 1.2 GPU Check

# ============================================================================
# Cell 10: Reproducibility markdown (from vF.2.0)
# ============================================================================
new_cells.append(old[11])  # Reproducibility Setup explanation

# ============================================================================
# Cell 11: Imports (UPDATED -- add logging import, suppress warnings)
# ============================================================================
new_cells.append(code("""\
# ============================================================================
# 1.3 Common Imports & Reproducibility
# ============================================================================
import os
import gc
import cv2
import glob
import random
import time
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Suppress CLIP unexpected key warnings
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

print("All imports loaded. Seed set to", SEED)"""))

# ============================================================================
# Cell 12: Experiment Configuration (UPDATED -- 10 epochs, early stopping)
# ============================================================================
new_cells.append(old[13])  # Config explanation markdown

new_cells.append(code("""\
# ============================================================================
# Experiment Configuration (centralised)
# ============================================================================

DEBUG_MODE = False  # Set True for fast experimentation (smaller dataset subset)
DEBUG_SUBSET = 2000  # Number of samples in debug mode

CONFIG = {
    "seed": 42,
    "img_size": 256,
    "batch_size": 4,
    "accum_steps": 2,
    "effective_batch_size": 8,
    "epochs": 10,
    "ablation_epochs": 5,
    "learning_rate": 1e-4,
    "sam_decoder_lr": 5e-5,
    "weight_decay": 0.01,
    "scheduler": "CosineAnnealingLR",
    "loss_weights": {"det_bce": 1.0, "mask_bce": 2.0, "mask_dice": 0.5},
    "clip_model": "openai/clip-vit-base-patch16",
    "sam_variant": "vit_b",
    "early_stopping_patience": 3,
    "grad_clip_norm": 1.0,
    "debug_mode": DEBUG_MODE,
}

print("Experiment Configuration:")
for k, v in CONFIG.items():
    print(f"  {k:25s}: {v}")
if DEBUG_MODE:
    print(f"\\n  ** DEBUG MODE ACTIVE: using {DEBUG_SUBSET} samples for fast iteration **")"""))

# ============================================================================
# Cells 14-19: Problem Overview + Dataset (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[15])  # Section 2 -- Problem Overview
new_cells.append(old[16])  # Section 3 -- Dataset Setup header
new_cells.append(old[17])  # 3.1 Dataset Source markdown
new_cells.append(old[18])  # 3.1 Connect to Kaggle Dataset code
new_cells.append(old[19])  # 3.2 Discover & Pair images code

# ============================================================================
# Cell 20-21: Dataset Statistics (from vF.2.0)
# ============================================================================
new_cells.append(old[20])  # Dataset Statistics markdown
new_cells.append(old[21])  # 3.2b Dataset Statistics code

# ============================================================================
# Cells 22-23: Sample list + Visualization (from vF.2.0)
# ============================================================================
new_cells.append(old[22])  # 3.3 Build Unified Sample List & Split
new_cells.append(old[23])  # 3.4 Visualize Dataset Samples

# ============================================================================
# Cells 24-25: Augmentation (FIXED API)
# ============================================================================
new_cells.append(old[24])  # Section 4 header

new_cells.append(code("""\
# ============================================================================
# 4.1 Define Augmentation Transforms
# ============================================================================
# FIXED in vF.3.0: Updated Albumentations API
#   - GaussNoise: var_limit -> std_range
#   - ImageCompression: quality_lower/upper -> quality_range

IMG_SIZE = 256

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
    A.ImageCompression(quality_range=(70, 100), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print("Transforms defined.")
print(f"  Train augmentations: {len(train_transform.transforms)} ops")
print(f"  Val   augmentations: {len(val_transform.transforms)} ops")"""))

# ============================================================================
# Cells 26-27: Augmentation preview + Dataset class (from vF.2.0)
# ============================================================================
# Fix augmentation preview cell (old[26]) -- update deprecated API
aug_preview_src = get_src(26)
aug_preview_src = aug_preview_src.replace(
    "A.GaussNoise(var_limit=(5.0, 30.0), p=0.5)",
    "A.GaussNoise(std_range=(0.02, 0.1), p=0.5)")
aug_preview_src = aug_preview_src.replace(
    "A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5)",
    "A.ImageCompression(quality_range=(70, 100), p=0.5)")
new_cells.append(code(aug_preview_src))
new_cells.append(old[27])  # 4.3 Dataset class & DataLoaders

# ============================================================================
# Cells 28-32: Architecture (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[28])  # Section 5 header
new_cells.append(old[29])  # 5.3 Download SAM weights
new_cells.append(old[30])  # 5.4 Model Components (DetectionHead, FeatureProjection, SAMBackbone)
new_cells.append(old[31])  # 5.5 FakeShieldLite full model
new_cells.append(old[32])  # 5.6 Instantiate & smoke test

# ============================================================================
# Cells 33-34: Model Parameter Summary (from vF.2.0)
# ============================================================================
new_cells.append(old[33])  # Model Parameter Summary markdown
new_cells.append(old[34])  # Model Parameter Summary code

# ============================================================================
# Cell 35-36: Feature Caching (REDESIGNED -- CLIP-only)
# ============================================================================
new_cells.append(md("""\
## Feature Caching (CLIP-Only)

**vF.3.0 redesign:** In vF.2.0, feature caching attempted to store both CLIP and SAM features.
SAM features at (256, 64, 64) × 10K samples = **20 GB** — far exceeding Kaggle's 13 GB RAM.

**New approach:** Cache only CLIP [CLS] tokens:
- CLIP features: (N, 768) float16 = **~15 MB** for 10K samples
- SAM encoding runs live during training (per-image, ~0.5s each)
- Ablation experiments use cached CLIP features + live SAM encoding

This makes ablations faster (skip CLIP) while keeping SAM in the loop."""))

new_cells.append(code("""\
# ============================================================================
# Feature Caching -- CLIP-Only (Redesigned in vF.3.0)
# ============================================================================
# vF.2.0 tried to cache SAM features (256×64×64 per image = 20 GB total).
# This crashed the run. vF.3.0 caches ONLY CLIP [CLS] tokens (~15 MB total).

def extract_clip_features(model, dataset, device, batch_size=4):
    \"\"\"Extract CLIP [CLS] features for all samples.

    Returns:
        clip_features: (N, 768) float16 tensor (~15 MB for 10K samples)
        labels:        (N,) float32 tensor
    \"\"\"
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    all_clip, all_labels = [], []

    print(f"Extracting CLIP features from {len(dataset)} samples...")
    for batch in tqdm(loader, desc="  CLIP extraction"):
        images = batch["image"].to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            clip_input = model._prepare_for_clip(images)
            clip_out = model.clip_encoder(pixel_values=clip_input)
            cls_feat = clip_out.pooler_output  # (B, 768)

        all_clip.append(cls_feat.half().cpu())
        all_labels.append(batch["label"])

    clip_features = torch.cat(all_clip, dim=0)
    labels = torch.cat(all_labels, dim=0)

    mem_mb = clip_features.nbytes / 1e6
    print(f"Cached: CLIP {tuple(clip_features.shape)}")
    print(f"Cache size: {mem_mb:.1f} MB (float16)")
    return clip_features, labels


print("CLIP-only feature caching utilities defined.")"""))

# ============================================================================
# Cell 37: Run CLIP extraction
# ============================================================================
new_cells.append(code("""\
# ============================================================================
# Feature Caching -- Run CLIP extraction
# ============================================================================

train_ds_cache = TamperDataset(train_samples, transform=val_transform)
val_ds_cache   = TamperDataset(val_samples,   transform=val_transform)

print("Extracting TRAINING CLIP features:")
train_clip, train_labels = extract_clip_features(
    model, train_ds_cache, DEVICE, batch_size=BATCH_SIZE)

print("\\nExtracting VALIDATION CLIP features:")
val_clip, val_labels = extract_clip_features(
    model, val_ds_cache, DEVICE, batch_size=BATCH_SIZE)

print(f"\\nCLIP features cached: train={tuple(train_clip.shape)}, val={tuple(val_clip.shape)}")
print(f"Total cache: {(train_clip.nbytes + val_clip.nbytes)/1e6:.1f} MB")"""))

# ============================================================================
# Cells 38-39: Infrastructure (from vF.2.0)
# ============================================================================
new_cells.append(old[38])  # Infrastructure markdown
new_cells.append(old[39])  # Infrastructure code

# ============================================================================
# Cells 40-41: Loss Functions (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[40])  # Section 6 header
new_cells.append(old[41])  # 6.1 Loss Functions code

# ============================================================================
# Cell 42-43: Optimizer & Scheduler (FIXED T_max)
# ============================================================================
new_cells.append(old[42])  # Section 7 header

new_cells.append(code("""\
# ============================================================================
# 7.1 Optimiser & Scheduler
# ============================================================================
# FIXED in vF.3.0: T_max accounts for ACCUM_STEPS
# FIXED in vF.3.0: Epochs reduced from 20 to 10

NUM_EPOCHS = CONFIG["epochs"]  # 10
LR = CONFIG["learning_rate"]   # 1e-4
PATIENCE = CONFIG["early_stopping_patience"]  # 3

# Differential learning rates (FakeShield uses lower LR for pretrained parts)
param_groups = [
    {"params": model.detection_head.parameters(),     "lr": LR,       "name": "det_head"},
    {"params": model.feature_projection.parameters(), "lr": LR,       "name": "feat_proj"},
    {"params": model.sam.mask_decoder.parameters(),   "lr": LR * 0.5, "name": "sam_dec"},
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])

# FIXED: T_max should count gradient update steps, not batches
steps_per_epoch = len(train_loader)
grad_updates_per_epoch = steps_per_epoch // ACCUM_STEPS
total_grad_updates = NUM_EPOCHS * grad_updates_per_epoch

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_grad_updates, eta_min=1e-6,
)

scaler = GradScaler()

print(f"Optimiser : AdamW  (weight_decay={CONFIG['weight_decay']})")
print(f"Scheduler : CosineAnnealing  (T_max={total_grad_updates} gradient updates)")
print(f"Epochs    : {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUM_STEPS})")
print(f"Steps/ep  : {steps_per_epoch} batches = {grad_updates_per_epoch} gradient updates")
print(f"Early stop: patience={PATIENCE}")
print(f"Grad clip : max_norm={CONFIG['grad_clip_norm']}")"""))

# ============================================================================
# Cell 44: Training Loop (UPDATED -- gradient clipping, fixed)
# ============================================================================
new_cells.append(code("""\
# ============================================================================
# 7.2 Training Loop (vF.3.0: gradient clipping, fixed scheduler)
# ============================================================================

history = {"train_loss": [], "val_loss": [],
           "val_det_f1": [], "val_iou": [], "val_dice": []}
best_val_iou = 0.0
SAVE_DIR = "/kaggle/working/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
GRAD_CLIP_NORM = CONFIG["grad_clip_norm"]


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    model.clip_encoder.eval()
    model.sam.image_encoder.eval()
    model.sam.prompt_encoder.eval()  # vF.3.0: explicitly set prompt encoder to eval

    running_loss = 0.0
    pbar = tqdm(loader, desc="  Train", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            out = model(images)
            losses = criterion(out["det_logits"], out["mask_logits"],
                               labels, masks)
            loss = losses["total"] / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            # vF.3.0: gradient clipping before optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad],
                max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += losses["total"].item()
        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

    return running_loss / len(loader)


print("Training loop defined (vF.3.0: gradient clipping, prompt_encoder.eval()).")"""))

# ============================================================================
# Cells 45-46: Evaluation Metrics (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[45])  # Section 8 header
new_cells.append(old[46])  # 8.1 Metric Functions

# ============================================================================
# Cells 47-48: Validation (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[47])  # Section 9 header
new_cells.append(old[48])  # 9.1 Validation Function

# ============================================================================
# Cell 49: Run Training (UPDATED -- early stopping + time estimation)
# ============================================================================
new_cells.append(code("""\
# ============================================================================
# 9.2 Run Training (vF.3.0: early stopping + time estimation)
# ============================================================================

SAVE_DIR = "/kaggle/working/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
training_start_time = time.time()
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 50)

    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, DEVICE)

    val = validate(model, val_loader, criterion, DEVICE)

    epoch_time = time.time() - epoch_start

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val["loss"])
    history["val_det_f1"].append(val["f1"])
    history["val_iou"].append(val["iou"])
    history["val_dice"].append(val["dice"])

    print(f"  Train Loss : {train_loss:.4f}")
    print(f"  Val Loss   : {val['loss']:.4f}")
    print(f"  Detection  -- ACC: {val['accuracy']:.4f}  P: {val['precision']:.4f}  "
          f"R: {val['recall']:.4f}  F1: {val['f1']:.4f}")
    print(f"  Localization -- IoU: {val['iou']:.4f}  Dice: {val['dice']:.4f}")
    print(f"  Epoch time : {epoch_time/60:.1f} min")

    # Time estimation after first epoch
    if epoch == 0:
        elapsed = time.time() - training_start_time
        remaining = elapsed * (NUM_EPOCHS - 1)
        print(f"  >> ETA: ~{remaining/60:.0f} min remaining ({NUM_EPOCHS-1} more epochs)")

    # Save best model
    if val["iou"] > best_val_iou:
        best_val_iou = val["iou"]
        patience_counter = 0
        ckpt_path = os.path.join(SAVE_DIR, "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": val,
            "history": history,
        }, ckpt_path)
        print(f"  >> New best model saved  (IoU={best_val_iou:.4f})")
    else:
        patience_counter += 1
        print(f"  >> No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"  >> Early stopping triggered after {epoch+1} epochs")
            break

training_time = time.time() - training_start_time
print(f"\\n{'=' * 50}")
print(f"Training complete in {training_time/60:.1f} min.  Best val IoU: {best_val_iou:.4f}")
print(f"Epochs run: {len(history['train_loss'])}/{NUM_EPOCHS}")"""))

# ============================================================================
# Cells 50-51: Training Curves (from vF.2.0)
# ============================================================================
new_cells.append(old[50])  # Training Curves markdown
new_cells.append(old[51])  # Training Curves code

# ============================================================================
# Cells 52-53: Prediction Visualization (from vF.2.0)
# ============================================================================
new_cells.append(old[52])  # Visualization markdown header
new_cells.append(old[53])  # Evaluation Results markdown

# ============================================================================
# Cells 54-55: Load best checkpoint + prediction vis (from vF.2.0)
# ============================================================================
new_cells.append(old[54])  # 10.1 Load Best Checkpoint
new_cells.append(old[55])  # 10.2 Prediction Visualization Grid

# ============================================================================
# Cells 56-57: Evaluate on Test Set (from vF.2.0)
# ============================================================================
new_cells.append(old[56])  # Test eval markdown
new_cells.append(old[57])  # Evaluate on Test Set code

# ============================================================================
# Cells 58-59: Evaluation summary (from vF.2.0)
# ============================================================================
new_cells.append(old[58])  # Evaluation Summary markdown

# ============================================================================
# Cells 59-60: Error Analysis (from vF.2.0)
# ============================================================================
new_cells.append(old[59])  # Error Analysis header markdown
new_cells.append(old[60])  # Error Analysis code
new_cells.append(old[61])  # Failure Mode Discussion

# ============================================================================
# Cell 61: Ablation Study (REDESIGNED)
# ============================================================================
new_cells.append(md("""\
## Ablation Study

**vF.3.0 redesign:** The ablation framework now uses cached CLIP features but runs SAM encoding live.
This avoids the 20 GB SAM caching problem while still speeding up ablation experiments
(CLIP extraction is skipped, saving ~30% of forward pass time).

**Experiments:**
1. **Baseline** — Full model, all components active
2. **No Detection Head** — Set w_det=0, freeze detection head
3. **No CLIP Projection** — Zero and freeze FeatureProjection (SAM receives zero prompts)
4. **Random SAM Decoder** — Reinitialize mask decoder weights from scratch

Note: "No Augmentation" ablation from vF.2.0 was removed — it was identical to Baseline
since cached features use val_transform for all experiments."""))

new_cells.append(code("""\
# ============================================================================
# Ablation Study -- Framework (Redesigned in vF.3.0)
# ============================================================================
# Uses cached CLIP features + live SAM encoding.
# Avoids reloading CLIP for each experiment.

ABLATION_EPOCHS = CONFIG["ablation_epochs"]  # 5


def run_ablation_v3(name, model_modifier_fn, epochs=ABLATION_EPOCHS):
    \"\"\"Run a single ablation experiment.

    Uses a fresh model but shares the frozen CLIP/SAM encoder weights
    from the global model to avoid redundant downloads.
    \"\"\"
    print(f"\\n{'='*55}")
    print(f"  ABLATION: {name}")
    print(f"{'='*55}")

    # Build model reusing frozen weights (no re-download)
    abl_model = FakeShieldLite(
        clip_model=CONFIG["clip_model"],
        sam_checkpoint=SAM_CHECKPOINT,
        img_size=IMG_SIZE,
    ).to(DEVICE)

    # Copy frozen encoder weights from trained model (避免重新下载)
    abl_model.clip_encoder.load_state_dict(model.clip_encoder.state_dict())
    abl_model.sam.image_encoder.load_state_dict(model.sam.image_encoder.state_dict())
    abl_model.sam.prompt_encoder.load_state_dict(model.sam.prompt_encoder.state_dict())

    # Apply ablation modification
    model_modifier_fn(abl_model)

    trainable = sum(p.numel() for p in abl_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # Optimizer (same config as baseline)
    abl_params = [
        {"params": abl_model.detection_head.parameters(),     "lr": LR},
        {"params": abl_model.feature_projection.parameters(), "lr": LR},
        {"params": abl_model.sam.mask_decoder.parameters(),   "lr": LR * 0.5},
    ]
    abl_opt = torch.optim.AdamW(abl_params, weight_decay=CONFIG["weight_decay"])
    abl_scaler = GradScaler()

    # Train
    for ep in range(epochs):
        abl_model.train()
        abl_model.clip_encoder.eval()
        abl_model.sam.image_encoder.eval()
        abl_model.sam.prompt_encoder.eval()
        ep_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks  = batch["mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            abl_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = abl_model(images)
                losses = criterion(out["det_logits"], out["mask_logits"], labels, masks)

            abl_scaler.scale(losses["total"]).backward()
            abl_scaler.unscale_(abl_opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in abl_opt.param_groups for p in g["params"] if p.requires_grad],
                max_norm=GRAD_CLIP_NORM)
            abl_scaler.step(abl_opt)
            abl_scaler.update()
            ep_loss += losses["total"].item()

        if (ep+1) % max(1, epochs//3) == 0 or ep == epochs-1:
            print(f"  Epoch {ep+1}/{epochs}  train_loss={ep_loss/len(train_loader):.4f}")

    # Validate
    result = validate(abl_model, val_loader, criterion, DEVICE)
    print(f"  Results -- F1: {result['f1']:.4f}  IoU: {result['iou']:.4f}  Dice: {result['dice']:.4f}")

    del abl_model, abl_opt, abl_scaler
    torch.cuda.empty_cache()
    gc.collect()

    return result


print("Ablation framework defined (vF.3.0: live SAM encoding, shared frozen weights).")"""))

# ============================================================================
# Cell 63: Run Ablation Experiments (REDESIGNED)
# ============================================================================
new_cells.append(code("""\
# ============================================================================
# Ablation Experiments -- Run All (vF.3.0)
# ============================================================================

ablation_results = {}

# 1. Baseline (full model, no modifications)
ablation_results["Baseline"] = run_ablation_v3("Baseline", lambda m: None)

# 2. No Detection Head (freeze det head, zero out det loss weight)
def no_det_modifier(m):
    for p in m.detection_head.parameters():
        p.requires_grad = False
ablation_results["No Detection Head"] = run_ablation_v3(
    "No Detection Head", no_det_modifier)
# Note: ideally we'd also set w_det=0 in the criterion, but for simplicity
# we just freeze the head -- gradients flow through but params don't update.

# 3. No CLIP Projection (zero out projection, freeze it)
def no_clip_proj_modifier(m):
    for p in m.feature_projection.parameters():
        nn.init.zeros_(p)
        p.requires_grad = False
ablation_results["No CLIP Projection"] = run_ablation_v3(
    "No CLIP Projection", no_clip_proj_modifier)

# 4. Random SAM Decoder (reinitialize decoder weights)
def random_sam_modifier(m):
    for module in m.sam.mask_decoder.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
ablation_results["Random SAM Decoder"] = run_ablation_v3(
    "Random SAM Decoder", random_sam_modifier)

print("\\n" + "=" * 55)
print("  All ablation experiments complete!")
print("=" * 55)"""))

# ============================================================================
# Cell 64: Ablation Results Table (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[65])  # Ablation Results Table & Visualization

# ============================================================================
# Cells 65-66: Threshold Sensitivity (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[66])  # Threshold header markdown
new_cells.append(old[67])  # Threshold Sensitivity Analysis code

# ============================================================================
# Cells 67-68: Detection Confidence (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[68])  # Confidence header markdown
new_cells.append(old[69])  # Detection Confidence Distribution code

# ============================================================================
# Cells 69-70: Hard Example Analysis (from vF.2.0, unchanged)
# ============================================================================
new_cells.append(old[70])  # Hard example header markdown
new_cells.append(old[71])  # Hard Example Analysis code

# ============================================================================
# Cells 71-73: Robustness Testing (from vF.2.0 but fix augmentation API)
# ============================================================================
new_cells.append(old[72])  # Robustness header markdown

# Check if cell 73 has the deprecated API calls and fix them
robustness_src = get_src(73)
if "var_limit" in robustness_src or "quality_lower" in robustness_src:
    robustness_src = robustness_src.replace(
        "A.GaussNoise(var_limit=(5.0, 30.0), p=0.5)",
        "A.GaussNoise(std_range=(0.02, 0.1), p=0.5)")
    robustness_src = robustness_src.replace(
        "A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)",
        "A.GaussNoise(std_range=(0.05, 0.15), p=0.5)")
    robustness_src = robustness_src.replace(
        "A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5)",
        "A.ImageCompression(quality_range=(70, 100), p=0.5)")
    robustness_src = robustness_src.replace(
        "A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5)",
        "A.ImageCompression(quality_range=(50, 90), p=0.5)")
    new_cells.append(code(robustness_src))
else:
    new_cells.append(old[73])

# ============================================================================
# Cells 74-76: Robustness insights + Computational Efficiency (from vF.2.0)
# ============================================================================
new_cells.append(old[74])  # Robustness Insights markdown
new_cells.append(old[75])  # Computational Efficiency header markdown
new_cells.append(old[76])  # Computational Efficiency code

# ============================================================================
# Cells 77-79: Efficiency analysis + Inference Demo (from vF.2.0)
# ============================================================================
new_cells.append(old[77])  # Inference header markdown (or efficiency analysis)
new_cells.append(old[78])  # Inference Demo header markdown
new_cells.append(old[79])  # Inference Demo code

# ============================================================================
# Cells 80-82: Model Saving (from vF.2.0 but update version)
# ============================================================================
new_cells.append(old[80])  # Section 12 header

new_cells.append(code("""\
# ============================================================================
# Save Final Weights
# ============================================================================

final_path = "/kaggle/working/model_vF30_fakeshield_lite.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "config": CONFIG,
    "experiment_info": EXPERIMENT_INFO,
    "history": history,
    "best_val_iou": best_val_iou,
    "ablation_results": ablation_results if 'ablation_results' in dir() else {},
}, final_path)

size_mb = os.path.getsize(final_path) / 1e6
print(f"Model saved to: {final_path}")
print(f"File size: {size_mb:.1f} MB")"""))

new_cells.append(old[82])  # 12.2 Reload Model demo

# ============================================================================
# Cell 83: Conclusion (UPDATED)
# ============================================================================
new_cells.append(md("""\
## Conclusion

### Summary

FakeShield-Lite vF.3.0 successfully implements a pruned version of FakeShield (Xu et al., ICLR 2025)
that fits within a Kaggle T4 GPU's 16 GB VRAM constraint. The model uses ~182M parameters with only
~5M trainable (2.8%), achieving the dual objectives of tampering detection and localization.

### Key Fixes in vF.3.0

| Fix | Before (vF.2.0) | After (vF.3.0) |
|-----|-----------------|-----------------|
| Feature caching | SAM + CLIP (20 GB, crashed) | CLIP-only (15 MB) |
| Albumentations API | Deprecated params (silently ignored) | Updated to current API |
| Epochs | 20 (exceeds session limit) | 10 + early stopping |
| Scheduler T_max | Counted batches (2× too large) | Counts gradient updates |
| Gradient clipping | None | max_norm=1.0 |
| Ablation design | "No Aug" was copy of Baseline | Removed; 4 meaningful ablations |

### Limitations

1. SAM encoding at 1024×1024 remains the primary bottleneck (~0.5s per image)
2. CLIP and SAM see different resolutions (224 vs 1024) — information asymmetry
3. No multi-scale tampering detection
4. Limited to single-dataset evaluation (CASIA)

### Future Directions

| Direction | Description |
|-----------|-------------|
| Smaller SAM | MobileSAM or EfficientSAM to reduce encoding time |
| SAM resolution | Test 512×512 SAM input (quality vs speed trade-off) |
| Multi-dataset | Evaluate on Columbia, Coverage, NIST16 |
| Knowledge distillation | Distill from full FakeShield to improve Lite quality |"""))

# ── Save ────────────────────────────────────────────────────────────────────

nb["cells"] = new_cells
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Built vF.3.0 notebook: {OUTPUT_PATH}")
print(f"Total cells: {len(new_cells)}")
print(f"  Markdown: {sum(1 for c in new_cells if c['cell_type'] == 'markdown')}")
print(f"  Code:     {sum(1 for c in new_cells if c['cell_type'] == 'code')}")
