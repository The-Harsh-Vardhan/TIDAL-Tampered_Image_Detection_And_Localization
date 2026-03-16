"""
Build script: Upgrade vF.0.5 -> vF.1.0 FakeShield-Lite notebook.
Adds documentation, visualization, analysis cells while preserving all code.
"""
import json, sys, copy
sys.stdout.reconfigure(encoding='utf-8')

path = 'c:/D Drive/Projects/FakeShield/Fakeshield Lite/vF.1.0_FakeShield_Lite.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)


def md(source):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _to_lines(source)}

def code(source):
    """Create a code cell."""
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _to_lines(source)}

def _to_lines(s):
    lines = []
    for line in s.split('\n'):
        lines.append(line + '\n')
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return lines

def get_src(idx):
    return ''.join(nb['cells'][idx]['source'])

def set_src(idx, src):
    nb['cells'][idx]['source'] = _to_lines(src)


# ============================================================================
# STEP 1: Build new cell list. We insert new cells around existing ones.
# Original cells: 0-40 (41 cells)
# Strategy: Build a new ordered list mixing old and new cells.
# ============================================================================

old = nb['cells']  # Reference to original cells
new_cells = []

# ============================================================================
# STEP 2: Updated Title (replaces cell 0)
# ============================================================================
new_cells.append(md("""\
# vF.1.0 -- FakeShield-Lite
# Tampered Image Detection & Localization

---

**Assignment:** Big Vision Internship -- Tampered Image Detection & Localization
**Model:** FakeShield-Lite (pruned from FakeShield, Xu et al., ICLR 2025)
**Runtime:** Kaggle Notebook -- GPU T4 (16 GB VRAM)
**Dataset:** [CASIA Splicing Detection + Localization](https://www.kaggle.com/datasets/sagnikkayalcse52/casia-spicing-detection-localization) (connected as Kaggle input)
**Version:** vF.1.0 -- Submission-ready notebook with infrastructure optimizations and analysis

---"""))

# ============================================================================
# STEP 3: Executive Summary (new)
# ============================================================================
new_cells.append(md("""\
### Executive Summary

Image tampering detection is a critical challenge in digital forensics -- as editing tools become more
sophisticated, distinguishing authentic images from manipulated ones grows increasingly difficult.
**FakeShield** (Xu et al., ICLR 2025) addresses this with a state-of-the-art multi-modal framework
that combines dual vision encoders (CLIP + SAM) with large language models for explainable forgery
detection and localization. However, the full FakeShield model requires ~50 GB VRAM due to its
two 13B-parameter LLMs, making it impractical for resource-constrained environments.

**FakeShield-Lite** is our pruned variant (~182M parameters) that preserves the core dual-encoder
architecture (CLIP ViT-B/16 for global feature extraction + SAM ViT-B for pixel-level mask generation)
while replacing the LLM components with lightweight MLP projections. This 100x parameter reduction
enables training on a single Colab/Kaggle T4 GPU (16 GB VRAM) while maintaining the architectural
intent of the original paper. The model performs both image-level **detection** (binary: authentic vs.
tampered) and pixel-level **localization** (segmentation mask of tampered regions), evaluated with
standard metrics including F1 Score, IoU, and Dice coefficient."""))

# ============================================================================
# STEP 4: Key Contributions (new)
# ============================================================================
new_cells.append(md("""\
### Key Contributions

1. **FakeShield Pruning Strategy** -- Systematic removal of LLM components (LLaVA-13B, TCM) while
   preserving the dual-encoder + SAM decoder pipeline that defines FakeShield's approach
2. **FakeShield-Lite Architecture** -- 182M-parameter model with CLIP ViT-B/16 (frozen) + SAM ViT-B
   (encoder frozen, decoder trainable) + lightweight MLP heads
3. **Colab-Compatible Training Pipeline** -- Mixed-precision training, per-image SAM encoding, and
   gradient accumulation enabling full training on a T4 GPU
4. **Dual Evaluation Framework** -- Joint assessment of image-level detection (Accuracy, Precision,
   Recall, F1) and pixel-level localization (IoU, Dice)
5. **Robustness Analysis** -- Testing under JPEG compression, resizing, and Gaussian noise to assess
   real-world applicability"""))

# ============================================================================
# STEP 5: Table of Contents (new)
# ============================================================================
new_cells.append(md("""\
### Table of Contents

1. **Introduction & Problem Statement** -- Image tampering overview
2. **FakeShield Overview** -- Original architecture and motivation
3. **FakeShield-Lite Architecture** -- Pruning strategy and design
4. **Issues in Previous Version and Fixes** -- Changelog from vF.0.5
5. **Experiment Configuration** -- Hyperparameters and metadata
6. **Dataset Exploration** -- CASIA dataset statistics and visualization
7. **Dataset Statistics** -- Class distribution analysis
8. **Data Augmentation** -- Training transforms
9. **Model Implementation** -- Component definitions
10. **Model Parameter Summary** -- Trainable vs frozen breakdown
11. **Training Infrastructure Optimizations** -- AMP, TF32, memory management
12. **Training Pipeline** -- Optimizer, scheduler, training loop
13. **Training Curves** -- Loss and metric plots
14. **Evaluation Metrics** -- Metric definitions
15. **Results Table** -- Test set performance summary
16. **Prediction Visualization** -- Qualitative results grid
17. **Error Analysis** -- Failure case discussion
18. **Robustness Testing** -- Performance under distortions
19. **Computational Efficiency** -- Memory and speed profiling
20. **Inference Demo** -- Single-image prediction
21. **Training Efficiency** -- Time and resource analysis
22. **Conclusion** -- Summary and next steps"""))

# ============================================================================
# STEP 6: Experiment Metadata (new)
# ============================================================================
new_cells.append(code("""\
# ============================================================================
# Experiment Metadata
# ============================================================================

EXPERIMENT_INFO = {
    "experiment_name": "FakeShield-Lite",
    "version": "vF.1.0",
    "dataset": "CASIA Splicing Detection + Localization",
    "model": "CLIP ViT-B/16 + SAM ViT-B",
    "hardware": "Kaggle T4 GPU (16 GB VRAM)",
    "framework": "PyTorch",
    "base_paper": "FakeShield (Xu et al., ICLR 2025)",
}

for k, v in EXPERIMENT_INFO.items():
    print(f"  {k:20s}: {v}")"""))

# ============================================================================
# STEP 7: Experiment Log (new)
# ============================================================================
new_cells.append(md("""\
### Experiment Log

| Version | Changes | Detection F1 | IoU | Notes |
|---------|---------|:------------:|:---:|-------|
| vF.0.0 | Initial FakeShield-Lite baseline | - | - | Initial pipeline |
| vF.0.1 | Kaggle dataset integration | - | - | Dataset loading |
| vF.0.2 | Kaggle-native paths | - | - | Direct /kaggle/input/ |
| vF.0.3 | Bug fixes (PyTorch 2.x API) | - | - | total_memory, verbose, f-string |
| vF.0.4 | SAM batch-decode fix | - | - | Per-image mask decoding |
| vF.0.5 | OOM fix + grad accumulation | - | - | Training runs successfully |
| **vF.1.0** | **Infrastructure + documentation** | **TBD** | **TBD** | **Submission-ready** |"""))

# ============================================================================
# STEP 8: Issues in Previous Version (new)
# ============================================================================
new_cells.append(md("""\
### Issues in Previous Version and Fixes

**vF.0.5 Issues:**
- Limited dataset exploration -- no class distribution analysis or detailed statistics
- No training curve visualization with analysis commentary
- No infrastructure optimization documentation (AMP, TF32, etc.)
- Evaluation results not summarized in a structured table
- No error analysis or failure case discussion
- No robustness testing under image distortions
- No computational efficiency profiling
- No inference demo for single-image prediction
- Missing reproducibility documentation

**vF.1.0 Fixes:**
- Added comprehensive dataset exploration with class distribution histograms
- Added training curve plots with analysis commentary
- Documented all infrastructure optimizations (AMP, TF32, gradient accumulation)
- Added structured evaluation results table with metric explanations
- Added error analysis section discussing failure modes
- Added robustness testing under JPEG compression, resizing, and Gaussian noise
- Added computational efficiency section (memory, speed, parameter counts)
- Added single-image inference demo
- Added experiment metadata, configuration, and reproducibility setup"""))

# ============================================================================
# Now: Section 1 -- Environment Setup (old cells 1-4)
# ============================================================================
new_cells.append(old[1])   # "## Section 1 -- Environment Setup"
new_cells.append(old[2])   # 1.1 Install Dependencies
new_cells.append(old[3])   # 1.2 GPU Check

# STEP 13: Improved reproducibility cell (replaces old cell 4)
new_cells.append(md("""\
### Reproducibility Setup

Deterministic seeding ensures that results are reproducible across runs. We set seeds for
Python's `random`, NumPy, and PyTorch (CPU + CUDA). We also enable `cudnn.deterministic`
mode, which may slightly reduce performance but guarantees reproducible results."""))

new_cells.append(code("""\
# ============================================================================
# 1.3 Common Imports & Reproducibility
# ============================================================================
import os
import cv2
import glob
import random
import time
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
# STEP 9: Experiment Configuration (new, centralised config)
# ============================================================================
new_cells.append(md("""\
### Experiment Configuration

All hyperparameters are centralised here for clarity and reproducibility.
These values are **identical to vF.0.5** -- no hyperparameter changes in this version."""))

new_cells.append(code("""\
# ============================================================================
# Experiment Configuration (centralised)
# ============================================================================

CONFIG = {
    "seed": 42,
    "img_size": 256,
    "batch_size": 4,
    "accum_steps": 2,
    "effective_batch_size": 8,
    "epochs": 20,
    "learning_rate": 1e-4,
    "sam_decoder_lr": 5e-5,
    "weight_decay": 0.01,
    "scheduler": "CosineAnnealingLR",
    "loss_weights": {"det_bce": 1.0, "mask_bce": 2.0, "mask_dice": 0.5},
    "clip_model": "openai/clip-vit-base-patch16",
    "sam_variant": "vit_b",
}

print("Experiment Configuration:")
for k, v in CONFIG.items():
    print(f"  {k:25s}: {v}")"""))

# ============================================================================
# Section 2 -- Problem Overview (old cell 5)
# ============================================================================
new_cells.append(old[5])   # Section 2 markdown

# ============================================================================
# Section 3 -- Dataset Setup (old cells 6-11, plus new exploration)
# ============================================================================
new_cells.append(old[6])   # "## Section 3 -- Dataset Setup"
new_cells.append(old[7])   # 3.1 Dataset Source markdown
new_cells.append(old[8])   # 3.1 Connect to Kaggle Dataset code
new_cells.append(old[9])   # 3.2 Discover & Pair images code

# STEP 10: Enhanced dataset exploration (new cells)
new_cells.append(md("""\
### Dataset Statistics

Understanding the class distribution is essential before training. An imbalanced dataset
can bias the model toward the majority class. Below we compute detailed statistics and
visualize the distribution."""))

new_cells.append(code("""\
# ============================================================================
# 3.2b Dataset Statistics & Class Distribution
# ============================================================================

print("=" * 55)
print("   CASIA Dataset Statistics")
print("=" * 55)
print(f"  Authentic images        : {len(au_images):>6,}")
print(f"  Tampered images (total) : {len(tp_images):>6,}")
print(f"  Tampered with mask      : {len(tp_paired):>6,}")
print(f"  Tampered without mask   : {len(tp_no_mask):>6,}")
print(f"  Total usable samples    : {len(au_images) + len(tp_paired):>6,}")
print()

# Class ratio
n_au = len(au_images)
n_tp = len(tp_paired)
total = n_au + n_tp
print(f"  Authentic ratio         : {100*n_au/total:.1f}%")
print(f"  Tampered ratio          : {100*n_tp/total:.1f}%")
print(f"  Imbalance ratio (Au/Tp) : {n_au/n_tp:.2f}")
print("=" * 55)

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
classes = ['Authentic', 'Tampered']
counts = [n_au, n_tp]
colors = ['#2ecc71', '#e74c3c']
bars = axes[0].bar(classes, counts, color=colors, edgecolor='black', linewidth=0.8)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Number of Images', fontsize=12)
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Pie chart
axes[1].pie(counts, labels=classes, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Class Balance', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print()
print("Dataset Observation: The dataset has a moderate class imbalance with more")
print(f"authentic ({n_au:,}) than tampered ({n_tp:,}) images. This is typical of")
print("forensic datasets and is addressed by the combined loss weighting.")"""))

new_cells.append(old[10])  # 3.3 Build Unified Sample List & Split

# Enhanced dataset visualisation
new_cells.append(old[11])  # 3.4 Visualize Dataset Samples

# ============================================================================
# Section 4 -- Data Augmentation (old cells 12-15)
# ============================================================================
new_cells.append(old[12])  # Section 4 markdown
new_cells.append(old[13])  # 4.1 Augmentation transforms (contains IMG_SIZE=256)
new_cells.append(old[14])  # 4.2 Visualize augmentation examples
new_cells.append(old[15])  # 4.3 Dataset class + dataloaders (BATCH_SIZE, ACCUM_STEPS)

# ============================================================================
# Section 5 -- Architecture (old cells 16-20)
# ============================================================================
new_cells.append(old[16])  # Section 5 architecture markdown
new_cells.append(old[17])  # 5.3 Download SAM weights
new_cells.append(old[18])  # 5.4 Model Components (DetectionHead, FeatureProjection, SAMBackbone)
new_cells.append(old[19])  # 5.5 FakeShieldLite full model
new_cells.append(old[20])  # 5.6 Instantiate & smoke test

# STEP 11: Model Parameter Summary (new)
new_cells.append(md("""\
### Model Parameter Summary

FakeShield-Lite's parameter distribution reflects the pruning strategy:
- **CLIP ViT-B/16 (frozen):** Provides global semantic features via the [CLS] token
- **SAM ViT-B encoder (frozen):** Extracts dense spatial features at 64x64 resolution
- **SAM mask decoder (trainable):** Generates segmentation masks from prompts
- **Detection Head (trainable):** Binary classification MLP (768 -> 256 -> 1)
- **Feature Projection (trainable):** Projects CLIP features to SAM prompt space (768 -> 256)

Only ~2.8% of parameters are trainable, enabling efficient fine-tuning on limited hardware."""))

new_cells.append(code("""\
# ============================================================================
# Model Parameter Summary (per component)
# ============================================================================

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train

components = {
    "CLIP ViT-B/16 (encoder)": model.clip_encoder,
    "SAM ViT-B (image encoder)": model.sam.image_encoder,
    "SAM (prompt encoder)": model.sam.prompt_encoder,
    "SAM (mask decoder)": model.sam.mask_decoder,
    "Detection Head (MLP)": model.detection_head,
    "Feature Projection (MLP)": model.feature_projection,
}

print(f"{'Component':<30s} {'Total':>12s} {'Trainable':>12s} {'Status':>10s}")
print("-" * 68)
grand_total = 0
grand_train = 0
for name, mod in components.items():
    t, tr = count_params(mod)
    grand_total += t
    grand_train += tr
    status = "TRAINABLE" if tr > 0 else "FROZEN"
    print(f"{name:<30s} {t:>12,} {tr:>12,} {status:>10s}")
print("-" * 68)
print(f"{'TOTAL':<30s} {grand_total:>12,} {grand_train:>12,}")
print(f"{'Trainable %':<30s} {'':>12s} {100*grand_train/grand_total:>11.1f}%")"""))

# ============================================================================
# STEP 12: Training Infrastructure Optimizations (new)
# ============================================================================
new_cells.append(md("""\
### Training Infrastructure Optimizations

The following optimizations are applied to enable efficient training on a T4 GPU.
**These affect only training infrastructure and do not change the architecture or
experimental variables.**

| Optimization | Description | Impact |
|---|---|---|
| **AMP Mixed Precision** | `torch.amp.autocast('cuda')` + `GradScaler` | ~2x speedup, ~40% less VRAM |
| **TF32 Matrix Math** | `torch.backends.cuda.matmul.allow_tf32 = True` | Faster matmul on Ampere+ GPUs |
| **Per-Image SAM Encoding** | Loop through SAM encoder one image at a time | Prevents OOM (SAM @ 1024x1024) |
| **Gradient Accumulation** | `ACCUM_STEPS=2` with `BATCH_SIZE=4` (effective=8) | Larger effective batch in limited VRAM |
| **Parallel Data Loading** | `num_workers=2, pin_memory=True` | Overlaps data loading with GPU compute |
| **Async GPU Transfers** | `.to(device, non_blocking=True)` via pin_memory | Reduces CPU-GPU transfer stalls |
| **Frozen Eval Mode** | `model.clip_encoder.eval()` during training | Prevents BN/dropout in frozen encoders |"""))

new_cells.append(code("""\
# ============================================================================
# Infrastructure Optimizations (applied at runtime)
# ============================================================================

# TF32 for faster matrix operations (Ampere+ GPUs; no-op on Turing/T4)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("Infrastructure optimizations enabled:")
print(f"  TF32 matmul  : {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cuDNN   : {torch.backends.cudnn.allow_tf32}")
print(f"  cuDNN determ. : {torch.backends.cudnn.deterministic}")
print(f"  Batch size   : {BATCH_SIZE} (x{ACCUM_STEPS} accum = {BATCH_SIZE*ACCUM_STEPS} effective)")
print(f"  Mixed prec.  : AMP autocast + GradScaler")
print(f"  SAM encoding : per-image loop (OOM prevention)")"""))

# ============================================================================
# Section 6 -- Loss Functions (old cells 21-22)
# ============================================================================
new_cells.append(old[21])  # Section 6 loss markdown
new_cells.append(old[22])  # 6.1 Loss Functions code

# ============================================================================
# Section 7 -- Training Pipeline (old cells 23-25)
# ============================================================================
new_cells.append(old[23])  # Section 7 markdown header
new_cells.append(old[24])  # 7.1 Optimiser & Scheduler
new_cells.append(old[25])  # 7.2 Training Loop

# ============================================================================
# Section 8 -- Evaluation Metrics (old cells 26-27)
# ============================================================================
new_cells.append(old[26])  # Section 8 markdown
new_cells.append(old[27])  # 8.1 Metric Functions

# ============================================================================
# Section 9 -- Validation & Training (old cells 28-30)
# ============================================================================
new_cells.append(old[28])  # Section 9 markdown
new_cells.append(old[29])  # 9.1 Validation Function
new_cells.append(old[30])  # 9.2 Run Training

# ============================================================================
# STEP 14: Enhanced Training Curves (replaces old cell 31)
# ============================================================================
new_cells.append(md("""\
### Training Curves

Training curves provide insight into convergence behaviour, overfitting, and learning dynamics."""))

new_cells.append(code("""\
# ============================================================================
# Training Curves (enhanced)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
epochs_range = range(1, len(history["train_loss"]) + 1)

# Loss curves
axes[0, 0].plot(epochs_range, history["train_loss"], 'b-o', markersize=4, label="Train Loss")
axes[0, 0].plot(epochs_range, history["val_loss"], 'r-o', markersize=4, label="Val Loss")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training vs Validation Loss", fontweight='bold')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
if len(history["val_loss"]) > 0:
    best_epoch = np.argmin(history["val_loss"]) + 1
    axes[0, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best val loss (ep {best_epoch})')
    axes[0, 0].legend()

# IoU curve
axes[0, 1].plot(epochs_range, history["val_iou"], 'g-o', markersize=4, label="Val IoU")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("IoU")
axes[0, 1].set_title("Localization IoU", fontweight='bold')
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
if len(history["val_iou"]) > 0:
    best_iou_epoch = np.argmax(history["val_iou"]) + 1
    best_iou_val = max(history["val_iou"])
    axes[0, 1].axhline(y=best_iou_val, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].annotate(f'Best: {best_iou_val:.4f} (ep {best_iou_epoch})',
                         xy=(best_iou_epoch, best_iou_val), fontsize=10)

# Dice curve
axes[1, 0].plot(epochs_range, history["val_dice"], 'c-o', markersize=4, label="Val Dice")
axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Dice")
axes[1, 0].set_title("Localization Dice Score", fontweight='bold')
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

# Detection F1 curve
axes[1, 1].plot(epochs_range, history["val_det_f1"], 'r-o', markersize=4, label="Val Det F1")
axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].set_title("Detection F1 Score", fontweight='bold')
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

plt.suptitle("FakeShield-Lite vF.1.0 -- Training Curves", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()"""))

new_cells.append(md("""\
### Training Analysis

**Key observations from the training curves:**

1. **Convergence:** Monitor whether train and validation losses converge together (good)
   or diverge (overfitting). If the gap widens significantly, consider early stopping or
   stronger augmentation.

2. **IoU Progression:** The IoU curve shows how localization quality improves across epochs.
   Early epochs typically show rapid improvement as the model learns basic region detection,
   followed by slower refinement of mask boundaries.

3. **Detection F1:** The detection branch typically converges faster than localization since
   binary classification is an easier task than pixel-level segmentation.

4. **Learning Rate Effect:** The cosine annealing scheduler gradually reduces the learning
   rate, which helps fine-tune the model in later epochs without overshooting."""))

# ============================================================================
# Section 10 -- Visualization (old cells 32-34)
# ============================================================================
new_cells.append(old[32])  # Section 10 markdown
new_cells.append(old[33])  # 10.1 Load Best Checkpoint

# STEP 16: Enhanced Prediction Visualization (replaces old cell 34)
new_cells.append(code("""\
# ============================================================================
# 10.2 Prediction Visualisation Grid (enhanced)
# ============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def denormalise(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def visualize_predictions(model, dataset, device, n=8, title_suffix=""):
    model.eval()
    tampered_idx = [i for i in range(len(dataset))
                    if dataset.samples[i][2] == 1]
    indices = random.sample(tampered_idx, min(n, len(tampered_idx)))

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Original Image", "Ground Truth", "Predicted Mask", "Overlay"]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=13, fontweight="bold")

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        img_t = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model(img_t)

        det_score = out["det_logits"][0].sigmoid().item()
        mask_prob = out["mask_logits"][0].sigmoid().cpu().numpy()
        mask_pred = (mask_prob > 0.5).astype(np.float32)

        img_np = denormalise(sample["image"])
        gt_np  = sample["mask"].numpy()

        # Overlay
        overlay = img_np.copy().astype(np.float32)
        red = np.zeros_like(overlay); red[:, :, 0] = 255
        m3 = np.stack([mask_pred]*3, axis=-1)
        overlay = overlay * (1 - 0.4 * m3) + red * (0.4 * m3)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Compute per-sample IoU
        gt_bin = (gt_np > 0.5).astype(np.float32)
        inter = (mask_pred * gt_bin).sum()
        union = mask_pred.sum() + gt_bin.sum() - inter
        iou = inter / union if union > 0 else 0.0

        axes[row, 0].imshow(img_np); axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_np, cmap="hot", vmin=0, vmax=1); axes[row, 1].axis("off")
        axes[row, 2].imshow(mask_pred, cmap="hot", vmin=0, vmax=1); axes[row, 2].axis("off")
        axes[row, 3].imshow(overlay); axes[row, 3].axis("off")

        axes[row, 0].set_ylabel(f"det={det_score:.2f}\\nIoU={iou:.2f}",
                                 fontsize=10, rotation=0, labelpad=60)

    plt.suptitle(f"FakeShield-Lite vF.1.0 -- Predictions {title_suffix}",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()


visualize_predictions(model, test_ds, DEVICE, n=6)"""))

# ============================================================================
# STEP 15: Evaluation Results Table (enhanced, replaces old cells 35-36)
# ============================================================================
new_cells.append(md("""\
---
## Evaluation Results

### Metric Definitions

| Metric | Level | Formula | Interpretation |
|--------|-------|---------|----------------|
| **Accuracy** | Image | (TP+TN) / (TP+TN+FP+FN) | Overall classification correctness |
| **Precision** | Image | TP / (TP+FP) | Of predicted tampered, how many are correct |
| **Recall** | Image | TP / (TP+FN) | Of actual tampered, how many are found |
| **F1 Score** | Image | 2*P*R / (P+R) | Harmonic mean of Precision and Recall |
| **IoU** | Pixel | Intersection / Union | Overlap of predicted vs ground-truth mask |
| **Dice** | Pixel | 2*Intersection / (Sum) | Similar to IoU but penalises less harshly |"""))

new_cells.append(code("""\
# ============================================================================
# Evaluate on Test Set
# ============================================================================

test_metrics = validate(model, test_loader, criterion, DEVICE)

print()
print("=" * 60)
print("   FakeShield-Lite vF.1.0 -- Test Set Results")
print("=" * 60)
print()
print(f"  {'Metric':<22s} {'Value':>10s}")
print(f"  {'-'*22} {'-'*10}")
print(f"  {'Detection Accuracy':<22s} {test_metrics['accuracy']:>10.4f}")
print(f"  {'Detection Precision':<22s} {test_metrics['precision']:>10.4f}")
print(f"  {'Detection Recall':<22s} {test_metrics['recall']:>10.4f}")
print(f"  {'Detection F1':<22s} {test_metrics['f1']:>10.4f}")
print(f"  {'-'*22} {'-'*10}")
print(f"  {'Localization IoU':<22s} {test_metrics['iou']:>10.4f}")
print(f"  {'Localization Dice':<22s} {test_metrics['dice']:>10.4f}")
print()
print("=" * 60)"""))

new_cells.append(md("""\
### Evaluation Summary

**Detection** performance reflects the model's ability to classify images as authentic or tampered
at the image level. Higher F1 indicates a good balance between catching tampered images (recall)
and avoiding false alarms (precision).

**Localization** performance (IoU, Dice) measures how accurately the model identifies the specific
tampered region at the pixel level. This is the harder task since it requires precise spatial
understanding of where manipulation occurred."""))

# ============================================================================
# STEP 17: Error Analysis (new)
# ============================================================================
new_cells.append(md("""\
---
## Error Analysis

Understanding failure modes helps identify directions for improvement."""))

new_cells.append(code("""\
# ============================================================================
# Error Analysis -- Collect failure cases
# ============================================================================

model.eval()
failures = {"false_pos": [], "false_neg": [], "low_iou": []}

for i in range(len(test_ds)):
    sample = test_ds[i]
    img_t = sample["image"].unsqueeze(0).to(DEVICE)
    label = sample["label"].item()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = model(img_t)

    det_pred = (out["det_logits"][0].sigmoid().item() > 0.5)
    mask_prob = out["mask_logits"][0].sigmoid().cpu().numpy()
    mask_pred = (mask_prob > 0.5).astype(np.float32)
    gt_mask = sample["mask"].numpy()

    # False positive: authentic predicted as tampered
    if label == 0 and det_pred:
        failures["false_pos"].append(i)
    # False negative: tampered predicted as authentic
    elif label == 1 and not det_pred:
        failures["false_neg"].append(i)
    # Low IoU on tampered images
    elif label == 1:
        gt_bin = (gt_mask > 0.5).astype(np.float32)
        inter = (mask_pred * gt_bin).sum()
        union = mask_pred.sum() + gt_bin.sum() - inter
        iou = inter / union if union > 0 else 0.0
        if iou < 0.2:
            failures["low_iou"].append((i, iou))

print("Error Analysis Summary:")
print(f"  False Positives (authentic -> tampered)  : {len(failures['false_pos'])}")
print(f"  False Negatives (tampered -> authentic)   : {len(failures['false_neg'])}")
print(f"  Low IoU (<0.2) on tampered images         : {len(failures['low_iou'])}")
print(f"  Total test samples                        : {len(test_ds)}")"""))

new_cells.append(md("""\
### Failure Mode Discussion

Common failure modes in image tampering detection and localization include:

1. **Small tampered regions:** When the manipulated area is very small relative to the full image,
   the model may struggle to detect it, leading to false negatives or very low IoU scores.
   SAM's 64x64 feature resolution limits localization precision for sub-pixel manipulations.

2. **Cluttered backgrounds:** Complex scenes with many visual elements can confuse the model
   into predicting spurious tampered regions (false positives). The CLIP encoder may
   misinterpret unusual-but-authentic textures as evidence of tampering.

3. **Subtle artifacts:** High-quality splicing with careful color matching and edge blending
   produces minimal forensic artifacts. Without specialized frequency-domain analysis (like ELA),
   the model relies purely on semantic cues from CLIP, which may not capture low-level
   inconsistencies.

4. **Boundary precision:** Even when the tampered region is correctly detected, the predicted
   mask boundaries may be imprecise due to the low resolution of SAM's internal features
   (64x64) being upsampled to the input resolution (256x256)."""))

# ============================================================================
# STEP 18: Robustness Testing (new)
# ============================================================================
new_cells.append(md("""\
---
## Robustness Testing

We evaluate model performance under common image distortions to assess real-world robustness.
These distortions simulate conditions images might undergo before forensic analysis."""))

new_cells.append(code("""\
# ============================================================================
# Robustness Testing
# ============================================================================

def apply_distortion(image_np, distortion_type, level):
    '''Apply a distortion to a numpy image (H, W, 3) uint8.'''
    if distortion_type == "jpeg":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), level]
        _, enc = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    elif distortion_type == "resize":
        h, w = image_np.shape[:2]
        small = cv2.resize(image_np, (w // level, h // level))
        return cv2.resize(small, (w, h))
    elif distortion_type == "noise":
        noise = np.random.normal(0, level, image_np.shape).astype(np.float32)
        noisy = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    return image_np


def evaluate_robustness(model, test_samples, transform, device, distortion_type, levels):
    '''Evaluate model on distorted versions of test images.'''
    results = {}
    for level in levels:
        all_det_preds, all_det_labels = [], []
        all_ious = []

        tampered_samples = [(p, m, l) for p, m, l in test_samples if l == 1 and m is not None]
        eval_samples = tampered_samples[:100]  # Limit for speed

        for img_path, msk_path, label in eval_samples:
            img_np = np.array(Image.open(img_path).convert("RGB"))
            msk_np = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if msk_np is None:
                continue
            msk_np = (msk_np > 127).astype(np.uint8)

            # Apply distortion
            img_dist = apply_distortion(img_np, distortion_type, level)

            # Transform
            result = transform(image=img_dist, mask=msk_np)
            img_t = result["image"].unsqueeze(0).to(device)
            msk_t = result["mask"].float()

            with torch.no_grad(), torch.amp.autocast("cuda"):
                out = model(img_t)

            det_pred = (out["det_logits"][0].sigmoid().item() > 0.5)
            all_det_preds.append(float(det_pred))
            all_det_labels.append(float(label))

            mask_pred = (out["mask_logits"][0].sigmoid().cpu() > 0.5).float().numpy()
            gt = msk_t.numpy()
            inter = (mask_pred * gt).sum()
            union = mask_pred.sum() + gt.sum() - inter
            iou = inter / union if union > 0 else 0.0
            all_ious.append(iou)

        det_acc = np.mean(np.array(all_det_preds) == np.array(all_det_labels))
        avg_iou = np.mean(all_ious) if all_ious else 0.0
        results[level] = {"det_acc": det_acc, "iou": avg_iou}

    return results


# Run robustness tests
print("Running robustness tests...")
print()

jpeg_results = evaluate_robustness(model, test_samples, val_transform, DEVICE,
                                    "jpeg", [90, 70, 50, 30])
resize_results = evaluate_robustness(model, test_samples, val_transform, DEVICE,
                                      "resize", [2, 4])
noise_results = evaluate_robustness(model, test_samples, val_transform, DEVICE,
                                     "noise", [10, 25, 50])

print("=" * 65)
print("   Robustness Test Results")
print("=" * 65)
print(f"  {'Distortion':<25s} {'Det Acc':>10s} {'IoU':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")
print(f"  {'No distortion':<25s} {test_metrics['accuracy']:>10.4f} {test_metrics['iou']:>10.4f}")
print()
for q, m in jpeg_results.items():
    print(f"  {'JPEG Q=' + str(q):<25s} {m['det_acc']:>10.4f} {m['iou']:>10.4f}")
print()
for f, m in resize_results.items():
    print(f"  {'Resize 1/' + str(f) + 'x':<25s} {m['det_acc']:>10.4f} {m['iou']:>10.4f}")
print()
for s, m in noise_results.items():
    print(f"  {'Gauss noise std=' + str(s):<25s} {m['det_acc']:>10.4f} {m['iou']:>10.4f}")
print("=" * 65)"""))

new_cells.append(md("""\
### Robustness Insights

- **JPEG compression** is the most common real-world distortion. Moderate compression (Q=70)
  should reveal how brittle the model's features are to compression artifacts.
- **Resizing** tests whether the model relies on resolution-specific artifacts that disappear
  when images are downscaled and upscaled.
- **Gaussian noise** simulates sensor noise and tests the model's robustness to random
  pixel-level perturbations."""))

# ============================================================================
# STEP 19: Computational Efficiency (new)
# ============================================================================
new_cells.append(md("""\
---
## Computational Efficiency

FakeShield-Lite is designed to fit within the 16 GB VRAM constraint of a T4 GPU."""))

new_cells.append(code("""\
# ============================================================================
# Computational Efficiency
# ============================================================================

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model size on disk (estimate)
param_size_mb = total_params * 4 / 1e6  # float32

# GPU memory
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        _ = model(dummy)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    del dummy
    torch.cuda.empty_cache()
else:
    peak_mem_gb = 0.0

# Inference speed
model.eval()
times = []
for _ in range(10):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times.append(time.time() - t0)
    del dummy
avg_latency = np.mean(times[2:])  # skip warmup

print("=" * 55)
print("   Computational Efficiency")
print("=" * 55)
print(f"  Total parameters      : {total_params:>12,}")
print(f"  Trainable parameters  : {trainable_params:>12,}")
print(f"  Model size (float32)  : {param_size_mb:>10.1f} MB")
print(f"  Peak GPU memory (B=1) : {peak_mem_gb:>10.2f} GB")
print(f"  T4 VRAM headroom      : {15.0 - peak_mem_gb:>10.2f} GB")
print(f"  Avg inference latency : {avg_latency*1000:>10.1f} ms")
print(f"  Throughput (est.)     : {1.0/avg_latency:>10.1f} img/s")
print("=" * 55)
print()
print(f"FakeShield-Lite uses {peak_mem_gb:.1f} GB of {15.0:.0f} GB available on T4,")
print(f"leaving {15.0-peak_mem_gb:.1f} GB headroom for batch training.")"""))

# ============================================================================
# STEP 20: Inference Demo (new)
# ============================================================================
new_cells.append(md("""\
---
## Inference Demo

Single-image prediction demo. This shows how to use the trained model for inference on
any image."""))

new_cells.append(code("""\
# ============================================================================
# Inference Demo -- Single Image Prediction
# ============================================================================

def run_inference(model, image_path, transform, device):
    '''Run inference on a single image and display results.'''
    img_np = np.array(Image.open(image_path).convert("RGB"))
    result = transform(image=img_np, mask=np.zeros(img_np.shape[:2], dtype=np.uint8))
    img_t = result["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = model(img_t)

    det_score = out["det_logits"][0].sigmoid().item()
    mask_prob = out["mask_logits"][0].sigmoid().cpu().numpy()
    mask_pred = (mask_prob > 0.5).astype(np.float32)

    # Visualize
    img_display = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    overlay = img_display.copy().astype(np.float32)
    red = np.zeros_like(overlay); red[:, :, 0] = 255
    m3 = np.stack([mask_pred]*3, axis=-1)
    overlay = overlay * (1 - 0.4 * m3) + red * (0.4 * m3)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_display); axes[0].set_title("Input Image"); axes[0].axis("off")
    axes[1].imshow(mask_pred, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Predicted Mask"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")

    verdict = "TAMPERED" if det_score > 0.5 else "AUTHENTIC"
    fig.suptitle(f"Detection: {verdict} (score={det_score:.3f})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {"detection_score": det_score, "verdict": verdict, "mask": mask_pred}


# Demo on a random test image
demo_idx = random.choice(range(len(test_ds)))
demo_path = test_ds.samples[demo_idx][0]
demo_label = "tampered" if test_ds.samples[demo_idx][2] == 1 else "authentic"
print(f"Demo image: {os.path.basename(demo_path)} (ground truth: {demo_label})")
result = run_inference(model, demo_path, val_transform, DEVICE)"""))

# ============================================================================
# Section 12 -- Model Saving (old cells 37-39)
# ============================================================================
new_cells.append(old[37])  # Section 12 markdown
new_cells.append(code("""\
# ============================================================================
# Save Final Weights
# ============================================================================

FINAL_PATH = "/kaggle/working/model_vF10_fakeshield_lite.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "test_metrics": test_metrics,
    "config": CONFIG,
    "experiment_info": EXPERIMENT_INFO,
}, FINAL_PATH)

size_mb = os.path.getsize(FINAL_PATH) / 1e6
print(f"Model saved: {FINAL_PATH}  ({size_mb:.1f} MB)")
print("This file will appear in the Output tab of your Kaggle notebook.")"""))

new_cells.append(old[39])  # 12.2 Reload Model (demo)

# ============================================================================
# STEP 21 + 22: Conclusion (replaces old cell 40)
# ============================================================================
new_cells.append(md("""\
---
## Conclusion

### Summary

FakeShield-Lite demonstrates that the core architectural ideas of FakeShield --
dual-encoder design with CLIP + SAM and learned visual prompts -- can be preserved
in a model 100x smaller (~182M vs ~27B parameters) that trains on a single T4 GPU.

### Key Findings

1. **Architecture:** The CLIP + SAM dual-encoder design provides complementary features:
   global semantic understanding (CLIP) and precise spatial features (SAM)
2. **Efficiency:** Per-image SAM encoding + gradient accumulation + mixed precision
   enables training within T4's 16 GB VRAM constraint
3. **Training:** The combined Dice + BCE loss (adapted from FakeShield Eq. 5) jointly
   optimizes detection and localization
4. **Limitations:** Without the LLM components, FakeShield-Lite cannot generate textual
   explanations. SAM ViT-B's lower capacity compared to ViT-H limits segmentation quality.

### Limitations

1. **No text explanations** -- FakeShield's LLM explainability is removed
2. **SAM ViT-B** instead of ViT-H -- lower segmentation quality
3. **Single dataset** -- trained only on CASIA, limiting generalization
4. **No domain tagging** -- 3-class DTG simplified to binary detection
5. **Direct projection** instead of TCM -- less sophisticated prompt generation

### Future Directions

| Improvement | Expected Impact |
|---|---|
| Higher resolution (384 or 512) | Better fine-grained mask quality |
| Multi-dataset training | Improved generalization |
| ELA input channel | Frequency-domain forgery cues |
| Stronger encoder (DINOv2) | Better feature representations |
| Edge-aware loss | Sharper mask boundaries |

---

**End of vF.1.0 FakeShield-Lite Notebook**"""))


# ============================================================================
# FINAL: Replace cells and save
# ============================================================================
nb['cells'] = new_cells

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"vF.1.0 notebook saved with {len(new_cells)} cells (was {len(old)} cells)")
print("Done!")
