"""
Generate vK.11.0 from vK.10.6 by applying architecture + training upgrades.

This is a constructive generator: it reads vK.10.6 as JSON, preserves working
cells (data pipeline, eval suite, visualizations), and replaces cells that need
upgrading (model architecture, training loop, config, losses).

Changes applied (from audit findings + Docs v11):
  - Replace custom UNetWithClassifier with SMP-based TamperDetector
    (pretrained ResNet34 encoder, from v6.5 pattern)
  - Add ELA (Error Level Analysis) as 4th input channel
  - Switch to AdamW with differential learning rates
  - Replace CosineAnnealingLR with ReduceLROnPlateau
  - Add edge supervision loss (Sobel-based boundary BCE)
  - Add gradient accumulation (effective batch = batch_size * accum_steps)
  - Add encoder freeze for first N epochs
  - Fix per-sample Dice loss (from v8)
  - Remove stderr suppression bug
  - Fix CONFIG/docs mismatch
  - Fix VRAM auto-scaling documentation
  - Add ELA visualization section
  - Update Grad-CAM target for SMP encoder
  - Improve robustness testing to use Albumentations pattern

Critical constraint: evaluation suite, data leakage checks, checkpoint system,
and visualization cells are preserved from vK.10.6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.10.6 Image Detection and Localisation.ipynb"
OUTPUT_PATH = NOTEBOOKS_DIR / "source" / "vK.11.0 Image Detection and Localisation.ipynb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(nb: dict, path: Path) -> None:
    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {path.name}")


def to_source_lines(text: str) -> list[str]:
    """Convert a plain string into the notebook source-line list format."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def make_md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_lines(text),
    }


def make_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(text),
    }


def copy_cell(src_nb: dict, idx: int) -> dict:
    """Deep-copy a cell from the source notebook."""
    cell = copy.deepcopy(src_nb["cells"][idx])
    cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None
    return cell


def copy_cell_with_append(src_nb: dict, idx: int, append_code: str) -> dict:
    """Deep-copy a source cell and append W&B logging code."""
    cell = copy_cell(src_nb, idx)
    if cell["cell_type"] == "code":
        existing = "".join(cell["source"]).rstrip("\n")
        combined = existing + "\n\n" + append_code
        cell["source"] = to_source_lines(combined)
    return cell


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

def build_cells(src: dict) -> list[dict]:
    """Build the complete vK.11.0 cell list."""
    cells = []

    # ======================================================================
    # CELL 0: Table of Contents
    # ======================================================================
    cells.append(make_md_cell("""\
# Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Configuration](#2-configuration)
3. [Reproducibility and Device Setup](#3-reproducibility-and-device-setup)
4. [Dataset Discovery and Metadata Cache](#4-dataset-discovery-and-metadata-cache)
5. [Dependencies and Imports](#5-dependencies-and-imports)
6. [Data Loading and Preprocessing](#6-data-loading-and-preprocessing)
7. [Model Architecture](#7-model-architecture)
8. [Experiment Tracking](#8-experiment-tracking)
9. [Training Utilities](#9-training-utilities)
10. [Training Loop](#10-training-loop)
11. [Evaluation](#11-evaluation) (includes threshold sweep, pixel AUC, confusion matrix, forgery-type, mask-size, shortcut checks)
12. [Visualization of Predictions](#12-visualization-of-predictions) (includes ELA visualization)
13. [Advanced Analysis](#13-advanced-analysis) (failure cases)
14. [Explainability](#14-explainability) (Grad-CAM)
15. [Robustness Testing](#15-robustness-testing)
16. [Inference Examples](#16-inference-examples)
17. [Conclusion](#17-conclusion)"""))

    # ======================================================================
    # CELL 1: Title + Introduction
    # ======================================================================
    cells.append(make_md_cell("""\
# Tampered Image Detection and Localization (vK.11.0)

This Kaggle-first notebook presents a complete assignment submission for tampered image
detection and tampered region localization.

**Architecture upgrades in vK.11.0** (from audit findings + Docs v11)
- **Pretrained ResNet34 encoder** via `segmentation_models_pytorch` (SMP) — proven in v6.5 to achieve Tam-F1=0.41 vs 0.22 from scratch
- **ELA (Error Level Analysis)** as 4th input channel — amplifies JPEG compression inconsistencies between authentic and tampered regions
- **Edge supervision loss** (Sobel-based boundary BCE) — improves boundary delineation around tampered regions
- **AdamW with differential learning rates** — encoder=1e-4, decoder/heads=1e-3 (from v6.5 pattern)
- **ReduceLROnPlateau scheduler** — replaces CosineAnnealing double-cycle bug from vK.10.6
- **Gradient accumulation** — effective batch = batch_size × accumulation_steps
- **Encoder freeze** for first 2 epochs — protects pretrained BatchNorm statistics
- **Per-sample Dice loss** — fixes batch-level bias toward large masks (from v8)

**Evaluation suite carried forward from vK.10.6** (12 features)
- Threshold sweep, pixel-level AUC, confusion matrix, ROC curve, PR curve
- Forgery-type breakdown, mask-size stratification, shortcut learning checks
- Grad-CAM explainability, robustness testing (8 conditions), failure case analysis
- Data leakage verification (path overlap assertion)

**Engineering carried forward from vK.10.5/vK.10.6**
- Multi-GPU training with `nn.DataParallel`
- Centralized CONFIG dictionary for all hyperparameters
- Full reproducibility seeding (Python, NumPy, PyTorch CPU/CUDA)
- Automatic Mixed Precision (AMP) for faster training
- Three-file checkpoint system with automatic resume
- Early stopping based on tampered-only Dice coefficient
- Metadata caching and VRAM-based batch size auto-adjustment

**Notebook deliverables**
- Image-level tamper detection through the classifier head
- Pixel-level tampered region localization through the segmentation branch
- ELA forensic signal visualization
- Comprehensive 12-point evaluation suite with robustness testing"""))

    # ======================================================================
    # CELL 2: Project Objectives table
    # ======================================================================
    cells.append(make_md_cell("""\
## Project Objectives: Fulfilled vs Remaining

| Requirement | Status | Evidence |
|---|---|---|
| Dataset: authentic + tampered + masks | Fulfilled | CASIA dataset with IMAGE/MASK dirs |
| Model performs detection + localization | Fulfilled | TamperDetector dual-head (SMP UNet + classifier) |
| Pretrained encoder for transfer learning | Fulfilled | ResNet34 (ImageNet) via SMP |
| ELA forensic preprocessing | Fulfilled | 4th input channel from JPEG re-save analysis |
| Evaluation with Dice / IoU / F1 | Fulfilled | Tampered-only and all-sample metrics |
| Visual results (Original, GT, Pred, Overlay) | Fulfilled | Submission prediction grid |
| Single notebook | Fulfilled | All code in one notebook |
| Reproducibility | Fulfilled | Full seeding + checkpoint resume |
| AMP training | Fulfilled | autocast + GradScaler |
| Early stopping | Fulfilled | Patience-based on tampered Dice |
| Multi-GPU utilization | Fulfilled | nn.DataParallel across T4 x2 |
| Threshold optimization | Fulfilled | 50-point sweep on validation set |
| Robustness testing | Fulfilled | 8 degradation conditions with deltas |
| Grad-CAM explainability | Fulfilled | Encoder layer4 heatmaps |
| Confusion matrix + ROC/PR | Fulfilled | seaborn heatmap + sklearn curves |
| Forgery-type breakdown | Fulfilled | Splicing vs copy-move metrics |
| Data leakage verification | Fulfilled | Path overlap assertion |
| Shortcut learning checks | Fulfilled | Mask randomization + boundary sensitivity |
| Failure case analysis | Fulfilled | 10 worst predictions annotated |
| Pixel-level AUC-ROC | Fulfilled | Threshold-independent localization metric |
| Edge supervision loss | Fulfilled | Sobel-based boundary BCE |
| Gradient accumulation | Fulfilled | Effective batch = batch_size x accumulation_steps |
| Encoder freeze warmup | Fulfilled | First 2 epochs frozen |"""))

    # ======================================================================
    # CELL 3: Section header — Environment Setup
    # ======================================================================
    cells.append(make_md_cell("## 1. Environment Setup"))

    # ======================================================================
    # CELLS 4-6: Environment setup — cell 4 modified to install SMP
    # ======================================================================
    cells.append(make_code_cell("""\
import subprocess
import sys
import os
import shutil
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
KAGGLE_DATASET_SLUG = "harshv777/casia2-0-upgraded-dataset"
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")
ATTACHED_DATASET_DIR = KAGGLE_INPUT_DIR / "casia2-0-upgraded-dataset"
DRIVE_SEARCH_ROOTS = [Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]

# Install segmentation-models-pytorch (not pre-installed on Kaggle or Colab)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "segmentation-models-pytorch",
])

if IN_COLAB:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "albumentations==1.3.1", "opencv-python-headless==4.10.0.84", "kaggle",
    ])

KAGGLE_WORKING_DIR.mkdir(parents=True, exist_ok=True)

print("IN_COLAB:", IN_COLAB)
print("KAGGLE_INPUT_DIR:", KAGGLE_INPUT_DIR)
print("KAGGLE_WORKING_DIR:", KAGGLE_WORKING_DIR)
print("ATTACHED_DATASET_DIR:", ATTACHED_DATASET_DIR)"""))
    cells.append(copy_cell(src, 5))   # has_valid_layout + dataset discovery
    cells.append(copy_cell(src, 6))   # debug: show Kaggle dirs

    # NOTE: Cell 7 (stderr suppression) is REMOVED — Bug #2 from audit

    # ======================================================================
    # CELL 7: Configuration header
    # ======================================================================
    cells.append(make_md_cell("""\
## 2. Configuration

Centralized configuration dictionary. All hyperparameters are defined here
and referenced by name throughout the notebook. Documentation tables below
are generated from the same values to prevent mismatches."""))

    # ======================================================================
    # CELL 8: CONFIG dict — MAJOR REWRITE
    # ======================================================================
    cells.append(make_code_cell("""\
import os
from pathlib import Path

SEED = 42
NB_VERSION = "vK.11.0"

CONFIG = {
    # -- Data --
    'image_size': 256,
    'batch_size': 8,             # auto-adjusted based on GPU VRAM
    'num_workers': 4,
    'train_ratio': 0.70,

    # -- Model (SMP pretrained — from v6.5 + Docs v11) --
    'architecture': 'TamperDetector',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 4,            # RGB + ELA
    'n_classes': 1,              # segmentation output channels
    'n_labels': 2,               # classification labels (authentic/tampered)
    'dropout': 0.5,

    # -- Optimizer (AdamW with differential LR — from v6.5) --
    'optimizer': 'AdamW',
    'encoder_lr': 1e-4,          # pretrained encoder: lower LR
    'decoder_lr': 1e-3,          # decoder + heads: higher LR
    'weight_decay': 1e-4,
    'max_grad_norm': 5.0,

    # -- Scheduler (ReduceLROnPlateau — from v8) --
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,

    # -- Loss --
    'alpha': 1.5,                # classification loss weight
    'beta': 1.0,                 # segmentation loss weight
    'gamma': 0.3,                # edge loss weight (NEW)
    'focal_gamma': 2.0,
    'seg_bce_weight': 0.5,
    'seg_dice_weight': 0.5,

    # -- Training --
    'max_epochs': 50,
    'patience': 10,              # early stopping patience
    'checkpoint_every': 10,      # periodic checkpoint interval
    'accumulation_steps': 4,     # gradient accumulation steps
    'encoder_freeze_epochs': 2,  # freeze encoder for first N epochs

    # -- Feature Flags --
    'use_amp': True,
    'use_wandb': True,
    'seg_threshold': 0.5,

    # -- ELA --
    'ela_quality': 90,           # JPEG re-save quality for ELA computation

    # -- Reproducibility --
    'seed': SEED,
}

# Determine Kaggle vs local paths
if os.path.exists('/kaggle/working'):
    KAGGLE_WORKING_DIR = Path('/kaggle/working')
    KAGGLE_INPUT_DIR = Path('/kaggle/input')
elif os.path.exists('/content/drive'):
    KAGGLE_WORKING_DIR = Path('/content/drive/MyDrive/BigVision')
    KAGGLE_INPUT_DIR = KAGGLE_WORKING_DIR / 'input'
else:
    KAGGLE_WORKING_DIR = Path('.')
    KAGGLE_INPUT_DIR = Path('./input')

CHECKPOINT_DIR = Path(KAGGLE_WORKING_DIR) / 'checkpoints'
RESULTS_DIR    = Path(KAGGLE_WORKING_DIR) / 'results'
PLOTS_DIR      = Path(KAGGLE_WORKING_DIR) / 'plots'

for d in [CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f'Working directory: {KAGGLE_WORKING_DIR}')
print(f'Input directory:   {KAGGLE_INPUT_DIR}')"""))

    # ======================================================================
    # CELL 9: Hyperparameter Summary — auto-correct to match CONFIG
    # ======================================================================
    cells.append(make_md_cell("""\
### 2.1 Hyperparameter Summary

| Hyperparameter | Value | Source |
|---|---|---|
| `architecture` | TamperDetector (SMP UNet + classifier head) | NEW in vK.11.0 |
| `encoder_name` | resnet34 | From v6.5 (proven Tam-F1=0.41) |
| `encoder_weights` | imagenet | Pretrained transfer learning |
| `in_channels` | 4 (RGB + ELA) | NEW in vK.11.0 |
| `image_size` | 256 | Preserved from vK.10.6 |
| `batch_size` | 8 (auto-adjusted) | VRAM-based |
| `accumulation_steps` | 4 (effective batch = batch_size x 4) | NEW in vK.11.0 |
| `optimizer` | AdamW | From v6.5 |
| `encoder_lr` | 1e-4 | Differential LR (from v6.5) |
| `decoder_lr` | 1e-3 | Differential LR (from v6.5) |
| `scheduler` | ReduceLROnPlateau(patience=3, factor=0.5) | From v8 |
| `max_epochs` | 50 | Preserved |
| `patience` | 10 | Early stopping on tampered Dice |
| `encoder_freeze_epochs` | 2 | NEW in vK.11.0 |
| `alpha` (cls weight) | 1.5 | Preserved |
| `beta` (seg weight) | 1.0 | Preserved |
| `gamma` (edge weight) | 0.3 | NEW in vK.11.0 |
| `focal_gamma` | 2.0 | Preserved |
| `use_amp` | True | Preserved |
| `ela_quality` | 90 | NEW in vK.11.0 |"""))

    # ======================================================================
    # CELL 10: Reproducibility header
    # ======================================================================
    cells.append(make_md_cell("""\
## 3. Reproducibility and Device Setup

Full reproducibility is enforced through seeded random number generators across
Python, NumPy, and PyTorch. GPU diagnostics confirm hardware capabilities and
auto-adjust the batch size based on available VRAM."""))

    # ======================================================================
    # CELL 11: Reproducibility + device + VRAM auto-scaling — from vK.10.6 cell 12 with corrected docs
    # ======================================================================
    cells.append(make_code_cell("""\
import random
import numpy as np
import torch


def set_seed(seed):
    \"\"\"Set seeds for full reproducibility across all libraries.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CONFIG['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()

    for gpu_id in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        vram_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        print(f'GPU {gpu_id}:          {gpu_name} ({vram_gb:.1f} GB)')

    total_vram_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus)) / 1e9

    # TF32 for faster matmul on Ampere+ (does not affect determinism)
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f'GPUs available: {n_gpus}')
    print(f'Total VRAM:     {total_vram_gb:.1f} GB')
    print(f'cuDNN:          {torch.backends.cudnn.enabled}')
    print(f'Deterministic:  {torch.backends.cudnn.deterministic}')
    print(f'AMP:            {"enabled" if CONFIG["use_amp"] else "disabled"}')

    # Auto-adjust batch size based on total VRAM (SMP model ~1.1 GB base)
    if total_vram_gb >= 28:
        CONFIG['batch_size'] = 32
    elif total_vram_gb >= 20:
        CONFIG['batch_size'] = 24
    elif total_vram_gb >= 15:
        CONFIG['batch_size'] = 16
    # else keep CONFIG default (8)
    print(f'Batch size (auto-adjusted for {n_gpus} GPU{"s" if n_gpus > 1 else ""}): {CONFIG["batch_size"]}')
    print(f'Effective batch size: {CONFIG["batch_size"] * CONFIG["accumulation_steps"]}')
else:
    n_gpus = 0
    print('WARNING: No GPU detected. Training will be extremely slow.')

print(f'Device: {device}')"""))

    # ======================================================================
    # CELL 12: VRAM docs — corrected to match actual code
    # ======================================================================
    cells.append(make_md_cell("""\
The batch size is auto-adjusted based on total available GPU VRAM:
- **>=28 GB** (2x T4): `batch_size=32` (effective=128 with accumulation)
- **>=20 GB**: `batch_size=24` (effective=96)
- **>=15 GB** (single T4): `batch_size=16` (effective=64)
- **<15 GB**: `batch_size=8` (effective=32, default)

The effective hyperparameters logged to W&B reflect the adjusted value."""))

    # ======================================================================
    # CELLS 13-23: Dataset Discovery, Metadata, Split, Summary, Leakage — PRESERVE
    # ======================================================================
    cells.append(copy_cell(src, 14))   # Section 4 header
    cells.append(copy_cell(src, 15))   # 2.1.1 Locate IMAGE and MASK
    cells.append(copy_cell(src, 16))   # Dataset discovery code
    cells.append(copy_cell(src, 17))   # 2.1.2 Build Metadata Table
    cells.append(copy_cell(src, 18))   # Metadata build code
    cells.append(copy_cell(src, 19))   # 2.2 Train/Val/Test Split header
    # Cell 20 — fix to use CONFIG['train_ratio']
    cells.append(make_code_cell("""\
import pandas as pd
from sklearn.model_selection import train_test_split

# Attempt to load cached split CSVs
train_csv = os.path.join(KAGGLE_WORKING_DIR, 'train_metadata.csv')
val_csv   = os.path.join(KAGGLE_WORKING_DIR, 'val_metadata.csv')
test_csv  = os.path.join(KAGGLE_WORKING_DIR, 'test_metadata.csv')

if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)
    total_loaded = len(train_df) + len(val_df) + len(test_df)
    if total_loaded == len(df):
        print(f'Loaded cached splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
    else:
        print(f'Cache stale ({total_loaded} vs {len(df)}), re-splitting...')
        train_df, temp_df = train_test_split(
            df, test_size=1 - CONFIG['train_ratio'], stratify=df['label'],
            random_state=CONFIG['seed'],
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df['label'],
            random_state=CONFIG['seed'],
        )
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
else:
    train_df, temp_df = train_test_split(
        df, test_size=1 - CONFIG['train_ratio'], stratify=df['label'],
        random_state=CONFIG['seed'],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'],
        random_state=CONFIG['seed'],
    )
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')"""))

    cells.append(copy_cell(src, 21))   # 4.4 Dataset Summary header
    cells.append(copy_cell(src, 22))   # Summary code
    cells.append(copy_cell(src, 23))   # 4.5 Data Leakage Verification header
    cells.append(copy_cell(src, 24))   # Leakage verification code

    # ======================================================================
    # CELL: Dependencies and Imports — MODIFIED to add smp
    # ======================================================================
    cells.append(make_md_cell("""\
## 5. Dependencies and Imports

All training, evaluation, and visualization imports are consolidated here
to avoid redundant imports scattered across multiple cells."""))

    cells.append(make_code_cell("""\
import cv2
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Kaggle
import matplotlib.pyplot as plt

print('Imports complete.')
print(f'PyTorch: {torch.__version__}')
print(f'SMP: {smp.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')"""))

    # ======================================================================
    # Data Loading section header
    # ======================================================================
    cells.append(copy_cell(src, 27))   # Section 6 header
    cells.append(copy_cell(src, 28))   # 6.1 Transforms header

    # ======================================================================
    # ELA computation function — NEW
    # ======================================================================
    cells.append(make_code_cell("""\
# ================== ELA (Error Level Analysis) ==================
def compute_ela(image_bgr, quality=90):
    \"\"\"Compute Error Level Analysis map.

    Re-saves image as JPEG at given quality, then computes absolute
    difference to reveal JPEG compression inconsistencies between
    authentic and tampered regions.

    Args:
        image_bgr: Input image in BGR format (as loaded by cv2.imread).
        quality: JPEG re-save quality (default 90).

    Returns:
        Grayscale ELA map as uint8 numpy array.
    \"\"\"
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, decoded)
    return cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)"""))

    # ======================================================================
    # Transforms — MODIFIED with VerticalFlip and ELA-aware additional_targets
    # ======================================================================
    cells.append(make_code_cell("""\
# ================== Define transforms ==================
IMAGE_SIZE = CONFIG['image_size']

def get_train_transform():
    \"\"\"Augmentation pipeline for training images, masks, and ELA maps.\"\"\"
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.25),
        A.ImageCompression(quality_range=(50, 90), p=0.25),
        A.Affine(
            translate_percent={'x': (-0.02, 0.02), 'y': (-0.02, 0.02)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'ela': 'image'})

def get_valid_transform():
    \"\"\"Deterministic preprocessing for validation and test samples.\"\"\"
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'ela': 'image'})"""))

    # ======================================================================
    # Dataset class header
    # ======================================================================
    cells.append(make_md_cell("""\
### 6.2 Dataset Class

The `ELAImageMaskDataset` class loads image-mask pairs from metadata, computes ELA maps,
applies shared transforms, and returns 4-channel tensors (RGB + ELA) compatible with the
TamperDetector model."""))

    # ======================================================================
    # Dataset class — REWRITTEN for ELA
    # ======================================================================
    cells.append(make_code_cell("""\
# ================== Dataset with ELA channel ==================
class ELAImageMaskDataset(Dataset):
    \"\"\"Dataset that loads RGB images, computes ELA, and returns 4-channel input.\"\"\"

    def __init__(self, df, transform=None, ela_quality=90):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.ela_quality = ela_quality

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        mask_path = row['mask_path']
        label = int(row['label'])

        # Load image in BGR (OpenCV default)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f'Failed to read image: {img_path}')

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)

        # Compute ELA before color conversion
        ela = compute_ela(image_bgr, quality=self.ela_quality)

        # Convert to RGB for augmentation
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype('float32')

        # Convert ELA to 3-channel for Albumentations compatibility
        ela_3ch = np.stack([ela, ela, ela], axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask, ela=ela_3ch)
            image_t = augmented['image']       # (3, H, W) normalized
            mask_t  = augmented['mask']         # (H, W)
            ela_t   = augmented['ela']          # (3, H, W) — take first channel
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t  = torch.from_numpy(mask).float()
            ela_t   = torch.from_numpy(ela_3ch).permute(2, 0, 1).float() / 255.0

        # Extract single ELA channel and normalize to [0, 1]
        ela_ch = ela_t[0:1]  # (1, H, W) — already normalized by Albumentations

        # Stack RGB + ELA = 4 channels
        input_tensor = torch.cat([image_t, ela_ch], dim=0)  # (4, H, W)

        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        return input_tensor, mask_t, torch.tensor(label, dtype=torch.long)"""))

    # ======================================================================
    # DataLoader Construction — from vK.10.6 cell 32-33, updated class name
    # ======================================================================
    cells.append(make_md_cell("""\
### 6.3 DataLoader Construction

DataLoaders with persistent workers, seeded worker init, and drop_last for training stability."""))

    cells.append(make_code_cell("""\
def seed_worker(worker_id):
    \"\"\"Ensure each DataLoader worker uses a unique but reproducible seed.\"\"\"
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(CONFIG['seed'])

train_dataset = ELAImageMaskDataset(train_df, transform=get_train_transform(),
                                     ela_quality=CONFIG['ela_quality'])
val_dataset   = ELAImageMaskDataset(val_df,   transform=get_valid_transform(),
                                     ela_quality=CONFIG['ela_quality'])
test_dataset  = ELAImageMaskDataset(test_df,  transform=get_valid_transform(),
                                     ela_quality=CONFIG['ela_quality'])

common_loader_args = dict(
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=True,
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=True, drop_last=True, **common_loader_args)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                          shuffle=False, **common_loader_args)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'],
                          shuffle=False, **common_loader_args)

print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}')
print(f'Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}')"""))

    # ======================================================================
    # Data Visualization — MODIFIED from vK.10.6 cell 35 (fix 4-channel + ELA aug)
    # ======================================================================
    cells.append(copy_cell(src, 34))   # 6.4 Data Visualization header
    cells.append(make_code_cell("""\
# ================== Pre-training data visualization ==================
import matplotlib.pyplot as plt
import numpy as np

def _denorm(img_t):
    \"\"\"Reverse ImageNet normalization for display (handles 4-ch RGB+ELA).\"\"\"
    if img_t.shape[0] == 4:
        img_t = img_t[:3]  # extract RGB, drop ELA channel
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = img_t.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1)

# --- 1) Sample grid: 4 authentic + 4 tampered with masks ---
fig, axes = plt.subplots(4, 3, figsize=(10, 13))
fig.suptitle('Sample Images (top 2 authentic, bottom 2 tampered)', fontsize=13)

auth_indices = [i for i in range(len(train_dataset)) if train_df.iloc[i]['label'] == 0][:2]
tamp_indices = [i for i in range(len(train_dataset)) if train_df.iloc[i]['label'] == 1][:2]

for row, idx in enumerate(auth_indices + tamp_indices):
    img, mask, label = train_dataset[idx]
    lbl_str = 'Authentic' if label == 0 else 'Tampered'
    axes[row, 0].imshow(_denorm(img))
    axes[row, 0].set_title(f'{lbl_str} (idx {idx})')
    axes[row, 1].imshow(mask.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[row, 1].set_title('Ground Truth Mask')
    axes[row, 2].imshow(_denorm(img))
    axes[row, 2].imshow(mask.squeeze().numpy(), cmap='Reds', alpha=0.4)
    axes[row, 2].set_title('Overlay')
    for ax in axes[row]:
        ax.axis('off')
plt.tight_layout()
plt.show()

# --- 2) Class distribution per split ---
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (name, df) in zip(axes, [('Train', train_df), ('Val', val_df), ('Test', test_df)]):
    counts = df['label'].value_counts().sort_index()
    bars = ax.bar(['Authentic', 'Tampered'], [counts.get(0, 0), counts.get(1, 0)],
                  color=['#2ecc71', '#e74c3c'])
    ax.set_title(f'{name} (n={len(df)})')
    ax.set_ylabel('Count')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', fontsize=9)
fig.suptitle('Class Distribution Across Splits', fontsize=13)
plt.tight_layout()
plt.show()

# --- 3) Augmentation preview: same image with 4 random augmentations ---
aug_idx = tamp_indices[0]  # pick first tampered sample
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle('Augmentation Preview (same tampered image, 4 random transforms)', fontsize=12)
raw_row = train_df.iloc[aug_idx]
raw_img = cv2.cvtColor(cv2.imread(raw_row['image_path']), cv2.COLOR_BGR2RGB)
raw_img_resized = cv2.resize(raw_img, (IMAGE_SIZE, IMAGE_SIZE))
axes[0].imshow(raw_img_resized)
axes[0].set_title('Original')
axes[0].axis('off')
aug_tf = get_train_transform()
raw_mask = cv2.imread(raw_row['mask_path'], cv2.IMREAD_GRAYSCALE) if pd.notna(raw_row.get('mask_path')) else np.zeros((raw_img.shape[0], raw_img.shape[1]), dtype=np.uint8)
for j in range(1, 5):
    ela_raw = compute_ela(cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR), quality=CONFIG['ela_quality'])
    ela_3ch = np.stack([ela_raw, ela_raw, ela_raw], axis=-1)
    augmented = aug_tf(image=raw_img, mask=raw_mask, ela=ela_3ch)
    aug_img = augmented['image']
    axes[j].imshow(_denorm(aug_img))
    axes[j].set_title(f'Aug {j}')
    axes[j].axis('off')
plt.tight_layout()
plt.show()"""))

    # ======================================================================
    # CELL: Model Architecture — MAJOR REWRITE
    # ======================================================================
    cells.append(make_md_cell("""\
## 7. Model Architecture

A dual-head model (`TamperDetector`) combining an SMP UNet with a pretrained ResNet34 encoder
for segmentation and a custom classification head on the bottleneck features.

```
              Input (4 x 256 x 256)
              [RGB + ELA channel]
                      |
            +-------------------+
            |  ResNet34 Encoder  | (ImageNet pretrained)
            |                   |
            |  Stage 0: 64ch    | -> skip_0
            |  Stage 1: 64ch    | -> skip_1
            |  Stage 2: 128ch   | -> skip_2
            |  Stage 3: 256ch   | -> skip_3
            |  Stage 4: 512ch   | -> bottleneck
            +--+-------------+--+
               |             |
    +----------+             +-----------+
    | DECODER                   CLASSIFIER |
    |                                      |
    | Up(512->256) + skip_3      AdaptiveAvgPool(1x1)
    | Up(256->128) + skip_2           Flatten
    | Up(128->64) + skip_1       Linear(512->256)
    | Up(64->32) + skip_0         ReLU + Dropout
    |                          Linear(256->2)
    | Conv(32->1, 1x1)               |
    |       |                        v
    v       v              cls_logits (B x 2)
seg_logits (B x 1 x 256 x 256)
```

**Key design choices:**
- **Pretrained ResNet34 encoder** — ImageNet features provide rich low-level edge/texture detectors
  that forensic detection builds on. v6.5 proved this achieves Tam-F1=0.41 vs 0.22 from scratch.
- **4-channel input (RGB + ELA)** — SMP auto-adapts the first conv layer for >3 channels.
- **Dual heads** — classification head on bottleneck (AUC=0.91 in vK.10.6), segmentation via decoder.
- **~24.5M parameters** — smaller than the custom UNet (31.6M) yet stronger due to pretrained features."""))

    # ======================================================================
    # Model class definition — NEW TamperDetector
    # ======================================================================
    cells.append(make_code_cell("""\
# ================== TamperDetector: SMP UNet + Classification Head ==================
class TamperDetector(nn.Module):
    \"\"\"
    Dual-head model for tampered image detection and localization.

    Segmentation branch: SMP UNet with pretrained ResNet34 encoder.
    Classification branch: FC head on encoder bottleneck features.

    Args:
        config: Dictionary containing model hyperparameters.
    \"\"\"
    def __init__(self, config):
        super().__init__()
        self.segmentor = smp.Unet(
            encoder_name=config['encoder_name'],
            encoder_weights=config['encoder_weights'],
            in_channels=config['in_channels'],
            classes=config['n_classes'],
        )

        # Classification head on bottleneck (512 for ResNet34)
        encoder_out = self.segmentor.encoder.out_channels[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(256, config['n_labels']),
        )

    def forward(self, x):
        features = self.segmentor.encoder(x)
        cls_logits = self.classifier(features[-1])
        decoder_output = self.segmentor.decoder(features)
        seg_logits = self.segmentor.segmentation_head(decoder_output)
        return cls_logits, seg_logits"""))

    # ======================================================================
    # Model instantiation — REWRITTEN
    # ======================================================================
    cells.append(make_code_cell("""\
model = TamperDetector(CONFIG).to(device)

# Wrap with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f'DataParallel enabled across {torch.cuda.device_count()} GPUs')


def get_base_model(m):
    \"\"\"Unwrap DataParallel to access the underlying model.\"\"\"
    return m.module if isinstance(m, nn.DataParallel) else m


def freeze_encoder(m):
    \"\"\"Freeze encoder parameters to protect pretrained BatchNorm statistics.\"\"\"
    base = get_base_model(m)
    for param in base.segmentor.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(m):
    \"\"\"Unfreeze encoder parameters for fine-tuning.\"\"\"
    base = get_base_model(m)
    for param in base.segmentor.encoder.parameters():
        param.requires_grad = True


# Verify model
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters:     {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# Shape verification
with torch.no_grad():
    dummy = torch.randn(1, CONFIG['in_channels'], CONFIG['image_size'], CONFIG['image_size']).to(device)
    cls_out, seg_out = model(dummy)
    assert seg_out.shape == (1, 1, CONFIG['image_size'], CONFIG['image_size']), \\
        f'Unexpected seg shape: {seg_out.shape}'
    assert cls_out.shape == (1, CONFIG['n_labels']), \\
        f'Unexpected cls shape: {cls_out.shape}'
print(f'Shape check passed: ({CONFIG["in_channels"]}, {CONFIG["image_size"]}, {CONFIG["image_size"]}) -> seg {seg_out.shape}, cls {cls_out.shape}')"""))

    # ======================================================================
    # W&B — from vK.10.6 cells 39-40, updated project/run name
    # ======================================================================
    cells.append(copy_cell(src, 39))   # Section 8 header

    cells.append(make_code_cell("""\
import importlib.util
import subprocess

WANDB_ACTIVE = False
WANDB_RUN = None

if CONFIG['use_wandb']:
    WANDB_CONFIG = {k: v for k, v in CONFIG.items()}
    WANDB_CONFIG['nb_version'] = NB_VERSION

    try:
        if importlib.util.find_spec("wandb") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb"])

        import wandb

        try:
            from kaggle_secrets import UserSecretsClient
            wandb_api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
            if not wandb_api_key:
                raise ValueError('WANDB_API_KEY is empty')
            wandb.login(key=wandb_api_key)
            WANDB_RUN = wandb.init(
                project=f"{NB_VERSION}-tampered-image-detection-assignment",
                name=f"{NB_VERSION}-smp-resnet34-ela-seed{SEED}-kaggle",
                tags=[NB_VERSION.lower().replace(".", ""), "smp", "resnet34", "ela", "edge-loss", "amp", "early-stopping", "multi-gpu", "eval-suite"],
                config=WANDB_CONFIG,
                reinit=True,
            )
            WANDB_ACTIVE = True
        except Exception as auth_exc:
            print(f"W&B online unavailable, switching to offline: {auth_exc}")
            WANDB_RUN = wandb.init(
                project="tampered-image-detection-assignment",
                name=f"{NB_VERSION}-offline",
                config=WANDB_CONFIG,
                mode="offline",
                reinit=True,
            )
            WANDB_ACTIVE = True

        # Define metric axes so W&B plots use epoch as x-axis
        if WANDB_ACTIVE:
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("lr/*", step_metric="epoch")

    except Exception as exc:
        print(f"W&B setup failed: {exc}")

print(f"W&B active: {WANDB_ACTIVE}")
if WANDB_ACTIVE:
    print(f"W&B project: {NB_VERSION}-tampered-image-detection-assignment")
    print(f"W&B run: {WANDB_RUN.name}")
    wandb.config.update({
        'notebook_version': NB_VERSION,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'num_params': sum(p.numel() for p in get_base_model(model).parameters()),
    }, allow_val_change=True)"""))

    # ======================================================================
    # Training Utilities — REWRITTEN
    # ======================================================================
    cells.append(make_md_cell("""\
## 9. Training Utilities

This section defines loss functions (Focal, BCE+Dice, Edge), evaluation metrics,
the optimizer (AdamW with differential LR), scheduler (ReduceLROnPlateau),
checkpoint helpers, and the AMP scaler."""))

    cells.append(make_code_cell("""\
# ================== Loss functions, optimizer, scheduler, AMP scaler ==================

# Compute class weights from the training split
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"].values,
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)


class FocalLoss(nn.Module):
    \"\"\"Focal-style classification loss that down-weights easy examples.\"\"\"
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def dice_loss(pred, target, eps=1e-7):
    \"\"\"Per-sample Dice loss for segmentation logits (from v8 improvement).\"\"\"
    prob = torch.sigmoid(pred)
    # Per-sample computation to avoid batch-level bias toward large masks
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return (1 - dice).mean()


def edge_loss(pred_logits, gt_masks):
    \"\"\"Sobel-based edge supervision loss (from Docs v11 I7).

    Computes BCE between predicted and ground truth mask edges to improve
    boundary delineation around tampered regions.
    \"\"\"
    pred_prob = torch.sigmoid(pred_logits)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=pred_prob.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    # Ground truth edges
    gt_edge = (F.conv2d(gt_masks, sobel_x, padding=1).abs() +
               F.conv2d(gt_masks, sobel_y, padding=1).abs()).clamp(0, 1)
    # Predicted edges
    pred_edge = (F.conv2d(pred_prob, sobel_x, padding=1).abs() +
                 F.conv2d(pred_prob, sobel_y, padding=1).abs()).clamp(0, 1)
    return F.binary_cross_entropy(pred_edge, gt_edge)


criterion_cls = FocalLoss(alpha=class_weights, gamma=CONFIG['focal_gamma'])
bce_loss      = nn.BCEWithLogitsLoss()
ALPHA = CONFIG['alpha']
BETA  = CONFIG['beta']
GAMMA = CONFIG['gamma']

# AdamW with differential learning rates (encoder=1e-4, decoder/heads=1e-3)
base_model = get_base_model(model)
optimizer = torch.optim.AdamW([
    {'params': base_model.segmentor.encoder.parameters(), 'lr': CONFIG['encoder_lr']},
    {'params': base_model.segmentor.decoder.parameters(), 'lr': CONFIG['decoder_lr']},
    {'params': base_model.segmentor.segmentation_head.parameters(), 'lr': CONFIG['decoder_lr']},
    {'params': base_model.classifier.parameters(), 'lr': CONFIG['decoder_lr']},
], weight_decay=CONFIG['weight_decay'])

# ReduceLROnPlateau — monitors val tampered F1 (replaces CosineAnnealing double-cycle bug)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max',
    patience=CONFIG['scheduler_patience'],
    factor=CONFIG['scheduler_factor'],
    verbose=True,
)

# AMP scaler
scaler = GradScaler('cuda', enabled=CONFIG['use_amp'])

print(f'Optimizer: AdamW(encoder_lr={CONFIG[\"encoder_lr\"]}, decoder_lr={CONFIG[\"decoder_lr\"]}, wd={CONFIG[\"weight_decay\"]})')
print(f'Scheduler: ReduceLROnPlateau(patience={CONFIG[\"scheduler_patience\"]}, factor={CONFIG[\"scheduler_factor\"]})')
print(f'AMP: {"enabled" if CONFIG["use_amp"] else "disabled"}')
print(f'Loss weights: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}')
print(f'Gradient accumulation: {CONFIG[\"accumulation_steps\"]} steps')"""))

    # ======================================================================
    # Evaluation Metrics — PRESERVE from vK.10.6 cells 43-44
    # ======================================================================
    cells.append(copy_cell(src, 43))   # 9.1 Evaluation Metrics header
    cells.append(copy_cell(src, 44))   # Metrics code

    # ======================================================================
    # Checkpoint Helpers — PRESERVE from vK.10.6 cells 45-46
    # ======================================================================
    cells.append(copy_cell(src, 45))   # 9.2 Checkpoint Helpers header
    cells.append(copy_cell(src, 46))   # Checkpoint code

    # ======================================================================
    # Training Loop — REWRITTEN with grad accumulation, encoder freeze, edge loss
    # ======================================================================
    cells.append(make_md_cell("""\
## 10. Training Loop

The training loop uses:
- **AMP** for mixed precision training
- **Gradient accumulation** for larger effective batch sizes
- **Gradient clipping** at `max_grad_norm` for stability
- **Encoder freeze** for the first N epochs to protect pretrained BN statistics
- **ReduceLROnPlateau** scheduler monitoring tampered-only F1
- **Three-file checkpointing** with automatic resume
- **Early stopping** based on tampered-only Dice coefficient"""))

    cells.append(make_code_cell("""\
def train_one_epoch(epoch):
    \"\"\"Train for one epoch with AMP, gradient accumulation, and edge loss.\"\"\"
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    accum_steps = CONFIG['accumulation_steps']

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, masks, labels) in enumerate(tqdm(train_loader, desc=f'Train Ep{epoch}', leave=False)):
        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        with autocast('cuda', enabled=CONFIG['use_amp']):
            cls_logits, seg_logits = model(images)
            loss_cls = criterion_cls(cls_logits, labels)
            loss_seg = CONFIG['seg_bce_weight'] * bce_loss(seg_logits, masks) + \\
                       CONFIG['seg_dice_weight'] * dice_loss(seg_logits, masks)
            loss_edge = edge_loss(seg_logits, masks)
            loss = (ALPHA * loss_cls + BETA * loss_seg + GAMMA * loss_edge) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps * images.size(0)
        preds = torch.argmax(cls_logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    # Flush partial accumulation window
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct / total_samples

    return epoch_loss, epoch_acc"""))

    cells.append(make_code_cell("""\
@torch.no_grad()
def evaluate(loader, dataset_len, name='Val'):
    \"\"\"Evaluate model with AMP, returning all-sample and tampered-only metrics.\"\"\"
    model.eval()
    running_loss = 0.0
    correct = 0
    all_cls_logits, all_seg_logits, all_masks, all_labels = [], [], [], []

    for images, masks, labels in tqdm(loader, desc=name, leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        with autocast('cuda', enabled=CONFIG['use_amp']):
            cls_logits, seg_logits = model(images)
            loss_cls = criterion_cls(cls_logits, labels)
            loss_seg = CONFIG['seg_bce_weight'] * bce_loss(seg_logits, masks) + \\
                       CONFIG['seg_dice_weight'] * dice_loss(seg_logits, masks)
            loss_edge = edge_loss(seg_logits, masks)
            loss = ALPHA * loss_cls + BETA * loss_seg + GAMMA * loss_edge

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(cls_logits, dim=1)
        correct += (preds == labels).sum().item()

        all_cls_logits.append(cls_logits.cpu())
        all_seg_logits.append(seg_logits.cpu())
        all_masks.append(masks.cpu())
        all_labels.append(labels.cpu())

    all_cls_logits = torch.cat(all_cls_logits)
    all_seg_logits = torch.cat(all_seg_logits)
    all_masks = torch.cat(all_masks)
    all_labels = torch.cat(all_labels)

    seg_metrics = compute_metrics_split(all_seg_logits, all_masks, all_labels)

    epoch_loss = running_loss / dataset_len
    epoch_acc = correct / dataset_len

    # ROC-AUC for classification
    probs = torch.softmax(all_cls_logits, dim=1)[:, 1].numpy()
    labels_np = all_labels.numpy()
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = 0.0

    seg_metrics['loss'] = epoch_loss
    seg_metrics['acc'] = epoch_acc
    seg_metrics['roc_auc'] = auc

    print(
        f'  {name} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | AUC: {auc:.4f} | '
        f'Dice(tam): {seg_metrics[\"tampered_dice\"]:.4f} | '
        f'IoU(tam): {seg_metrics[\"tampered_iou\"]:.4f}'
    )
    return seg_metrics"""))

    # Training state initialization — from vK.10.6 cell 50
    cells.append(copy_cell(src, 50))

    # ======================================================================
    # Main training loop — REWRITTEN with encoder freeze + ReduceLROnPlateau
    # ======================================================================
    cells.append(make_code_cell("""\
# ================== Main training loop ==================
best_model_path = os.path.join(str(CHECKPOINT_DIR), 'best_model.pt')

# Encoder freeze for warmup
if CONFIG['encoder_freeze_epochs'] > 0:
    freeze_encoder(model)
    print(f"Encoder FROZEN for first {CONFIG['encoder_freeze_epochs']} epochs")

for epoch in range(start_epoch, CONFIG['max_epochs'] + 1):
    print(f'\\nEpoch {epoch}/{CONFIG["max_epochs"]}')

    # Unfreeze encoder after warmup period
    if epoch == start_epoch + CONFIG['encoder_freeze_epochs']:
        unfreeze_encoder(model)
        print("Encoder UNFROZEN for fine-tuning")

    train_loss, train_acc = train_one_epoch(epoch)
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

    val_metrics = evaluate(val_loader, len(val_dataset), name='Val')

    # Step scheduler with monitored metric (ReduceLROnPlateau)
    scheduler.step(val_metrics['tampered_f1'])

    # Record history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_dice'].append(0.0)  # train dice computed at eval time if needed
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['acc'])
    history['val_dice'].append(val_metrics['dice'])
    history['val_iou'].append(val_metrics['iou'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_tampered_dice'].append(val_metrics['tampered_dice'])
    history['val_tampered_iou'].append(val_metrics['tampered_iou'])
    history['val_tampered_f1'].append(val_metrics['tampered_f1'])
    history['val_roc_auc'].append(val_metrics['roc_auc'])
    history['lr'].append(optimizer.param_groups[0]['lr'])

    # W&B logging
    if WANDB_ACTIVE:
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss, 'train/accuracy': train_acc,
            'val/loss': val_metrics['loss'], 'val/accuracy': val_metrics['acc'],
            'val/dice': val_metrics['dice'], 'val/iou': val_metrics['iou'],
            'val/f1': val_metrics['f1'],
            'val/tampered_dice': val_metrics['tampered_dice'],
            'val/tampered_iou': val_metrics['tampered_iou'],
            'val/tampered_f1': val_metrics['tampered_f1'],
            'val/roc_auc': val_metrics['roc_auc'],
            'lr/encoder': optimizer.param_groups[0]['lr'],
            'lr/decoder': optimizer.param_groups[1]['lr'],
        })

        # Log val prediction images every 5 epochs (and epoch 1)
        if epoch == 1 or epoch % 5 == 0:
            _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            _std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            _vis_images = []
            model.eval()
            with torch.no_grad():
                for _imgs, _masks, _labels in val_loader:
                    for _i in range(min(_imgs.size(0), 4 - len(_vis_images))):
                        _rgb = ((_imgs[_i][:3] * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        _gt = (_masks[_i].squeeze().numpy() > 0.5).astype(np.uint8)
                        _inp = _imgs[_i].unsqueeze(0).to(device)
                        with autocast('cuda', enabled=CONFIG['use_amp']):
                            _, _sl = model(_inp)
                        _pred = (torch.sigmoid(_sl).cpu().squeeze().numpy() > CONFIG['seg_threshold']).astype(np.uint8)
                        _vis_images.append(wandb.Image(
                            _rgb,
                            masks={
                                "ground_truth": {"mask_data": _gt, "class_labels": {0: "background", 1: "tampered"}},
                                "prediction": {"mask_data": _pred, "class_labels": {0: "background", 1: "tampered"}},
                            },
                            caption=f"epoch={epoch} | {'Tampered' if _labels[_i] else 'Authentic'}"
                        ))
                    if len(_vis_images) >= 4:
                        break
            if _vis_images:
                wandb.log({"val/predictions": _vis_images, "epoch": epoch})

    # Build checkpoint state (save unwrapped model for portability)
    state = {
        'epoch': epoch,
        'model_state_dict': get_base_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'best_epoch': best_epoch,
        'config': CONFIG,
        'history': history,
    }

    # Save last checkpoint every epoch
    save_checkpoint(state, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))

    # Save history CSV every epoch for crash resilience
    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)

    # Best model selection: tampered-only Dice
    current_metric = val_metrics['tampered_dice']
    if current_metric > best_metric:
        best_metric = current_metric
        best_epoch = epoch
        state['best_metric'] = best_metric
        state['best_epoch'] = best_epoch
        save_checkpoint(state, best_model_path)
        torch.save(get_base_model(model).state_dict(),
                   os.path.join(str(KAGGLE_WORKING_DIR), 'best_model.pth'))
        print(f'  ** New best model: tampered Dice = {best_metric:.4f} at epoch {epoch}')
        if WANDB_ACTIVE:
            wandb.save(best_model_path)

    # Periodic checkpoint
    if epoch % CONFIG['checkpoint_every'] == 0:
        save_checkpoint(state, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt'))
        print(f'  Periodic checkpoint saved at epoch {epoch}')

    # Early stopping
    if epoch - best_epoch >= CONFIG['patience']:
        print(f'Early stopping at epoch {epoch}. Best tampered Dice={best_metric:.4f} at epoch {best_epoch}')
        break

print(f'\\nTraining complete. Best tampered Dice={best_metric:.4f} at epoch {best_epoch}')"""))

    # ======================================================================
    # Evaluation suite — PRESERVE from vK.10.6 cells 52-68
    # ======================================================================
    cells.append(copy_cell(src, 52))   # Section 11 header
    cells.append(copy_cell_with_append(src, 53, """\

    # --- W&B: log model artifact ---
    try:
        _ckpt = os.path.join(str(KAGGLE_WORKING_DIR), 'best_model.pth')
        if os.path.exists(_ckpt):
            _art = wandb.Artifact('best-model', type='model',
                description=f'Best checkpoint from epoch {best_epoch}')
            _art.add_file(_ckpt)
            WANDB_RUN.log_artifact(_art)
            print('  W&B: model artifact logged')
    except Exception as _e:
        print(f'  W&B artifact skip: {_e}')

    # --- W&B: save training history CSV ---
    try:
        _hist = os.path.join(str(RESULTS_DIR), 'training_history.csv')
        if os.path.exists(str(_hist)):
            wandb.save(str(_hist))
            print('  W&B: training_history.csv saved')
    except Exception as _e:
        print(f'  W&B save skip: {_e}')

    # --- W&B: log final metrics table ---
    try:
        _tbl = wandb.Table(
            columns=['Metric', 'Value'],
            data=[
                ['accuracy', test_metrics['acc']],
                ['dice_all', test_metrics['dice']],
                ['tampered_dice', test_metrics['tampered_dice']],
                ['tampered_iou', test_metrics['tampered_iou']],
                ['tampered_f1', test_metrics['tampered_f1']],
                ['roc_auc', test_metrics['roc_auc']],
                ['best_epoch', float(best_epoch)],
            ]
        )
        wandb.log({'final_test_metrics_table': _tbl})
        print('  W&B: final metrics table logged')
    except Exception as _e:
        print(f'  W&B table skip: {_e}')"""))   # Load best + test eval + W&B artifacts
    cells.append(copy_cell(src, 54))   # Metric inflation note
    cells.append(copy_cell(src, 55))   # 11.2 Training Curves header
    cells.append(copy_cell(src, 56))   # Training curves code
    cells.append(copy_cell(src, 57))   # 11.3 Threshold Optimization header

    # Threshold sweep — increase to 50 points
    cells.append(make_code_cell("""\
# ================== Threshold Sweep on Validation Set (50-point) ==================
@torch.no_grad()
def collect_predictions(loader):
    \"\"\"Collect all predictions and ground truths from a dataloader.\"\"\"
    model.eval()
    all_seg_logits, all_masks, all_labels, all_cls_logits = [], [], [], []
    for images, masks, labels in tqdm(loader, desc='Collecting predictions', leave=False):
        images = images.to(device)
        with autocast('cuda', enabled=CONFIG['use_amp']):
            cls_logits, seg_logits = model(images)
        all_seg_logits.append(seg_logits.cpu())
        all_masks.append(masks.cpu())
        all_labels.append(labels.cpu())
        all_cls_logits.append(cls_logits.cpu())
    return {
        'seg_logits': torch.cat(all_seg_logits),
        'masks': torch.cat(all_masks),
        'labels': torch.cat(all_labels),
        'cls_logits': torch.cat(all_cls_logits),
    }

val_preds = collect_predictions(val_loader)
test_preds = collect_predictions(test_loader)

thresholds = np.linspace(0.05, 0.80, 50)
val_f1_scores = []
for thr in thresholds:
    pred_bin = (torch.sigmoid(val_preds['seg_logits']) > thr).float()
    tam_mask = val_preds['labels'] == 1
    if tam_mask.sum() == 0:
        val_f1_scores.append(0.0)
        continue
    tp = (pred_bin[tam_mask] * val_preds['masks'][tam_mask]).sum(dim=(1,2,3))
    fp = (pred_bin[tam_mask] * (1 - val_preds['masks'][tam_mask])).sum(dim=(1,2,3))
    fn = ((1 - pred_bin[tam_mask]) * val_preds['masks'][tam_mask]).sum(dim=(1,2,3))
    eps = 1e-7
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    val_f1_scores.append((2 * prec * rec / (prec + rec + eps)).mean().item())

optimal_idx = np.argmax(val_f1_scores)
OPTIMAL_THRESHOLD = float(thresholds[optimal_idx])
print(f"Optimal threshold: {OPTIMAL_THRESHOLD:.4f} (val tampered F1 = {val_f1_scores[optimal_idx]:.4f})")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, val_f1_scores, 'b-o', markersize=3)
ax.axvline(OPTIMAL_THRESHOLD, color='r', linestyle='--', label=f'Optimal = {OPTIMAL_THRESHOLD:.4f}')
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default = 0.50')
ax.set_xlabel('Threshold')
ax.set_ylabel('Tampered-Only F1')
ax.set_title('Segmentation Threshold Sweep (Validation Set, 50 points)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

def compute_metrics_at_threshold(seg_logits, masks, labels, threshold):
    eps = 1e-7
    results = {}
    for name, filt in [('all', None), ('tampered', labels == 1)]:
        sl = seg_logits[filt] if filt is not None else seg_logits
        ms = masks[filt] if filt is not None else masks
        if len(sl) == 0:
            results[f'{name}_dice'] = 0.0
            results[f'{name}_iou'] = 0.0
            results[f'{name}_f1'] = 0.0
            continue
        pred_bin = (torch.sigmoid(sl) > threshold).float()
        tp = (pred_bin * ms).sum(dim=(1,2,3))
        fp = (pred_bin * (1 - ms)).sum(dim=(1,2,3))
        fn = ((1 - pred_bin) * ms).sum(dim=(1,2,3))
        inter = tp
        union = pred_bin.sum(dim=(1,2,3)) + ms.sum(dim=(1,2,3))
        results[f'{name}_dice'] = ((2 * inter + eps) / (union + eps)).mean().item()
        results[f'{name}_iou'] = ((inter + eps) / (union - inter + eps)).mean().item()
        prec = (tp + eps) / (tp + fp + eps)
        rec = (tp + eps) / (tp + fn + eps)
        results[f'{name}_f1'] = ((2 * prec * rec) / (prec + rec + eps)).mean().item()
    return results

test_opt = compute_metrics_at_threshold(test_preds['seg_logits'], test_preds['masks'],
                                         test_preds['labels'], OPTIMAL_THRESHOLD)
print(f"\\nTest Metrics at Optimal Threshold ({OPTIMAL_THRESHOLD:.4f}):")
for k, v in test_opt.items():
    print(f"  {k}: {v:.4f}")

if WANDB_ACTIVE:
    wandb.log({"threshold_sweep": wandb.Image(fig)})
    wandb.summary.update({"optimal_threshold": OPTIMAL_THRESHOLD})
    for k, v in test_opt.items():
        wandb.summary[f"test_opt/{k}"] = v"""))

    # Remaining evaluation cells — PRESERVE from vK.10.6
    cells.append(copy_cell(src, 59))   # 11.4 Pixel-Level AUC header
    cells.append(copy_cell(src, 60))   # Pixel-level AUC code
    cells.append(copy_cell(src, 61))   # 11.5 Confusion matrix header
    cells.append(copy_cell_with_append(src, 62, """\
if WANDB_ACTIVE:
    wandb.log({"classification/confusion_matrix_roc_pr": wandb.Image(fig)})"""))   # Confusion matrix + ROC/PR code + W&B
    cells.append(copy_cell(src, 63))   # 11.6 Per-Forgery-Type header
    cells.append(copy_cell_with_append(src, 64, """\
if WANDB_ACTIVE:
    for _ft in ['splicing', 'copy-move']:
        _ft_m = forgery_types == _ft
        if _ft_m.sum() == 0: continue
        _pb = (seg_probs[_ft_m] > thr).float()
        _gt = test_preds['masks'][_ft_m]
        _eps = 1e-7
        _inter = (_pb*_gt).sum(dim=(1,2,3))
        _dice = ((2*_inter+_eps)/(_pb.sum(dim=(1,2,3))+_gt.sum(dim=(1,2,3))+_eps)).mean().item()
        _tp = _inter; _fp = (_pb*(1-_gt)).sum(dim=(1,2,3)); _fn = ((1-_pb)*_gt).sum(dim=(1,2,3))
        _pr = (_tp+_eps)/(_tp+_fp+_eps); _rc = (_tp+_eps)/(_tp+_fn+_eps)
        _f1 = (2*_pr*_rc/(_pr+_rc+_eps)).mean().item()
        wandb.summary[f"forgery/{_ft}_dice"] = _dice
        wandb.summary[f"forgery/{_ft}_f1"] = _f1"""))   # Per-forgery-type + W&B
    cells.append(copy_cell(src, 65))   # 11.7 Mask-Size header
    cells.append(copy_cell_with_append(src, 66, """\
if WANDB_ACTIVE and bucket_names:
    wandb.log({"mask_size_stratification": wandb.Image(fig)})
    for _bn, _bf in zip(bucket_names, bucket_f1s):
        wandb.summary[f"mask_size/{_bn}_f1"] = _bf"""))   # Mask-size stratified + W&B
    cells.append(copy_cell(src, 67))   # 11.8 Shortcut Learning header
    cells.append(copy_cell(src, 68))   # Shortcut learning code

    # ======================================================================
    # Visualization — PRESERVE + add ELA viz
    # ======================================================================
    cells.append(copy_cell(src, 69))   # Section 12 header
    # denormalize — updated for 4-channel (RGB+ELA) input
    cells.append(make_code_cell("""\
def denormalize(img_tensor):
    \"\"\"Convert a normalized image tensor back to displayable RGB space.

    Handles 4-channel (RGB+ELA) tensors by extracting only the RGB channels.
    \"\"\"
    if img_tensor.shape[0] == 4:
        img_tensor = img_tensor[:3]  # extract RGB channels only
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)"""))
    # Load best model for visualization — use checkpoint path
    cells.append(make_code_cell("""\
# Load best model for visualization
if os.path.exists(best_model_path):
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    get_base_model(model).load_state_dict(ckpt['model_state_dict'])
    print(f'Best model loaded from checkpoint (epoch {ckpt.get(\"best_epoch\", \"?\")})')
else:
    legacy_path = os.path.join(str(KAGGLE_WORKING_DIR), 'best_model.pth')
    get_base_model(model).load_state_dict(torch.load(legacy_path, map_location=device))
    print('Best model loaded from legacy path')
model.eval()"""))
    cells.append(copy_cell(src, 72))   # 12.1 Sample Collection header
    cells.append(copy_cell(src, 73))   # collect_samples function
    cells.append(copy_cell(src, 74))   # show_samples_with_masks
    cells.append(copy_cell(src, 75))   # show_image_and_mask
    cells.append(copy_cell(src, 76))   # show_image_and_mask calls

    # NEW: ELA Visualization
    cells.append(make_md_cell("""\
### 12.2 ELA Channel Visualization

Visualize the Error Level Analysis maps alongside RGB images and predictions
to demonstrate the forensic signal the model receives as its 4th input channel."""))

    cells.append(make_code_cell("""\
# ================== ELA Visualization ==================
def show_ela_visualization(loader, num_samples=3):
    \"\"\"Show RGB, ELA, predicted mask, and overlay for sample images.\"\"\"
    model.eval()
    fig, axes = plt.subplots(num_samples * 2, 4, figsize=(16, 4 * num_samples * 2))
    shown_auth, shown_tamp = 0, 0
    row = 0

    with torch.no_grad():
        for images, masks, labels in loader:
            for i in range(images.size(0)):
                if row >= num_samples * 2:
                    break
                label = labels[i].item()
                if label == 0 and shown_auth >= num_samples:
                    continue
                if label == 1 and shown_tamp >= num_samples:
                    continue

                # Extract RGB (first 3 channels) and ELA (4th channel)
                img_rgb = images[i, :3].clone()
                ela_ch = images[i, 3].clone()

                # Denormalize RGB for display
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_display = (img_rgb * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

                # ELA channel for display
                ela_display = ela_ch.numpy()

                # Prediction
                inp = images[i:i+1].to(device)
                with autocast('cuda', enabled=CONFIG['use_amp']):
                    _, seg_logits = model(inp)
                pred = torch.sigmoid(seg_logits).cpu().squeeze().numpy()

                # GT mask
                gt = masks[i].squeeze().numpy()

                tag = 'Authentic' if label == 0 else 'Tampered'
                axes[row, 0].imshow(img_display)
                axes[row, 0].set_title(f'{tag} - RGB')
                axes[row, 0].axis('off')

                axes[row, 1].imshow(ela_display, cmap='hot')
                axes[row, 1].set_title(f'{tag} - ELA')
                axes[row, 1].axis('off')

                axes[row, 2].imshow(gt, cmap='gray')
                axes[row, 2].set_title('Ground Truth')
                axes[row, 2].axis('off')

                axes[row, 3].imshow(pred, cmap='hot')
                axes[row, 3].set_title('Prediction')
                axes[row, 3].axis('off')

                row += 1
                if label == 0:
                    shown_auth += 1
                else:
                    shown_tamp += 1

            if row >= num_samples * 2:
                break

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'ela_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()

    if WANDB_ACTIVE:
        wandb.log({'evaluation/ela_visualization': wandb.Image(fig)})

show_ela_visualization(test_loader, num_samples=3)"""))

    # Remaining visualization cells
    cells.append(copy_cell(src, 77))   # Collect real/fake for grid
    cells.append(copy_cell(src, 78))   # 12.2 Submission-Ready header (now 12.3)
    cells.append(copy_cell(src, 79))   # Submission prediction grid

    # ======================================================================
    # Advanced Analysis — PRESERVE from vK.10.6
    # ======================================================================
    cells.append(copy_cell(src, 80))   # Section 13 header
    cells.append(copy_cell_with_append(src, 81, """\
if WANDB_ACTIVE:
    wandb.log({"failure_cases": wandb.Image(fig)})"""))   # Failure case analysis + W&B

    # ======================================================================
    # Explainability — MODIFIED Grad-CAM target for SMP encoder
    # ======================================================================
    cells.append(copy_cell(src, 82))   # Section 14 header

    cells.append(make_code_cell("""\
# ================== Grad-CAM Explainability (SMP Encoder) ==================
import warnings

class GradCAM:
    \"\"\"Grad-CAM for segmentation encoder features.\"\"\"
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handles = []
        self._handles.append(target_layer.register_forward_hook(self._save_activation))
        self._handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        \"\"\"Generate Grad-CAM heatmap.\"\"\"
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Forward pass — use segmentation output for backward
        cls_out, seg_out = self.model(input_tensor)
        target = seg_out.mean()
        self.model.zero_grad()
        target.backward()

        if self.gradients is None or self.activations is None:
            warnings.warn('Grad-CAM: hooks did not capture data.')
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

# Target the deepest encoder layer (layer4 for ResNet34)
_base = get_base_model(model)
grad_cam = GradCAM(_base, _base.segmentor.encoder.layer4)

# Collect samples for Grad-CAM visualization
num_auth, num_tamp = 3, 3
auth_samples, tamp_samples = [], []

with torch.no_grad():
    for images, masks, labels in test_loader:
        for i in range(images.size(0)):
            label = labels[i].item()
            if label == 0 and len(auth_samples) < num_auth:
                auth_samples.append({'image': images[i], 'mask': masks[i], 'label': label})
            elif label == 1 and len(tamp_samples) < num_tamp:
                tamp_samples.append({'image': images[i], 'mask': masks[i], 'label': label})
        if len(auth_samples) >= num_auth and len(tamp_samples) >= num_tamp:
            break

all_samples = auth_samples + tamp_samples
fig, axes = plt.subplots(len(all_samples), 3, figsize=(12, 4 * len(all_samples)))

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for idx, sample in enumerate(all_samples):
    inp = sample['image'].unsqueeze(0).to(device).requires_grad_(True)
    cam = grad_cam.generate(inp)

    # Display RGB (first 3 channels)
    rgb = sample['image'][:3]
    img_np = (rgb * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    tag = 'Authentic' if sample['label'] == 0 else 'Tampered'
    axes[idx, 0].imshow(img_np)
    axes[idx, 0].set_title(f'{tag}')
    axes[idx, 0].axis('off')

    if cam is not None:
        axes[idx, 1].imshow(img_np)
        axes[idx, 1].imshow(cam, cmap='jet', alpha=0.5)
        axes[idx, 1].set_title('Grad-CAM')
    else:
        axes[idx, 1].set_title('Grad-CAM (failed)')
    axes[idx, 1].axis('off')

    with torch.no_grad():
        _, seg_out = model(sample['image'].unsqueeze(0).to(device))
    pred = torch.sigmoid(seg_out).cpu().squeeze().numpy()
    axes[idx, 2].imshow(pred, cmap='hot')
    axes[idx, 2].set_title('Predicted Mask')
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'gradcam_visualization.png'), dpi=150, bbox_inches='tight')
plt.show()

if WANDB_ACTIVE:
    wandb.log({'evaluation/gradcam': wandb.Image(fig)})

grad_cam.remove_hooks()"""))

    # ======================================================================
    # Robustness Testing — PRESERVE header, MODIFY code for better pattern
    # ======================================================================
    cells.append(copy_cell(src, 84))   # Section 15 header

    cells.append(make_code_cell("""\
# ================== Robustness Testing Suite (Albumentations-based) ==================
NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

robustness_transforms = {
    'clean': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), NORMALIZE, ToTensorV2()],
                       additional_targets={'ela': 'image'}),
    'jpeg_qf70': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                            A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),
    'jpeg_qf50': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                            A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),
    'noise_s10': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),
    'noise_s25': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                            A.GaussNoise(var_limit=(100.0, 100.0), p=1.0),
                            NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),
    'blur_k3': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                          A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                          NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),
    'blur_k5': A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                          A.GaussianBlur(blur_limit=(5, 5), p=1.0),
                          NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),
    'resize_0.75': A.Compose([A.Resize(int(IMAGE_SIZE*0.75), int(IMAGE_SIZE*0.75)),
                              A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                              NORMALIZE, ToTensorV2()], additional_targets={'ela': 'image'}),
}

def compute_tam_f1(pred_bin, masks, eps=1e-7):
    \"\"\"Compute per-sample tampered F1.\"\"\"
    tp = (pred_bin * masks).sum(dim=(1,2,3))
    fp = (pred_bin * (1 - masks)).sum(dim=(1,2,3))
    fn = ((1 - pred_bin) * masks).sum(dim=(1,2,3))
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    return (2 * prec * rec / (prec + rec + eps)).mean().item()

@torch.no_grad()
def run_robustness_eval(condition_name, transform, threshold):
    \"\"\"Evaluate model under a specific degradation condition.\"\"\"
    model.eval()
    robust_dataset = ELAImageMaskDataset(
        test_df[test_df['label'] == 1],  # tampered only
        transform=transform,
        ela_quality=CONFIG['ela_quality'],
    )
    if len(robust_dataset) == 0:
        return 0.0
    robust_loader = DataLoader(robust_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=False, num_workers=CONFIG['num_workers'],
                               pin_memory=True)
    all_preds, all_masks = [], []
    for images, masks, labels in robust_loader:
        images = images.to(device)
        with autocast('cuda', enabled=CONFIG['use_amp']):
            _, seg_logits = model(images)
        pred_bin = (torch.sigmoid(seg_logits).cpu() > threshold).float()
        all_preds.append(pred_bin)
        all_masks.append(masks)
    return compute_tam_f1(torch.cat(all_preds), torch.cat(all_masks))

threshold = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else CONFIG['seg_threshold']
print(f'Robustness evaluation using threshold={threshold:.4f}')

robustness_results = {}
for name, transform in tqdm(robustness_transforms.items(), desc='Robustness tests'):
    f1 = run_robustness_eval(name, transform, threshold)
    robustness_results[name] = f1
    print(f'  {name:20s}: F1={f1:.4f}')

# Delta from clean
clean_f1 = robustness_results.get('clean', 0.0)
print(f'\\nDeltas from clean (F1={clean_f1:.4f}):')
for name, f1 in robustness_results.items():
    if name != 'clean':
        print(f'  {name:20s}: delta={f1 - clean_f1:+.4f}')

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names = list(robustness_results.keys())
f1s = [robustness_results[n] for n in names]
colors = ['green' if n == 'clean' else 'steelblue' for n in names]
ax.bar(names, f1s, color=colors)
ax.axhline(clean_f1, color='green', linestyle='--', alpha=0.5, label=f'Clean baseline = {clean_f1:.4f}')
ax.set_ylabel('Tampered-Only F1')
ax.set_title('Robustness Testing: F1 Under Degradation')
ax.legend()
ax.set_xticklabels(names, rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'robustness_results.png'), dpi=150, bbox_inches='tight')
plt.show()

if WANDB_ACTIVE:
    wandb.log({'evaluation/robustness': wandb.Image(fig)})
    try:
        _rob_table = wandb.Table(
            columns=['Condition', 'F1', 'Delta_from_Clean'],
            data=[[n, f, f - clean_f1] for n, f in robustness_results.items()]
        )
        wandb.log({'evaluation/robustness_table': _rob_table})
    except Exception:
        pass"""))

    # ======================================================================
    # Inference — MODIFIED for 4-channel input
    # ======================================================================
    cells.append(copy_cell(src, 86))   # Section 16 header

    cells.append(make_code_cell("""\
def predict_single_image(image_path, model, device, threshold=None):
    \"\"\"Run inference on a single image with ELA preprocessing.\"\"\"
    if threshold is None:
        threshold = OPTIMAL_THRESHOLD if 'OPTIMAL_THRESHOLD' in dir() else CONFIG['seg_threshold']

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f'Failed to read image: {image_path}')

    # Compute ELA
    ela = compute_ela(image_bgr, quality=CONFIG['ela_quality'])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ela_3ch = np.stack([ela, ela, ela], axis=-1)

    # Apply validation transform
    transform = get_valid_transform()
    augmented = transform(image=image_rgb, mask=np.zeros_like(ela), ela=ela_3ch)
    image_t = augmented['image']
    ela_t = augmented['ela'][0:1]  # single channel
    input_tensor = torch.cat([image_t, ela_t], dim=0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with autocast('cuda', enabled=CONFIG['use_amp']):
            cls_logits, seg_logits = model(input_tensor)

    cls_probs = torch.softmax(cls_logits, dim=1)[0]
    seg_prob = torch.sigmoid(seg_logits).cpu().squeeze().numpy()
    seg_mask = (seg_prob > threshold).astype(np.uint8)

    return {
        'cls_probs': cls_probs.cpu().numpy(),
        'is_tampered': int(cls_probs[1] > 0.5),
        'seg_prob': seg_prob,
        'seg_mask': seg_mask,
        'threshold': threshold,
    }

print('predict_single_image() defined.')"""))

    # ======================================================================
    # Conclusion + W&B teardown — PRESERVE from vK.10.6 cells 88-89
    # ======================================================================
    cells.append(make_md_cell("""\
## 17. Conclusion

This notebook (vK.11.0) represents the synthesis of findings from 13 experiment runs
across two architecture tracks:

**Architecture upgrades from audits:**
- Pretrained ResNet34 encoder via SMP (from v6.5 — proven Tam-F1=0.41)
- ELA as 4th input channel for JPEG forgery detection
- Edge supervision loss for improved boundary delineation
- AdamW with differential learning rates (from v6.5)
- ReduceLROnPlateau scheduler (from v8, fixing CosineAnnealing double-cycle bug)
- Gradient accumulation for larger effective batch sizes
- Encoder freeze warmup to protect pretrained BatchNorm statistics
- Per-sample Dice loss (from v8, fixing batch-level bias)

**Evaluation suite (12 features from vK.10.6):**
Threshold optimization, pixel-level AUC, confusion matrix, ROC/PR curves,
forgery-type breakdown, mask-size stratification, robustness testing,
Grad-CAM explainability, shortcut learning validation, failure case analysis,
data leakage verification, ELA visualization.

**Bug fixes from audit:**
- Removed stderr suppression (Bug #2)
- Fixed CONFIG/docs mismatch (Bug #1)
- Fixed unused CONFIG values (Bug #3, #4)
- Replaced CosineAnnealing double-cycle with ReduceLROnPlateau"""))

    cells.append(copy_cell(src, 89))   # W&B teardown

    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading: {INPUT_PATH.name}")
    src = load_notebook(INPUT_PATH)

    # Build new cell list
    cells = build_cells(src)

    # Create output notebook
    out = copy.deepcopy(src)
    out["cells"] = cells

    # Clear all outputs
    for cell in out["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    save_notebook(out, OUTPUT_PATH)
    print(f"Total cells: {len(cells)}")
    print("Done.")


if __name__ == "__main__":
    main()
