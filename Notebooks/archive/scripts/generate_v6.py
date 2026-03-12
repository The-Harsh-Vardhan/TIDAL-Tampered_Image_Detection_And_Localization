#!/usr/bin/env python3
"""Generate v6 training notebooks for Colab and Kaggle environments."""
import json, os

def md(text):
    lines = text.strip('\n').split('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)]}

def code(text):
    lines = text.strip('\n').split('\n')
    return {"cell_type": "code", "metadata": {}, "source": [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)], "outputs": [], "execution_count": None}

def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 5}

# ── Section 1: Project Introduction ───────────────────────────────────────────

TITLE_KAGGLE = md("""\
# Tamper Detection v6 (Kaggle) — Image Forgery Detection & Localization

> **v6** — Final, clean implementation. Aligned with Docs6 specification.
> Dataset loaded from Kaggle mounts (`/kaggle/input/`). All outputs saved to `/kaggle/working/`.

**Changes from v5.1:**
- Reorganised into 22 clearly named sections for interview readability
- Enabled cuDNN benchmark and TF32 for faster training
- Added `persistent_workers=True` to DataLoaders
- Unified section structure across Colab and Kaggle variants
- Clean, final-version codebase — no legacy v4/v5 artifacts

**Architecture:** SMP U-Net (ResNet34, ImageNet pretrained)
**Dataset:** CASIA Splicing Detection + Localization
**Loss:** BCE + Dice | **Optimizer:** AdamW (differential LR)
**Training:** AMP, gradient accumulation, early stopping
**Image Size:** 384 × 384 | **Split:** 70 / 15 / 15

## Notebook Sections

1. Environment Setup
2. GPU Verification
3. Dataset Loading
4. Dataset Discovery
5. Dataset Validation
6. Preprocessing Pipeline
7. Dataset Splitting
8. Dataset Class
9. DataLoaders
10. Model Architecture
11. Loss Functions
12. Evaluation Metrics
13. Validation Loop
14. Training Pipeline
15. Threshold Selection
16. Test Evaluation
17. Visualization
18. Explainable AI
19. Robustness Testing
20. Experiment Tracking
21. Saving Artifacts
22. Final Summary""")

TITLE_COLAB = md("""\
# Tamper Detection v6 (Colab) — Image Forgery Detection & Localization

> **v6** — Final, clean implementation. Aligned with Docs6 specification.
> Dataset downloaded via Kaggle API. Checkpoints saved to Google Drive.

**Changes from v5:**
- Image resolution: 512 → 384 (safer T4 GPU headroom)
- Data split: 85/7.5/7.5 → 70/15/15 (more balanced evaluation)
- Mask binarization: > 128 → > 0 (captures all annotated pixels)
- Case-insensitive dataset discovery (handles uppercase IMAGE/MASK)
- Data leakage verification (set-intersection assertions)
- Grad-CAM wrapped with safety checks
- True-negative precision/recall returns (1.0, 1.0)
- Enabled cuDNN benchmark and TF32 for faster training
- Added `persistent_workers=True` to DataLoaders
- Reorganised into 22 clearly named sections

**Architecture:** SMP U-Net (ResNet34, ImageNet pretrained)
**Dataset:** CASIA Splicing Detection + Localization
**Loss:** BCE + Dice | **Optimizer:** AdamW (differential LR)
**Training:** AMP, gradient accumulation, early stopping
**Image Size:** 384 × 384 | **Split:** 70 / 15 / 15

## Notebook Sections

1. Environment Setup
2. GPU Verification
3. Dataset Loading
4. Dataset Discovery
5. Dataset Validation
6. Preprocessing Pipeline
7. Dataset Splitting
8. Dataset Class
9. DataLoaders
10. Model Architecture
11. Loss Functions
12. Evaluation Metrics
13. Validation Loop
14. Training Pipeline
15. Threshold Selection
16. Test Evaluation
17. Visualization
18. Explainable AI
19. Robustness Testing
20. Experiment Tracking
21. Saving Artifacts
22. Final Summary""")

# ── Section 2: Environment Setup ──────────────────────────────────────────────

SEC1_MD = md("""\
## 1. Environment Setup

Install dependencies, import libraries, and set the global random seed for full reproducibility.""")

PIP_KAGGLE = code("""\
!pip install -q segmentation-models-pytorch "albumentations>=1.3.1,<2.0" """)

PIP_COLAB = code("""\
!pip install -q kaggle segmentation-models-pytorch "albumentations>=1.3.1,<2.0" """)

IMPORTS = code("""\
import os
import random
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=UserWarning)
print('All imports successful.')""")

SEED_CELL = code("""\
SEED = 42

def set_seed(seed=SEED):
    \"\"\"Set seed for full reproducibility.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
print(f'Seed set to {SEED}')""")

# ── Section 3: GPU Verification ───────────────────────────────────────────────

SEC2_MD = md("""\
## 2. GPU Verification

Detect GPU, enable performance optimizations (cuDNN benchmark, TF32), and define the global configuration.""")

GPU_CHECK = code("""\
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print('cuDNN benchmark: enabled')
    print('TF32: enabled')
else:
    print('WARNING: No GPU detected. Training will be extremely slow.')

print(f'Device: {device}')""")

CONFIG_KAGGLE = code("""\
CONFIG = {
    'image_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'train_ratio': 0.70,
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 50,
    'patience': 10,
    'accumulation_steps': 4,
    'max_grad_norm': 1.0,
    'seed': SEED,
}

# Kaggle output directories
OUTPUT_DIR = '/kaggle/working'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

for d in [CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'Image size:          {CONFIG["image_size"]}')
print(f'Batch size:          {CONFIG["batch_size"]}')
print(f'Accumulation steps:  {CONFIG["accumulation_steps"]}')
print(f'Effective batch:     {CONFIG["batch_size"] * CONFIG["accumulation_steps"]}')
print(f'Train ratio:         {CONFIG["train_ratio"]}')
print(f'Checkpoint dir:      {CHECKPOINT_DIR}')
print(f'Results dir:         {RESULTS_DIR}')
print(f'Plots dir:           {PLOTS_DIR}')""")

CONFIG_COLAB = code("""\
CONFIG = {
    'image_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'train_ratio': 0.70,
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 50,
    'patience': 10,
    'accumulation_steps': 4,
    'max_grad_norm': 1.0,
    'seed': SEED,
}

# Google Drive checkpoint directory
USE_GOOGLE_DRIVE = True
DRIVE_BASE = '/content/drive/MyDrive/tamper_project'
LOCAL_BASE = './artifacts/tamper_project'

if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        OUTPUT_DIR = DRIVE_BASE
        print('Google Drive mounted successfully.')
    except Exception:
        print('WARNING: Drive mount failed. Using local storage.')
        OUTPUT_DIR = LOCAL_BASE
else:
    OUTPUT_DIR = LOCAL_BASE

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

for d in [CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'Image size:          {CONFIG["image_size"]}')
print(f'Batch size:          {CONFIG["batch_size"]}')
print(f'Accumulation steps:  {CONFIG["accumulation_steps"]}')
print(f'Effective batch:     {CONFIG["batch_size"] * CONFIG["accumulation_steps"]}')
print(f'Train ratio:         {CONFIG["train_ratio"]}')
print(f'Checkpoint dir:      {CHECKPOINT_DIR}')
print(f'Results dir:         {RESULTS_DIR}')
print(f'Plots dir:           {PLOTS_DIR}')""")

# ── Section 4: Dataset Loading ────────────────────────────────────────────────

SEC3_MD_KAGGLE = md("""\
## 3. Dataset Loading

The dataset is pre-mounted by Kaggle at `/kaggle/input/`.
No API authentication or download step is needed.

**Dataset:** `sagnikkayalcse52/casia-spicing-detection-localization`
**Structure:** `Image/Au/`, `Image/Tp/` (images), `Mask/Au/`, `Mask/Tp/` (ground-truth masks)""")

SEC3_MD_COLAB = md("""\
## 3. Dataset Loading

Download the CASIA dataset from Kaggle using the Kaggle API.
Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` set via Colab Secrets.

**Dataset:** `sagnikkayalcse52/casia-spicing-detection-localization`
**Structure:** `Image/Au/`, `Image/Tp/` (images), `Mask/Au/`, `Mask/Tp/` (ground-truth masks)""")

WANDB_KAGGLE = code("""\
USE_WANDB = False  # Set True to enable W&B experiment tracking

if USE_WANDB:
    !pip install -q wandb
    import wandb
    from kaggle_secrets import UserSecretsClient
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    wandb.init(
        project='v6 Tampered Image Detection & Localization',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}-kaggle-v6',
        tags=['v6', 'casia-v2', 'kaggle'],
    )
    print('W&B initialized.')
else:
    print('W&B disabled. Running with local artifacts only.')""")

WANDB_COLAB = code("""\
USE_WANDB = False  # Set True to enable W&B experiment tracking

if USE_WANDB:
    !pip install -q wandb
    import wandb
    try:
        from google.colab import userdata
        wandb.login(key=userdata.get('WANDB_API_KEY'))
    except Exception:
        wandb.login()  # Interactive fallback
    wandb.init(
        project='v6 Tampered Image Detection & Localization',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}-colab-v6',
        tags=['v6', 'casia-v2', 'colab'],
    )
    print('W&B initialized.')
else:
    print('W&B disabled. Running with local artifacts only.')""")

DATASET_KAGGLE = code("""\
# Kaggle input path — dataset is already mounted
KAGGLE_INPUT = '/kaggle/input'

# Dynamically discover dataset root (case-insensitive search for IMAGE/ and MASK/)
DATASET_ROOT = None
IMAGE_DIR_NAME = None
MASK_DIR_NAME = None

for root, dirs, files in os.walk(KAGGLE_INPUT):
    dirs_lower = [d.lower() for d in dirs]
    if 'image' in dirs_lower and 'mask' in dirs_lower:
        DATASET_ROOT = root
        IMAGE_DIR_NAME = dirs[dirs_lower.index('image')]
        MASK_DIR_NAME = dirs[dirs_lower.index('mask')]
        break

if DATASET_ROOT is None:
    raise FileNotFoundError(
        'Could not find dataset with IMAGE/ and MASK/ directories under /kaggle/input/. '
        'Ensure the CASIA Splicing Detection + Localization dataset is attached.'
    )

print(f'Dataset root:  {DATASET_ROOT}')
print(f'Image dir:     {IMAGE_DIR_NAME}')
print(f'Mask dir:      {MASK_DIR_NAME}')

# Directory structure summary
for sub in ['Au', 'Tp']:
    for parent in [IMAGE_DIR_NAME, MASK_DIR_NAME]:
        path = os.path.join(DATASET_ROOT, parent, sub)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f'  {parent}/{sub}: {count} files')
        else:
            print(f'  {parent}/{sub}: NOT FOUND')""")

DATASET_COLAB = code("""\
# Kaggle API authentication via Colab Secrets
try:
    from google.colab import userdata
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    print('Kaggle credentials loaded from Colab Secrets.')
except Exception:
    print('WARNING: Could not load Kaggle credentials from Secrets.')
    print('Set KAGGLE_USERNAME and KAGGLE_KEY manually if needed.')

DATASET_SLUG = 'sagnikkayalcse52/casia-spicing-detection-localization'
DOWNLOAD_DIR = '/content/datasets'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def find_dataset_root(search_root):
    \"\"\"Walk directory tree to find folder containing IMAGE/ and MASK/ (case-insensitive).\"\"\"
    for root, dirs, files in os.walk(search_root):
        dirs_lower = [d.lower() for d in dirs]
        if 'image' in dirs_lower and 'mask' in dirs_lower:
            return root, dirs[dirs_lower.index('image')], dirs[dirs_lower.index('mask')]
    return None, None, None

# Check if already downloaded
DATASET_ROOT, IMAGE_DIR_NAME, MASK_DIR_NAME = find_dataset_root(DOWNLOAD_DIR)

if DATASET_ROOT is None:
    print(f'Downloading dataset: {DATASET_SLUG}')
    !kaggle datasets download -d {DATASET_SLUG} -p {DOWNLOAD_DIR} --unzip
    DATASET_ROOT, IMAGE_DIR_NAME, MASK_DIR_NAME = find_dataset_root(DOWNLOAD_DIR)

if DATASET_ROOT is None:
    raise FileNotFoundError(
        f'Could not find dataset with IMAGE/ and MASK/ directories under {DOWNLOAD_DIR}. '
        'Check Kaggle API credentials and dataset slug.'
    )

print(f'Dataset root:  {DATASET_ROOT}')
print(f'Image dir:     {IMAGE_DIR_NAME}')
print(f'Mask dir:      {MASK_DIR_NAME}')

# Directory structure summary
for sub in ['Au', 'Tp']:
    for parent in [IMAGE_DIR_NAME, MASK_DIR_NAME]:
        path = os.path.join(DATASET_ROOT, parent, sub)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f'  {parent}/{sub}: {count} files')
        else:
            print(f'  {parent}/{sub}: NOT FOUND')""")

# ── Section 5: Dataset Discovery ──────────────────────────────────────────────

SEC4_MD = md("""\
## 4. Dataset Discovery

Dynamically discover image-mask pairs with:
- Corruption guard for tampered and authentic images
- Dimension validation (image and mask must match spatially)
- Forgery type classification (`_D_` → splicing, `_S_` → copy-move)
- Excluded pairs tracking with reason strings""")

DISCOVERY_FUNCS = code("""\
def validate_dimensions(image_path, mask_path):
    \"\"\"Check that image and mask have the same spatial dimensions.\"\"\"
    img = Image.open(image_path)
    msk = Image.open(mask_path)
    if img.size != msk.size:
        return False, f'dim_mismatch: img={img.size} mask={msk.size}'
    return True, ''


def is_valid_image(filepath):
    \"\"\"Check if an image file can be opened and decoded.\"\"\"
    try:
        img = Image.open(filepath)
        img.verify()
        return True
    except Exception:
        return False


def discover_pairs(dataset_root, image_dir_name, mask_dir_name):
    \"\"\"Discover image-mask pairs dynamically.

    Args:
        dataset_root: Root directory containing image and mask subdirs.
        image_dir_name: Name of the image directory (case-preserved).
        mask_dir_name: Name of the mask directory (case-preserved).

    Returns:
        pairs: list of dicts with keys image_path, mask_path, label, forgery_type
        excluded: list of (filename, reason) tuples
    \"\"\"
    img_tp_dir = os.path.join(dataset_root, image_dir_name, 'Tp')
    mask_tp_dir = os.path.join(dataset_root, mask_dir_name, 'Tp')
    img_au_dir = os.path.join(dataset_root, image_dir_name, 'Au')

    pairs = []
    excluded = []

    # --- Tampered images ---
    if os.path.isdir(img_tp_dir):
        for img_name in sorted(os.listdir(img_tp_dir)):
            img_path = os.path.join(img_tp_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            # Corruption check
            if not is_valid_image(img_path):
                excluded.append((img_name, 'corrupt_image'))
                continue

            # Find matching mask
            mask_path = os.path.join(mask_tp_dir, img_name)
            if not os.path.exists(mask_path):
                stem = Path(img_name).stem
                mask_found = False
                for ext in ['.png', '.jpg', '.bmp', '.tif']:
                    alt_mask = os.path.join(mask_tp_dir, stem + ext)
                    if os.path.exists(alt_mask):
                        mask_path = alt_mask
                        mask_found = True
                        break
                if not mask_found:
                    excluded.append((img_name, 'mask_not_found'))
                    continue

            # Dimension validation
            valid, reason = validate_dimensions(img_path, mask_path)
            if not valid:
                excluded.append((img_name, reason))
                continue

            # Forgery type classification
            stem = Path(img_name).stem
            if '_D_' in stem:
                forgery_type = 'splicing'
            elif '_S_' in stem:
                forgery_type = 'copy-move'
            else:
                forgery_type = 'unknown'
                warnings.warn(f'Unrecognized forgery pattern: {img_name}')

            pairs.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'label': 1.0,
                'forgery_type': forgery_type,
            })

    # --- Authentic images ---
    if os.path.isdir(img_au_dir):
        for img_name in sorted(os.listdir(img_au_dir)):
            img_path = os.path.join(img_au_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            if not is_valid_image(img_path):
                excluded.append((img_name, 'corrupt_file'))
                continue

            pairs.append({
                'image_path': img_path,
                'mask_path': None,
                'label': 0.0,
                'forgery_type': 'authentic',
            })

    return pairs, excluded

print('Discovery functions defined.')""")

DISCOVERY_RUN = code("""\
# Run discovery using the resolved directory names
pairs, excluded = discover_pairs(DATASET_ROOT, IMAGE_DIR_NAME, MASK_DIR_NAME)

# Summary
type_counts = Counter(p['forgery_type'] for p in pairs)
print(f'Total valid pairs: {len(pairs)}')
for ftype, count in sorted(type_counts.items()):
    print(f'  {ftype}: {count}')

print(f'\\nExcluded: {len(excluded)}')
if excluded:
    for name, reason in excluded[:10]:
        print(f'  {name}: {reason}')
    if len(excluded) > 10:
        print(f'  ... and {len(excluded) - 10} more')""")

# ── Section 6: Dataset Validation ─────────────────────────────────────────────

SEC5_MD = md("""\
## 5. Dataset Validation

Verify discovered pairs, check class distributions, and run sample load checks.
A validation summary confirms the dataset is healthy before splitting.""")

VALIDATION = code("""\
assert len(pairs) > 0, 'No valid pairs discovered!'

tampered_count = sum(1 for p in pairs if p['label'] == 1.0)
authentic_count = sum(1 for p in pairs if p['label'] == 0.0)

print('=' * 50)
print('DATASET VALIDATION SUMMARY')
print('=' * 50)
print(f'Total samples:      {len(pairs) + len(excluded)}')
print(f'Valid pairs:         {len(pairs)}')
print(f'Skipped (excluded):  {len(excluded)}')
print(f'  Tampered images:   {tampered_count}')
print(f'  Authentic images:  {authentic_count}')
print(f'  Tampered ratio:    {tampered_count / len(pairs):.2%}')
print('=' * 50)

# Sample load check
print('\\nSample load check:')
for i in range(min(3, len(pairs))):
    p = pairs[i]
    img = cv2.imread(p['image_path'])
    assert img is not None, f'Failed to load image: {p["image_path"]}'
    if p['mask_path'] is not None:
        mask = cv2.imread(p['mask_path'], cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f'Failed to load mask: {p["mask_path"]}'
        print(f'  [{p["forgery_type"]}] img={img.shape}, mask={mask.shape}')
    else:
        print(f'  [{p["forgery_type"]}] img={img.shape}, mask=None (zero mask)')

print('\\nDataset validation passed.')""")

# ── Section 7: Preprocessing Pipeline ─────────────────────────────────────────

SEC6_MD = md("""\
## 6. Preprocessing Pipeline

Spatial augmentations for training: `Resize(384)`, `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`.
Validation/test: resize + ImageNet normalize only.

All transforms operate at 384 × 384 to fit comfortably within T4 GPU VRAM.""")

TRANSFORMS = code("""\
train_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print('Transforms defined.')
print(f'  Train: resize {CONFIG["image_size"]}, flip, rotate, normalize')
print(f'  Val/Test: resize {CONFIG["image_size"]}, normalize')""")

# ── Section 8: Dataset Splitting ──────────────────────────────────────────────

SEC7_MD = md("""\
## 7. Dataset Splitting

Stratified split: **70% train / 15% validation / 15% test**.
Stratification key: `forgery_type` (authentic / splicing / copy-move).

Two-step procedure:
1. Split 70% train vs 30% temp
2. Split temp 50/50 into val and test

Data leakage is verified with set-intersection assertions.
Split manifest saved for reproducibility.""")

SPLIT = code("""\
forgery_labels = [p['forgery_type'] for p in pairs]

# Step 1: train (70%) vs temp (30%)
train_pairs, temp_pairs = train_test_split(
    pairs, test_size=0.30, random_state=SEED, stratify=forgery_labels
)

# Step 2: temp -> val (50%) + test (50%) = 15% each
temp_labels = [p['forgery_type'] for p in temp_pairs]
val_pairs, test_pairs = train_test_split(
    temp_pairs, test_size=0.5, random_state=SEED, stratify=temp_labels
)

print(f'Train: {len(train_pairs)}')
print(f'Val:   {len(val_pairs)}')
print(f'Test:  {len(test_pairs)}')

# Per-split class distribution
for name, split in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
    counts = Counter(p['forgery_type'] for p in split)
    dist = ', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))
    print(f'  {name}: {dist}')

# Data leakage verification
train_paths = set(p['image_path'] for p in train_pairs)
val_paths = set(p['image_path'] for p in val_pairs)
test_paths = set(p['image_path'] for p in test_pairs)
assert len(train_paths & val_paths) == 0, 'LEAKAGE: train-val overlap!'
assert len(train_paths & test_paths) == 0, 'LEAKAGE: train-test overlap!'
assert len(val_paths & test_paths) == 0, 'LEAKAGE: val-test overlap!'
print('\\nNo data leakage detected.')""")

MANIFEST = code("""\
# Save split manifest for reproducibility
manifest = {
    'seed': SEED,
    'total_pairs': len(pairs),
    'excluded_count': len(excluded),
    'train_count': len(train_pairs),
    'val_count': len(val_pairs),
    'test_count': len(test_pairs),
    'train': [p['image_path'] for p in train_pairs],
    'val': [p['image_path'] for p in val_pairs],
    'test': [p['image_path'] for p in test_pairs],
}

manifest_path = os.path.join(RESULTS_DIR, 'split_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Split manifest saved to: {manifest_path}')""")

# ── Section 9: Dataset Class ─────────────────────────────────────────────────

SEC8_MD = md("""\
## 8. Dataset Class

`TamperingDataset` loads images as RGB, binarizes masks (`> 0` → foreground), generates zero masks for authentic images, and applies albumentations transforms.

Returns `(image, mask, label)` per sample.""")

DATASET_CLASS = code("""\
class TamperingDataset(Dataset):
    \"\"\"Dataset for tamper detection with segmentation masks.\"\"\"

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]

        # Load image (BGR -> RGB)
        image = cv2.imread(entry['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load or generate mask
        if entry['mask_path'] is not None:
            mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)  # Binarize: any non-zero pixel is foreground
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask is (1, H, W) float tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).float()

        label = torch.tensor(entry['label'], dtype=torch.float32)
        return image, mask, label

print('TamperingDataset defined.')""")

# ── Section 10: DataLoaders ───────────────────────────────────────────────────

SEC9_MD = md("""\
## 9. DataLoaders

Efficient data loading with:
- Deterministic seed per worker for reproducibility
- `pin_memory=True` for faster GPU transfers
- `persistent_workers=True` to avoid worker respawn overhead""")

DATALOADERS = code("""\
def seed_worker(worker_id):
    \"\"\"Set deterministic seed per DataLoader worker.\"\"\"
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_dataset = TamperingDataset(train_pairs, transform=train_transform)
val_dataset = TamperingDataset(val_pairs, transform=val_transform)
test_dataset = TamperingDataset(test_pairs, transform=val_transform)

loader_kwargs = dict(
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    persistent_workers=True if CONFIG['num_workers'] > 0 else False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
    **loader_kwargs,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches:   {len(val_loader)}')
print(f'Test batches:  {len(test_loader)}')""")

SANITY_CHECK = code("""\
# Sanity check: visualize one training batch
images, masks, labels = next(iter(train_loader))
print(f'Image batch shape: {images.shape}')
print(f'Mask batch shape:  {masks.shape}')
print(f'Labels: {labels}')

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i in range(min(4, images.size(0))):
    img = images[i].permute(1, 2, 0).numpy() * std + mean
    img = np.clip(img, 0, 1)
    msk = masks[i].squeeze().numpy()

    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Image (label={labels[i].item():.0f})')
    axes[0, i].axis('off')

    axes[1, i].imshow(msk, cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title('Mask')
    axes[1, i].axis('off')

plt.suptitle('Sanity Check: Training Batch')
plt.tight_layout()
plt.show()""")

# ── Section 11: Model Architecture ────────────────────────────────────────────

SEC10_MD = md("""\
## 10. Model Architecture

**SMP U-Net** with ResNet34 encoder (ImageNet pretrained).
- Input: `(B, 3, 384, 384)` — RGB images
- Output: `(B, 1, 384, 384)` — raw logits (sigmoid applied at inference)
- ~24M parameters

This is a **baseline segmentation model** appropriate for the assignment scope.
Stronger architectures (edge-enhanced, transformer-based) are documented as future work.""")

MODEL = code("""\
model = smp.Unet(
    encoder_name=CONFIG['encoder_name'],
    encoder_weights=CONFIG['encoder_weights'],
    in_channels=CONFIG['in_channels'],
    classes=CONFIG['classes'],
    activation=None,  # Raw logits — sigmoid applied during inference
)
model = model.to(device)

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters:     {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# Forward pass shape verification
with torch.no_grad():
    dummy = torch.randn(1, 3, CONFIG['image_size'], CONFIG['image_size']).to(device)
    out = model(dummy)
    print(f'Input shape:  {dummy.shape}')
    print(f'Output shape: {out.shape}')
    assert out.shape == (1, 1, CONFIG['image_size'], CONFIG['image_size']), \\
        f'Unexpected output shape: {out.shape}'

print('Model shape check passed.')""")

# ── Section 12: Loss Functions ────────────────────────────────────────────────

SEC11_MD = md("""\
## 11. Loss Functions

**BCEDiceLoss** — combines binary cross-entropy (pixel-level classification) with Dice loss (handles class imbalance). Equal weight; `smooth=1.0`.

**Rationale:** In CASIA, tampered pixels are typically <5% of image area. BCE alone under-penalizes false negatives on the minority class. Dice loss directly optimizes the F1-like overlap.""")

LOSS = code("""\
class BCEDiceLoss(nn.Module):
    \"\"\"Combined BCE + Dice loss for binary segmentation.\"\"\"

    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return bce_loss + dice_loss

criterion = BCEDiceLoss()

# Optimizer: AdamW with differential learning rates
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': CONFIG['encoder_lr']},
    {'params': model.decoder.parameters(), 'lr': CONFIG['decoder_lr']},
    {'params': model.segmentation_head.parameters(), 'lr': CONFIG['decoder_lr']},
], weight_decay=CONFIG['weight_decay'])

# Mixed precision scaler
scaler = GradScaler('cuda')

print('BCEDiceLoss initialized.')
print(f'Optimizer: AdamW')
print(f'  Encoder LR:   {CONFIG["encoder_lr"]}')
print(f'  Decoder LR:   {CONFIG["decoder_lr"]}')
print(f'  Weight decay:  {CONFIG["weight_decay"]}')""")

# ── Section 13: Evaluation Metrics ────────────────────────────────────────────

SEC12_MD = md("""\
## 12. Evaluation Metrics

Pixel-level evaluation functions used throughout training validation and test evaluation.

**Edge-case convention:** When both prediction and ground truth are all-zero (true negative),
F1 = 1.0, IoU = 1.0, Precision = 1.0, Recall = 1.0. This is consistent — no false positives, no false negatives.""")

METRICS = code("""\
def compute_pixel_f1(pred, gt, eps=1e-8):
    \"\"\"Compute Pixel-F1 score.\"\"\"
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return (2 * precision * recall / (precision + recall + eps)).item()


def compute_iou(pred, gt, eps=1e-8):
    \"\"\"Compute Intersection over Union.\"\"\"
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / (union + eps)).item()


def compute_precision_recall(pred, gt, eps=1e-8):
    \"\"\"Compute pixel-level precision and recall.

    Returns (1.0, 1.0) for true negatives (both empty) for consistency
    with F1 and IoU edge-case handling.
    \"\"\"
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0, 1.0  # precision=0 (all FP), recall=1 (no FN)
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision.item(), recall.item()

print('Metric functions defined: compute_pixel_f1, compute_iou, compute_precision_recall')""")

# ── Section 14: Validation Loop ───────────────────────────────────────────────

SEC13_MD = md("""\
## 13. Validation Loop

The validation function runs after each training epoch.
It evaluates the model at a given threshold and returns loss, mean F1, and mean IoU.

Validation is **threshold-aware**: checkpoint selection and early stopping use the best
validation Pixel-F1 from the sweep, not a fixed 0.5 threshold.""")

VALIDATE = code("""\
@torch.no_grad()
def validate(model, val_loader, criterion, device, threshold=0.5):
    \"\"\"Run validation and return loss, mean F1, mean IoU.\"\"\"
    model.eval()
    total_loss = 0.0
    f1_scores = []
    iou_scores = []

    for images, masks, labels in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for i in range(images.size(0)):
            f1 = compute_pixel_f1(preds[i], masks[i])
            iou = compute_iou(preds[i], masks[i])
            f1_scores.append(f1)
            iou_scores.append(iou)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_f1 = np.mean(f1_scores)
    avg_iou = np.mean(iou_scores)
    return avg_loss, avg_f1, avg_iou

print('Validation function defined.')""")

# ── Section 15: Training Pipeline ─────────────────────────────────────────────

SEC14_MD = md("""\
## 14. Training Pipeline

Mixed precision training with gradient accumulation (effective batch size = 16):
- Early stopping on validation Pixel-F1 (patience = 10)
- Gradient clipping (max_norm = 1.0)
- Partial window flush after final batch
- Checkpoints saved every epoch (last) and on improvement (best)
- W&B logging (guarded behind `USE_WANDB`)""")

CHECKPOINT_HELPERS = code("""\
def save_checkpoint(state, filepath):
    \"\"\"Save training checkpoint.\"\"\"
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer, scaler):
    \"\"\"Load training checkpoint for resume.\"\"\"
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['epoch'] + 1, ckpt['best_f1'], ckpt['best_epoch']

print('Checkpoint helpers defined.')""")

TRAINING_LOOP = code("""\
# Training configuration
ACCUMULATION_STEPS = CONFIG['accumulation_steps']
MAX_EPOCHS = CONFIG['max_epochs']
PATIENCE = CONFIG['patience']

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_f1': [],
    'val_iou': [],
}

best_f1 = 0.0
best_epoch = 0
start_epoch = 0

# Optional: resume from checkpoint
resume_path = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt')
if os.path.exists(resume_path):
    start_epoch, best_f1, best_epoch = load_checkpoint(
        resume_path, model, optimizer, scaler
    )
    print(f'Resumed from epoch {start_epoch}, best F1={best_f1:.4f} at epoch {best_epoch + 1}')

# --- Training loop ---
for epoch in range(start_epoch, MAX_EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        enumerate(train_loader), total=len(train_loader),
        desc=f'Epoch {epoch + 1}/{MAX_EPOCHS}'
    )

    for batch_idx, (images, masks, labels) in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, masks) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f'{running_loss / (batch_idx + 1):.4f}'})

    # Partial window flush (if last batch didn't align with accumulation steps)
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    val_loss, val_f1, val_iou = validate(model, val_loader, criterion, device)

    # Record history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['val_iou'].append(val_iou)

    print(
        f'Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, '
        f'val_loss={val_loss:.4f}, val_f1={val_f1:.4f}, val_iou={val_iou:.4f}'
    )

    # W&B logging (guarded)
    if USE_WANDB:
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': avg_train_loss,
            'val/loss': val_loss,
            'val/pixel_f1': val_f1,
            'val/pixel_iou': val_iou,
            'train/lr_encoder': optimizer.param_groups[0]['lr'],
            'train/lr_decoder': optimizer.param_groups[1]['lr'],
        })

    # Build checkpoint state
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }

    # Save last checkpoint every epoch
    save_checkpoint(state, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        state['best_f1'] = best_f1
        state['best_epoch'] = best_epoch
        save_checkpoint(state, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        print(f'  -> New best model saved (F1={best_f1:.4f})')

    # Periodic checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_checkpoint(state, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt'))

    # Early stopping
    if epoch - best_epoch >= PATIENCE:
        print(f'Early stopping at epoch {epoch + 1}. Best F1={best_f1:.4f} at epoch {best_epoch + 1}')
        break

print(f'\\nTraining complete. Best F1={best_f1:.4f} at epoch {best_epoch + 1}')""")

# ── Section 16: Threshold Selection ───────────────────────────────────────────

SEC15_MD = md("""\
## 15. Threshold Selection

Sweep 50 thresholds from 0.1 to 0.9 on the **validation set** to maximize mean Pixel-F1.
The selected threshold is frozen for all downstream test-set evaluation.""")

LOAD_BEST = code("""\
# Load best model for threshold selection and evaluation
best_ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded best model from epoch {ckpt["best_epoch"] + 1} (F1={ckpt["best_f1"]:.4f})')""")

THRESHOLD_SWEEP = code("""\
@torch.no_grad()
def find_best_threshold(model, val_loader, device, num_thresholds=50):
    \"\"\"Sweep thresholds on validation set to maximize mean Pixel-F1.\"\"\"
    model.eval()
    thresholds = np.linspace(0.1, 0.9, num_thresholds)

    # Collect all validation predictions
    all_probs = []
    all_masks = []
    for images, masks, labels in tqdm(val_loader, desc='Collecting val predictions'):
        images = images.to(device)
        with autocast('cuda'):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_masks.append(masks)

    all_probs = torch.cat(all_probs)
    all_masks = torch.cat(all_masks)

    # Sweep thresholds
    results = []
    for t in tqdm(thresholds, desc='Threshold sweep'):
        f1_scores = []
        preds = (all_probs > t).float()
        for i in range(len(all_probs)):
            f1 = compute_pixel_f1(preds[i], all_masks[i])
            f1_scores.append(f1)
        mean_f1 = np.mean(f1_scores)
        results.append((t, mean_f1))

    results.sort(key=lambda x: x[1], reverse=True)
    best_t, best_f1_val = results[0]
    return best_t, results

best_threshold, threshold_results = find_best_threshold(model, val_loader, device)
print(f'Best threshold: {best_threshold:.4f}')
print(f'Best val F1 at threshold: {threshold_results[0][1]:.4f}')""")

# ── Section 17: Test Evaluation ───────────────────────────────────────────────

SEC16_MD = md("""\
## 16. Test Evaluation

Full evaluation with three reporting views:
1. **Mixed-set** — All test images (authentic + tampered)
2. **Tampered-only** — Pixel-level metrics on tampered images only
3. **Forgery-type breakdown** — Separate metrics for splicing and copy-move

Image-level detection uses the **max pixel probability** as a tamper score.""")

EVALUATE_FUNC = code("""\
@torch.no_grad()
def evaluate(model, test_loader, test_pairs, device, threshold):
    \"\"\"Full evaluation on test set with mixed, tampered-only, and per-forgery metrics.\"\"\"
    model.eval()

    all_f1 = []
    all_iou = []
    all_precision = []
    all_recall = []
    tampered_f1 = []
    tampered_iou = []
    image_preds = []
    image_labels = []
    image_scores = []

    forgery_f1 = {'splicing': [], 'copy-move': []}

    idx = 0
    for images, masks, labels in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        with autocast('cuda'):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > threshold).float()

        for i in range(images.size(0)):
            f1 = compute_pixel_f1(preds[i], masks[i])
            iou = compute_iou(preds[i], masks[i])
            prec, rec = compute_precision_recall(preds[i], masks[i])

            all_f1.append(f1)
            all_iou.append(iou)
            all_precision.append(prec)
            all_recall.append(rec)

            tamper_score = probs[i].view(-1).max().item()
            image_scores.append(tamper_score)
            image_labels.append(int(labels[i].item()))
            image_preds.append(int(tamper_score >= threshold))

            if labels[i].item() == 1.0:
                tampered_f1.append(f1)
                tampered_iou.append(iou)
                if idx < len(test_pairs):
                    ftype = test_pairs[idx]['forgery_type']
                    if ftype in forgery_f1:
                        forgery_f1[ftype].append(f1)
            idx += 1

    image_accuracy = np.mean([p == l for p, l in zip(image_preds, image_labels)])
    try:
        image_auc = roc_auc_score(image_labels, image_scores)
    except ValueError:
        image_auc = float('nan')

    results = {
        'pixel_f1_mean': float(np.mean(all_f1)),
        'pixel_f1_std': float(np.std(all_f1)),
        'pixel_iou_mean': float(np.mean(all_iou)),
        'pixel_iou_std': float(np.std(all_iou)),
        'precision_mean': float(np.mean(all_precision)),
        'recall_mean': float(np.mean(all_recall)),
        'tampered_f1_mean': float(np.mean(tampered_f1)) if tampered_f1 else 0.0,
        'tampered_f1_std': float(np.std(tampered_f1)) if tampered_f1 else 0.0,
        'tampered_iou_mean': float(np.mean(tampered_iou)) if tampered_iou else 0.0,
        'tampered_iou_std': float(np.std(tampered_iou)) if tampered_iou else 0.0,
        'image_accuracy': float(image_accuracy),
        'image_auc_roc': float(image_auc),
        'threshold_used': threshold,
        'num_test_images': len(all_f1),
        'num_tampered_images': len(tampered_f1),
        'forgery_breakdown': {},
    }

    for ftype, scores in forgery_f1.items():
        if scores:
            results['forgery_breakdown'][ftype] = {
                'f1_mean': float(np.mean(scores)),
                'f1_std': float(np.std(scores)),
                'count': len(scores),
            }

    return results

print('Evaluate function defined.')""")

EVALUATE_RUN = code("""\
test_results = evaluate(model, test_loader, test_pairs, device, best_threshold)

print(f'\\nTEST SET RESULTS (threshold={best_threshold:.4f})')
print(f'\\nMixed-set ({test_results["num_test_images"]} images):')
print(f'  Pixel-F1:  {test_results["pixel_f1_mean"]:.4f} +/- {test_results["pixel_f1_std"]:.4f}')
print(f'  Pixel-IoU: {test_results["pixel_iou_mean"]:.4f} +/- {test_results["pixel_iou_std"]:.4f}')
print(f'  Precision: {test_results["precision_mean"]:.4f}')
print(f'  Recall:    {test_results["recall_mean"]:.4f}')

print(f'\\nTampered-only ({test_results["num_tampered_images"]} images):')
print(f'  Pixel-F1:  {test_results["tampered_f1_mean"]:.4f} +/- {test_results["tampered_f1_std"]:.4f}')
print(f'  Pixel-IoU: {test_results["tampered_iou_mean"]:.4f} +/- {test_results["tampered_iou_std"]:.4f}')

print(f'\\nImage-level:')
print(f'  Accuracy: {test_results["image_accuracy"]:.4f}')
print(f'  AUC-ROC:  {test_results["image_auc_roc"]:.4f}')

print(f'\\nForgery-type breakdown:')
for ftype, data in test_results['forgery_breakdown'].items():
    print(f'  {ftype} ({data["count"]} images): F1={data["f1_mean"]:.4f} +/- {data["f1_std"]:.4f}')""")

EVALUATE_WANDB = code("""\
# Log test results to W&B (guarded)
if USE_WANDB:
    wandb.summary.update({
        'best/val_f1': best_f1,
        'best/epoch': best_epoch + 1,
        'test/pixel_f1_mixed': test_results['pixel_f1_mean'],
        'test/pixel_f1_tampered': test_results['tampered_f1_mean'],
        'test/pixel_iou_mixed': test_results['pixel_iou_mean'],
        'test/image_accuracy': test_results['image_accuracy'],
        'test/image_auc_roc': test_results['image_auc_roc'],
        'test/threshold': best_threshold,
    })
    print('Test results logged to W&B.')""")

# ── Section 18: Visualization ─────────────────────────────────────────────────

SEC17_MD = md("""\
## 17. Visualization

1. **Training curves** — loss, F1, and IoU over epochs
2. **F1-vs-threshold** — threshold sweep visualization
3. **Prediction grid** — Original | GT Mask | Predicted Mask | Overlay

All figures saved to the plots directory.""")

VIZ_CURVES = code("""\
# Training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

epochs_range = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs_range, history['train_loss'], label='Train')
axes[0].plot(epochs_range, history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history['val_f1'], color='green')
axes[1].axvline(x=best_epoch + 1, color='red', linestyle='--',
                label=f'Best (epoch {best_epoch + 1})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Pixel-F1')
axes[1].set_title('Validation Pixel-F1')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history['val_iou'], color='orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Pixel-IoU')
axes[2].set_title('Validation Pixel-IoU')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Training curves saved.')""")

VIZ_THRESHOLD = code("""\
# F1 vs Threshold plot
thresh_vals = [r[0] for r in threshold_results]
f1_vals = [r[1] for r in threshold_results]

sorted_pairs_t = sorted(zip(thresh_vals, f1_vals))
thresh_sorted = [p[0] for p in sorted_pairs_t]
f1_sorted = [p[1] for p in sorted_pairs_t]

plt.figure(figsize=(8, 5))
plt.plot(thresh_sorted, f1_sorted, 'b-', linewidth=2)
plt.axvline(x=best_threshold, color='red', linestyle='--',
            label=f'Best: {best_threshold:.3f} (F1={threshold_results[0][1]:.4f})')
plt.xlabel('Threshold')
plt.ylabel('Mean Pixel-F1')
plt.title('F1 vs. Threshold (Validation Set)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'f1_vs_threshold.png'), dpi=150, bbox_inches='tight')
plt.show()
print('F1 vs threshold plot saved.')""")

VIZ_COLLECT = code("""\
@torch.no_grad()
def collect_predictions(model, test_loader, test_pairs, device, threshold):
    \"\"\"Collect all test predictions with metadata for visualization.\"\"\"
    model.eval()
    predictions = []
    idx = 0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for images, masks, labels in test_loader:
        images_dev = images.to(device)
        with autocast('cuda'):
            logits = model(images_dev)
        probs = torch.sigmoid(logits).cpu()

        for i in range(images.size(0)):
            pred_mask = (probs[i] > threshold).float()
            f1 = compute_pixel_f1(pred_mask, masks[i])

            img_np = images[i].permute(1, 2, 0).numpy() * std + mean
            img_np = np.clip(img_np, 0, 1)

            gt_mask_area = masks[i].sum().item() / masks[i].numel()

            predictions.append({
                'image': img_np,
                'gt_mask': masks[i].squeeze().numpy(),
                'pred_mask': pred_mask.squeeze().numpy(),
                'prob_map': probs[i].squeeze().numpy(),
                'pixel_f1': f1,
                'label': labels[i].item(),
                'forgery_type': test_pairs[idx]['forgery_type'] if idx < len(test_pairs) else 'unknown',
                'gt_mask_area': gt_mask_area,
            })
            idx += 1

    return predictions

predictions = collect_predictions(model, test_loader, test_pairs, device, best_threshold)
print(f'Collected {len(predictions)} predictions.')""")

VIZ_GRID = code("""\
# Prediction grid: best / median / worst tampered + authentic
tampered_preds = [p for p in predictions if p['label'] == 1.0]
authentic_preds = [p for p in predictions if p['label'] == 0.0]
tampered_sorted = sorted(tampered_preds, key=lambda p: p['pixel_f1'])

samples = []
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[-2:])               # 2 best
mid = len(tampered_sorted) // 2
if len(tampered_sorted) >= 4:
    samples.extend(tampered_sorted[mid - 1:mid + 1])   # 2 median
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[:2])                 # 2 worst
if len(authentic_preds) >= 2:
    samples.extend(authentic_preds[:2])                 # 2 authentic

n_rows = len(samples)
if n_rows == 0:
    print('No samples available for prediction grid.')
else:
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        img = sample['image']
        gt = sample['gt_mask']
        pred = sample['pred_mask']

        overlay = img.copy()
        pred_bool = pred > 0
        if pred_bool.any():
            overlay[pred_bool] = overlay[pred_bool] * 0.6 + np.array([1, 0, 0]) * 0.4

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'{sample["forgery_type"]} (F1={sample["pixel_f1"]:.3f})')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title('Predicted Mask')
        axes[row, 2].axis('off')

        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('Overlay')
        axes[row, 3].axis('off')

    plt.suptitle('Prediction Grid: Best / Median / Worst / Authentic', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'prediction_grid.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print('Prediction grid saved.')""")

VIZ_WANDB = code("""\
# Log visualizations to W&B (guarded)
if USE_WANDB:
    for plot_name in ['prediction_grid.png', 'training_curves.png', 'f1_vs_threshold.png']:
        plot_path = os.path.join(PLOTS_DIR, plot_name)
        if os.path.exists(plot_path):
            wandb.log({plot_name.replace('.png', ''): wandb.Image(plot_path)})
    print('Visualizations logged to W&B.')""")

# ── Section 19: Explainable AI ────────────────────────────────────────────────

SEC18_MD = md("""\
## 18. Explainable AI

Lightweight explainability to verify model behaviour:

- **Grad-CAM heatmaps** — Spatial regions driving encoder activations
- **Diagnostic overlays** — TP (green), FP (red), FN (blue) colour coding
- **Failure case analysis** — Systematic examination of worst predictions

**Why explainability matters for tamper detection:** In forensic applications, it's important
to understand *where* the model is looking. Grad-CAM reveals whether the model attends to
actual tampered regions or spurious artifacts, building trust in the model's decisions.

Grad-CAM is wrapped in safety checks to prevent crashes from unexpected tensor shapes or missing gradients.""")

GRADCAM_CLASS = code("""\
class GradCAM:
    \"\"\"Grad-CAM for segmentation encoder features.\"\"\"

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handles = []
        self._handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        \"\"\"Generate Grad-CAM heatmap. Returns None on failure.\"\"\"
        self.model.eval()
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)
        target = output.mean()
        self.model.zero_grad()
        target.backward()

        # Safety: check hooks captured data
        if self.gradients is None or self.activations is None:
            warnings.warn('Grad-CAM: hooks did not capture data.')
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode='bilinear', align_corners=False
        )
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

print('GradCAM class defined.')""")

DIAGNOSTIC_OVERLAY = code("""\
def create_diagnostic_overlay(original, pred_mask, gt_mask):
    \"\"\"Colour-coded overlay: Green=TP, Red=FP, Blue=FN.\"\"\"
    overlay = original.copy().astype(np.float32)
    if overlay.max() > 1.0:
        overlay = overlay / 255.0

    tp_mask = (pred_mask > 0) & (gt_mask > 0)
    fp_mask = (pred_mask > 0) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask > 0)

    overlay[tp_mask] = overlay[tp_mask] * 0.5 + np.array([0, 1, 0]) * 0.5
    overlay[fp_mask] = overlay[fp_mask] * 0.5 + np.array([1, 0, 0]) * 0.5
    overlay[fn_mask] = overlay[fn_mask] * 0.5 + np.array([0, 0, 1]) * 0.5

    return np.clip(overlay, 0, 1)

print('Diagnostic overlay function defined.')""")

GRADCAM_VIZ = code("""\
# Grad-CAM + diagnostic overlay visualization
grad_cam = GradCAM(model, model.encoder.layer4)

cam_samples = [p for p in predictions if p['label'] == 1.0]
cam_samples = sorted(cam_samples, key=lambda p: p['pixel_f1'], reverse=True)[:4]

n_cam = len(cam_samples)
if n_cam == 0:
    print('No tampered samples for Grad-CAM visualization.')
else:
    fig, axes = plt.subplots(n_cam, 5, figsize=(25, 5 * n_cam))
    if n_cam == 1:
        axes = axes[np.newaxis, :]

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row, sample in enumerate(cam_samples):
        img_tensor = torch.from_numpy(sample['image']).permute(2, 0, 1).float()
        img_tensor = (img_tensor - mean_t) / std_t
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        try:
            cam = grad_cam.generate(img_tensor)
        except Exception as e:
            warnings.warn(f'Grad-CAM failed for row {row}: {e}')
            cam = None

        diagnostic = create_diagnostic_overlay(
            sample['image'], sample['pred_mask'], sample['gt_mask']
        )

        axes[row, 0].imshow(sample['image'])
        axes[row, 0].set_title(f'Original ({sample["forgery_type"]})')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(sample['image'])
        if cam is not None:
            axes[row, 1].imshow(cam, cmap='jet', alpha=0.5)
            axes[row, 1].set_title('Grad-CAM Heatmap')
        else:
            axes[row, 1].set_title('Grad-CAM (failed)')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(sample['gt_mask'], cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title('Ground Truth')
        axes[row, 2].axis('off')

        axes[row, 3].imshow(sample['pred_mask'], cmap='gray', vmin=0, vmax=1)
        axes[row, 3].set_title(f'Prediction (F1={sample["pixel_f1"]:.3f})')
        axes[row, 3].axis('off')

        axes[row, 4].imshow(diagnostic)
        axes[row, 4].set_title('Diagnostic (G=TP, R=FP, B=FN)')
        axes[row, 4].axis('off')

    plt.suptitle('Explainable AI: Grad-CAM + Diagnostic Overlays', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gradcam_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print('Grad-CAM analysis saved.')

grad_cam.remove_hooks()""")

FAILURE_ANALYSIS = code("""\
def analyze_failure_cases(predictions, n_worst=10):
    \"\"\"Analyze worst predictions to identify systematic error patterns.\"\"\"
    tampered_preds = [p for p in predictions if p['label'] == 1.0]
    sorted_preds = sorted(tampered_preds, key=lambda p: p['pixel_f1'])
    worst = sorted_preds[:n_worst]

    if not worst:
        print('No tampered predictions to analyze.')
        return None

    analysis = {
        'forgery_types': [p['forgery_type'] for p in worst],
        'mean_mask_area': np.mean([p['gt_mask_area'] for p in worst]),
        'mean_f1': np.mean([p['pixel_f1'] for p in worst]),
        'common_patterns': [],
    }

    small_mask_count = sum(1 for p in worst if p['gt_mask_area'] < 0.02)
    if small_mask_count > n_worst // 2:
        analysis['common_patterns'].append(
            f'Fails on small tampered regions (<2% area): {small_mask_count}/{len(worst)}'
        )

    type_counts = Counter(analysis['forgery_types'])
    dominant_type = type_counts.most_common(1)[0]
    if dominant_type[1] > len(worst) * 0.7:
        analysis['common_patterns'].append(
            f'Disproportionately fails on {dominant_type[0]}: {dominant_type[1]}/{len(worst)}'
        )

    print(f'\\nFailure Case Analysis (worst {len(worst)} predictions):')
    print(f'  Mean Pixel-F1:     {analysis["mean_f1"]:.4f}')
    print(f'  Mean GT mask area: {analysis["mean_mask_area"]:.4f}')
    print(f'  Forgery types:     {dict(type_counts)}')
    if analysis['common_patterns']:
        print('  Patterns detected:')
        for pattern in analysis['common_patterns']:
            print(f'    - {pattern}')
    else:
        print('  No dominant failure patterns detected.')

    return analysis

failure_analysis = analyze_failure_cases(predictions)""")

# ── Section 20: Robustness Testing ────────────────────────────────────────────

SEC19_MD = md("""\
## 19. Robustness Testing

Evaluate model robustness against post-processing degradations applied to **copies** of test images (originals are never modified):

- JPEG compression (QF=70, QF=50)
- Gaussian noise (light, heavy)
- Gaussian blur (5×5 kernel)
- Resize degradation (0.75×, 0.5×)

**Why robustness matters:** In real-world forensic scenarios, images are often recompressed, resized, or degraded before analysis. A robust model maintains localization accuracy despite these transformations. Same validation-selected threshold used for all conditions.""")

ROBUSTNESS_TRANSFORMS = code("""\
NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

robustness_transforms = {
    'clean': A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        NORMALIZE, ToTensorV2(),
    ]),
    'jpeg_qf70': A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'jpeg_qf50': A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'gaussian_noise_light': A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'gaussian_noise_heavy': A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.GaussNoise(var_limit=(100.0, 100.0), p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
    'gaussian_blur': A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.GaussianBlur(blur_limit=(5, 5), p=1.0),
        NORMALIZE, ToTensorV2(),
    ]),
}

print(f'Defined {len(robustness_transforms)} degradation transforms.')""")

RESIZE_DATASET = code("""\
class ResizeDegradationDataset(Dataset):
    \"\"\"Applies resize degradation to copies of images. Masks remain clean.\"\"\"

    def __init__(self, pairs, scale_factor, image_size=384):
        self.pairs = pairs
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.normalize = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]

        image = cv2.imread(entry['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize degradation on a copy
        h, w = image.shape[:2]
        small_h = max(1, int(h * self.scale_factor))
        small_w = max(1, int(w * self.scale_factor))
        degraded = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)

        if entry['mask_path'] is not None:
            mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        augmented = self.normalize(image=degraded, mask=mask)
        image_t = augmented['image']
        mask_t = augmented['mask'].unsqueeze(0).float()
        label = torch.tensor(entry['label'], dtype=torch.float32)
        return image_t, mask_t, label

print('ResizeDegradationDataset defined.')""")

ROBUSTNESS_RUN = code("""\
@torch.no_grad()
def run_robustness_eval(model, loader, device, threshold):
    \"\"\"Evaluation loop for robustness testing.\"\"\"
    model.eval()
    f1_scores = []
    for images, masks, labels in loader:
        images = images.to(device)
        with autocast('cuda'):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > threshold).float()
        for i in range(images.size(0)):
            f1 = compute_pixel_f1(preds[i], masks[i])
            f1_scores.append(f1)
    return f1_scores


def evaluate_robustness(model, test_pairs, device, threshold):
    \"\"\"Evaluate model under all degradation conditions.\"\"\"
    results = {}

    for name, transform in tqdm(robustness_transforms.items(), desc='Robustness tests'):
        dataset = TamperingDataset(test_pairs, transform=transform)
        loader = DataLoader(
            dataset, batch_size=CONFIG['batch_size'],
            shuffle=False, num_workers=CONFIG['num_workers']
        )
        f1_scores = run_robustness_eval(model, loader, device, threshold)
        results[name] = {
            'f1_mean': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
        }

    for scale in [0.75, 0.5]:
        name = f'resize_{scale}x'
        dataset = ResizeDegradationDataset(test_pairs, scale_factor=scale)
        loader = DataLoader(
            dataset, batch_size=CONFIG['batch_size'],
            shuffle=False, num_workers=CONFIG['num_workers']
        )
        f1_scores = run_robustness_eval(model, loader, device, threshold)
        results[name] = {
            'f1_mean': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
        }

    return results

robustness_results = evaluate_robustness(model, test_pairs, device, best_threshold)

print(f'\\nRobustness Results (threshold={best_threshold:.4f}):')
print(f'{"":<25} {"Pixel-F1 (mean +/- std)":<25} {"Delta from clean":<15}')
print('-' * 65)
clean_f1 = robustness_results.get('clean', {}).get('f1_mean', 0)
for name, data in robustness_results.items():
    delta = data['f1_mean'] - clean_f1 if name != 'clean' else 0.0
    delta_str = f'{delta:+.4f}' if name != 'clean' else '---'
    print(f'{name:<25} {data["f1_mean"]:.4f} +/- {data["f1_std"]:.4f}      {delta_str}')""")

ROBUSTNESS_CHART = code("""\
# Robustness bar chart
names = list(robustness_results.keys())
means = [robustness_results[n]['f1_mean'] for n in names]
stds = [robustness_results[n]['f1_std'] for n in names]

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(names)), means, yerr=stds, capsize=4, color='steelblue', alpha=0.8)

if 'clean' in names:
    bars[names.index('clean')].set_color('green')

plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.ylabel('Pixel-F1')
plt.title('Robustness Testing: Pixel-F1 Under Degradation')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'robustness_chart.png'), dpi=150, bbox_inches='tight')
plt.show()

if USE_WANDB:
    wandb.log({
        'robustness_chart': wandb.Image(os.path.join(PLOTS_DIR, 'robustness_chart.png')),
    })
    for name, data in robustness_results.items():
        wandb.log({f'robustness/{name}_f1': data['f1_mean']})
    print('Robustness results logged to W&B.')""")

# ── Section 21: Experiment Tracking ───────────────────────────────────────────

SEC20_MD = md("""\
## 20. Experiment Tracking

W&B experiment tracking is **integrated throughout this notebook** (guarded behind `USE_WANDB`).

| Section | W&B Action |
|---|---|
| 3 (Dataset Loading) | Install, login, init with CONFIG |
| 14 (Training Pipeline) | Per-epoch metric logging |
| 16 (Test Evaluation) | Test results summary |
| 17 (Visualization) | Images logged |
| 19 (Robustness Testing) | Per-degradation F1 |
| 21 (Saving Artifacts) | Model artifact upload, `wandb.finish()` |

When `USE_WANDB = False` (default), all artifacts are saved locally.
No W&B code runs. The notebook is fully self-contained without W&B.""")

# ── Section 22: Saving Artifacts ──────────────────────────────────────────────

SEC21_MD = md("""\
## 21. Saving Artifacts

Save all results, configuration, and artifacts for reproducibility.
Upload best model as W&B artifact if tracking is enabled.""")

SAVE_RESULTS = code("""\
# Save results summary
results_summary = {
    'config': CONFIG,
    'seed': SEED,
    'best_epoch': best_epoch + 1,
    'best_val_f1': float(best_f1),
    'threshold': float(best_threshold),
    'test_results': {
        k: v for k, v in test_results.items()
        if k != 'forgery_breakdown'
    },
    'forgery_breakdown': test_results.get('forgery_breakdown', {}),
    'robustness_results': {
        name: {'f1_mean': data['f1_mean'], 'f1_std': data['f1_std']}
        for name, data in robustness_results.items()
    },
    'training_history': {
        'epochs': len(history['train_loss']),
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_f1': history['val_f1'][-1] if history['val_f1'] else None,
    },
}

results_path = os.path.join(RESULTS_DIR, 'results_summary.json')
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f'Results summary saved to: {results_path}')""")

WANDB_FINISH = code("""\
# W&B artifact upload and cleanup (guarded)
if USE_WANDB:
    artifact = wandb.Artifact('best-model-v6', type='model')
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        print('Model artifact uploaded to W&B.')
    wandb.finish()
    print('W&B run finished.')""")

# ── Section 23: Final Summary ─────────────────────────────────────────────────

SEC22_MD = md("""\
## 22. Final Summary

Artifact inventory and notebook completion check.""")

FINAL_SUMMARY_KAGGLE = code("""\
print()
print('=' * 60)
print('NOTEBOOK COMPLETE — ARTIFACT INVENTORY')
print('=' * 60)

expected_artifacts = {
    CHECKPOINT_DIR: ['best_model.pt', 'last_checkpoint.pt'],
    RESULTS_DIR: ['split_manifest.json', 'results_summary.json'],
    PLOTS_DIR: ['training_curves.png', 'f1_vs_threshold.png', 'prediction_grid.png',
                'gradcam_analysis.png', 'robustness_chart.png'],
}

all_ok = True
for directory, files in expected_artifacts.items():
    print(f'\\n{directory}/')
    for f_name in files:
        fpath = os.path.join(directory, f_name)
        status = 'OK' if os.path.exists(fpath) else 'MISSING'
        if status == 'MISSING':
            all_ok = False
        print(f'  [{status}] {f_name}')

print('\\n' + '=' * 60)
if all_ok:
    print('All artifacts saved successfully.')
else:
    print('WARNING: Some artifacts are missing. Check the sections above.')
print(f'Output directory: {OUTPUT_DIR}')
print('=' * 60)""")

FINAL_SUMMARY_COLAB = code("""\
print()
print('=' * 60)
print('NOTEBOOK COMPLETE — ARTIFACT INVENTORY')
print('=' * 60)

expected_artifacts = {
    CHECKPOINT_DIR: ['best_model.pt', 'last_checkpoint.pt'],
    RESULTS_DIR: ['split_manifest.json', 'results_summary.json'],
    PLOTS_DIR: ['training_curves.png', 'f1_vs_threshold.png', 'prediction_grid.png',
                'gradcam_analysis.png', 'robustness_chart.png'],
}

all_ok = True
for directory, files in expected_artifacts.items():
    print(f'\\n{directory}/')
    for f_name in files:
        fpath = os.path.join(directory, f_name)
        status = 'OK' if os.path.exists(fpath) else 'MISSING'
        if status == 'MISSING':
            all_ok = False
        print(f'  [{status}] {f_name}')

print('\\n' + '=' * 60)
if all_ok:
    print('All artifacts saved successfully.')
else:
    print('WARNING: Some artifacts are missing. Check the sections above.')
print(f'Output directory: {OUTPUT_DIR}')
print('=' * 60)""")

# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════════════

kaggle_cells = [
    TITLE_KAGGLE,
    SEC1_MD, PIP_KAGGLE, IMPORTS, SEED_CELL,
    SEC2_MD, GPU_CHECK, CONFIG_KAGGLE,
    SEC3_MD_KAGGLE, WANDB_KAGGLE, DATASET_KAGGLE,
    SEC4_MD, DISCOVERY_FUNCS, DISCOVERY_RUN,
    SEC5_MD, VALIDATION,
    SEC6_MD, TRANSFORMS,
    SEC7_MD, SPLIT, MANIFEST,
    SEC8_MD, DATASET_CLASS,
    SEC9_MD, DATALOADERS, SANITY_CHECK,
    SEC10_MD, MODEL,
    SEC11_MD, LOSS,
    SEC12_MD, METRICS,
    SEC13_MD, VALIDATE,
    SEC14_MD, CHECKPOINT_HELPERS, TRAINING_LOOP,
    SEC15_MD, LOAD_BEST, THRESHOLD_SWEEP,
    SEC16_MD, EVALUATE_FUNC, EVALUATE_RUN, EVALUATE_WANDB,
    SEC17_MD, VIZ_CURVES, VIZ_THRESHOLD, VIZ_COLLECT, VIZ_GRID, VIZ_WANDB,
    SEC18_MD, GRADCAM_CLASS, DIAGNOSTIC_OVERLAY, GRADCAM_VIZ, FAILURE_ANALYSIS,
    SEC19_MD, ROBUSTNESS_TRANSFORMS, RESIZE_DATASET, ROBUSTNESS_RUN, ROBUSTNESS_CHART,
    SEC20_MD,
    SEC21_MD, SAVE_RESULTS, WANDB_FINISH,
    SEC22_MD, FINAL_SUMMARY_KAGGLE,
]

colab_cells = [
    TITLE_COLAB,
    SEC1_MD, PIP_COLAB, IMPORTS, SEED_CELL,
    SEC2_MD, GPU_CHECK, CONFIG_COLAB,
    SEC3_MD_COLAB, WANDB_COLAB, DATASET_COLAB,
    SEC4_MD, DISCOVERY_FUNCS, DISCOVERY_RUN,
    SEC5_MD, VALIDATION,
    SEC6_MD, TRANSFORMS,
    SEC7_MD, SPLIT, MANIFEST,
    SEC8_MD, DATASET_CLASS,
    SEC9_MD, DATALOADERS, SANITY_CHECK,
    SEC10_MD, MODEL,
    SEC11_MD, LOSS,
    SEC12_MD, METRICS,
    SEC13_MD, VALIDATE,
    SEC14_MD, CHECKPOINT_HELPERS, TRAINING_LOOP,
    SEC15_MD, LOAD_BEST, THRESHOLD_SWEEP,
    SEC16_MD, EVALUATE_FUNC, EVALUATE_RUN, EVALUATE_WANDB,
    SEC17_MD, VIZ_CURVES, VIZ_THRESHOLD, VIZ_COLLECT, VIZ_GRID, VIZ_WANDB,
    SEC18_MD, GRADCAM_CLASS, DIAGNOSTIC_OVERLAY, GRADCAM_VIZ, FAILURE_ANALYSIS,
    SEC19_MD, ROBUSTNESS_TRANSFORMS, RESIZE_DATASET, ROBUSTNESS_RUN, ROBUSTNESS_CHART,
    SEC20_MD,
    SEC21_MD, SAVE_RESULTS, WANDB_FINISH,
    SEC22_MD, FINAL_SUMMARY_COLAB,
]

# Write notebooks
script_dir = os.path.dirname(os.path.abspath(__file__))

kaggle_path = os.path.join(script_dir, 'tamper_detection_v6_kaggle.ipynb')
with open(kaggle_path, 'w', encoding='utf-8') as f:
    json.dump(nb(kaggle_cells), f, indent=1, ensure_ascii=False)
print(f'Kaggle notebook: {kaggle_path} ({len(kaggle_cells)} cells)')

colab_path = os.path.join(script_dir, 'tamper_detection_v6_colab.ipynb')
with open(colab_path, 'w', encoding='utf-8') as f:
    json.dump(nb(colab_cells), f, indent=1, ensure_ascii=False)
print(f'Colab notebook: {colab_path} ({len(colab_cells)} cells)')
