#!/usr/bin/env python3
"""Generate v6.5 engineering-improved notebooks for Colab and Kaggle."""
import json, os

def md(text):
    lines = text.strip('\n').split('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)]}

def code(text):
    lines = text.strip('\n').split('\n')
    return {"cell_type": "code", "metadata": {}, "source": [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)], "outputs": [], "execution_count": None}

def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 5}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

TITLE_KAGGLE = md("""\
# Tamper Detection v6.5 (Kaggle) — Image Forgery Detection & Localization

> **v6.5** — Engineering-improved implementation with hardware abstraction,
> config-driven pipeline, optional multi-GPU support, and flag-controlled AMP.
> Same training pipeline and results as v6, with professional ML engineering practices.

## Engineering Improvements over v6

| Feature | v6 | v6.5 |
|---|---|---|
| Configuration | Flat dict, scattered flags | Central `CONFIG` with feature flags |
| Device setup | Inline code | `setup_device()` abstraction |
| Model init | Inline | `setup_model()` with optional DataParallel |
| Training loop | Monolithic cell | `train_one_epoch()` helper |
| Validation | Standalone function | `validate_model()` with configurable AMP |
| Multi-GPU | Not supported | Optional `DataParallel` via flag |
| AMP control | Always on | `CONFIG['use_amp']` flag |
| DataLoader | Hardcoded kwargs | Config-driven workers, pin_memory, persistent |

**Architecture:** SMP U-Net (ResNet34, ImageNet pretrained)
**Dataset:** CASIA Splicing Detection + Localization
**Loss:** BCE + Dice | **Optimizer:** AdamW (differential LR)
**Training:** AMP (configurable), gradient accumulation, early stopping
**Image Size:** 384 × 384 | **Split:** 70 / 15 / 15

## Notebook Sections

1. Project Overview
2. Environment Setup
3. Dataset Loading
4. Dataset Validation
5. Preprocessing
6. Model Architecture
7. Training Pipeline
8. Evaluation
9. Visualization
10. Explainable AI
11. Robustness Testing
12. Experiment Tracking
13. Save Artifacts""")

TITLE_COLAB = md("""\
# Tamper Detection v6.5 (Colab) — Image Forgery Detection & Localization

> **v6.5** — Engineering-improved implementation with hardware abstraction,
> config-driven pipeline, optional multi-GPU support, and flag-controlled AMP.
> Same training pipeline and results as v6, with professional ML engineering practices.

## Engineering Improvements over v6

| Feature | v6 | v6.5 |
|---|---|---|
| Configuration | Flat dict, scattered flags | Central `CONFIG` with feature flags |
| Device setup | Inline code | `setup_device()` abstraction |
| Model init | Inline | `setup_model()` with optional DataParallel |
| Training loop | Monolithic cell | `train_one_epoch()` helper |
| Validation | Standalone function | `validate_model()` with configurable AMP |
| Multi-GPU | Not supported | Optional `DataParallel` via flag |
| AMP control | Always on | `CONFIG['use_amp']` flag |
| DataLoader | Hardcoded kwargs | Config-driven workers, pin_memory, persistent |

**Architecture:** SMP U-Net (ResNet34, ImageNet pretrained)
**Dataset:** CASIA Splicing Detection + Localization
**Loss:** BCE + Dice | **Optimizer:** AdamW (differential LR)
**Training:** AMP (configurable), gradient accumulation, early stopping
**Image Size:** 384 × 384 | **Split:** 70 / 15 / 15

## Notebook Sections

1. Project Overview
2. Environment Setup
3. Dataset Loading
4. Dataset Validation
5. Preprocessing
6. Model Architecture
7. Training Pipeline
8. Evaluation
9. Visualization
10. Explainable AI
11. Robustness Testing
12. Experiment Tracking
13. Save Artifacts""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

SEC2_MD = md("""\
## 2. Environment Setup

Install dependencies, import libraries, configure the central `CONFIG` dict, and set up device abstraction.

**v6.5 changes:**
- All hyperparameters and feature flags live in a single `CONFIG` block
- `setup_device()` handles GPU detection, cuDNN benchmark, TF32, and multi-GPU info
- `CONFIG['use_amp']`, `CONFIG['use_multi_gpu']`, `CONFIG['use_wandb']` control optional features""")

PIP_KAGGLE = code("""\
!pip install -q segmentation-models-pytorch "albumentations>=1.3.1,<2.0" """)

PIP_COLAB = code("""\
!pip install -q "kaggle>=1.6,<1.7" opendatasets segmentation-models-pytorch "albumentations>=1.3.1,<2.0" """)

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

SEED_AND_CONFIG_KAGGLE = code("""\
# ── Central Configuration ─────────────────────────────────────────────────────
# All hyperparameters and feature flags in one place.
# Change values here — every downstream cell reads from CONFIG.

SEED = 42

CONFIG = {
    # ── Data ──
    'image_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'train_ratio': 0.70,

    # ── Model ──
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,

    # ── Optimizer ──
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,

    # ── Training ──
    'max_epochs': 50,
    'patience': 10,
    'accumulation_steps': 4,
    'max_grad_norm': 1.0,

    # ── Feature Flags ──
    'use_amp': True,          # Automatic mixed precision
    'use_multi_gpu': True,    # DataParallel when multiple GPUs available
    'use_wandb': False,       # Weights & Biases experiment tracking

    # ── Reproducibility ──
    'seed': SEED,
}

# ── Kaggle output directories ──
OUTPUT_DIR = '/kaggle/working'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

for d in [CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

print('CONFIG:')
for k, v in CONFIG.items():
    print(f'  {k}: {v}')
print(f'\\nEffective batch size: {CONFIG["batch_size"] * CONFIG["accumulation_steps"]}')
print(f'Output: {OUTPUT_DIR}')""")

SEED_AND_CONFIG_COLAB = code("""\
# ── Central Configuration ─────────────────────────────────────────────────────
# All hyperparameters and feature flags in one place.
# Change values here — every downstream cell reads from CONFIG.

SEED = 42

CONFIG = {
    # ── Data ──
    'image_size': 384,
    'batch_size': 4,
    'num_workers': 2,
    'train_ratio': 0.70,

    # ── Model ──
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,

    # ── Optimizer ──
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,

    # ── Training ──
    'max_epochs': 50,
    'patience': 10,
    'accumulation_steps': 4,
    'max_grad_norm': 1.0,

    # ── Feature Flags ──
    'use_amp': True,          # Automatic mixed precision
    'use_multi_gpu': True,    # DataParallel when multiple GPUs available
    'use_wandb': False,       # Weights & Biases experiment tracking

    # ── Reproducibility ──
    'seed': SEED,
}

# ── Google Drive output directories ──
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

print('CONFIG:')
for k, v in CONFIG.items():
    print(f'  {k}: {v}')
print(f'\\nEffective batch size: {CONFIG["batch_size"] * CONFIG["accumulation_steps"]}')
print(f'Output: {OUTPUT_DIR}')""")

SETUP_DEVICE = code("""\
# ── Device Abstraction ────────────────────────────────────────────────────────

def set_seed(seed):
    \"\"\"Set seed for full reproducibility across all libraries.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(config):
    \"\"\"Detect hardware, enable optimizations, and return the training device.

    Prints GPU name, VRAM, multi-GPU availability, and optimization flags.
    Enables cuDNN benchmark and TF32 when a CUDA device is available.

    Returns:
        torch.device: The selected training device.
    \"\"\"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        n_gpus = torch.cuda.device_count()

        # Performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f'GPU:              {gpu_name}')
        print(f'VRAM:             {vram_gb:.1f} GB')
        print(f'GPUs available:   {n_gpus}')
        print(f'cuDNN benchmark:  enabled')
        print(f'TF32:             enabled')
        print(f'AMP:              {"enabled" if config["use_amp"] else "disabled"}')
        print(f'Multi-GPU:        {"enabled" if config["use_multi_gpu"] and n_gpus > 1 else "disabled"}')
    else:
        print('WARNING: No GPU detected. Training will be extremely slow.')

    print(f'Device:           {device}')
    return device

set_seed(SEED)
device = setup_device(CONFIG)""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

SEC3_MD_KAGGLE = md("""\
## 3. Dataset Loading

The dataset is pre-mounted by Kaggle at `/kaggle/input/`.

**Dataset:** `sagnikkayalcse52/casia-spicing-detection-localization`
**Structure:** `Image/Au/`, `Image/Tp/` (images), `Mask/Au/`, `Mask/Tp/` (ground-truth masks)""")

SEC3_MD_COLAB = md("""\
## 3. Dataset Loading

Download the CASIA dataset from Kaggle using the Kaggle API.
Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` set via Colab Secrets.

**Dataset:** `sagnikkayalcse52/casia-spicing-detection-localization`
**Structure:** `Image/Au/`, `Image/Tp/` (images), `Mask/Au/`, `Mask/Tp/` (ground-truth masks)""")

WANDB_KAGGLE = code("""\
# ── W&B Experiment Tracking (controlled by CONFIG) ────────────────────────────

if CONFIG['use_wandb']:
    !pip install -q wandb
    import wandb
    from kaggle_secrets import UserSecretsClient
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    wandb.init(
        project='v6.5 Tampered Image Detection & Localization',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}-kaggle-v6.5',
        tags=['v6.5', 'casia-v2', 'kaggle'],
    )
    print('W&B initialized.')
else:
    print('W&B disabled (CONFIG[\\'use_wandb\\'] = False). Local artifacts only.')""")

WANDB_COLAB = code("""\
# ── W&B Experiment Tracking (controlled by CONFIG) ────────────────────────────

if CONFIG['use_wandb']:
    !pip install -q wandb
    import wandb
    try:
        from google.colab import userdata
        wandb.login(key=userdata.get('WANDB_API_KEY'))
    except Exception:
        wandb.login()  # Interactive fallback
    wandb.init(
        project='v6.5 Tampered Image Detection & Localization',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}-colab-v6.5',
        tags=['v6.5', 'casia-v2', 'colab'],
    )
    print('W&B initialized.')
else:
    print('W&B disabled (CONFIG[\\'use_wandb\\'] = False). Local artifacts only.')""")

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
# Store your credentials in Colab Secrets (key icon in left sidebar):
#   Secret name: KAGGLE_USERNAME  -> value: your kaggle username
#   Secret name: KAGGLE_KEY       -> value: your kaggle API key
# Alternatively, you can use any secret names and update the code below.

_kaggle_creds_loaded = False
try:
    from google.colab import userdata
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    _kaggle_creds_loaded = True
    print('Kaggle credentials loaded from Colab Secrets.')
except Exception:
    pass

if not _kaggle_creds_loaded:
    # Fallback: write kaggle.json directly
    import getpass
    print('Colab Secrets not found. Enter Kaggle credentials manually:')
    _user = input('Kaggle username: ')
    _key = getpass.getpass('Kaggle API key: ')
    os.environ['KAGGLE_USERNAME'] = _user
    os.environ['KAGGLE_KEY'] = _key
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    import json as _json
    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as _f:
        _json.dump({'username': _user, 'key': _key}, _f)
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
    print('Kaggle credentials saved to ~/.kaggle/kaggle.json')

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
    # Method 1: kaggle CLI
    try:
        !kaggle datasets download -d {DATASET_SLUG} -p {DOWNLOAD_DIR} --unzip --force
        DATASET_ROOT, IMAGE_DIR_NAME, MASK_DIR_NAME = find_dataset_root(DOWNLOAD_DIR)
    except Exception as _e:
        print(f'kaggle CLI failed: {_e}')

    # Method 2: opendatasets fallback (handles kaggle package bugs)
    if DATASET_ROOT is None:
        print('Falling back to opendatasets...')
        import opendatasets as od
        od.download(f'https://www.kaggle.com/datasets/{DATASET_SLUG}', data_dir=DOWNLOAD_DIR)
        DATASET_ROOT, IMAGE_DIR_NAME, MASK_DIR_NAME = find_dataset_root(DOWNLOAD_DIR)

if DATASET_ROOT is None:
    raise FileNotFoundError(
        f'Could not find dataset with IMAGE/ and MASK/ directories under {DOWNLOAD_DIR}. '
        'Check Kaggle API credentials and dataset slug.'
    )

print(f'Dataset root:  {DATASET_ROOT}')
print(f'Image dir:     {IMAGE_DIR_NAME}')
print(f'Mask dir:      {MASK_DIR_NAME}')

for sub in ['Au', 'Tp']:
    for parent in [IMAGE_DIR_NAME, MASK_DIR_NAME]:
        path = os.path.join(DATASET_ROOT, parent, sub)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f'  {parent}/{sub}: {count} files')
        else:
            print(f'  {parent}/{sub}: NOT FOUND')""")

DISCOVERY_FUNCS = code("""\
# ── Dataset Discovery ─────────────────────────────────────────────────────────

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

            if not is_valid_image(img_path):
                excluded.append((img_name, 'corrupt_image'))
                continue

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

            valid, reason = validate_dimensions(img_path, mask_path)
            if not valid:
                excluded.append((img_name, reason))
                continue

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
pairs, excluded = discover_pairs(DATASET_ROOT, IMAGE_DIR_NAME, MASK_DIR_NAME)

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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DATASET VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

SEC4_MD = md("""\
## 4. Dataset Validation

Verify discovered pairs, check class distributions, run sample load checks,
and perform stratified splitting with data leakage verification.""")

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

SPLIT = code("""\
# ── Stratified Split: 70/15/15 ────────────────────────────────────────────────

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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

SEC5_MD = md("""\
## 5. Preprocessing

**Transforms** (config-driven image size):
- Train: `Resize` → `HFlip` → `VFlip` → `RandomRotate90` → `Normalize` → `ToTensorV2`
- Val/Test: `Resize` → `Normalize` → `ToTensorV2`

**Dataset class:** Loads RGB images, binarizes masks (`> 0`), generates zero masks for authentic images.

**DataLoaders:** Config-driven `num_workers`, `pin_memory`, `persistent_workers`.""")

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

print(f'Train: resize {CONFIG["image_size"]}, flip, rotate, normalize')
print(f'Val/Test: resize {CONFIG["image_size"]}, normalize')""")

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

        image = cv2.imread(entry['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if entry['mask_path'] is not None:
            mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).float()

        label = torch.tensor(entry['label'], dtype=torch.float32)
        return image, mask, label

print('TamperingDataset defined.')""")

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

# Config-driven DataLoader kwargs
_nw = CONFIG['num_workers']
loader_kwargs = dict(
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=_nw > 0,
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
print(f'Test batches:  {len(test_loader)}')
print(f'num_workers={_nw}, pin_memory={loader_kwargs["pin_memory"]}, persistent_workers={loader_kwargs["persistent_workers"]}')""")

SANITY_CHECK = code("""\
# Sanity check: visualize one training batch
images, masks, labels = next(iter(train_loader))
print(f'Image batch: {images.shape}  Mask batch: {masks.shape}  Labels: {labels}')

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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

SEC6_MD = md("""\
## 6. Model Architecture

**SMP U-Net** with ResNet34 encoder (ImageNet pretrained). ~24M parameters.

**v6.5 changes:**
- `setup_model()` encapsulates model creation, optional `DataParallel` wrapping, and shape verification
- When `CONFIG['use_multi_gpu'] = True` and multiple GPUs are detected, the model is automatically wrapped in `torch.nn.DataParallel`
- Loss, optimizer, and scaler creation remain unchanged from v6""")

SETUP_MODEL_FUNC = code("""\
def setup_model(config, device):
    \"\"\"Create model, optionally wrap in DataParallel, and verify output shape.

    Args:
        config: CONFIG dict with model and multi-GPU settings.
        device: torch.device for model placement.

    Returns:
        model: The (optionally wrapped) model on the target device.
        is_parallel: Whether DataParallel is active.
    \"\"\"
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=config['encoder_weights'],
        in_channels=config['in_channels'],
        classes=config['classes'],
        activation=None,  # Raw logits
    )
    model = model.to(device)

    # Optional multi-GPU
    is_parallel = False
    if (torch.cuda.device_count() > 1 and config['use_multi_gpu']):
        model = torch.nn.DataParallel(model)
        is_parallel = True
        print(f'DataParallel enabled across {torch.cuda.device_count()} GPUs.')
    else:
        print('Single-device training.')

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters:     {total_params:,}')
    print(f'Trainable parameters: {trainable:,}')

    # Shape verification
    with torch.no_grad():
        dummy = torch.randn(1, 3, config['image_size'], config['image_size']).to(device)
        out = model(dummy)
        assert out.shape == (1, 1, config['image_size'], config['image_size']), \\
            f'Unexpected output shape: {out.shape}'
    print(f'Shape check passed: (1, 3, {config["image_size"]}, {config["image_size"]}) -> {out.shape}')

    return model, is_parallel

model, is_parallel = setup_model(CONFIG, device)""")

LOSS_OPTIM = code("""\
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

# Access underlying model for parameter groups (handles DataParallel)
base_model = model.module if is_parallel else model

optimizer = torch.optim.AdamW([
    {'params': base_model.encoder.parameters(), 'lr': CONFIG['encoder_lr']},
    {'params': base_model.decoder.parameters(), 'lr': CONFIG['decoder_lr']},
    {'params': base_model.segmentation_head.parameters(), 'lr': CONFIG['decoder_lr']},
], weight_decay=CONFIG['weight_decay'])

# Mixed precision scaler (enabled/disabled by CONFIG)
scaler = GradScaler('cuda', enabled=CONFIG['use_amp'])

print(f'Loss: BCEDiceLoss')
print(f'Optimizer: AdamW (encoder_lr={CONFIG["encoder_lr"]}, decoder_lr={CONFIG["decoder_lr"]})')
print(f'AMP scaler enabled: {CONFIG["use_amp"]}')""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

SEC7_MD = md("""\
## 7. Training Pipeline

**v6.5 changes:**
- Extracted `train_one_epoch()` and `validate_model()` helper functions for cleaner structure
- AMP controlled by `CONFIG['use_amp']` — training works correctly with AMP on or off
- `GradScaler` disabled gracefully when `use_amp = False`
- Checkpoint save/load handles both plain and DataParallel models

Core training logic is **identical** to v6: gradient accumulation (effective batch = 16),
early stopping on val Pixel-F1 (patience = 10), gradient clipping (max_norm = 1.0).""")

METRICS = code("""\
# ── Evaluation Metrics ────────────────────────────────────────────────────────

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
    Returns (1.0, 1.0) for true negatives (both empty).
    \"\"\"
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0, 1.0
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision.item(), recall.item()

print('Metric functions defined.')""")

TRAIN_VALIDATE_HELPERS = code("""\
# ── Training & Validation Helpers ─────────────────────────────────────────────

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config):
    \"\"\"Run one training epoch with gradient accumulation and optional AMP.

    Args:
        model: The model (plain or DataParallel).
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler (enabled or disabled based on config).
        device: Training device.
        config: CONFIG dict.

    Returns:
        avg_loss: Mean training loss for the epoch.
    \"\"\"
    model.train()
    running_loss = 0.0
    accum_steps = config['accumulation_steps']
    use_amp = config['use_amp']
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for batch_idx, (images, masks, labels) in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{running_loss / (batch_idx + 1):.4f}'})

    # Partial window flush
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = running_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def validate_model(model, val_loader, criterion, device, config, threshold=0.5):
    \"\"\"Run validation with optional AMP and return loss, mean F1, mean IoU.

    Args:
        model: The model (plain or DataParallel).
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Training device.
        config: CONFIG dict.
        threshold: Binarization threshold for metrics.

    Returns:
        avg_loss, avg_f1, avg_iou
    \"\"\"
    model.eval()
    total_loss = 0.0
    f1_scores = []
    iou_scores = []
    use_amp = config['use_amp']

    for images, masks, labels in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        with autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for i in range(images.size(0)):
            f1_scores.append(compute_pixel_f1(preds[i], masks[i]))
            iou_scores.append(compute_iou(preds[i], masks[i]))

    avg_loss = total_loss / len(val_loader.dataset)
    avg_f1 = np.mean(f1_scores)
    avg_iou = np.mean(iou_scores)
    return avg_loss, avg_f1, avg_iou

print('train_one_epoch() and validate_model() defined.')""")

CHECKPOINT_HELPERS = code("""\
def save_checkpoint(state, filepath):
    \"\"\"Save training checkpoint.\"\"\"
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer, scaler, device, is_parallel):
    \"\"\"Load training checkpoint, handling DataParallel state dicts.\"\"\"
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    # Handle DataParallel prefix mismatch
    if is_parallel and not any(k.startswith('module.') for k in state_dict):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_parallel and any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['epoch'] + 1, ckpt['best_f1'], ckpt['best_epoch']

print('Checkpoint helpers defined.')""")

TRAINING_LOOP = code("""\
# ── Training Loop ─────────────────────────────────────────────────────────────

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
        resume_path, model, optimizer, scaler, device, is_parallel
    )
    print(f'Resumed from epoch {start_epoch}, best F1={best_f1:.4f} at epoch {best_epoch + 1}')

for epoch in range(start_epoch, CONFIG['max_epochs']):
    print(f'\\nEpoch {epoch + 1}/{CONFIG["max_epochs"]}')

    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, CONFIG)

    # Validate
    val_loss, val_f1, val_iou = validate_model(model, val_loader, criterion, device, CONFIG)

    # Record
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['val_iou'].append(val_iou)

    print(
        f'  train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
        f'val_f1={val_f1:.4f}, val_iou={val_iou:.4f}'
    )

    # W&B logging
    if CONFIG['use_wandb']:
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/pixel_f1': val_f1,
            'val/pixel_iou': val_iou,
            'train/lr_encoder': optimizer.param_groups[0]['lr'],
            'train/lr_decoder': optimizer.param_groups[1]['lr'],
        })

    # Checkpoint state (always save unwrapped state_dict for portability)
    model_state = model.module.state_dict() if is_parallel else model.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }

    save_checkpoint(state, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        state['best_f1'] = best_f1
        state['best_epoch'] = best_epoch
        save_checkpoint(state, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        print(f'  -> New best model saved (F1={best_f1:.4f})')

    if (epoch + 1) % 10 == 0:
        save_checkpoint(state, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pt'))

    if epoch - best_epoch >= CONFIG['patience']:
        print(f'Early stopping at epoch {epoch + 1}. Best F1={best_f1:.4f} at epoch {best_epoch + 1}')
        break

print(f'\\nTraining complete. Best F1={best_f1:.4f} at epoch {best_epoch + 1}')""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

SEC8_MD = md("""\
## 8. Evaluation

1. **Threshold selection** — Sweep 50 thresholds on validation set
2. **Test evaluation** — Mixed-set, tampered-only, forgery-type breakdown, image-level metrics""")

LOAD_BEST = code("""\
# Load best model for evaluation
best_ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)

# Load into unwrapped model, then re-wrap if needed
base_model = model.module if is_parallel else model
base_model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded best model from epoch {ckpt["best_epoch"] + 1} (F1={ckpt["best_f1"]:.4f})')""")

THRESHOLD_SWEEP = code("""\
@torch.no_grad()
def find_best_threshold(model, val_loader, device, config, num_thresholds=50):
    \"\"\"Sweep thresholds on validation set to maximize mean Pixel-F1.\"\"\"
    model.eval()
    thresholds = np.linspace(0.1, 0.9, num_thresholds)

    all_probs = []
    all_masks = []
    for images, masks, labels in tqdm(val_loader, desc='Collecting val predictions'):
        images = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_masks.append(masks)

    all_probs = torch.cat(all_probs)
    all_masks = torch.cat(all_masks)

    results = []
    for t in tqdm(thresholds, desc='Threshold sweep'):
        f1_scores = []
        preds = (all_probs > t).float()
        for i in range(len(all_probs)):
            f1_scores.append(compute_pixel_f1(preds[i], all_masks[i]))
        results.append((t, np.mean(f1_scores)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0], results

best_threshold, threshold_results = find_best_threshold(model, val_loader, device, CONFIG)
print(f'Best threshold: {best_threshold:.4f}')
print(f'Best val F1 at threshold: {threshold_results[0][1]:.4f}')""")

EVALUATE_FUNC = code("""\
@torch.no_grad()
def evaluate(model, test_loader, test_pairs, device, config, threshold):
    \"\"\"Full test evaluation with mixed, tampered-only, per-forgery, and image-level metrics.\"\"\"
    model.eval()

    all_f1, all_iou, all_precision, all_recall = [], [], [], []
    tampered_f1, tampered_iou = [], []
    image_preds, image_labels, image_scores = [], [], []
    forgery_f1 = {'splicing': [], 'copy-move': []}

    idx = 0
    for images, masks, labels in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
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

print('evaluate() defined.')""")

EVALUATE_RUN = code("""\
test_results = evaluate(model, test_loader, test_pairs, device, CONFIG, best_threshold)

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
if CONFIG['use_wandb']:
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

SEC9_MD = md("""\
## 9. Visualization

1. **Training curves** — loss, F1, and IoU over epochs
2. **F1-vs-threshold** — threshold sweep visualization
3. **Prediction grid** — Original | GT Mask | Predicted Mask | Overlay""")

VIZ_CURVES = code("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs_range, history['train_loss'], label='Train')
axes[0].plot(epochs_range, history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history['val_f1'], color='green')
axes[1].axvline(x=best_epoch + 1, color='red', linestyle='--',
                label=f'Best (epoch {best_epoch + 1})')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Pixel-F1')
axes[1].set_title('Validation Pixel-F1')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history['val_iou'], color='orange')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Pixel-IoU')
axes[2].set_title('Validation Pixel-IoU')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Training curves saved.')""")

VIZ_THRESHOLD = code("""\
thresh_vals = [r[0] for r in threshold_results]
f1_vals = [r[1] for r in threshold_results]
sorted_pairs_t = sorted(zip(thresh_vals, f1_vals))

plt.figure(figsize=(8, 5))
plt.plot([p[0] for p in sorted_pairs_t], [p[1] for p in sorted_pairs_t], 'b-', linewidth=2)
plt.axvline(x=best_threshold, color='red', linestyle='--',
            label=f'Best: {best_threshold:.3f} (F1={threshold_results[0][1]:.4f})')
plt.xlabel('Threshold'); plt.ylabel('Mean Pixel-F1')
plt.title('F1 vs. Threshold (Validation Set)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'f1_vs_threshold.png'), dpi=150, bbox_inches='tight')
plt.show()
print('F1 vs threshold plot saved.')""")

VIZ_COLLECT = code("""\
@torch.no_grad()
def collect_predictions(model, test_loader, test_pairs, device, config, threshold):
    \"\"\"Collect all test predictions with metadata for visualization.\"\"\"
    model.eval()
    predictions = []
    idx = 0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for images, masks, labels in test_loader:
        images_dev = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            logits = model(images_dev)
        probs = torch.sigmoid(logits).cpu()

        for i in range(images.size(0)):
            pred_mask = (probs[i] > threshold).float()
            f1 = compute_pixel_f1(pred_mask, masks[i])
            img_np = images[i].permute(1, 2, 0).numpy() * std + mean
            img_np = np.clip(img_np, 0, 1)

            predictions.append({
                'image': img_np,
                'gt_mask': masks[i].squeeze().numpy(),
                'pred_mask': pred_mask.squeeze().numpy(),
                'prob_map': probs[i].squeeze().numpy(),
                'pixel_f1': f1,
                'label': labels[i].item(),
                'forgery_type': test_pairs[idx]['forgery_type'] if idx < len(test_pairs) else 'unknown',
                'gt_mask_area': masks[i].sum().item() / masks[i].numel(),
            })
            idx += 1

    return predictions

predictions = collect_predictions(model, test_loader, test_pairs, device, CONFIG, best_threshold)
print(f'Collected {len(predictions)} predictions.')""")

VIZ_GRID = code("""\
tampered_preds = [p for p in predictions if p['label'] == 1.0]
authentic_preds = [p for p in predictions if p['label'] == 0.0]
tampered_sorted = sorted(tampered_preds, key=lambda p: p['pixel_f1'])

samples = []
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[-2:])
mid = len(tampered_sorted) // 2
if len(tampered_sorted) >= 4:
    samples.extend(tampered_sorted[mid - 1:mid + 1])
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[:2])
if len(authentic_preds) >= 2:
    samples.extend(authentic_preds[:2])

n_rows = len(samples)
if n_rows == 0:
    print('No samples available for prediction grid.')
else:
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        img, gt, pred = sample['image'], sample['gt_mask'], sample['pred_mask']
        overlay = img.copy()
        pred_bool = pred > 0
        if pred_bool.any():
            overlay[pred_bool] = overlay[pred_bool] * 0.6 + np.array([1, 0, 0]) * 0.4

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'{sample["forgery_type"]} (F1={sample["pixel_f1"]:.3f})')
        axes[row, 0].axis('off')
        axes[row, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title('Ground Truth'); axes[row, 1].axis('off')
        axes[row, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title('Predicted Mask'); axes[row, 2].axis('off')
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('Overlay'); axes[row, 3].axis('off')

    plt.suptitle('Prediction Grid: Best / Median / Worst / Authentic', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'prediction_grid.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print('Prediction grid saved.')""")

VIZ_WANDB = code("""\
if CONFIG['use_wandb']:
    for plot_name in ['prediction_grid.png', 'training_curves.png', 'f1_vs_threshold.png']:
        plot_path = os.path.join(PLOTS_DIR, plot_name)
        if os.path.exists(plot_path):
            wandb.log({plot_name.replace('.png', ''): wandb.Image(plot_path)})
    print('Visualizations logged to W&B.')""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════════════════════════

SEC10_MD = md("""\
## 10. Explainable AI

- **Grad-CAM** — Spatial attention heatmaps from encoder layer4
- **Diagnostic overlays** — TP (green), FP (red), FN (blue) colour coding
- **Failure case analysis** — Worst-prediction breakdown

Grad-CAM hooks target the **base model** (unwrapped from DataParallel if needed).""")

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
# Grad-CAM hooks must target the base (unwrapped) model
_base = model.module if is_parallel else model
grad_cam = GradCAM(_base, _base.encoder.layer4)

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
        axes[row, 2].set_title('Ground Truth'); axes[row, 2].axis('off')
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: ROBUSTNESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

SEC11_MD = md("""\
## 11. Robustness Testing

Evaluate under degradation conditions: JPEG compression, Gaussian noise, blur, resize.
Uses the same validation-selected threshold for all conditions.""")

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
def run_robustness_eval(model, loader, device, config, threshold):
    \"\"\"Robustness evaluation loop with optional AMP.\"\"\"
    model.eval()
    f1_scores = []
    for images, masks, labels in loader:
        images = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            logits = model(images)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > threshold).float()
        for i in range(images.size(0)):
            f1_scores.append(compute_pixel_f1(preds[i], masks[i]))
    return f1_scores


def evaluate_robustness(model, test_pairs, device, config, threshold):
    \"\"\"Evaluate model under all degradation conditions.\"\"\"
    results = {}

    for name, transform in tqdm(robustness_transforms.items(), desc='Robustness tests'):
        dataset = TamperingDataset(test_pairs, transform=transform)
        loader = DataLoader(
            dataset, batch_size=config['batch_size'],
            shuffle=False, num_workers=config['num_workers']
        )
        f1_scores = run_robustness_eval(model, loader, device, config, threshold)
        results[name] = {
            'f1_mean': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
        }

    for scale in [0.75, 0.5]:
        name = f'resize_{scale}x'
        dataset = ResizeDegradationDataset(test_pairs, scale_factor=scale)
        loader = DataLoader(
            dataset, batch_size=config['batch_size'],
            shuffle=False, num_workers=config['num_workers']
        )
        f1_scores = run_robustness_eval(model, loader, device, config, threshold)
        results[name] = {
            'f1_mean': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
        }

    return results

robustness_results = evaluate_robustness(model, test_pairs, device, CONFIG, best_threshold)

print(f'\\nRobustness Results (threshold={best_threshold:.4f}):')
print(f'{"":<25} {"Pixel-F1 (mean +/- std)":<25} {"Delta from clean":<15}')
print('-' * 65)
clean_f1 = robustness_results.get('clean', {}).get('f1_mean', 0)
for name, data in robustness_results.items():
    delta = data['f1_mean'] - clean_f1 if name != 'clean' else 0.0
    delta_str = f'{delta:+.4f}' if name != 'clean' else '---'
    print(f'{name:<25} {data["f1_mean"]:.4f} +/- {data["f1_std"]:.4f}      {delta_str}')""")

ROBUSTNESS_CHART = code("""\
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

if CONFIG['use_wandb']:
    wandb.log({'robustness_chart': wandb.Image(os.path.join(PLOTS_DIR, 'robustness_chart.png'))})
    for name, data in robustness_results.items():
        wandb.log({f'robustness/{name}_f1': data['f1_mean']})
    print('Robustness results logged to W&B.')""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: EXPERIMENT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

SEC12_MD = md("""\
## 12. Experiment Tracking

W&B tracking is integrated throughout, controlled by `CONFIG['use_wandb']`:

| Section | W&B Action |
|---|---|
| 3 (Dataset Loading) | Init with CONFIG |
| 7 (Training Pipeline) | Per-epoch metrics |
| 8 (Evaluation) | Test results summary |
| 9 (Visualization) | Plot images |
| 11 (Robustness) | Per-degradation F1 |
| 13 (Save Artifacts) | Model artifact, `wandb.finish()` |

When `use_wandb = False` (default), the notebook is fully self-contained with local artifacts only.""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

SEC13_MD = md("""\
## 13. Save Artifacts

Save all results, config, and upload model artifact to W&B if enabled.""")

SAVE_RESULTS = code("""\
results_summary = {
    'version': 'v6.5',
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
    'engineering_flags': {
        'use_amp': CONFIG['use_amp'],
        'use_multi_gpu': CONFIG['use_multi_gpu'],
        'is_parallel': is_parallel,
    },
}

results_path = os.path.join(RESULTS_DIR, 'results_summary.json')
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f'Results summary saved to: {results_path}')""")

WANDB_FINISH = code("""\
if CONFIG['use_wandb']:
    artifact = wandb.Artifact('best-model-v6.5', type='model')
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        print('Model artifact uploaded to W&B.')
    wandb.finish()
    print('W&B run finished.')""")

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINEERING IMPROVEMENTS SECTION + FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

ENGINEERING_MD = md("""\
---

## Engineering Improvements in v6.5

This section summarizes the engineering practices introduced in v6.5 and explains why they are
standard in professional ML systems.

### 1. Central Configuration System

All hyperparameters and feature flags live in a single `CONFIG` dict at the top of the notebook.
This eliminates magic numbers scattered throughout the code and makes it trivial to:
- Reproduce experiments with different settings
- Track what changed between runs
- Pass the full config to experiment trackers like W&B

### 2. Hardware Abstraction (`setup_device()`)

A dedicated function detects available hardware, enables GPU optimizations (cuDNN benchmark, TF32),
and reports device capabilities. This decouples infrastructure concerns from ML logic and makes
the notebook portable across different GPU environments without code changes.

### 3. Optional Multi-GPU Support (`DataParallel`)

When `CONFIG['use_multi_gpu'] = True` and multiple GPUs are detected, the model is automatically
wrapped in `torch.nn.DataParallel`. This provides data-parallel training without modifying the
training loop. Checkpoint save/load handles the `module.` prefix transparently.

**Why DataParallel over DDP?** In a notebook context, DataParallel is simpler — it runs in a
single process and doesn't require `torch.distributed.launch`. For production multi-node training,
DistributedDataParallel (DDP) would be preferred.

### 4. Flag-Controlled AMP (`CONFIG['use_amp']`)

Mixed precision training via `torch.amp.autocast` and `GradScaler` is controlled by a single
flag. The `GradScaler` is initialized with `enabled=CONFIG['use_amp']`, so when AMP is disabled,
scaling operations are no-ops. This lets users disable AMP for debugging numerical issues without
changing any training code.

### 5. Config-Driven DataLoaders

`num_workers`, `pin_memory`, and `persistent_workers` are derived from `CONFIG` and hardware
detection rather than being hardcoded. This ensures DataLoaders are optimally configured for the
runtime environment.

### 6. Modular Helper Functions

- `setup_device()` — Hardware detection and optimization
- `setup_model()` — Model creation, optional DataParallel, shape verification
- `train_one_epoch()` — Single epoch training with AMP and gradient accumulation
- `validate_model()` — Validation with configurable AMP

These functions improve readability, testability, and reuse. Each has a clear signature,
docstring, and single responsibility.""")

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
    print('WARNING: Some artifacts are missing.')
print(f'Output directory: {OUTPUT_DIR}')
print(f'Multi-GPU used:   {is_parallel}')
print(f'AMP used:         {CONFIG["use_amp"]}')
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
    print('WARNING: Some artifacts are missing.')
print(f'Output directory: {OUTPUT_DIR}')
print(f'Multi-GPU used:   {is_parallel}')
print(f'AMP used:         {CONFIG["use_amp"]}')
print('=' * 60)""")

# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════════════

kaggle_cells = [
    TITLE_KAGGLE,
    # 2. Environment Setup
    SEC2_MD, PIP_KAGGLE, IMPORTS, SEED_AND_CONFIG_KAGGLE, SETUP_DEVICE,
    # 3. Dataset Loading
    SEC3_MD_KAGGLE, WANDB_KAGGLE, DATASET_KAGGLE, DISCOVERY_FUNCS, DISCOVERY_RUN,
    # 4. Dataset Validation
    SEC4_MD, VALIDATION, SPLIT, MANIFEST,
    # 5. Preprocessing
    SEC5_MD, TRANSFORMS, DATASET_CLASS, DATALOADERS, SANITY_CHECK,
    # 6. Model Architecture
    SEC6_MD, SETUP_MODEL_FUNC, LOSS_OPTIM,
    # 7. Training Pipeline
    SEC7_MD, METRICS, TRAIN_VALIDATE_HELPERS, CHECKPOINT_HELPERS, TRAINING_LOOP,
    # 8. Evaluation
    SEC8_MD, LOAD_BEST, THRESHOLD_SWEEP, EVALUATE_FUNC, EVALUATE_RUN, EVALUATE_WANDB,
    # 9. Visualization
    SEC9_MD, VIZ_CURVES, VIZ_THRESHOLD, VIZ_COLLECT, VIZ_GRID, VIZ_WANDB,
    # 10. Explainable AI
    SEC10_MD, GRADCAM_CLASS, DIAGNOSTIC_OVERLAY, GRADCAM_VIZ, FAILURE_ANALYSIS,
    # 11. Robustness Testing
    SEC11_MD, ROBUSTNESS_TRANSFORMS, RESIZE_DATASET, ROBUSTNESS_RUN, ROBUSTNESS_CHART,
    # 12. Experiment Tracking
    SEC12_MD,
    # 13. Save Artifacts
    SEC13_MD, SAVE_RESULTS, WANDB_FINISH,
    # Engineering summary + Final
    ENGINEERING_MD, FINAL_SUMMARY_KAGGLE,
]

colab_cells = [
    TITLE_COLAB,
    # 2. Environment Setup
    SEC2_MD, PIP_COLAB, IMPORTS, SEED_AND_CONFIG_COLAB, SETUP_DEVICE,
    # 3. Dataset Loading
    SEC3_MD_COLAB, WANDB_COLAB, DATASET_COLAB, DISCOVERY_FUNCS, DISCOVERY_RUN,
    # 4. Dataset Validation
    SEC4_MD, VALIDATION, SPLIT, MANIFEST,
    # 5. Preprocessing
    SEC5_MD, TRANSFORMS, DATASET_CLASS, DATALOADERS, SANITY_CHECK,
    # 6. Model Architecture
    SEC6_MD, SETUP_MODEL_FUNC, LOSS_OPTIM,
    # 7. Training Pipeline
    SEC7_MD, METRICS, TRAIN_VALIDATE_HELPERS, CHECKPOINT_HELPERS, TRAINING_LOOP,
    # 8. Evaluation
    SEC8_MD, LOAD_BEST, THRESHOLD_SWEEP, EVALUATE_FUNC, EVALUATE_RUN, EVALUATE_WANDB,
    # 9. Visualization
    SEC9_MD, VIZ_CURVES, VIZ_THRESHOLD, VIZ_COLLECT, VIZ_GRID, VIZ_WANDB,
    # 10. Explainable AI
    SEC10_MD, GRADCAM_CLASS, DIAGNOSTIC_OVERLAY, GRADCAM_VIZ, FAILURE_ANALYSIS,
    # 11. Robustness Testing
    SEC11_MD, ROBUSTNESS_TRANSFORMS, RESIZE_DATASET, ROBUSTNESS_RUN, ROBUSTNESS_CHART,
    # 12. Experiment Tracking
    SEC12_MD,
    # 13. Save Artifacts
    SEC13_MD, SAVE_RESULTS, WANDB_FINISH,
    # Engineering summary + Final
    ENGINEERING_MD, FINAL_SUMMARY_COLAB,
]

# Write notebooks
script_dir = os.path.dirname(os.path.abspath(__file__))

kaggle_path = os.path.join(script_dir, 'tamper_detection_v6.5_kaggle.ipynb')
with open(kaggle_path, 'w', encoding='utf-8') as f:
    json.dump(nb(kaggle_cells), f, indent=1, ensure_ascii=False)
print(f'Kaggle notebook: {kaggle_path} ({len(kaggle_cells)} cells)')

colab_path = os.path.join(script_dir, 'tamper_detection_v6.5_colab.ipynb')
with open(colab_path, 'w', encoding='utf-8') as f:
    json.dump(nb(colab_cells), f, indent=1, ensure_ascii=False)
print(f'Colab notebook: {colab_path} ({len(colab_cells)} cells)')
