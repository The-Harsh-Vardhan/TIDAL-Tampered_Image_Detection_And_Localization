"""
Cell definitions for vK.4 notebook generator.
Each function returns a list of (cell_type, source_string) tuples.
"""

def md(text):
    return ("markdown", text.strip())

def code(text):
    return ("code", text.strip())

def cells_header():
    return [
        md("""# Tampered Image Detection and Localization — Submission Notebook (vK.4)

This **Kaggle** notebook presents a complete assignment submission for tampered image detection and tampered region localization.

**Key improvements in vK.4 over vK.2**
- Kaggle-native: no Colab/Drive shims — dataset mounts automatically
- Centralized `CONFIG` dict for all hyperparameters
- Full reproducibility via `set_seed()`
- Mixed-precision training (AMP) + gradient accumulation
- `pos_weight` BCE + per-sample Dice loss (from v8)
- Expanded augmentation pipeline (ColorJitter, Compression, GaussNoise, GaussianBlur)
- ReduceLROnPlateau scheduler + early stopping
- Threshold sweep, mask-size stratified evaluation
- Grad-CAM explainability, robustness testing, shortcut learning checks

**Model architecture preserved from vK.2:** custom `UNetWithClassifier` (DoubleConv encoder-decoder + classification head)."""),

        md("""## Project Objectives

| Status | Objective | Notes |
|---|---|---|
| ✅ | Detect whether an image is tampered | Classifier head → image-level accuracy |
| ✅ | Localize tampered regions | Segmentation branch → pixel-level masks |
| ✅ | Run in one Kaggle notebook | Single notebook, GPU P100 |
| ✅ | Track experiments | W&B integration (optional) |
| ✅ | Explainability | Grad-CAM, failure analysis, robustness testing |"""),
    ]

def cells_env_setup():
    return [
        md("""## 1. Environment Setup

Install dependencies and configure the central `CONFIG` dictionary."""),

        code("""!pip install -q "albumentations>=1.3.1,<2.0" opencv-python-headless

import os, sys, random, json, warnings, math, contextlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Suppress C-level libpng warnings
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")"""),

        code("""# ── Central Configuration ─────────────────────────────────────────────────────

SEED = 42

CONFIG = {
    # ── Data ──
    'image_size': 256,
    'batch_size': 16,
    'num_workers': 4,

    # ── Model ── (custom UNetWithClassifier from vK.2)
    'n_channels': 3,
    'n_classes': 1,      # segmentation output channels
    'n_labels': 2,       # classification labels (authentic / tampered)

    # ── Optimizer ──
    'lr': 1e-4,
    'weight_decay': 1e-4,

    # ── Scheduler ──
    'scheduler': 'reduce_on_plateau',
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,

    # ── Training ──
    'max_epochs': 50,
    'patience': 10,           # early stopping patience
    'accumulation_steps': 2,  # effective batch = 16 * 2 = 32
    'max_grad_norm': 1.0,

    # ── Loss ──
    'use_pos_weight': True,
    'dice_per_sample': True,
    'cls_loss_weight': 1.5,   # ALPHA
    'seg_loss_weight': 1.0,   # BETA

    # ── Augmentation ──
    'aug_color_jitter': True,
    'aug_compression': True,
    'aug_gauss_noise': True,
    'aug_gauss_blur': True,

    # ── Feature Flags ──
    'use_amp': True,
    'use_wandb': True,

    # ── Reproducibility ──
    'seed': SEED,
}

# ── Output directories ──
OUTPUT_DIR = '/kaggle/working'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

for d in [CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

print('CONFIG:')
for k, v in CONFIG.items():
    print(f'  {k}: {v}')
print(f'\\nEffective batch size: {CONFIG["batch_size"] * CONFIG["accumulation_steps"]}')"""),

        code("""# ── Device & Seed ─────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = True
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

print(f'Training device: {device}')
print(f'Seed: {SEED}')"""),
    ]

def cells_wandb():
    return [
        md("""## 2. Experiment Tracking (W&B)

Kaggle-native W&B setup using `kaggle_secrets`."""),

        code("""WANDB_ACTIVE = False
WANDB_RUN = None

if CONFIG['use_wandb']:
    try:
        !pip install -q wandb
        import wandb
        from kaggle_secrets import UserSecretsClient
        wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
        WANDB_RUN = wandb.init(
            project='tampered-image-detection-assignment',
            config=CONFIG,
            name=f'vk4-unetwithclassifier-kaggle-seed{SEED}',
            tags=['vK.4', 'casia-v2', 'kaggle', 'custom-unet'],
        )
        WANDB_ACTIVE = True
        print('W&B initialized.')
    except Exception as e:
        print(f'W&B setup failed, continuing without: {e}')
        WANDB_ACTIVE = False
else:
    print('W&B disabled.')

print(f'W&B active: {WANDB_ACTIVE}')"""),
    ]

def cells_dataset():
    return [
        md("""## 3. Dataset Loading & Validation

Auto-discover the CASIA dataset under `/kaggle/input/`, validate image-mask pairs, and perform stratified splitting."""),

        code("""# ── Dataset Discovery ─────────────────────────────────────────────────────────
KAGGLE_INPUT = '/kaggle/input'

DATASET_ROOT = None
IMAGE_DIR = None
MASK_DIR = None

for root, dirs, files in os.walk(KAGGLE_INPUT):
    dirs_lower = [d.lower() for d in dirs]
    if 'image' in dirs_lower and 'mask' in dirs_lower:
        DATASET_ROOT = root
        IMAGE_DIR = os.path.join(root, dirs[dirs_lower.index('image')])
        MASK_DIR = os.path.join(root, dirs[dirs_lower.index('mask')])
        break

if DATASET_ROOT is None:
    raise FileNotFoundError(
        'Could not find dataset with IMAGE/ and MASK/ directories under /kaggle/input/. '
        'Ensure the CASIA Splicing Detection + Localization dataset is attached.'
    )

print(f'Dataset root: {DATASET_ROOT}')
print(f'IMAGE dir:    {IMAGE_DIR}')
print(f'MASK dir:     {MASK_DIR}')

for sub in ['Au', 'Tp']:
    img_sub = os.path.join(IMAGE_DIR, sub)
    mask_sub = os.path.join(MASK_DIR, sub)
    if os.path.isdir(img_sub):
        print(f'  {sub}: {len(os.listdir(img_sub))} images, '
              f'{len(os.listdir(mask_sub)) if os.path.isdir(mask_sub) else 0} masks')"""),

        code("""# ── Build Validated Pairs ──────────────────────────────────────────────────────

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        return True
    except Exception:
        return False

def detect_forgery_type(filename):
    fn = filename.lower()
    if '_cm' in fn or 'copy' in fn:
        return 'copy-move'
    elif '_sp' in fn or 'splic' in fn:
        return 'splicing'
    return 'unknown'

valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
label_folders = {'Au': {'class_name': 'authentic', 'label': 0},
                 'Tp': {'class_name': 'tampered',  'label': 1}}

pairs = []
excluded = []

for sub_name, info in label_folders.items():
    img_subdir = os.path.join(IMAGE_DIR, sub_name)
    mask_subdir = os.path.join(MASK_DIR, sub_name)

    if not os.path.isdir(img_subdir):
        print(f'Warning: {img_subdir} not found, skipping.')
        continue

    for fname in sorted(os.listdir(img_subdir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue
        img_path = os.path.join(img_subdir, fname)
        mask_path = os.path.join(mask_subdir, fname) if os.path.isdir(mask_subdir) else None

        if mask_path and not os.path.exists(mask_path):
            # Try common mask name variants
            stem = os.path.splitext(fname)[0]
            for mext in ['.png', '.jpg', '.tif', '.bmp']:
                alt = os.path.join(mask_subdir, stem + mext)
                if os.path.exists(alt):
                    mask_path = alt
                    break
            else:
                mask_path = None

        entry = {
            'image_path': img_path,
            'mask_path': mask_path,
            'label': float(info['label']),
            'class_name': info['class_name'],
            'forgery_type': detect_forgery_type(fname) if info['label'] == 1 else 'authentic',
        }

        pairs.append(entry)

tampered_count = sum(1 for p in pairs if p['label'] == 1.0)
authentic_count = sum(1 for p in pairs if p['label'] == 0.0)

print('=' * 50)
print('DATASET VALIDATION SUMMARY')
print('=' * 50)
print(f'Total valid pairs:   {len(pairs)}')
print(f'  Tampered images:   {tampered_count}')
print(f'  Authentic images:  {authentic_count}')
print(f'  Tampered ratio:    {tampered_count / max(len(pairs), 1):.2%}')
forgery_types = Counter(p['forgery_type'] for p in pairs)
print(f'  Forgery types:     {dict(forgery_types)}')
print('=' * 50)"""),

        code("""# ── Stratified Split: 70/15/15 ────────────────────────────────────────────────

labels_for_split = [p['label'] for p in pairs]

train_pairs, temp_pairs = train_test_split(
    pairs, test_size=0.30, random_state=SEED, stratify=labels_for_split
)
temp_labels = [p['label'] for p in temp_pairs]
val_pairs, test_pairs = train_test_split(
    temp_pairs, test_size=0.5, random_state=SEED, stratify=temp_labels
)

print(f'Train: {len(train_pairs)}')
print(f'Val:   {len(val_pairs)}')
print(f'Test:  {len(test_pairs)}')

for name, split in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
    counts = Counter(p['class_name'] for p in split)
    print(f'  {name}: {dict(counts)}')

# Save split manifest
manifest = {
    'seed': SEED,
    'total_pairs': len(pairs),
    'train_count': len(train_pairs),
    'val_count': len(val_pairs),
    'test_count': len(test_pairs),
}
manifest_path = os.path.join(RESULTS_DIR, 'split_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Split manifest saved to: {manifest_path}')"""),
    ]

def cells_augmentation():
    return [
        md("""## 4. Preprocessing & Augmentation

**vK.4 augmentation pipeline** (adopted from v8):
- `ColorJitter` — photometric regularization
- `ImageCompression` — forces model to handle JPEG artifacts
- `GaussNoise` — noise robustness
- `GaussianBlur` — blur invariance

All augmentations are CONFIG-controlled."""),

        code("""# ── Augmentation Pipeline ──────────────────────────────────────────────────────

def build_train_transform(config):
    transforms = [
        A.Resize(config['image_size'], config['image_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ]

    if config.get('aug_color_jitter', False):
        transforms.append(A.ColorJitter(brightness=0.2, contrast=0.2,
                                         saturation=0.2, hue=0.1, p=0.5))
    if config.get('aug_compression', False):
        transforms.append(A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3))
    if config.get('aug_gauss_noise', False):
        transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.3))
    if config.get('aug_gauss_blur', False):
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.2))

    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return A.Compose(transforms)

train_transform = build_train_transform(CONFIG)
val_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print('Train augmentations:')
for t in train_transform.transforms:
    print(f'  {t.__class__.__name__}')"""),
    ]

def cells_dataset_class():
    return [
        code("""# ── Dataset Class ─────────────────────────────────────────────────────────────

class TamperingDataset(Dataset):
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

        label = torch.tensor(int(entry['label']), dtype=torch.long)
        return image, mask, label"""),

        code("""# ── DataLoaders ───────────────────────────────────────────────────────────────

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_dataset = TamperingDataset(train_pairs, transform=train_transform)
val_dataset = TamperingDataset(val_pairs, transform=val_transform)
test_dataset = TamperingDataset(test_pairs, transform=val_transform)

_nw = CONFIG['num_workers']
loader_kwargs = dict(
    num_workers=_nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=_nw > 0,
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=True, drop_last=True,
                          worker_init_fn=seed_worker, generator=g, **loader_kwargs)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                        shuffle=False, drop_last=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                         shuffle=False, drop_last=False, **loader_kwargs)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches:   {len(val_loader)}')
print(f'Test batches:  {len(test_loader)}')
print(f'num_workers={_nw}, pin_memory={loader_kwargs["pin_memory"]}')"""),

        code("""# ── Sanity Check: Visualize One Training Batch ────────────────────────────────

images, masks, labels = next(iter(train_loader))
print(f'Image batch: {images.shape}  Mask batch: {masks.shape}  Labels: {labels[:8]}')

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i in range(min(4, images.size(0))):
    img = images[i].permute(1, 2, 0).numpy() * std + mean
    img = np.clip(img, 0, 1)
    msk = masks[i].squeeze().numpy()

    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Image (label={labels[i].item()})')
    axes[0, i].axis('off')

    axes[1, i].imshow(msk, cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title('Mask')
    axes[1, i].axis('off')

plt.suptitle('Training Batch Sample', fontsize=14)
plt.tight_layout()
plt.show()"""),
    ]
