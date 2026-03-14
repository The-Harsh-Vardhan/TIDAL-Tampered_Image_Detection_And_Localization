"""Generate tamper_detection_v5_kaggle.ipynb — Kaggle-compatible version of v5."""
import json
import os

def to_source(text):
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result

cells = []

def md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': to_source(text)
    })

def code(text):
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'source': to_source(text),
        'execution_count': None,
        'outputs': []
    })


# ==============================================================================
# SECTION 0: KAGGLE ENVIRONMENT NOTE
# ==============================================================================

md("""# Tamper Detection v5 (Kaggle) — Image Forgery Detection & Localization

> **Kaggle-compatible version** of the Colab training notebook.
> The dataset is loaded directly from Kaggle dataset mounts (`/kaggle/input/`) rather than downloaded via the Kaggle API.
> All outputs are saved to `/kaggle/working/` instead of Google Drive.

**Architecture:** SMP U-Net (ResNet34, ImageNet pretrained)
**Dataset:** CASIA Splicing Detection + Localization
**Loss:** BCE + Dice
**Optimizer:** AdamW (differential LR)
**Training:** AMP, gradient accumulation, early stopping

## Notebook Sections

1. Setup & Environment
2. Dataset Loading
3. Dataset Discovery
4. Dataset Validation
5. Preprocessing & Split
6. Dataset Class
7. DataLoaders
8. Model Definition
9. Loss Function & Optimizer
10. Training Loop
11. Threshold Selection
12. Evaluation
13. Visualization
14. Explainable AI
15. Robustness Testing
16. Experiment Tracking
17. Save & Export""")


# ==============================================================================
# SECTION 1: SETUP & ENVIRONMENT
# ==============================================================================

md("## 1. Setup & Environment")

code("""!pip install -q segmentation-models-pytorch "albumentations>=1.3.1,<2.0" """)

code("""import os
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

code("""SEED = 42

def set_seed(seed=SEED):
    '''Set seed for full reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
print(f'Seed set to {SEED}')""")

code("""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected. Enable GPU via Settings > Accelerator.')
print(f'Device: {device}')""")

code("""CONFIG = {
    'image_size': 512,
    'batch_size': 4,
    'num_workers': 2,
    'train_ratio': 0.85,
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

print(f'Checkpoint directory: {CHECKPOINT_DIR}')
print(f'Results directory:    {RESULTS_DIR}')
print(f'Plots directory:      {PLOTS_DIR}')""")

code("""USE_WANDB = False  # Set True to enable W&B experiment tracking

if USE_WANDB:
    !pip install -q wandb
    import wandb
    wandb.login()
    wandb.init(
        project='tamper-detection',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}-kaggle',
        tags=['mvp', 'casia-v2', 'kaggle'],
    )
    print('W&B initialized.')
else:
    print('W&B disabled. Running with local artifacts only.')""")


# ==============================================================================
# SECTION 2: DATASET LOADING
# ==============================================================================

md("""## 2. Dataset Loading

The dataset is pre-mounted by Kaggle at `/kaggle/input/`.
No API authentication or download step is needed.

**Dataset:** `sagnikkayalcse52/casia-spicing-detection-localization`
**Structure:** `Image/Au/`, `Image/Tp/` (images), `Mask/Au/`, `Mask/Tp/` (ground-truth masks)""")

code("""# Kaggle input path — dataset is already mounted
KAGGLE_INPUT = '/kaggle/input'

# Dynamically discover dataset root (find directory with Image/ and Mask/)
DATASET_ROOT = None
for root, dirs, files in os.walk(KAGGLE_INPUT):
    if 'Image' in dirs and 'Mask' in dirs:
        DATASET_ROOT = root
        break

if DATASET_ROOT is None:
    raise FileNotFoundError(
        'Could not find dataset root with Image/ and Mask/ directories under /kaggle/input/. '
        'Ensure the CASIA Splicing Detection + Localization dataset is attached to this notebook.'
    )

print(f'Dataset root: {DATASET_ROOT}')

# List directory structure
for d in ['Image/Au', 'Image/Tp', 'Mask/Au', 'Mask/Tp']:
    path = os.path.join(DATASET_ROOT, d)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        print(f'  {d}: {count} files')
    else:
        print(f'  {d}: NOT FOUND')""")


# ==============================================================================
# SECTION 3: DATASET DISCOVERY
# ==============================================================================

md("""## 3. Dataset Discovery

Dynamically discover image–mask pairs with:
- Dimension validation (image and mask must match spatially)
- Forgery type classification (`_D_` → splicing, `_S_` → copy-move)
- Authentic image verification
- Excluded pairs tracking with reason strings""")

code("""def validate_dimensions(image_path, mask_path):
    '''Check that image and mask have the same spatial dimensions.'''
    img = Image.open(image_path)
    msk = Image.open(mask_path)
    if img.size != msk.size:
        return False, f'dim_mismatch: img={img.size} mask={msk.size}'
    return True, ''


def discover_pairs(dataset_root):
    '''Discover image-mask pairs dynamically.

    Returns:
        pairs: list of dicts with keys image_path, mask_path, label, forgery_type
        excluded: list of (filename, reason) tuples
    '''
    img_tp_dir = os.path.join(dataset_root, 'Image', 'Tp')
    mask_tp_dir = os.path.join(dataset_root, 'Mask', 'Tp')
    img_au_dir = os.path.join(dataset_root, 'Image', 'Au')

    pairs = []
    excluded = []

    # --- Tampered images ---
    for img_name in sorted(os.listdir(img_tp_dir)):
        img_path = os.path.join(img_tp_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        mask_path = os.path.join(mask_tp_dir, img_name)
        if not os.path.exists(mask_path):
            # Try common mask naming variations
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
    for img_name in sorted(os.listdir(img_au_dir)):
        img_path = os.path.join(img_au_dir, img_name)
        if not os.path.isfile(img_path):
            continue

        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            excluded.append((img_name, 'corrupt_file'))
            continue

        pairs.append({
            'image_path': img_path,
            'mask_path': None,  # Zero mask generated at load time
            'label': 0.0,
            'forgery_type': 'authentic',
        })

    return pairs, excluded

print('Discovery functions defined.')""")

code("""# Run discovery
pairs, excluded = discover_pairs(DATASET_ROOT)

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


# ==============================================================================
# SECTION 4: DATASET VALIDATION
# ==============================================================================

md("""## 4. Dataset Validation

Verify discovered pairs and check class distributions before proceeding to data splitting.""")

code("""# Validate minimum dataset requirements
assert len(pairs) > 0, 'No valid pairs discovered!'

tampered_count = sum(1 for p in pairs if p['label'] == 1.0)
authentic_count = sum(1 for p in pairs if p['label'] == 0.0)

print(f'Tampered images:  {tampered_count}')
print(f'Authentic images: {authentic_count}')
print(f'Total:            {len(pairs)}')
print(f'Tampered ratio:   {tampered_count / len(pairs):.2%}')

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


# ==============================================================================
# SECTION 5: PREPROCESSING & DATA SPLIT
# ==============================================================================

md("""## 5. Preprocessing & Split

Stratified split: **85% train / 7.5% validation / 7.5% test**.
Stratification key: `forgery_type` (authentic / splicing / copy-move).

Two-step procedure:
1. Split 85% train vs 15% temp
2. Split temp 50/50 into val and test""")

code("""forgery_labels = [p['forgery_type'] for p in pairs]

# Step 1: train (85%) vs temp (15%)
train_pairs, temp_pairs = train_test_split(
    pairs, test_size=0.15, random_state=SEED, stratify=forgery_labels
)

# Step 2: temp -> val (50%) + test (50%)
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
    print(f'  {name}: {dist}')""")

code("""# Persist split manifest for reproducibility
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


# ==============================================================================
# SECTION 6: DATASET CLASS
# ==============================================================================

md("""## 6. Dataset Class

`TamperingDataset` loads images as RGB, binarizes masks (threshold > 128), generates zero masks for authentic images, and applies albumentations transforms.

Returns `(image, mask, label)` per sample.""")

code("""class TamperingDataset(Dataset):
    '''Dataset for tamper detection with segmentation masks.'''

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
            mask = (mask > 128).astype(np.uint8)
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


# ==============================================================================
# SECTION 7: DATALOADERS
# ==============================================================================

md("""## 7. DataLoaders

**MVP transforms** — spatial augmentations only:
- `Resize(512, 512)`, `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`
- `Normalize` with ImageNet statistics
- Phase 2 adds photometric augmentations (brightness, contrast, noise, JPEG compression)""")

code("""train_transform = A.Compose([
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

print('Transforms defined.')""")

code("""train_dataset = TamperingDataset(train_pairs, transform=train_transform)
val_dataset = TamperingDataset(val_pairs, transform=val_transform)
test_dataset = TamperingDataset(test_pairs, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    drop_last=False,
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches:   {len(val_loader)}')
print(f'Test batches:  {len(test_loader)}')""")

code("""# Sanity check: visualize one training batch
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


# ==============================================================================
# SECTION 8: MODEL DEFINITION
# ==============================================================================

md("""## 8. Model Definition

**SMP U-Net** with ResNet34 encoder (ImageNet pretrained).
- Input: `(B, 3, 512, 512)` — RGB
- Output: `(B, 1, 512, 512)` — raw logits (sigmoid applied at inference)""")

code("""model = smp.Unet(
    encoder_name=CONFIG['encoder_name'],
    encoder_weights=CONFIG['encoder_weights'],
    in_channels=CONFIG['in_channels'],
    classes=CONFIG['classes'],
    activation=None,  # Raw logits — no sigmoid during training
)
model = model.to(device)

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters:     {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# Forward pass shape check
with torch.no_grad():
    dummy = torch.randn(1, 3, CONFIG['image_size'], CONFIG['image_size']).to(device)
    out = model(dummy)
    print(f'Input shape:  {dummy.shape}')
    print(f'Output shape: {out.shape}')
    assert out.shape == (1, 1, CONFIG['image_size'], CONFIG['image_size']), \\
        f'Unexpected output shape: {out.shape}'

print('Model shape check passed.')""")


# ==============================================================================
# SECTION 9: LOSS FUNCTION & OPTIMIZER
# ==============================================================================

md("""## 9. Loss Function & Optimizer

**Loss:** BCE + Dice (equal weight, smooth=1.0) — handles class imbalance where tampered pixels are typically <5% of image area.

**Optimizer:** AdamW with differential LR:
- Encoder: 1e-4 (preserve pretrained features)
- Decoder/Head: 1e-3 (train new layers faster)

**Metrics** defined before training so they can be used in validation.""")

code("""class BCEDiceLoss(nn.Module):
    '''Combined BCE + Dice loss for binary segmentation.'''

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
print('BCEDiceLoss initialized.')""")

code("""optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': CONFIG['encoder_lr']},
    {'params': model.decoder.parameters(), 'lr': CONFIG['decoder_lr']},
    {'params': model.segmentation_head.parameters(), 'lr': CONFIG['decoder_lr']},
], weight_decay=CONFIG['weight_decay'])

scaler = GradScaler('cuda')

print('Optimizer: AdamW')
print(f'  Encoder LR:   {CONFIG["encoder_lr"]}')
print(f'  Decoder LR:   {CONFIG["decoder_lr"]}')
print(f'  Weight decay:  {CONFIG["weight_decay"]}')""")

code("""def compute_pixel_f1(pred, gt, eps=1e-8):
    '''Compute Pixel-F1 score.'''
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
    '''Compute Intersection over Union.'''
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / (union + eps)).item()


def compute_precision_recall(pred, gt, eps=1e-8):
    '''Compute pixel-level precision and recall.'''
    pred, gt = pred.flatten(), gt.flatten()
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision.item(), recall.item()

print('Metric functions defined: compute_pixel_f1, compute_iou, compute_precision_recall')""")


# ==============================================================================
# SECTION 10: TRAINING LOOP
# ==============================================================================

md("""## 10. Training Loop

Mixed precision training with gradient accumulation (effective batch size = 16).
- Early stopping on validation Pixel-F1 (patience = 10)
- Gradient clipping (max_norm = 1.0)
- Partial window flush after final batch
- Checkpoints saved to `/kaggle/working/checkpoints/`
- W&B logging (guarded behind `USE_WANDB`)""")

code("""def save_checkpoint(state, filepath):
    '''Save training checkpoint.'''
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer, scaler):
    '''Load training checkpoint for resume.'''
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['epoch'] + 1, ckpt['best_f1'], ckpt['best_epoch']

print('Checkpoint helpers defined.')""")

code("""@torch.no_grad()
def validate(model, val_loader, criterion, device, threshold=0.5):
    '''Run validation and return loss, mean F1, mean IoU.'''
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

code("""# Training configuration
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

    # Partial window flush (if final batch not aligned with accumulation boundary)
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


# ==============================================================================
# SECTION 11: THRESHOLD SELECTION
# ==============================================================================

md("""## 11. Threshold Selection

Sweep 50 thresholds from 0.1 to 0.9 on the **validation set** to maximize mean Pixel-F1.
The selected threshold is frozen for all downstream test-set evaluation.""")

code("""# Load best model for threshold selection and evaluation
best_ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded best model from epoch {ckpt["best_epoch"] + 1} (F1={ckpt["best_f1"]:.4f})')""")

code("""@torch.no_grad()
def find_best_threshold(model, val_loader, device, num_thresholds=50):
    '''Sweep thresholds on validation set to maximize mean Pixel-F1.'''
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

    # Find best threshold
    results.sort(key=lambda x: x[1], reverse=True)
    best_t, best_f1_val = results[0]
    return best_t, results

best_threshold, threshold_results = find_best_threshold(model, val_loader, device)
print(f'Best threshold: {best_threshold:.4f}')
print(f'Best val F1 at threshold: {threshold_results[0][1]:.4f}')""")


# ==============================================================================
# SECTION 12: EVALUATION ON TEST SET
# ==============================================================================

md("""## 12. Evaluation

Full evaluation with three reporting views:
1. **Mixed-set** — All test images (authentic + tampered)
2. **Tampered-only** — Pixel-level metrics on tampered images only
3. **Forgery-type breakdown** — Separate metrics for splicing and copy-move""")

code("""@torch.no_grad()
def evaluate(model, test_loader, test_pairs, device, threshold):
    '''Full evaluation on test set with mixed, tampered-only, and per-forgery metrics.'''
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

    # Per-forgery tracking
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

            # Image-level detection: max(prob_map) >= threshold
            tamper_score = probs[i].view(-1).max().item()
            image_scores.append(tamper_score)
            image_labels.append(int(labels[i].item()))
            image_preds.append(int(tamper_score >= threshold))

            # Track tampered-only and per-forgery
            if labels[i].item() == 1.0:
                tampered_f1.append(f1)
                tampered_iou.append(iou)
                if idx < len(test_pairs):
                    ftype = test_pairs[idx]['forgery_type']
                    if ftype in forgery_f1:
                        forgery_f1[ftype].append(f1)
            idx += 1

    # Image-level metrics
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

code("""test_results = evaluate(model, test_loader, test_pairs, device, best_threshold)

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

code("""# Log test results to W&B (guarded)
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


# ==============================================================================
# SECTION 13: VISUALIZATION
# ==============================================================================

md("""## 13. Visualization

1. Training curves (loss, F1, IoU)
2. F1-vs-threshold sweep
3. Prediction grid (Original | GT Mask | Predicted Mask | Overlay)""")

code("""# Training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

epochs_range = range(1, len(history['train_loss']) + 1)

# Loss
axes[0].plot(epochs_range, history['train_loss'], label='Train')
axes[0].plot(epochs_range, history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation F1
axes[1].plot(epochs_range, history['val_f1'], color='green')
axes[1].axvline(x=best_epoch + 1, color='red', linestyle='--',
                label=f'Best (epoch {best_epoch + 1})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Pixel-F1')
axes[1].set_title('Validation Pixel-F1')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Validation IoU
axes[2].plot(epochs_range, history['val_iou'], color='orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Pixel-IoU')
axes[2].set_title('Validation Pixel-IoU')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Training curves saved.')""")

code("""# F1 vs Threshold
thresh_vals = [r[0] for r in threshold_results]
f1_vals = [r[1] for r in threshold_results]

# Sort by threshold for clean plot
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

code("""@torch.no_grad()
def collect_predictions(model, test_loader, test_pairs, device, threshold):
    '''Collect all test predictions with metadata for visualization.'''
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

            # Denormalize image for display
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

code("""# Select representative samples for prediction grid
tampered_preds = [p for p in predictions if p['label'] == 1.0]
authentic_preds = [p for p in predictions if p['label'] == 0.0]
tampered_sorted = sorted(tampered_preds, key=lambda p: p['pixel_f1'])

samples = []

# 2 best tampered
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[-2:])

# 2 median tampered
mid = len(tampered_sorted) // 2
if len(tampered_sorted) >= 4:
    samples.extend(tampered_sorted[mid - 1:mid + 1])

# 2 worst tampered
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[:2])

# 2 authentic
if len(authentic_preds) >= 2:
    samples.extend(authentic_preds[:2])

n_rows = len(samples)
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

for row, sample in enumerate(samples):
    img = sample['image']
    gt = sample['gt_mask']
    pred = sample['pred_mask']

    # Overlay: red on predicted tampered regions
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

code("""# Log visualizations to W&B (guarded)
if USE_WANDB:
    wandb.log({
        'predictions': wandb.Image(os.path.join(PLOTS_DIR, 'prediction_grid.png')),
    })
    wandb.log({
        'training_curves': wandb.Image(os.path.join(PLOTS_DIR, 'training_curves.png')),
    })
    print('Visualizations logged to W&B.')""")


# ==============================================================================
# SECTION 14: EXPLAINABLE AI
# ==============================================================================

md("""## 14. Explainable AI

Lightweight explainability methods to verify model behavior:

- **Grad-CAM heatmaps** — Show which spatial regions drive encoder activations. Helps verify the model attends to tampered regions rather than irrelevant patterns.
- **Diagnostic overlays** — Color-coded TP (green), FP (red), FN (blue) regions for spatial error analysis.
- **Failure case analysis** — Systematic examination of worst predictions to identify error patterns.

Heavy XAI frameworks (SHAP, LIME) are excluded — they add significant overhead without clear benefit for dense pixel-level prediction.""")

code("""class GradCAM:
    '''Grad-CAM for segmentation encoder features.

    Generates heatmaps showing which spatial regions of the encoder
    contribute most to the model output.
    '''

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
        '''Generate Grad-CAM heatmap for the segmentation output.'''
        self.model.eval()
        output = self.model(input_tensor)
        # Use mean of segmentation logits as scalar target
        target = output.mean()
        self.model.zero_grad()
        target.backward()

        # Compute weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        # Resize to input resolution
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode='bilinear', align_corners=False
        )
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        '''Clean up registered hooks.'''
        for h in self._handles:
            h.remove()

print('GradCAM class defined.')""")

code("""def create_diagnostic_overlay(original, pred_mask, gt_mask):
    '''Create a color-coded overlay showing TP, FP, FN regions.

    Colors: Green=TP (correct detection), Red=FP (false alarm), Blue=FN (missed)
    '''
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

code("""# Grad-CAM + diagnostic overlay visualization on selected test samples
grad_cam = GradCAM(model, model.encoder.layer4)

# Select a few tampered samples (best predictions) for visualization
cam_samples = [p for p in predictions if p['label'] == 1.0]
cam_samples = sorted(cam_samples, key=lambda p: p['pixel_f1'], reverse=True)[:4]

n_cam = len(cam_samples)
fig, axes = plt.subplots(n_cam, 5, figsize=(25, 5 * n_cam))
if n_cam == 1:
    axes = axes[np.newaxis, :]

mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for row, sample in enumerate(cam_samples):
    # Re-normalize image to tensor for Grad-CAM
    img_tensor = torch.from_numpy(sample['image']).permute(2, 0, 1).float()
    img_tensor = (img_tensor - mean_t) / std_t
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    cam = grad_cam.generate(img_tensor)
    diagnostic = create_diagnostic_overlay(
        sample['image'], sample['pred_mask'], sample['gt_mask']
    )

    axes[row, 0].imshow(sample['image'])
    axes[row, 0].set_title(f'Original ({sample["forgery_type"]})')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(sample['image'])
    axes[row, 1].imshow(cam, cmap='jet', alpha=0.5)
    axes[row, 1].set_title('Grad-CAM Heatmap')
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

grad_cam.remove_hooks()
print('Grad-CAM analysis saved.')""")

code("""def analyze_failure_cases(predictions, n_worst=10):
    '''Analyze worst predictions to identify systematic error patterns.'''
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

    # Check for small-region failures
    small_mask_count = sum(1 for p in worst if p['gt_mask_area'] < 0.02)
    if small_mask_count > n_worst // 2:
        analysis['common_patterns'].append(
            f'Fails on small tampered regions (<2%% area): {small_mask_count}/{len(worst)}'
        )

    # Check for forgery-type bias
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


# ==============================================================================
# SECTION 15: ROBUSTNESS TESTING
# ==============================================================================

md("""## 15. Robustness Testing

Evaluate model robustness against post-processing degradations:
- JPEG compression (QF=70, QF=50)
- Gaussian noise (light, heavy)
- Gaussian blur (5x5 kernel)
- Resize degradation (0.75x, 0.5x)

**Protocol:**
- Degradations applied to test images only; masks remain clean
- No retraining or fine-tuning
- Same validation-selected threshold used for all conditions""")

code("""NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

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

code("""class ResizeDegradationDataset(Dataset):
    '''Applies resize degradation to images only. Masks remain clean.'''

    def __init__(self, pairs, scale_factor, image_size=512):
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

        # Load image
        image = cv2.imread(entry['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply resize degradation to image only
        h, w = image.shape[:2]
        small_h = max(1, int(h * self.scale_factor))
        small_w = max(1, int(w * self.scale_factor))
        degraded = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)

        # Load mask (clean path - no degradation)
        if entry['mask_path'] is not None:
            mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 128).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Normalize and tensorize
        augmented = self.normalize(image=degraded, mask=mask)
        image_t = augmented['image']
        mask_t = augmented['mask'].unsqueeze(0).float()
        label = torch.tensor(entry['label'], dtype=torch.float32)
        return image_t, mask_t, label

print('ResizeDegradationDataset defined.')""")

code("""@torch.no_grad()
def run_robustness_eval(model, loader, device, threshold):
    '''Run evaluation loop for robustness testing, returns per-image F1 scores.'''
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
    '''Evaluate model under all degradation conditions.'''
    results = {}

    # Albumentations-based degradations
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

    # Resize degradations
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

# Print results table
print(f'\\nRobustness Results (threshold={best_threshold:.4f}):')
print(f'{"":<25} {"Pixel-F1 (mean +/- std)":<25} {"Delta from clean":<15}')
print('-' * 65)
clean_f1 = robustness_results.get('clean', {}).get('f1_mean', 0)
for name, data in robustness_results.items():
    delta = data['f1_mean'] - clean_f1 if name != 'clean' else 0.0
    delta_str = f'{delta:+.4f}' if name != 'clean' else '---'
    print(f'{name:<25} {data["f1_mean"]:.4f} +/- {data["f1_std"]:.4f}      {delta_str}')""")

code("""# Robustness bar chart
names = list(robustness_results.keys())
means = [robustness_results[n]['f1_mean'] for n in names]
stds = [robustness_results[n]['f1_std'] for n in names]

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(names)), means, yerr=stds, capsize=4, color='steelblue', alpha=0.8)

# Highlight clean baseline in green
if 'clean' in names:
    bars[names.index('clean')].set_color('green')

plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.ylabel('Pixel-F1')
plt.title('Robustness Testing: Pixel-F1 Under Degradation')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'robustness_chart.png'), dpi=150, bbox_inches='tight')
plt.show()

# W&B logging (guarded)
if USE_WANDB:
    wandb.log({
        'robustness_chart': wandb.Image(os.path.join(PLOTS_DIR, 'robustness_chart.png')),
    })
    for name, data in robustness_results.items():
        wandb.log({f'robustness/{name}_f1': data['f1_mean']})
    print('Robustness results logged to W&B.')""")


# ==============================================================================
# SECTION 16: EXPERIMENT TRACKING
# ==============================================================================

md("""## 16. Experiment Tracking

W&B experiment tracking is **integrated throughout this notebook** — not isolated in a single section.

When `USE_WANDB = True`, the following actions occur automatically:

| Section | W&B Action |
|---|---|
| 1 (Setup) | Install, login, init with CONFIG |
| 10 (Training) | Per-epoch metric logging (loss, F1, IoU, LR) |
| 12 (Evaluation) | Test results summary update |
| 13 (Visualization) | Prediction grid + training curves logged as images |
| 15 (Robustness) | Per-degradation F1 + bar chart logged |
| 17 (Export) | Model artifact upload, `wandb.finish()` |

When `USE_WANDB = False` (default), the notebook produces local artifacts in `/kaggle/working/`:

| Artifact | Path |
|---|---|
| Best model weights | `checkpoints/best_model.pt` |
| Resume checkpoint | `checkpoints/last_checkpoint.pt` |
| Split manifest | `results/split_manifest.json` |
| Results summary | `results/results_summary.json` |
| Training curves | `plots/training_curves.png` |
| Prediction grid | `plots/prediction_grid.png` |
| Grad-CAM analysis | `plots/gradcam_analysis.png` |
| Robustness chart | `plots/robustness_chart.png` |""")


# ==============================================================================
# SECTION 17: SAVE & EXPORT RESULTS
# ==============================================================================

md("""## 17. Save & Export

Save all results, configuration, and artifacts for reproducibility.
All outputs are saved under `/kaggle/working/`.""")

code("""# Save comprehensive results summary
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

code("""# W&B artifact upload and cleanup (guarded)
if USE_WANDB:
    artifact = wandb.Artifact('best-model', type='model')
    artifact.add_file(os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    wandb.log_artifact(artifact)
    print('Model artifact uploaded to W&B.')

    wandb.finish()
    print('W&B run finished.')
else:
    print('W&B disabled. All artifacts saved locally to /kaggle/working/.')

print()
print('=' * 60)
print('Notebook complete!')
print()
print(f'Checkpoints:     {CHECKPOINT_DIR}/')
for f_name in ['best_model.pt', 'last_checkpoint.pt']:
    fpath = os.path.join(CHECKPOINT_DIR, f_name)
    status = 'OK' if os.path.exists(fpath) else 'MISSING'
    print(f'  [{status}] {f_name}')

print(f'\\nResults:         {RESULTS_DIR}/')
for f_name in ['split_manifest.json', 'results_summary.json']:
    fpath = os.path.join(RESULTS_DIR, f_name)
    status = 'OK' if os.path.exists(fpath) else 'MISSING'
    print(f'  [{status}] {f_name}')

print(f'\\nPlots:           {PLOTS_DIR}/')
for f_name in ['training_curves.png', 'f1_vs_threshold.png', 'prediction_grid.png',
               'gradcam_analysis.png', 'robustness_chart.png']:
    fpath = os.path.join(PLOTS_DIR, f_name)
    status = 'OK' if os.path.exists(fpath) else 'MISSING'
    print(f'  [{status}] {f_name}')

print('=' * 60)""")


# ==============================================================================
# BUILD AND SAVE NOTEBOOK
# ==============================================================================

notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0',
            'mimetype': 'text/x-python',
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'pygments_lexer': 'ipython3',
            'file_extension': '.py',
        },
        'kaggle': {
            'accelerator': 'gpu',
            'dataSources': [
                {
                    'sourceId': 0,
                    'sourceType': 'datasetVersion',
                    'datasetSlug': 'casia-spicing-detection-localization',
                    'sourceSlug': 'sagnikkayalcse52/casia-spicing-detection-localization',
                }
            ],
            'isGpuEnabled': True,
            'isInternetEnabled': True,
        },
    },
    'cells': cells,
}

output_path = os.path.join(
    r'c:\D Drive\Projects\BigVision Assignment',
    'notebooks',
    'tamper_detection_v5_kaggle.ipynb'
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Notebook saved to: {output_path}')
print(f'Total cells: {len(cells)}')
md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
code_count = sum(1 for c in cells if c['cell_type'] == 'code')
print(f'  Markdown cells: {md_count}')
print(f'  Code cells: {code_count}')
