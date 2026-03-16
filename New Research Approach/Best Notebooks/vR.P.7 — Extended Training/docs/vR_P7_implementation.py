# vR.P.7 — Full PyTorch Implementation
# ELA + Extended Training (50 epochs, patience 10)
# Parent: vR.P.3 | Encoder: ResNet-34 (frozen body, BN unfrozen) | Input: ELA 384x384

# =============================================================================
# CELL 0: Markdown — Title & Pipeline
# =============================================================================
# vR.P.7 — ELA + Extended Training
#
# Pipeline: Image → ELA (Q=90) → Resize(384) → Normalize(ELA stats) → UNet(ResNet-34) → Binary Mask
#
# Change from P.3: EPOCHS 25→50, PATIENCE 7→10, NUM_WORKERS 2→4
# Hypothesis: P.3 was still improving at epoch 25 (best=last). More training time = better results.


# =============================================================================
# CELL 1: Markdown — Changelog
# =============================================================================
# ## Changelog
# - **vR.P.7:** Extended training (50 epochs, patience 10). P.3's best epoch was 25/25
#   (val loss still decreasing, Pixel F1 still increasing). This version gives the model
#   more time to converge. Also increases NUM_WORKERS to 4 for faster data loading.
#   Fixes P.3's NameError bug (denormalize → denormalize_ela).


# =============================================================================
# CELL 2: Configuration & Imports
# =============================================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
VERSION = 'vR.P.7'
CHANGE = 'Extended training (50 epochs, patience 10) — P.3 was still improving at epoch 25'

# --- CHANGED from P.3 ---
EPOCHS = 50              # was 25
PATIENCE = 10            # was 7
NUM_WORKERS = 4          # was 2
PREFETCH_FACTOR = 2      # explicit (was default)

# --- UNCHANGED from P.3 ---
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 3          # ELA is 3-channel RGB
NUM_CLASSES = 1
LEARNING_RATE = 1e-3
ELA_QUALITY = 90
USE_GT_MASKS = True
MASK_AREA_THRESHOLD = 100  # min tampered pixels for image-level classification

CHECKPOINT_PATH = f'{VERSION}_checkpoint.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    # TF32 for faster math on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f'Version: {VERSION}')
print(f'Change: {CHANGE}')
print(f'Device: {DEVICE}')
print(f'PyTorch: {torch.__version__}')
print(f'SMP: {smp.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'TF32: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}')


# =============================================================================
# CELL 3: Dataset Discovery
# =============================================================================
# (Standard CASIA dataset discovery — same as P.3)
# User must set DATASET_ROOT to the actual Kaggle dataset path

DATASET_ROOT = '/kaggle/input/casia-spicing-detection-localization'
# Adjust this path for your environment:
# DATASET_ROOT = '/path/to/casia-dataset'

def discover_dataset(root):
    """Walk dataset directory and categorize images into Au/Tp with GT mask mapping."""
    image_paths = []
    labels = []
    gt_map = {}

    # Find GT masks
    gt_dir = os.path.join(root, 'CASIA 2 Groundtruth')
    if os.path.isdir(gt_dir):
        for f in sorted(os.listdir(gt_dir)):
            if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif')):
                stem = Path(f).stem.lower()
                gt_map[stem] = os.path.join(gt_dir, f)
                # Also store variants
                for variant in [stem + '_gt', stem.replace('tp', 'gt'), stem.replace('Tp', 'Gt')]:
                    gt_map[variant] = os.path.join(gt_dir, f)

    # Find images
    for category in ['Au', 'Tp']:
        cat_dir = os.path.join(root, 'CASIA2', category)
        if not os.path.isdir(cat_dir):
            # Try alternate structure
            for alt in [f'CASIA 2.0/{category}', f'CASIA2.0/{category}', category]:
                alt_dir = os.path.join(root, alt)
                if os.path.isdir(alt_dir):
                    cat_dir = alt_dir
                    break

        if os.path.isdir(cat_dir):
            for f in sorted(os.listdir(cat_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    image_paths.append(os.path.join(cat_dir, f))
                    labels.append(0 if category == 'Au' else 1)

    return image_paths, labels, gt_map

image_paths, labels, gt_map = discover_dataset(DATASET_ROOT)
print(f'Total images: {len(image_paths)}')
print(f'  Authentic: {labels.count(0)}')
print(f'  Tampered:  {labels.count(1)}')
print(f'  GT masks:  {len(gt_map)}')


# =============================================================================
# CELL 4: Data Splitting (70/15/15, stratified, seed=42)
# =============================================================================
from sklearn.model_selection import train_test_split

# First split: 70% train, 30% temp
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.30, random_state=SEED, stratify=labels
)

# Second split: 50/50 of temp → 15% val, 15% test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
)

print(f'Train: {len(train_paths)} ({sum(train_labels)} Tp)')
print(f'Val:   {len(val_paths)} ({sum(val_labels)} Tp)')
print(f'Test:  {len(test_paths)} ({sum(test_labels)} Tp)')

# Data leakage assertion
assert len(set(train_paths) & set(val_paths)) == 0, 'Train-Val overlap!'
assert len(set(train_paths) & set(test_paths)) == 0, 'Train-Test overlap!'
assert len(set(val_paths) & set(test_paths)) == 0, 'Val-Test overlap!'
print('No data leakage detected.')


# =============================================================================
# CELL 5: ELA Computation
# =============================================================================
def compute_ela_image(image_path, quality=90):
    """Compute ELA map from an image path. Returns PIL RGB Image."""
    try:
        original = Image.open(image_path).convert('RGB')
        buffer = BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        ela = ImageChops.difference(original, resaved)
        extrema = ela.getextrema()
        max_diff = max(val[1] for val in extrema)
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        return ela
    except Exception:
        return Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))


def generate_pseudo_mask(image_path, quality=90, threshold=50):
    """Generate a binary pseudo-mask from ELA (fallback when GT mask unavailable)."""
    ela = compute_ela_image(image_path, quality)
    ela_gray = np.array(ela.convert('L'))
    mean_val = ela_gray.mean()
    std_val = ela_gray.std()
    adaptive_thresh = max(threshold, mean_val + 2 * std_val)
    mask = (ela_gray > adaptive_thresh).astype(np.float32)
    return mask


# =============================================================================
# CELL 6: ELA Normalization Statistics
# =============================================================================
def compute_ela_statistics(image_paths, n_samples=500, size=384):
    """Compute channel mean and std of ELA maps from a sample of images."""
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(image_paths), min(n_samples, len(image_paths)), replace=False)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for idx in tqdm(sample_idx, desc='ELA stats', leave=False):
        ela = compute_ela_image(image_paths[idx], quality=ELA_QUALITY)
        ela_resized = ela.resize((size, size), Image.BILINEAR)
        arr = np.array(ela_resized, dtype=np.float64) / 255.0
        channel_sum += arr.reshape(-1, 3).sum(axis=0)
        channel_sq_sum += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += arr.shape[0] * arr.shape[1]

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    std = np.maximum(std, 1e-5)
    return mean.tolist(), std.tolist()

# Compute from training set
ELA_MEAN, ELA_STD = compute_ela_statistics(train_paths, n_samples=500, size=IMAGE_SIZE)
print(f'ELA normalization statistics (from 500 training samples):')
print(f'  Mean: [{ELA_MEAN[0]:.4f}, {ELA_MEAN[1]:.4f}, {ELA_MEAN[2]:.4f}]')
print(f'  Std:  [{ELA_STD[0]:.4f}, {ELA_STD[1]:.4f}, {ELA_STD[2]:.4f}]')


# =============================================================================
# CELL 7: GT Mask Loading
# =============================================================================
def get_gt_mask(image_path, target_size):
    """Get ground truth mask for an image.
    - Authentic images: all-zero mask
    - Tampered images: GT mask if available, else ELA pseudo-mask
    """
    is_tampered = '/Tp/' in image_path or '\\Tp\\' in image_path

    if not is_tampered:
        return np.zeros((target_size, target_size), dtype=np.float32)

    if USE_GT_MASKS:
        stem = Path(image_path).stem.lower()
        variants = [stem, stem + '_gt', stem.replace('tp', 'gt'), stem.replace('Tp', 'Gt')]
        for v in variants:
            if v in gt_map:
                mask = Image.open(gt_map[v]).convert('L')
                mask = mask.resize((target_size, target_size), Image.NEAREST)
                mask_arr = np.array(mask, dtype=np.float32)
                if mask_arr.max() > 1:
                    mask_arr = mask_arr / 255.0
                mask_arr = (mask_arr > 0.5).astype(np.float32)
                return mask_arr

    # Fallback: ELA pseudo-mask
    try:
        mask = generate_pseudo_mask(image_path)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((target_size, target_size), Image.NEAREST)
        return np.array(mask_pil, dtype=np.float32) / 255.0
    except Exception:
        return np.zeros((target_size, target_size), dtype=np.float32)


# =============================================================================
# CELL 8: Dataset Class
# =============================================================================
class CASIASegmentationDataset(Dataset):
    """CASIA v2.0 dataset for tampered region segmentation — ELA input."""

    def __init__(self, image_paths, labels, ela_mean, ela_std,
                 mask_size=384, ela_quality=90):
        self.image_paths = image_paths
        self.labels = labels
        self.mask_size = mask_size
        self.ela_quality = ela_quality
        self.resize = transforms.Resize((mask_size, mask_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=ela_mean, std=ela_std)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # Compute ELA instead of loading RGB
        ela = compute_ela_image(path, quality=self.ela_quality)
        ela = self.resize(ela)
        ela_tensor = self.to_tensor(ela)          # [0, 1]
        ela_tensor = self.normalize(ela_tensor)    # ELA-specific mean/std

        mask = get_gt_mask(path, self.mask_size)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return ela_tensor, mask, label


# =============================================================================
# CELL 9: DataLoaders
# =============================================================================
train_dataset = CASIASegmentationDataset(
    train_paths, train_labels, ELA_MEAN, ELA_STD,
    mask_size=IMAGE_SIZE, ela_quality=ELA_QUALITY
)
val_dataset = CASIASegmentationDataset(
    val_paths, val_labels, ELA_MEAN, ELA_STD,
    mask_size=IMAGE_SIZE, ela_quality=ELA_QUALITY
)
test_dataset = CASIASegmentationDataset(
    test_paths, test_labels, ELA_MEAN, ELA_STD,
    mask_size=IMAGE_SIZE, ela_quality=ELA_QUALITY
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR
)

print(f'Train batches: {len(train_loader)} ({len(train_dataset)} images)')
print(f'Val batches:   {len(val_loader)} ({len(val_dataset)} images)')
print(f'Test batches:  {len(test_loader)} ({len(test_dataset)} images)')
print(f'Workers: {NUM_WORKERS}, Prefetch: {PREFETCH_FACTOR}, Pin memory: True')


# =============================================================================
# CELL 10: Sample Visualization (ELA Input)
# =============================================================================
def denormalize_ela(tensor, mean=ELA_MEAN, std=ELA_STD):
    """Reverse ELA normalization for display."""
    t = tensor.clone()
    for ch in range(3):
        t[ch] = t[ch] * std[ch] + mean[ch]
    return t.clamp(0, 1)


fig, axes = plt.subplots(4, 4, figsize=(16, 16))

sample_indices = []
au_shown, tp_shown = 0, 0
for i in range(len(train_dataset)):
    if train_labels[i] == 0 and au_shown < 2:
        sample_indices.append(i)
        au_shown += 1
    elif train_labels[i] == 1 and tp_shown < 2:
        sample_indices.append(i)
        tp_shown += 1
    if au_shown >= 2 and tp_shown >= 2:
        break

for row, idx in enumerate(sample_indices):
    ela_tensor, mask, label = train_dataset[idx]
    ela_display = denormalize_ela(ela_tensor).permute(1, 2, 0).numpy()

    axes[row, 0].imshow(ela_display)
    axes[row, 0].set_title(f'ELA Input ({"Au" if label==0 else "Tp"})', fontsize=10)
    axes[row, 0].axis('off')

    try:
        orig = Image.open(train_paths[idx]).convert('RGB')
        orig = orig.resize((IMAGE_SIZE, IMAGE_SIZE))
        axes[row, 1].imshow(np.array(orig))
    except Exception:
        axes[row, 1].text(0.5, 0.5, 'Load failed', ha='center', va='center')
    axes[row, 1].set_title('Original RGB (reference)', fontsize=10)
    axes[row, 1].axis('off')

    mask_display = mask.squeeze(0).numpy()
    axes[row, 2].imshow(mask_display, cmap='hot', vmin=0, vmax=1)
    axes[row, 2].set_title(f'GT Mask (sum={mask_display.sum():.0f})', fontsize=10)
    axes[row, 2].axis('off')

    overlay = ela_display.copy()
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[:, :, 0] = mask_display
    overlay = overlay * 0.7 + mask_rgb * 0.3
    overlay = np.clip(overlay, 0, 1)
    axes[row, 3].imshow(overlay)
    axes[row, 3].set_title('ELA + Mask Overlay', fontsize=10)
    axes[row, 3].axis('off')

plt.suptitle(f'{VERSION} — Sample Images: ELA Input | Original RGB | GT Mask | Overlay',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.show()


# =============================================================================
# CELL 11: Model Creation + Freeze Strategy
# =============================================================================
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
    activation=None  # Raw logits — sigmoid applied in loss/postprocessing
)

# Step 1: Freeze ALL encoder parameters
for param in model.encoder.parameters():
    param.requires_grad = False

# Step 2: Unfreeze ONLY BatchNorm layers in encoder (domain adaptation)
bn_param_count = 0
for module in model.encoder.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        for param in module.parameters():
            param.requires_grad = True
            bn_param_count += param.numel()
        module.track_running_stats = True

model = model.to(DEVICE)

# Report parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
head_params = sum(p.numel() for p in model.segmentation_head.parameters() if p.requires_grad)

print(f'Total parameters:       {total_params:,}')
print(f'Trainable parameters:   {trainable_params:,}')
print(f'  Encoder BN params:       {bn_param_count:,}  (BatchNorm only, lr={LEARNING_RATE})')
print(f'  Decoder:              {decoder_params:,}  (lr={LEARNING_RATE})')
print(f'  Segmentation head:        {head_params:,}  (lr={LEARNING_RATE})')
print(f'Frozen parameters:     {frozen_params:,}  (all conv/fc weights)')
print(f'Trainable ratio:      {trainable_params/total_params:.1%}')


# =============================================================================
# CELL 12: Loss Function
# =============================================================================
bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def criterion(pred, target):
    """Combined BCE + Dice loss (equal weight)."""
    return bce_loss_fn(pred, target) + dice_loss_fn(pred, target)


# =============================================================================
# CELL 13: Optimizer + Scheduler
# =============================================================================
trainable_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params_list, lr=LEARNING_RATE, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

print(f'Optimizer: Adam(lr={LEARNING_RATE}, weight_decay=1e-5)')
print(f'Scheduler: ReduceLROnPlateau(factor=0.5, patience=3)')
print(f'Early stopping: patience={PATIENCE}, monitor=val_loss')


# =============================================================================
# CELL 14: Training Functions (with AMP)
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch with AMP mixed precision. Returns average loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    for images, masks, labels in tqdm(loader, desc='Train', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate with AMP. Returns average loss and pixel-level metrics."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_masks = []

    for images, masks, labels in tqdm(loader, desc='Val', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, masks)

        total_loss += loss.item()
        num_batches += 1

        # Collect predictions (cast to float32 for numpy)
        probs = torch.sigmoid(predictions.float())
        all_preds.append(probs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Compute pixel-level metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    binary_preds = (all_preds > 0.5).astype(np.float32)

    pred_flat = binary_preds.flatten()
    mask_flat = all_masks.flatten()

    eps = 1e-7
    tp = (pred_flat * mask_flat).sum()
    fp = (pred_flat * (1 - mask_flat)).sum()
    fn = ((1 - pred_flat) * mask_flat).sum()

    pixel_f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    pixel_iou = tp / (tp + fp + fn + eps)

    return avg_loss, pixel_f1, pixel_iou


# =============================================================================
# CELL 15: Training Loop (with checkpoint save/resume)
# =============================================================================
scaler = GradScaler('cuda')

history = {
    'train_loss': [], 'val_loss': [], 'val_pixel_f1': [], 'val_pixel_iou': [],
    'lr': []
}

best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0
best_model_state = None
start_epoch = 1

# ── Checkpoint Resume ─────────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    print(f'Checkpoint found: {CHECKPOINT_PATH}')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    best_epoch = ckpt['best_epoch']
    patience_counter = ckpt['patience_counter']
    history = ckpt['history']
    if ckpt.get('best_model_state') is not None:
        best_model_state = ckpt['best_model_state']
    print(f'  Resuming from epoch {start_epoch} (best_epoch={best_epoch}, '
          f'best_val_loss={best_val_loss:.4f})')
else:
    print('No checkpoint found — starting fresh.')

# ── Training Loop ─────────────────────────────────────────────────────────────
print(f'Starting training: epochs {start_epoch}-{EPOCHS}, patience={PATIENCE}')
print(f'LR: {LEARNING_RATE} | Input: ELA (Q={ELA_QUALITY}) | AMP: Enabled')
print(f'{"="*80}')

for epoch in range(start_epoch, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]['lr']

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
    val_loss, val_f1, val_iou = validate(model, val_loader, criterion, DEVICE)

    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_pixel_f1'].append(val_f1)
    history['val_pixel_iou'].append(val_iou)
    history.get('lr', history.get('lr_encoder', [])).append(current_lr)

    # Check for improvement
    improved = ''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        improved = ' *'
    else:
        patience_counter += 1

    print(f'Epoch {epoch:>2}/{EPOCHS} | Train Loss: {train_loss:.4f} | '
          f'Val Loss: {val_loss:.4f} | Pixel F1: {val_f1:.4f} | '
          f'IoU: {val_iou:.4f} | LR: {current_lr:.2e}{improved}')

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'best_model_state': best_model_state,
        'history': history,
    }, CHECKPOINT_PATH)

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}')
        break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model = model.to(DEVICE)
    print(f'\nRestored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})')
else:
    print('\nNo improvement during training — using final weights')

print(f'{"="*80}')
print(f'Training complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}')


# =============================================================================
# CELL 16: Training Curves
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_range = range(1, len(history['train_loss']) + 1)

# Loss curves
axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
if len(history['train_loss']) > 25:
    axes[0, 0].axvline(x=25, color='orange', linestyle=':', alpha=0.5,
                        label='P.3 limit (epoch 25)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training vs Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Pixel F1
axes[0, 1].plot(epochs_range, history['val_pixel_f1'], 'g-', linewidth=2)
axes[0, 1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
if len(history['val_pixel_f1']) > 25:
    axes[0, 1].axvline(x=25, color='orange', linestyle=':', alpha=0.5,
                        label='P.3 limit')
    axes[0, 1].legend()
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Pixel F1')
axes[0, 1].set_title('Validation Pixel F1')
axes[0, 1].grid(True, alpha=0.3)

# Pixel IoU
axes[1, 0].plot(epochs_range, history['val_pixel_iou'], 'm-', linewidth=2)
axes[1, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
if len(history['val_pixel_iou']) > 25:
    axes[1, 0].axvline(x=25, color='orange', linestyle=':', alpha=0.5,
                        label='P.3 limit')
    axes[1, 0].legend()
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Pixel IoU')
axes[1, 0].set_title('Validation Pixel IoU')
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(epochs_range, history.get('lr', history.get('lr_encoder', [])), 'k-', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'{VERSION} — Training History ({len(history["train_loss"])} epochs)', fontsize=14)
plt.tight_layout()
plt.show()


# =============================================================================
# CELL 17: Test Evaluation — Pixel-Level Metrics
# =============================================================================
@torch.no_grad()
def evaluate_test(model, loader, device):
    """Full evaluation on test set."""
    model.eval()
    all_probs = []
    all_masks = []
    all_labels = []

    for images, masks, labels in tqdm(loader, desc='Test Eval'):
        images = images.to(device, non_blocking=True)
        predictions = model(images)
        probs = torch.sigmoid(predictions.float())

        all_probs.append(probs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_labels = np.array(all_labels)

    return all_probs, all_masks, all_labels


test_probs, test_masks, test_labels_arr = evaluate_test(model, test_loader, DEVICE)
test_preds_binary = (test_probs > 0.5).astype(np.float32)

# ── Pixel-Level Metrics ──────────────────────────────────────────────────────
pred_flat = test_preds_binary.flatten()
mask_flat = test_masks.flatten()
prob_flat = test_probs.flatten()

eps = 1e-7
tp = (pred_flat * mask_flat).sum()
fp = (pred_flat * (1 - mask_flat)).sum()
fn = ((1 - pred_flat) * mask_flat).sum()
tn = ((1 - pred_flat) * (1 - mask_flat)).sum()

pixel_f1 = (2 * tp) / (2 * tp + fp + fn + eps)
pixel_iou = tp / (tp + fp + fn + eps)
pixel_precision = tp / (tp + fp + eps)
pixel_recall = tp / (tp + fn + eps)

# Pixel AUC (subsample for speed)
n_pixels = len(prob_flat)
if n_pixels > 5_000_000:
    sample_idx = np.random.choice(n_pixels, 5_000_000, replace=False)
    pixel_auc = roc_auc_score(mask_flat[sample_idx], prob_flat[sample_idx])
else:
    pixel_auc = (roc_auc_score(mask_flat, prob_flat)
                 if mask_flat.sum() > 0 and (1 - mask_flat).sum() > 0 else 0.0)

print(f'{"="*60}')
print(f'{VERSION} — PIXEL-LEVEL TEST RESULTS')
print(f'{"="*60}')
print(f'Pixel F1:        {pixel_f1:.4f}')
print(f'Pixel IoU:       {pixel_iou:.4f}')
print(f'Pixel AUC:       {pixel_auc:.4f}')
print(f'Pixel Precision: {pixel_precision:.4f}')
print(f'Pixel Recall:    {pixel_recall:.4f}')


# =============================================================================
# CELL 18: Test Evaluation — Image-Level Classification (Derived)
# =============================================================================
image_pred_labels = []
image_pred_scores = []

for i in range(len(test_probs)):
    prob_map = test_probs[i, 0]
    binary_map = (prob_map > 0.5).astype(np.float32)
    tampered_pixel_count = binary_map.sum()

    pred_label = 1 if tampered_pixel_count >= MASK_AREA_THRESHOLD else 0
    image_pred_labels.append(pred_label)
    image_pred_scores.append(prob_map.max())

image_pred_labels = np.array(image_pred_labels)
image_pred_scores = np.array(image_pred_scores)

# Classification metrics
cls_accuracy = accuracy_score(test_labels_arr, image_pred_labels)
cls_report = classification_report(
    test_labels_arr, image_pred_labels,
    target_names=['Authentic', 'Tampered'], output_dict=True
)
cls_macro_f1 = f1_score(test_labels_arr, image_pred_labels, average='macro')
cls_auc = (roc_auc_score(test_labels_arr, image_pred_scores)
           if len(np.unique(test_labels_arr)) > 1 else 0.0)

print(f'\n{"="*60}')
print(f'{VERSION} — IMAGE-LEVEL TEST RESULTS')
print(f'{"="*60}')
print(f'Image Accuracy:  {cls_accuracy:.4f} ({cls_accuracy:.2%})')
print(f'Image Macro F1:  {cls_macro_f1:.4f}')
print(f'Image ROC-AUC:   {cls_auc:.4f}')
print(f'\nClassification Report:')
print(classification_report(
    test_labels_arr, image_pred_labels,
    target_names=['Authentic', 'Tampered']
))


# =============================================================================
# CELL 19: Confusion Matrix + ROC Curve
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(test_labels_arr, image_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Authentic', 'Tampered'],
            yticklabels=['Authentic', 'Tampered'], ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Confusion Matrix (Acc={cls_accuracy:.2%})')

tn_cls, fp_cls, fn_cls, tp_cls = cm.ravel()
print(f'Confusion Matrix:')
print(f'  TN={tn_cls}, FP={fp_cls}, FN={fn_cls}, TP={tp_cls}')
print(f'  FP Rate: {fp_cls/(tn_cls+fp_cls)*100:.1f}%')
print(f'  FN Rate: {fn_cls/(fn_cls+tp_cls)*100:.1f}%')

# ROC Curve
if len(np.unique(test_labels_arr)) > 1:
    fpr, tpr, _ = roc_curve(test_labels_arr, image_pred_scores)
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {cls_auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve (AUC={cls_auc:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.suptitle(f'{VERSION} — Classification Performance', fontsize=14)
plt.tight_layout()
plt.show()


# =============================================================================
# CELL 20: Per-Image Metrics Distribution
# =============================================================================
per_image_f1 = []
per_image_iou = []
per_image_labels = []

for i in range(len(test_probs)):
    pred = (test_probs[i].flatten() > 0.5).astype(np.float32)
    mask = test_masks[i].flatten()

    tp_i = (pred * mask).sum()
    fp_i = (pred * (1 - mask)).sum()
    fn_i = ((1 - pred) * mask).sum()

    f1_i = (2 * tp_i) / (2 * tp_i + fp_i + fn_i + 1e-7)
    iou_i = tp_i / (tp_i + fp_i + fn_i + 1e-7)

    per_image_f1.append(f1_i)
    per_image_iou.append(iou_i)
    per_image_labels.append(test_labels_arr[i])

per_image_f1 = np.array(per_image_f1)
per_image_iou = np.array(per_image_iou)
per_image_labels = np.array(per_image_labels)

# Stats for tampered images only
tp_mask = per_image_labels == 1
print(f'\nPer-image F1 (tampered only): mean={per_image_f1[tp_mask].mean():.4f}, '
      f'median={np.median(per_image_f1[tp_mask]):.4f}')
print(f'Per-image IoU (tampered only): mean={per_image_iou[tp_mask].mean():.4f}, '
      f'median={np.median(per_image_iou[tp_mask]):.4f}')


# =============================================================================
# CELL 21: Prediction Visualization
# =============================================================================
def visualize_predictions(model, dataset, indices, device, title='Predictions'):
    """Display Original | GT Mask | Predicted Mask | Overlay for each index."""
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    model.eval()

    for row, idx in enumerate(indices):
        img_tensor, gt_mask, label = dataset[idx]
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device))
            pred_prob = torch.sigmoid(pred.float()).cpu().squeeze().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.float32)

        # Denormalize ELA for display — FIXED (was 'denormalize' in P.3)
        img_display = denormalize_ela(img_tensor).permute(1, 2, 0).numpy()
        gt_display = gt_mask.squeeze(0).numpy()

        # Col 0: ELA Input
        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f'ELA Input ({"Tampered" if label==1 else "Authentic"})',
                                fontsize=11)
        axes[row, 0].axis('off')

        # Col 1: Ground Truth
        axes[row, 1].imshow(gt_display, cmap='hot', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Ground Truth (sum={gt_display.sum():.0f})', fontsize=11)
        axes[row, 1].axis('off')

        # Col 2: Predicted Mask
        axes[row, 2].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].set_title(f'Predicted (sum={pred_binary.sum():.0f})', fontsize=11)
        axes[row, 2].axis('off')

        # Col 3: Overlay (green=GT, red=pred)
        overlay = img_display.copy()
        overlay_mask = np.zeros_like(overlay)
        overlay_mask[:, :, 1] = gt_display * 0.4
        overlay_mask[:, :, 0] = pred_binary * 0.4
        combined = np.clip(overlay * 0.6 + overlay_mask, 0, 1)
        axes[row, 3].imshow(combined)
        axes[row, 3].set_title('Overlay (green=GT, red=pred)', fontsize=11)
        axes[row, 3].axis('off')

    plt.suptitle(f'{VERSION} — {title}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# Select sample images
tampered_indices = [i for i, l in enumerate(test_labels_arr) if l == 1]
authentic_indices = [i for i, l in enumerate(test_labels_arr) if l == 0]

print('--- Tampered Image Predictions ---')
visualize_predictions(model, test_dataset, tampered_indices[:4], DEVICE, 'Tampered Images')

print('\n--- Authentic Image Predictions ---')
visualize_predictions(model, test_dataset, authentic_indices[:2], DEVICE, 'Authentic Images')


# =============================================================================
# CELL 22: Tracking Table (Hardcoded Parent + Live Results)
# =============================================================================
print(f'\n{"="*80}')
print(f'TRACKING TABLE — Pretrained Localization Track')
print(f'{"="*80}')
print(f'| Version | Change | Pixel F1 | IoU | Pixel AUC | Img Acc | Epochs | Verdict |')
print(f'|---------|--------|----------|-----|-----------|---------|--------|---------|')
print(f'| vR.P.3  | ELA input (BN unfrozen) | 0.6920 | 0.5291 | 0.9528 | 86.79% | 25 (25) | STRONG POSITIVE |')
print(f'| vR.P.4  | 4ch RGB+ELA | 0.7053 | 0.5447 | 0.9433 | 84.42% | 25 (24) | NEUTRAL |')
print(f'| **{VERSION}** | **{CHANGE}** | **{pixel_f1:.4f}** | '
      f'**{pixel_iou:.4f}** | **{pixel_auc:.4f}** | **{cls_accuracy:.2%}** | '
      f'**{len(history["train_loss"])} ({best_epoch})** | **TBD** |')

# Verdict
delta_from_parent = pixel_f1 - 0.6920
if delta_from_parent >= 0.058:
    verdict = 'STRONG POSITIVE'
elif delta_from_parent >= 0.02:
    verdict = 'POSITIVE'
elif delta_from_parent >= -0.02:
    verdict = 'NEUTRAL'
else:
    verdict = 'NEGATIVE'

print(f'\nDelta from P.3 (Pixel F1): {delta_from_parent:+.4f} ({delta_from_parent*100:+.2f}pp)')
print(f'Verdict: {verdict}')


# =============================================================================
# CELL 23: Model Save
# =============================================================================
model_filename = f'{VERSION}_unet_resnet34_model.pth'

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'history': history,
    'config': {
        'version': VERSION,
        'change': CHANGE,
        'encoder': ENCODER,
        'encoder_weights': ENCODER_WEIGHTS,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs_trained': len(history['train_loss']),
        'best_epoch': best_epoch,
        'patience': PATIENCE,
        'ela_quality': ELA_QUALITY,
        'ela_mean': ELA_MEAN,
        'ela_std': ELA_STD,
        'seed': SEED,
        'num_workers': NUM_WORKERS,
    },
    'results': {
        'pixel_f1': float(pixel_f1),
        'pixel_iou': float(pixel_iou),
        'pixel_auc': float(pixel_auc),
        'pixel_precision': float(pixel_precision),
        'pixel_recall': float(pixel_recall),
        'image_accuracy': float(cls_accuracy),
        'image_macro_f1': float(cls_macro_f1),
        'image_auc': float(cls_auc),
    },
}, model_filename)

file_size = os.path.getsize(model_filename) / (1024 * 1024)
print(f'Model saved: {model_filename} ({file_size:.1f} MB)')
print(f'Contains: model weights, optimizer state, scheduler state, training history, config, results')


# =============================================================================
# CELL 24: Final Summary
# =============================================================================
print(f'\n{"="*80}')
print(f'{VERSION} — FINAL SUMMARY')
print(f'{"="*80}')
print(f'Change: {CHANGE}')
print(f'Parent: vR.P.3 (Pixel F1=0.6920, best epoch=25/25)')
print(f'')
print(f'PIXEL-LEVEL RESULTS:')
print(f'  Pixel F1:        {pixel_f1:.4f}  (P.3: 0.6920, delta: {pixel_f1-0.6920:+.4f})')
print(f'  Pixel IoU:       {pixel_iou:.4f}  (P.3: 0.5291, delta: {pixel_iou-0.5291:+.4f})')
print(f'  Pixel AUC:       {pixel_auc:.4f}  (P.3: 0.9528, delta: {pixel_auc-0.9528:+.4f})')
print(f'  Pixel Precision: {pixel_precision:.4f}')
print(f'  Pixel Recall:    {pixel_recall:.4f}')
print(f'')
print(f'IMAGE-LEVEL RESULTS:')
print(f'  Image Accuracy:  {cls_accuracy:.2%}  (P.3: 86.79%, delta: {(cls_accuracy-0.8679)*100:+.2f}pp)')
print(f'  Image Macro F1:  {cls_macro_f1:.4f}  (P.3: 0.8560, delta: {cls_macro_f1-0.8560:+.4f})')
print(f'  Image ROC-AUC:   {cls_auc:.4f}  (P.3: 0.9502, delta: {cls_auc-0.9502:+.4f})')
print(f'')
print(f'TRAINING:')
print(f'  Epochs trained:  {len(history["train_loss"])}')
print(f'  Best epoch:      {best_epoch}')
print(f'  Best val loss:   {best_val_loss:.4f}')
print(f'  Final LR:        {history["lr"][-1]:.2e}')
print(f'')
print(f'CONFUSION MATRIX:')
print(f'  TN={tn_cls}, FP={fp_cls}, FN={fn_cls}, TP={tp_cls}')
print(f'  FP Rate: {fp_cls/(tn_cls+fp_cls)*100:.1f}%  (P.3: 2.7%)')
print(f'  FN Rate: {fn_cls/(fn_cls+tp_cls)*100:.1f}%  (P.3: 28.6%)')
print(f'')
print(f'VERDICT: {verdict}')
print(f'Model saved: {model_filename} ({file_size:.1f} MB)')
