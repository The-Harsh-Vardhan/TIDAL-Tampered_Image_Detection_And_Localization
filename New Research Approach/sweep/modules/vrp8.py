#!/usr/bin/env python
# Auto-generated from: vR.P.8 Image Detection and Localisation.ipynb
# Do not edit directly -- regenerate with convert_notebooks.py

import sys, os
# Add sweep directory to path for path_config import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ============================================================
# Cell 2
# ============================================================
# ============================================================
# 1. SETUP
# ============================================================

# Install segmentation-models-pytorch
# [SHELL] !pip install -q segmentation-models-pytorch
# [SHELL] !pip install -q wandb

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
import segmentation_models_pytorch as smp

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# ============================================================
# Configuration
# ============================================================
VERSION = 'vR.P.8'
CHANGE = 'ELA + gradual encoder unfreeze (progressive: frozen → layer4 → layer3+layer4)'
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 3  # ELA is 3-channel RGB
NUM_CLASSES = 1
LEARNING_RATE = 1e-3   # Decoder LR
ENCODER_LR = 1e-5      # Encoder LR (100x lower — prevents catastrophic forgetting)
ELA_QUALITY = 90
EPOCHS = 40             # Up from 25 (need 26+ for Stage 2)
PATIENCE = 7            # Reset at stage transitions
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
CHECKPOINT_PATH = f'{VERSION}_checkpoint.pth'
# --- Kaggle Persistence: Files only ---
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
RESULTS_DIR = '/kaggle/working/results'
LOGS_DIR = '/kaggle/working/logs'
for _d in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)
RESUME = True
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Progressive unfreeze schedule
UNFREEZE_SCHEDULE = {
0: {'epochs': (1, 10),  'desc': 'Encoder frozen, BN unfrozen'},
1: {'epochs': (11, 25), 'desc': 'layer4 unfrozen'},
2: {'epochs': (26, 40), 'desc': 'layer3 + layer4 unfrozen'},
}

# Reproducibility

# --- Weights & Biases Experiment Tracking ---
USE_WANDB = True
WANDB_PROJECT = 'tamper-detection-ablation'
DATASET_NAME = 'CASIA2'

import re as _re
_nb_dir = os.path.basename(os.getcwd()).lower()
_run_match = _re.search(r'run-(\d+)', _nb_dir)
RUN_ID = f'run{_run_match.group(1)}' if _run_match else 'run01'
EXPERIMENT_ID = VERSION.lower().replace('.', '').replace(' ', '')

# Infer feature flags from notebook config
_change_lower = CHANGE.lower()
_input_lower = globals().get('INPUT_TYPE', '').lower()
FEATURE_SET = 'rgb'
if 'multi-q' in _input_lower or 'multi-quality' in _change_lower:
    FEATURE_SET = 'multi_quality_ela'
elif 'ela' in _change_lower or 'ela' in _input_lower:
    FEATURE_SET = 'ela'
if 'dct' in _input_lower:
    FEATURE_SET = 'dct' if FEATURE_SET == 'rgb' else f'{FEATURE_SET}+dct'
if 'srm' in _input_lower:
    FEATURE_SET = 'srm_noise'
if 'ycbcr' in _input_lower:
    FEATURE_SET = 'ycbcr'
if 'noiseprint' in _input_lower:
    FEATURE_SET = 'noiseprint'
USE_TTA = 'tta' in _change_lower
JPEG_AUG = 'jpeg' in _change_lower and 'augment' in _change_lower
EDGE_SUPERVISION = 'edge' in _change_lower
NOISE_FEATURES = 'noise' in _input_lower or 'srm' in _input_lower

if USE_WANDB:
    try:
        import wandb
        wandb.login()
        wandb.init(
        project=WANDB_PROJECT,
        name=f'{EXPERIMENT_ID}_{RUN_ID}',
        config={
        'experiment': EXPERIMENT_ID, 'version': VERSION, 'change': CHANGE,
        'run': RUN_ID, 'dataset': DATASET_NAME, 'feature_set': FEATURE_SET,
        'input_type': globals().get('INPUT_TYPE', FEATURE_SET),
        'tta': USE_TTA, 'jpeg_aug': JPEG_AUG,
        'edge_supervision': EDGE_SUPERVISION, 'noise_features': NOISE_FEATURES,
        'encoder': ENCODER, 'in_channels': IN_CHANNELS,
        'img_size': IMG_SIZE, 'batch_size': BATCH_SIZE,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'patience': PATIENCE,
        },
        reinit=True,
        )
        print(f'W&B run initialized: {EXPERIMENT_ID}_{RUN_ID}')
    except Exception as e:
        print(f'W&B init failed ({e}), continuing without tracking')
        USE_WANDB = False

print(f'Experiment: {EXPERIMENT_ID} | Run: {RUN_ID} | Features: {FEATURE_SET}')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    # TF32 for faster math on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# GPU info
print(f"{'='*60}")
print(f"  {VERSION} — {CHANGE}")
print(f"{'='*60}")
print(f"Device:   {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch:  {torch.__version__}")
print(f"SMP:      {smp.__version__}")
print(f"Image:    {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Input:    ELA (Q={ELA_QUALITY})")
print(f"Encoder:  {ENCODER} ({ENCODER_WEIGHTS}, progressive unfreeze)")
print(f"Batch:    {BATCH_SIZE}")
print(f"Dec LR:   {LEARNING_RATE}")
print(f"Enc LR:   {ENCODER_LR} (active from Stage 1)")
print(f"Epochs:   {EPOCHS}")
print(f"Patience: {PATIENCE} (reset at stage transitions)")
print(f"Seed:     {SEED}")
print(f"Workers:  {NUM_WORKERS} (prefetch={PREFETCH_FACTOR})")
print(f"AMP:      Enabled (mixed precision)")
print(f"TF32:     Enabled")
print()
print("Progressive Unfreeze Schedule:")
for stage, info in UNFREEZE_SCHEDULE.items():
    print(f"  Stage {stage}: epochs {info['epochs'][0]}-{info['epochs'][1]} — {info['desc']}")

# ============================================================
# Cell 4
# ============================================================
# ============================================================
# 2.1 Dataset Path Discovery (FIXED in vR.P.1)
# ============================================================

def find_dataset():
    """Search /kaggle/input/ for Au/ and Tp/ directories.
    
    FIXED in vR.P.1: Collects ALL candidate dirs containing Au+Tp,
    then prefers the one with 'image' in the path (not 'mask').
    Also detects sibling MASK directory as ground truth.
    
    Returns: (dataset_root, au_dir, tp_dir, gt_mask_dir_or_None)
    """
    search_roots = ['/kaggle/input', '/content/drive/MyDrive']
    candidates = []  # list of (dirpath, au_path, tp_path)
    
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if 'Au' in dirnames and 'Tp' in dirnames:
                candidates.append((
                dirpath,
                os.path.join(dirpath, 'Au'),
                os.path.join(dirpath, 'Tp')
                ))
    
    if not candidates:
        return None, None, None, None
    
    # Separate IMAGE vs MASK candidates
    image_candidates = [c for c in candidates if 'mask' not in c[0].lower()]
    mask_candidates = [c for c in candidates if 'mask' in c[0].lower()]
    
    # Prefer IMAGE directory; fall back to first candidate if no IMAGE found
    if image_candidates:
        # Among image candidates, prefer the one with 'image' in path
        explicit_image = [c for c in image_candidates if 'image' in c[0].lower()]
        chosen = explicit_image[0] if explicit_image else image_candidates[0]
    else:
        chosen = candidates[0]
    
    # Detect GT mask directory
    gt_dir = None
    if mask_candidates:
        # Use the MASK candidate that has Au/Tp structure (for tampered masks)
        gt_dir = mask_candidates[0][0]  # root of MASK/{Au,Tp}
    
    return chosen[0], chosen[1], chosen[2], gt_dir

DATASET_ROOT, AU_DIR, TP_DIR, GT_DIR_ROOT = find_dataset()

if DATASET_ROOT is None:
    for base in ['/kaggle/input']:
        if os.path.isdir(base):
            print(f'Contents of {base}:')
            for dirpath, dirnames, filenames in os.walk(base):
                depth = dirpath.replace(base, '').count(os.sep)
                print(f'{"  " * depth}{os.path.basename(dirpath)}/')
                if depth >= 3:
                    break
    raise FileNotFoundError('Could not find Au/ and Tp/ directories.')

# Resolve GT mask directory
# GT_DIR_ROOT may be MASK/ which contains Au/ and Tp/ subdirs.
# For tampered images, GT masks are in MASK/Tp/
# We need to check if MASK/Tp/ contains actual mask images.
GT_DIR = None
if GT_DIR_ROOT is not None:
    # Check if MASK/Tp/ has image files (those are GT masks for tampered images)
    gt_tp_dir = os.path.join(GT_DIR_ROOT, 'Tp')
    if os.path.isdir(gt_tp_dir):
        gt_files = [f for f in os.listdir(gt_tp_dir)
        if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}]
        if gt_files:
            GT_DIR = GT_DIR_ROOT  # MASK directory with Au/Tp structure
            print(f'GT mask structure detected: {GT_DIR}')
            print(f'  MASK/Au: {len(os.listdir(os.path.join(GT_DIR, "Au")))} files')
            print(f'  MASK/Tp: {len(gt_files)} mask files')

# If GT_DIR_ROOT didn't work, search for other GT mask directories
if GT_DIR is None:
    # Search within dataset neighborhood
    search_base = os.path.dirname(DATASET_ROOT)
    for root, dirs, files in os.walk(search_base):
        for d in dirs:
            if any(pat in d.lower() for pat in ['groundtruth', 'gt', 'mask']):
                candidate = os.path.join(root, d)
                if any(Path(f).suffix.lower() in {'.jpg','.jpeg','.png','.tif','.bmp'}
                for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))):
                    GT_DIR = candidate
                    break
        if GT_DIR:
            break
    
    # Search separate Kaggle datasets
    if GT_DIR is None:
        input_dir = '/kaggle/input'
        if os.path.isdir(input_dir):
            for d in sorted(os.listdir(input_dir)):
                if any(pat in d.lower() for pat in ['groundtruth', 'gt', 'mask']):
                    for root, dirs, files in os.walk(os.path.join(input_dir, d)):
                        img_files = [f for f in files if Path(f).suffix.lower() in {'.jpg','.jpeg','.png','.tif','.bmp'}]
                        if img_files:
                            GT_DIR = root
                            break
                    if GT_DIR:
                        break

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

print(f'\nDataset root:  {DATASET_ROOT}')
print(f'Authentic dir: {AU_DIR}  ({len(os.listdir(AU_DIR))} files)')
print(f'Tampered dir:  {TP_DIR}  ({len(os.listdir(TP_DIR))} files)')
if GT_DIR:
    print(f'GT mask dir:   {GT_DIR}')
else:
    print(f'GT mask dir:   NOT FOUND \u2014 will generate pseudo-masks from ELA')


# ============================================================
# Cell 5
# ============================================================
# ============================================================
# 2.2 Collect Image Paths and Ground Truth Masks (UPDATED)
# ============================================================

def collect_paths(directory):
    """Collect sorted image paths from a directory."""
    paths = []
    for f in sorted(os.listdir(directory)):
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(directory, f))
    return paths

au_paths = collect_paths(AU_DIR)
tp_paths = collect_paths(TP_DIR)

# Build GT mask lookup
# The GT directory may have MASK/Au/ and MASK/Tp/ structure
# OR it may be a flat directory with mask files
gt_map = {}
if GT_DIR:
    gt_tp_dir = os.path.join(GT_DIR, 'Tp')
    gt_au_dir = os.path.join(GT_DIR, 'Au')
    
    # If GT has Au/Tp structure, collect from Tp subdirectory
    if os.path.isdir(gt_tp_dir):
        for f in sorted(os.listdir(gt_tp_dir)):
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                stem = Path(f).stem.lower()
                gt_map[stem] = os.path.join(gt_tp_dir, f)
        print(f'GT masks loaded from MASK/Tp/: {len(gt_map)}')
    else:
        # Flat directory
        for f in sorted(os.listdir(GT_DIR)):
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                stem = Path(f).stem.lower()
                gt_map[stem] = os.path.join(GT_DIR, f)
        print(f'GT masks loaded (flat): {len(gt_map)}')

# Match tampered images to GT masks
tp_with_gt = 0
tp_without_gt = 0
for tp in tp_paths:
    stem = Path(tp).stem.lower()
    # Try exact match and common variants
    variants = [stem, stem + '_gt', stem.replace('tp', 'gt'), stem.replace('Tp', 'Gt')]
    found = any(v in gt_map for v in variants)
    if found:
        tp_with_gt += 1
    else:
        tp_without_gt += 1

USE_GT_MASKS = GT_DIR is not None and tp_with_gt > len(tp_paths) * 0.5

print(f'\nAuthentic images:  {len(au_paths)}')
print(f'Tampered images:   {len(tp_paths)}')
print(f'Total:             {len(au_paths) + len(tp_paths)}')
print(f'Class ratio (Au:Tp): {len(au_paths)/len(tp_paths):.2f}:1')
if GT_DIR:
    print(f'\nTampered with GT mask:    {tp_with_gt}')
    print(f'Tampered without GT mask: {tp_without_gt}')
print(f'\nUsing GT masks: {USE_GT_MASKS}')
if not USE_GT_MASKS:
    print('  -> Will generate pseudo-masks from ELA thresholding')


# ============================================================
# Cell 6
# ============================================================
# ============================================================
# 2.3 ELA Pseudo-Mask Generation (Fallback)
# ============================================================

def compute_ela(image_path, quality=90):
    """Compute Error Level Analysis map."""
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

def generate_pseudo_mask(image_path, quality=90, threshold=50):
    """Generate a binary pseudo-mask from ELA.
    Pixels with ELA brightness above threshold are marked as tampered.
    """
    ela = compute_ela(image_path, quality)
    ela_gray = np.array(ela.convert('L'))
    # Adaptive threshold: use mean + 2*std of the ELA map
    mean_val = ela_gray.mean()
    std_val = ela_gray.std()
    adaptive_thresh = max(threshold, mean_val + 2 * std_val)
    mask = (ela_gray > adaptive_thresh).astype(np.float32)
    return mask

def get_gt_mask(image_path, target_size):
    """Get ground truth mask for an image.
    - Authentic images: all-zero mask
    - Tampered images: GT mask if available, else ELA pseudo-mask
    """
    is_tampered = '/Tp/' in image_path or '\\Tp\\' in image_path

    if not is_tampered:
        # Authentic â€” all zeros (no tampering)
        return np.zeros((target_size, target_size), dtype=np.float32)

    if USE_GT_MASKS:
        # Try to find matching GT mask
        stem = Path(image_path).stem.lower()
        variants = [stem, stem + '_gt', stem.replace('tp', 'gt'), stem.replace('Tp', 'Gt')]
        for v in variants:
            if v in gt_map:
                mask = Image.open(gt_map[v]).convert('L')
                mask = mask.resize((target_size, target_size), Image.NEAREST)
                mask_arr = np.array(mask, dtype=np.float32)
                # Normalize to [0, 1]
                if mask_arr.max() > 1:
                    mask_arr = mask_arr / 255.0
                # Binarize
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

# Quick test
test_au = au_paths[0]
test_tp = tp_paths[0]
mask_au = get_gt_mask(test_au, IMAGE_SIZE)
mask_tp = get_gt_mask(test_tp, IMAGE_SIZE)
print(f'Authentic mask â€” shape: {mask_au.shape}, sum: {mask_au.sum():.0f} (should be 0)')
print(f'Tampered mask  â€” shape: {mask_tp.shape}, sum: {mask_tp.sum():.0f} (should be > 0)')

# ============================================================
# Cell 8
# ============================================================
# ============================================================
# 3.1 PyTorch Dataset and Transforms (CHANGED: ELA input)
# ============================================================

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

# ---- Compute ELA-specific normalization statistics ----
print('Computing ELA normalization statistics from training set sample...')

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
        arr = np.array(ela_resized, dtype=np.float64) / 255.0  # [0, 1]
        channel_sum += arr.reshape(-1, 3).sum(axis=0)
        channel_sq_sum += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += arr.shape[0] * arr.shape[1]

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)
    # Clamp std to avoid division by zero
    std = np.maximum(std, 1e-5)
    return mean.tolist(), std.tolist()

# We'll compute stats after data splitting (need train_paths)
# Placeholder â€” will be set after split
ELA_MEAN = [0.5, 0.5, 0.5]
ELA_STD = [0.25, 0.25, 0.25]

class CASIASegmentationDataset(Dataset):
    """CASIA v2.0 dataset for tampered region segmentation â€” ELA input."""

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
        ela_tensor = self.to_tensor(ela)  # [0, 1]
        ela_tensor = self.normalize(ela_tensor)  # ELA-specific mean/std

        mask = get_gt_mask(path, self.mask_size)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return ela_tensor, mask, label

print(f'Dataset class ready (ELA input, Q={ELA_QUALITY}).')
print(f'ELA stats will be computed after data splitting.')

# ============================================================
# Cell 9
# ============================================================
# ============================================================
# 3.2 Data Splitting (70/15/15 Stratified) + ELA Stats
# ============================================================

all_paths = au_paths + tp_paths
all_labels = [0] * len(au_paths) + [1] * len(tp_paths)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=SEED)

val_paths, test_paths, val_labels, test_labels = train_test_split(
temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED)

# Compute ELA normalization statistics from training set
ELA_MEAN, ELA_STD = compute_ela_statistics(train_paths, n_samples=500, size=IMAGE_SIZE)
print(f'ELA normalization statistics (from 500 training samples):')
print(f'  Mean: [{ELA_MEAN[0]:.4f}, {ELA_MEAN[1]:.4f}, {ELA_MEAN[2]:.4f}]')
print(f'  Std:  [{ELA_STD[0]:.4f}, {ELA_STD[1]:.4f}, {ELA_STD[2]:.4f}]')

train_dataset = CASIASegmentationDataset(
train_paths, train_labels, ela_mean=ELA_MEAN, ela_std=ELA_STD,
mask_size=IMAGE_SIZE, ela_quality=ELA_QUALITY)
val_dataset = CASIASegmentationDataset(
val_paths, val_labels, ela_mean=ELA_MEAN, ela_std=ELA_STD,
mask_size=IMAGE_SIZE, ela_quality=ELA_QUALITY)
test_dataset = CASIASegmentationDataset(
test_paths, test_labels, ela_mean=ELA_MEAN, ela_std=ELA_STD,
mask_size=IMAGE_SIZE, ela_quality=ELA_QUALITY)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
num_workers=NUM_WORKERS, pin_memory=True,
persistent_workers=NUM_WORKERS > 0,
prefetch_factor=PREFETCH_FACTOR, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
num_workers=NUM_WORKERS, pin_memory=True,
persistent_workers=NUM_WORKERS > 0,
prefetch_factor=PREFETCH_FACTOR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
num_workers=NUM_WORKERS, pin_memory=True,
persistent_workers=NUM_WORKERS > 0,
prefetch_factor=PREFETCH_FACTOR)

print(f'\nTrain: {len(train_dataset):>6} images  (Au: {sum(1 for l in train_labels if l==0)}, Tp: {sum(1 for l in train_labels if l==1)})')
print(f'Val:   {len(val_dataset):>6} images  (Au: {sum(1 for l in val_labels if l==0)}, Tp: {sum(1 for l in val_labels if l==1)})')
print(f'Test:  {len(test_dataset):>6} images  (Au: {sum(1 for l in test_labels if l==0)}, Tp: {sum(1 for l in test_labels if l==1)})')
print(f'\nTrain batches: {len(train_loader)}  (drop_last=True)')
print(f'Val batches:   {len(val_loader)}')
print(f'Test batches:  {len(test_loader)}')
print(f'Workers:       {NUM_WORKERS}  (persistent={NUM_WORKERS > 0}, prefetch={PREFETCH_FACTOR})')

# ============================================================
# Cell 10 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 3.3 Sample Visualization (ELA Input)
    # ============================================================
    
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
    
        # Column 0: ELA input (what the model sees)
        axes[row, 0].imshow(ela_display)
        axes[row, 0].set_title(f'ELA Input ({"Au" if label==0 else "Tp"})', fontsize=10)
        axes[row, 0].axis('off')
    
        # Column 1: Original RGB (for reference)
        try:
            orig = Image.open(train_paths[idx]).convert('RGB')
            orig = orig.resize((IMAGE_SIZE, IMAGE_SIZE))
            axes[row, 1].imshow(np.array(orig))
        except Exception:
            axes[row, 1].text(0.5, 0.5, 'Load failed', ha='center', va='center')
        axes[row, 1].set_title('Original RGB (reference)', fontsize=10)
        axes[row, 1].axis('off')
    
        # Column 2: GT Mask
        mask_display = mask.squeeze(0).numpy()
        axes[row, 2].imshow(mask_display, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].set_title(f'GT Mask (sum={mask_display.sum():.0f})', fontsize=10)
        axes[row, 2].axis('off')
    
        # Column 3: ELA + Mask overlay
        overlay = ela_display.copy()
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 0] = mask_display
        overlay = overlay * 0.7 + mask_rgb * 0.3
        overlay = np.clip(overlay, 0, 1)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('ELA + Mask Overlay', fontsize=10)
        axes[row, 3].axis('off')
    
    plt.suptitle('Sample Images: ELA Input | Original RGB | GT Mask | Overlay', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
except Exception as _viz_err:
    print(f'[Cell 10] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 12
# ============================================================
# ============================================================
# 4.1 Build Model + Progressive Unfreeze Helpers
# ============================================================

model = smp.Unet(
encoder_name=ENCODER,
encoder_weights=ENCODER_WEIGHTS,
in_channels=IN_CHANNELS,
classes=NUM_CLASSES,
activation=None
)

# --- Helper Functions for Progressive Unfreeze ---

def freeze_encoder(model):
    """Freeze all encoder params, then unfreeze BN for domain adaptation."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    for module in model.encoder.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            for param in module.parameters():
                param.requires_grad = True
            module.track_running_stats = True

def unfreeze_layer4(model):
    """Unfreeze layer4 of ResNet-34 encoder."""
    for param in model.encoder.layer4.parameters():
        param.requires_grad = True
    n = sum(p.numel() for p in model.encoder.layer4.parameters())
    print(f'  Unfroze encoder.layer4: {n:,} params')

def unfreeze_layer3_and_layer4(model):
    """Unfreeze layer3 + layer4 of ResNet-34 encoder."""
    for param in model.encoder.layer3.parameters():
        param.requires_grad = True
    for param in model.encoder.layer4.parameters():
        param.requires_grad = True
    n3 = sum(p.numel() for p in model.encoder.layer3.parameters())
    n4 = sum(p.numel() for p in model.encoder.layer4.parameters())
    print(f'  Unfroze encoder.layer3: {n3:,} params + encoder.layer4: {n4:,} params')

def get_current_stage(epoch):
    """Return stage number (0, 1, or 2) based on epoch."""
    if epoch <= 10:
        return 0
    elif epoch <= 25:
        return 1
    else:
        return 2

def rebuild_optimizer(model, encoder_lr, decoder_lr, stage):
    """Rebuild optimizer with correct param groups for current stage."""
    decoder_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    if stage > 0:
        # Stage 1 or 2: dual param groups (encoder conv+BN at low LR, decoder at high LR)
        encoder_trainable = [p for p in model.encoder.parameters() if p.requires_grad]
        new_optimizer = optim.Adam([
        {'params': encoder_trainable, 'lr': encoder_lr, 'weight_decay': 1e-5},
        {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': 1e-5},
        ])
    else:
        # Stage 0: single param group (BN + decoder at decoder LR)
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        new_optimizer = optim.Adam(all_trainable, lr=decoder_lr, weight_decay=1e-5)
    return new_optimizer

# --- Initial state: Stage 0 (frozen encoder + BN unfrozen) ---
freeze_encoder(model)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
seghead_trainable = sum(p.numel() for p in model.segmentation_head.parameters() if p.requires_grad)

print(f'Model: UNet + {ENCODER} ({ENCODER_WEIGHTS}) — PROGRESSIVE UNFREEZE')
print(f'Initial state: Stage 0 (frozen encoder + BN unfrozen)')
print(f'Total parameters:     {total_params:>12,}')
print(f'Trainable parameters: {trainable_params:>12,}')
print(f'  Encoder BN params:  {encoder_trainable:>12,}  (BatchNorm only)')
print(f'  Decoder:            {decoder_trainable:>12,}  (lr={LEARNING_RATE})')
print(f'  Segmentation head:  {seghead_trainable:>12,}  (lr={LEARNING_RATE})')
print(f'Frozen parameters:    {frozen_params:>12,}  (all conv/fc weights)')
print(f'Trainable ratio:      {trainable_params/total_params*100:.1f}% (will increase at transitions)')
print()
print('Unfreeze schedule:')
for stage, info in UNFREEZE_SCHEDULE.items():
    print(f"  Stage {stage}: epochs {info['epochs'][0]}-{info['epochs'][1]} — {info['desc']}")

# ============================================================
# Cell 14
# ============================================================
# ============================================================
# 5.1 Loss, Optimizer, Scheduler
# ============================================================

bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def criterion(pred, target):
    return bce_loss_fn(pred, target) + dice_loss_fn(pred, target)

# Stage 0: single param group (decoder + encoder BN at LEARNING_RATE)
trainable_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params_list, lr=LEARNING_RATE, weight_decay=1e-5)

print(f'Optimizer: Adam (Stage 0 — single LR)')
print(f'  Trainable params: {sum(p.numel() for p in trainable_params_list):,}')
print(f'  LR: {LEARNING_RATE}')
print(f'  (Encoder LR {ENCODER_LR} activates at Stage 1)')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
optimizer, mode='min', factor=0.5, patience=3
)

# ============================================================
# 5.2 Training and Validation Functions (AMP-enabled)
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
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
        probs = torch.sigmoid(predictions.float())
        all_preds.append(probs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
    avg_loss = total_loss / num_batches
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

print('Training functions ready (AMP-enabled).')

# ============================================================
# Cell 15
# ============================================================
# ============================================================
# 5.3 Training Loop (progressive unfreeze + checkpoint + AMP)
# ============================================================

scaler = GradScaler('cuda')

history = {
'train_loss': [], 'val_loss': [], 'val_pixel_f1': [], 'val_pixel_iou': [],
'lr_encoder': [], 'lr_decoder': []
}

best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0
best_model_state = None
start_epoch = 1
current_stage = 0

# --- Checkpoint resume with stage awareness ---
if RESUME and os.path.exists(LATEST_CHECKPOINT):
    print(f'Checkpoint found: {LATEST_CHECKPOINT}')
    ckpt = torch.load(LATEST_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)

    # Re-apply freeze state based on saved stage
    saved_stage = ckpt.get('current_stage', 0)
    current_stage = saved_stage
    freeze_encoder(model)  # Start frozen + BN
    if saved_stage >= 1:
        unfreeze_layer4(model)
    if saved_stage >= 2:
        unfreeze_layer3_and_layer4(model)

    # Rebuild optimizer with correct param groups, then load state
    optimizer = rebuild_optimizer(model, ENCODER_LR, LEARNING_RATE, current_stage)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
    )
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
    print(f'  Resuming from epoch {start_epoch} (stage={current_stage}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})')
else:
    print('No checkpoint found — starting fresh.')

print(f'Starting training: epochs {start_epoch}-{EPOCHS}, patience={PATIENCE}')
print(f'Dec LR: {LEARNING_RATE} | Enc LR: {ENCODER_LR} (from Stage 1) | Input: ELA (Q={ELA_QUALITY}) | AMP: Enabled')
print(f'Current stage: {current_stage} — {UNFREEZE_SCHEDULE[current_stage]["desc"]}')
print(f'{"="*80}')

for epoch in range(start_epoch, EPOCHS + 1):
    # --- Stage transition check ---
    new_stage = get_current_stage(epoch)
    if new_stage != current_stage:
        old_stage = current_stage
        current_stage = new_stage
        print(f'\n{"="*60}')
        print(f'  STAGE TRANSITION: {old_stage} → {current_stage} at epoch {epoch}')
        print(f'  {UNFREEZE_SCHEDULE[current_stage]["desc"]}')
        print(f'{"="*60}')

        if current_stage == 1:
            unfreeze_layer4(model)
        elif current_stage == 2:
            unfreeze_layer3_and_layer4(model)

        # Rebuild optimizer and scheduler for new param groups
        optimizer = rebuild_optimizer(model, ENCODER_LR, LEARNING_RATE, current_stage)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
        )
        scaler = GradScaler('cuda')  # Fresh scaler for new gradient magnitudes

        # Reset early stopping patience (don't penalize transition dips)
        patience_counter = 0

        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Optimizer rebuilt | Trainable: {trainable_now:,} | Patience reset')
        print(f'{"="*60}')

    # --- Record LR ---
    if len(optimizer.param_groups) == 2:
        enc_lr = optimizer.param_groups[0]['lr']
        dec_lr = optimizer.param_groups[1]['lr']
    else:
        enc_lr = 0.0
        dec_lr = optimizer.param_groups[0]['lr']

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
    val_loss, val_f1, val_iou = validate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_pixel_f1'].append(val_f1)
    history['val_pixel_iou'].append(val_iou)
    history['lr_encoder'].append(enc_lr)
    history['lr_decoder'].append(dec_lr)
    if USE_WANDB:
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'val_pixel_f1': val_f1, 'val_pixel_iou': val_iou,
            'lr_encoder': enc_lr, 'lr_decoder': dec_lr})

    improved = ''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_model_state, BEST_MODEL_PATH)
        improved = ' *'
    else:
        patience_counter += 1

    stage_tag = f'[S{current_stage}]'
    print(f'Epoch {epoch:>2}/{EPOCHS} {stage_tag} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
    f'F1: {val_f1:.4f} | IoU: {val_iou:.4f} | EncLR: {enc_lr:.1e} | DecLR: {dec_lr:.1e}{improved}')

    # Checkpoint with stage info
    torch.save({
    'epoch': epoch, 'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
    'patience_counter': patience_counter, 'best_model_state': best_model_state,
    'history': history, 'current_stage': current_stage,
    }, LATEST_CHECKPOINT)

    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch} (Stage {current_stage}). Best epoch: {best_epoch}')
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model = model.to(DEVICE)
    print(f'\nRestored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})')
else:
    print('\nNo improvement during training — using final weights')

print(f'{"="*80}')
print(f'Training complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}')

# ============================================================
# Cell 17
# ============================================================
# ============================================================
# 6.1 Test Set Evaluation â€” Pixel-Level Metrics
# ============================================================

@torch.no_grad()
def evaluate_test(model, loader, device):
    """Full evaluation on test set."""
    model.eval()
    all_probs = []
    all_masks = []
    all_labels = []

    for images, masks, labels in tqdm(loader, desc='Test Eval'):
        images = images.to(device)
        predictions = model(images)
        probs = torch.sigmoid(predictions)

        all_probs.append(probs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (N, 1, H, W)
    all_masks = np.concatenate(all_masks, axis=0)   # (N, 1, H, W)
    all_labels = np.array(all_labels)

    return all_probs, all_masks, all_labels

test_probs, test_masks, test_labels = evaluate_test(model, test_loader, DEVICE)
test_preds_binary = (test_probs > 0.5).astype(np.float32)

# Pixel-level metrics (flatten all pixels)
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
pixel_dice = pixel_f1  # Dice = F1 for binary
pixel_precision = tp / (tp + fp + eps)
pixel_recall = tp / (tp + fn + eps)

# Pixel AUC (subsample for speed if needed)
n_pixels = len(prob_flat)
if n_pixels > 5_000_000:
    sample_idx = np.random.choice(n_pixels, 5_000_000, replace=False)
    pixel_auc = roc_auc_score(mask_flat[sample_idx], prob_flat[sample_idx])
else:
    pixel_auc = roc_auc_score(mask_flat, prob_flat) if mask_flat.sum() > 0 and (1-mask_flat).sum() > 0 else 0.0

print(f'{"="*60}')
print(f'  PIXEL-LEVEL METRICS (Test Set)')
print(f'{"="*60}')
print(f'  Pixel Precision:  {pixel_precision:.4f}')
print(f'  Pixel Recall:     {pixel_recall:.4f}')
print(f'  Pixel F1 (Dice):  {pixel_f1:.4f}')
print(f'  Pixel IoU:        {pixel_iou:.4f}')
print(f'  Pixel AUC:        {pixel_auc:.4f}')
print(f'{"="*60}')

if USE_WANDB:
    wandb.log({'pixel_f1': pixel_f1, 'pixel_iou': pixel_iou,
    'pixel_precision': pixel_precision, 'pixel_recall': pixel_recall,
    'pixel_auc': pixel_auc})


# ============================================================
# Cell 18
# ============================================================
# ============================================================
# 6.2 Test Set Evaluation â€” Image-Level Classification
# ============================================================

# Derive image-level classification from masks:
# An image is classified as "tampered" if any predicted pixel > threshold
MASK_AREA_THRESHOLD = 100  # minimum number of tampered pixels to classify as tampered

image_pred_labels = []
image_pred_scores = []

for i in range(len(test_probs)):
    prob_map = test_probs[i, 0]  # (H, W)
    binary_map = (prob_map > 0.5).astype(np.float32)
    tampered_pixel_count = binary_map.sum()

    # Classification: tampered if enough pixels predicted as tampered
    pred_label = 1 if tampered_pixel_count >= MASK_AREA_THRESHOLD else 0
    image_pred_labels.append(pred_label)

    # Score: max probability in the mask (for ROC-AUC)
    image_pred_scores.append(prob_map.max())

image_pred_labels = np.array(image_pred_labels)
image_pred_scores = np.array(image_pred_scores)

# Classification metrics
cls_accuracy = accuracy_score(test_labels, image_pred_labels)
cls_report = classification_report(test_labels, image_pred_labels,
target_names=['Authentic', 'Tampered'],
output_dict=True)
cls_macro_f1 = f1_score(test_labels, image_pred_labels, average='macro')
cls_auc = roc_auc_score(test_labels, image_pred_scores) if len(np.unique(test_labels)) > 1 else 0.0

print(f'{"="*60}')
print(f'  IMAGE-LEVEL CLASSIFICATION (Test Set)')
print(f'{"="*60}')
print(f'  Test Accuracy:    {cls_accuracy:.4f}  ({cls_accuracy*100:.2f}%)')
print(f'  Macro F1:         {cls_macro_f1:.4f}')
print(f'  ROC-AUC:          {cls_auc:.4f}')
print(f'')
print(f'  Per-Class Results:')
print(f'  {"":>12} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
for cls_name in ['Authentic', 'Tampered']:
    r = cls_report[cls_name]
    print(f'  {cls_name:>12} {r["precision"]:>10.4f} {r["recall"]:>10.4f} {r["f1-score"]:>10.4f} {r["support"]:>10.0f}')
print(f'{"="*60}')

# Classification report (full)
print('\nFull Classification Report:')
print(classification_report(test_labels, image_pred_labels, target_names=['Authentic', 'Tampered']))

if USE_WANDB:
    wandb.log({'image_accuracy': cls_accuracy, 'image_macro_f1': cls_macro_f1,
    'image_roc_auc': cls_auc})


# ============================================================
# Cell 19 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 6.3 Confusion Matrix and ROC Curve
    # ============================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, image_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Authentic', 'Tampered'],
    yticklabels=['Authentic', 'Tampered'], ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Confusion Matrix (Acc={cls_accuracy:.2%})')
    
    # Print confusion details
    tn, fp, fn, tp_cls = cm.ravel()
    total = cm.sum()
    print(f'Confusion Matrix:')
    print(f'  TN={tn}, FP={fp}, FN={fn}, TP={tp_cls}')
    print(f'  FP Rate: {fp/(tn+fp)*100:.1f}%')
    print(f'  FN Rate: {fn/(fn+tp_cls)*100:.1f}%')
    
    # ROC Curve
    if len(np.unique(test_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(test_labels, image_pred_scores)
        axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {cls_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve (AUC={cls_auc:.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{VERSION} â€” Classification Performance', fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as _viz_err:
    print(f'[Cell 19] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 20 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 6.4 Training Curves (with stage transition markers)
    # ============================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    n_epochs = len(history['train_loss'])
    
    # Stage transition lines on all subplots
    for ax in axes.flat:
        if n_epochs > 10:
            ax.axvline(x=10.5, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
        if n_epochs > 25:
            ax.axvline(x=25.5, color='purple', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Loss curves
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pixel F1
    axes[0, 1].plot(epochs_range, history['val_pixel_f1'], 'g-', linewidth=2)
    axes[0, 1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pixel F1')
    axes[0, 1].set_title('Validation Pixel F1')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pixel IoU
    axes[1, 0].plot(epochs_range, history['val_pixel_iou'], 'm-', linewidth=2)
    axes[1, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Pixel IoU')
    axes[1, 0].set_title('Validation Pixel IoU')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate (dual)
    axes[1, 1].plot(epochs_range, history['lr_decoder'], 'r-', label='Decoder LR', linewidth=2)
    axes[1, 1].plot(epochs_range, history['lr_encoder'], 'b-', label='Encoder LR', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add stage labels
    if n_epochs > 10:
        axes[0, 0].text(5.5, axes[0, 0].get_ylim()[1] * 0.95, 'S0', fontsize=9, ha='center', color='orange', fontweight='bold')
    if n_epochs > 15:
        axes[0, 0].text(18, axes[0, 0].get_ylim()[1] * 0.95, 'S1', fontsize=9, ha='center', color='orange', fontweight='bold')
    if n_epochs > 30:
        axes[0, 0].text(33, axes[0, 0].get_ylim()[1] * 0.95, 'S2', fontsize=9, ha='center', color='purple', fontweight='bold')
    
    plt.suptitle(f'{VERSION} — Training History (Progressive Unfreeze)', fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as _viz_err:
    print(f'[Cell 20] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 22 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 7.1 Original / Ground Truth / Predicted / Overlay Grid
    # ============================================================
    
    def visualize_predictions(model, dataset, indices, device, title='Predictions'):
        """Display Original | GT Mask | Predicted Mask | Overlay for each index."""
        n = len(indices)
        fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
    
        model.eval()
    
        for row, idx in enumerate(indices):
            img_tensor, gt_mask, label = dataset[idx]
    
            # Predict
            with torch.no_grad():
                pred_logit = model(img_tensor.unsqueeze(0).to(device))
                pred_prob = torch.sigmoid(pred_logit).cpu().squeeze().numpy()
    
            pred_binary = (pred_prob > 0.5).astype(np.float32)
    
            # Denormalize image for display
            img_display = denormalize_ela(img_tensor).permute(1, 2, 0).numpy()
            gt_display = gt_mask.squeeze(0).numpy()
    
            # Col 0: Original
            axes[row, 0].imshow(img_display)
            axes[row, 0].set_title(f'Original ({"Tampered" if label==1 else "Authentic"})', fontsize=11)
            axes[row, 0].axis('off')
    
            # Col 1: Ground Truth
            axes[row, 1].imshow(gt_display, cmap='hot', vmin=0, vmax=1)
            axes[row, 1].set_title(f'Ground Truth (sum={gt_display.sum():.0f})', fontsize=11)
            axes[row, 1].axis('off')
    
            # Col 2: Predicted Mask
            axes[row, 2].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
            axes[row, 2].set_title(f'Predicted (sum={pred_binary.sum():.0f})', fontsize=11)
            axes[row, 2].axis('off')
    
            # Col 3: Overlay
            overlay = img_display.copy()
            # Green for GT, Red for Predicted
            overlay_mask = np.zeros_like(overlay)
            overlay_mask[:, :, 1] = gt_display * 0.4       # Green = GT
            overlay_mask[:, :, 0] = pred_binary * 0.4       # Red = Predicted
            combined = np.clip(overlay * 0.6 + overlay_mask, 0, 1)
            axes[row, 3].imshow(combined)
            axes[row, 3].set_title('Overlay (green=GT, red=pred)', fontsize=11)
            axes[row, 3].axis('off')
    
        plt.suptitle(f'{VERSION} â€” {title}', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.show()
    
    # Select sample images: 3 tampered + 2 authentic from test set
    tampered_indices = [i for i, l in enumerate(test_labels) if l == 1]
    authentic_indices = [i for i, l in enumerate(test_labels) if l == 0]
    
    sample_tp = tampered_indices[:4]
    sample_au = authentic_indices[:2]
    
    print('--- Tampered Image Predictions ---')
    visualize_predictions(model, test_dataset, sample_tp, DEVICE, 'Tampered Images')
    
    print('\n--- Authentic Image Predictions ---')
    visualize_predictions(model, test_dataset, sample_au, DEVICE, 'Authentic Images')
    
    if USE_WANDB:
        try:
            wandb.log({'prediction_examples': wandb.Image(plt.gcf())})
        except Exception:
            pass
    
except Exception as _viz_err:
    print(f'[Cell 22] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 23 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 7.2 Per-Image Metric Distribution
    # ============================================================
    
    # Compute per-image pixel F1
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
        per_image_labels.append(test_labels[i])
    
    per_image_f1 = np.array(per_image_f1)
    per_image_iou = np.array(per_image_iou)
    per_image_labels = np.array(per_image_labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 distribution by class
    for cls, name in [(0, 'Authentic'), (1, 'Tampered')]:
        mask_cls = per_image_labels == cls
        axes[0].hist(per_image_f1[mask_cls], bins=30, alpha=0.6, label=f'{name} (n={mask_cls.sum()})')
    axes[0].set_xlabel('Pixel F1 Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Per-Image Pixel F1 Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU distribution by class
    for cls, name in [(0, 'Authentic'), (1, 'Tampered')]:
        mask_cls = per_image_labels == cls
        axes[1].hist(per_image_iou[mask_cls], bins=30, alpha=0.6, label=f'{name} (n={mask_cls.sum()})')
    axes[1].set_xlabel('Pixel IoU Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Per-Image Pixel IoU Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{VERSION} â€” Per-Image Metric Distributions', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Summary stats
    print(f'Per-Image Pixel F1:')
    print(f'  Tampered:  mean={per_image_f1[per_image_labels==1].mean():.4f}, '
    f'std={per_image_f1[per_image_labels==1].std():.4f}')
    print(f'  Authentic: mean={per_image_f1[per_image_labels==0].mean():.4f}, '
    f'std={per_image_f1[per_image_labels==0].std():.4f}')
except Exception as _viz_err:
    print(f'[Cell 23] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 25
# ============================================================
# ============================================================
# 8.1 Results Tracking Table
# ============================================================

print(f'{"="*80}')
print(f'  RESULTS SUMMARY — {VERSION}')
print(f'{"="*80}')
print()

# Ablation comparison table
print('Cross-Track Comparison:')
print(f'{"Version":<10} {"Track":<12} {"Encoder":<12} {"Input":<10} {"Test Acc":<10} '
f'{"Macro F1":<10} {"AUC":<8} {"Tam-F1":<10} {"Pixel-F1":<10} {"IoU":<8}')
print('-' * 100)

# ETASR results (from vR.1.1)
print(f'{"vR.1.1":<10} {"ETASR":<12} {"2-layer CNN":<12} {"ELA 128²":<10} '
f'{"88.38%":<10} {"0.8805":<10} {"0.9601":<8} {"0.8606":<10} {"N/A":<10} {"N/A":<8}')

# P.3 baseline for comparison
print(f'{"vR.P.3":<10} {"Pretrained":<12} {"ResNet-34":<12} {"ELA 384²":<10} '
f'{"(pending)":<10} {"(pending)":<10} {"(pend)":<8} {"(pending)":<10} {"0.6920":<10} {"(pend)":<8}')

# This run
print(f'{VERSION:<10} {"Pretrained":<12} {"ResNet-34":<12} {"ELA 384²":<10} '
f'{cls_accuracy*100:.2f}{"%-":<7} {cls_macro_f1:<10.4f} {cls_auc:<8.4f} '
f'{cls_report["Tampered"]["f1-score"]:<10.4f} {pixel_f1:<10.4f} {pixel_iou:<8.4f}')
print()

# Full metrics summary
print(f'Pixel-Level Metrics:')
print(f'  Pixel F1 (Dice): {pixel_f1:.4f}')
print(f'  Pixel IoU:       {pixel_iou:.4f}')
print(f'  Pixel Precision: {pixel_precision:.4f}')
print(f'  Pixel Recall:    {pixel_recall:.4f}')
print(f'  Pixel AUC:       {pixel_auc:.4f}')
print()
print(f'Image-Level Metrics:')
print(f'  Accuracy:        {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)')
print(f'  Macro F1:        {cls_macro_f1:.4f}')
print(f'  ROC-AUC:         {cls_auc:.4f}')
print(f'  Au Precision:    {cls_report["Authentic"]["precision"]:.4f}')
print(f'  Au Recall:       {cls_report["Authentic"]["recall"]:.4f}')
print(f'  Au F1:           {cls_report["Authentic"]["f1-score"]:.4f}')
print(f'  Tp Precision:    {cls_report["Tampered"]["precision"]:.4f}')
print(f'  Tp Recall:       {cls_report["Tampered"]["recall"]:.4f}')
print(f'  Tp F1:           {cls_report["Tampered"]["f1-score"]:.4f}')
print()
print(f'Training:')
print(f'  Best epoch:      {best_epoch}')
print(f'  Epochs trained:  {len(history["train_loss"])}')
print(f'  Best val loss:   {best_val_loss:.4f}')
print(f'  Final stage:     {get_current_stage(len(history["train_loss"]))}')
print(f'{"="*80}')

# ============================================================
# Cell 27
# ============================================================
# ============================================================
# 10. Save Model
# ============================================================

model_filename = f'{VERSION}_unet_resnet34_model.pth'
torch.save({
'model_state_dict': model.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
'scheduler_state_dict': scheduler.state_dict(),
'best_epoch': best_epoch,
'best_val_loss': best_val_loss,
'history': history,
'config': {
'version': VERSION, 'encoder': ENCODER, 'encoder_weights': ENCODER_WEIGHTS,
'image_size': IMAGE_SIZE, 'batch_size': BATCH_SIZE,
'learning_rate': LEARNING_RATE, 'encoder_lr': ENCODER_LR,
'ela_quality': ELA_QUALITY,
'input_type': 'ELA',
'unfreeze_schedule': 'progressive: frozen → layer4 → layer3+layer4',
'epochs_trained': len(history['train_loss']),
'seed': SEED,
}
}, model_filename)

print(f'Model saved: {model_filename}')
print(f'File size: {os.path.getsize(model_filename) / 1e6:.1f} MB')
print(f'\n{VERSION} complete.')

# --- Save Experiment Artifacts ---
import json as _json
from datetime import datetime as _dt

# Save metrics.json
_metrics = {
'version': VERSION,
'best_epoch': best_epoch,
'best_val_loss': float(best_val_loss),
'epochs_trained': len(history['train_loss']),
'history': {k: [float(v) for v in vals] for k, vals in history.items()},
}
with open(os.path.join(RESULTS_DIR, f'{VERSION}_metrics.json'), 'w') as _f:
    _json.dump(_metrics, _f, indent=2)
print(f'Metrics saved to {RESULTS_DIR}/{VERSION}_metrics.json')

# Save experiment metadata to CSV (append)
_csv_path = os.path.join(RESULTS_DIR, 'experiment_results.csv')
_row = {
'experiment_id': VERSION,
'run_id': f'{VERSION}_{_dt.now().strftime("%Y%m%d_%H%M%S")}',
'dataset': 'CASIA_v2.0',
'model': 'UNet_ResNet34',
'seed': SEED,
'timestamp': _dt.now().isoformat(),
'best_epoch': best_epoch,
'best_val_loss': float(best_val_loss),
'epochs_trained': len(history['train_loss']),
}
try:
    _row['pixel_f1'] = float(pixel_f1)
    _row['pixel_iou'] = float(pixel_iou)
    _row['cls_accuracy'] = float(cls_accuracy)
except NameError:
    pass
_header = not os.path.exists(_csv_path)
with open(_csv_path, 'a') as _f:
    if _header:
        _f.write(','.join(_row.keys()) + '\n')
    _f.write(','.join(str(v) for v in _row.values()) + '\n')
print(f'Experiment results appended to {_csv_path}')

# Save best model to persistent location
if best_model_state is not None:
    torch.save(best_model_state, BEST_MODEL_PATH)
    print(f'Best model saved to {BEST_MODEL_PATH}')


if USE_WANDB:
    try:
        artifact = wandb.Artifact(name=f'{EXPERIMENT_ID}_{RUN_ID}_model', type='model')
        if os.path.exists(BEST_MODEL_PATH):
            artifact.add_file(BEST_MODEL_PATH)
            wandb.log_artifact(artifact)
    except Exception as e:
        print(f'W&B artifact logging failed: {e}')
    wandb.finish()
    print('W&B run finished.')

