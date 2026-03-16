#!/usr/bin/env python
# Auto-generated from: vR.P.18 Image Detection and Localisation.ipynb
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
VERSION = 'vR.P.18'
CHANGE = 'Compression robustness testing (no training, evaluate P.3 under recompression)'
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 3
NUM_CLASSES = 1
ELA_QUALITY = 90  # Single quality for ELA computation
QUALITY_FACTORS = [95, 90, 80, 70]  # Recompression levels to test
CHECKPOINT_DIR = '/kaggle/input'  # Adjust to where P.3 weights are uploaded
NUM_WORKERS = 0  # Kaggle compatibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Kaggle Persistence: Files only ---
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
RESULTS_DIR = "/kaggle/working/results"
LOGS_DIR = "/kaggle/working/logs"
for _d in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)
RESUME = True  # Kaggle persistence: auto-resume from checkpoints
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# Reproducibility

# --- Weights & Biases Experiment Tracking ---
USE_WANDB = True
WANDB_PROJECT = 'Tampered Image Detection & Localization'
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
        name=VERSION,
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
print(f"  {VERSION} \u2014 {CHANGE}")
print(f"{'='*60}")
print(f"Device:     {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch:    {torch.__version__}")
print(f"SMP:        {smp.__version__}")
print(f"Image:      {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"ELA Q:      {ELA_QUALITY}")
print(f"Test Q:     {QUALITY_FACTORS}")
print(f"Encoder:    {ENCODER} (loading P.3 weights)")
print(f"Batch:      {BATCH_SIZE}")
print(f"Seed:       {SEED}")
print(f"Workers:    {NUM_WORKERS}")
print(f"Training:   NONE (evaluation only)")
print(f"AMP:        Enabled (inference)")


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
        # Authentic — all zeros (no tampering)
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
print(f'Authentic mask \u2014 shape: {mask_au.shape}, sum: {mask_au.sum():.0f} (should be 0)')
print(f'Tampered mask  \u2014 shape: {mask_tp.shape}, sum: {mask_tp.sum():.0f} (should be > 0)')

# ============================================================
# Cell 8
# ============================================================
# ============================================================
# 3.1 ELA Functions and Robustness Dataset
# ============================================================

def compute_ela_image(image_path, quality=90):
    """Standard ELA computation."""
    try:
        original = Image.open(image_path).convert('RGB')
        buffer = BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        ela = ImageChops.difference(original, resaved)
        extrema = ela.getextrema()
        max_diff = max(val[1] for val in extrema)
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        return ela
    except Exception:
        return Image.new('RGB', (384, 384), (0, 0, 0))


def compute_ela_with_recompression(image_path, recompress_quality, ela_quality=90):
    """Recompress image at given quality, then compute ELA."""
    try:
        original = Image.open(image_path).convert('RGB')
        # Step 1: Recompress at test quality
        buf1 = BytesIO()
        original.save(buf1, 'JPEG', quality=recompress_quality)
        buf1.seek(0)
        recompressed = Image.open(buf1).convert('RGB')
        # Step 2: Compute ELA on recompressed image
        buf2 = BytesIO()
        recompressed.save(buf2, 'JPEG', quality=ela_quality)
        buf2.seek(0)
        resaved = Image.open(buf2).convert('RGB')
        ela = ImageChops.difference(recompressed, resaved)
        extrema = ela.getextrema()
        max_diff = max(val[1] for val in extrema)
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        return ela
    except Exception:
        return Image.new('RGB', (384, 384), (0, 0, 0))


def compute_ela_statistics(paths, n_samples=500, target_size=384, quality=90):
    """Compute ELA normalization statistics (mean, std) from image paths."""
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(paths), min(n_samples, len(paths)), replace=False)
    tensors = []
    for idx in tqdm(sample_idx, desc='Computing ELA stats'):
        ela = compute_ela_image(paths[idx], quality=quality)
        ela = ela.resize((target_size, target_size), Image.BILINEAR)
        tensors.append(transforms.ToTensor()(ela))
    stack = torch.stack(tensors)
    return stack.mean(dim=(0, 2, 3)).tolist(), stack.std(dim=(0, 2, 3)).tolist()


class CASIARobustnessDataset(Dataset):
    """Test dataset with optional JPEG recompression before ELA."""
    def __init__(self, image_paths, labels, ela_mean, ela_std,
    mask_size=384, ela_quality=90, recompress_quality=None):
        self.image_paths = image_paths
        self.labels = labels
        self.mask_size = mask_size
        self.ela_quality = ela_quality
        self.recompress_quality = recompress_quality
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=ela_mean, std=ela_std)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        if self.recompress_quality is not None:
            ela = compute_ela_with_recompression(path, self.recompress_quality, self.ela_quality)
        else:
            ela = compute_ela_image(path, quality=self.ela_quality)
        ela = ela.resize((self.mask_size, self.mask_size), Image.BILINEAR)
        ela_tensor = self.normalize(self.to_tensor(ela))
        mask = get_gt_mask(path, self.mask_size)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return ela_tensor, mask, label


print(f'Dataset class ready (Robustness testing, ELA Q={ELA_QUALITY}).')
print(f'Test conditions: Original + Q={QUALITY_FACTORS}')


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

# Compute ELA normalization statistics from training set (original, no recompression)
# This matches P.3's training conditions
ELA_MEAN, ELA_STD = compute_ela_statistics(
train_paths, n_samples=500, target_size=IMAGE_SIZE, quality=ELA_QUALITY)
print(f'ELA normalization statistics (Q={ELA_QUALITY}, from 500 training samples):')
print(f'  Mean: [{ELA_MEAN[0]:.4f}, {ELA_MEAN[1]:.4f}, {ELA_MEAN[2]:.4f}]')
print(f'  Std:  [{ELA_STD[0]:.4f}, {ELA_STD[1]:.4f}, {ELA_STD[2]:.4f}]')

# No train/val loaders needed --- evaluation only
# Test loaders will be created per-condition in the evaluation loop

print(f'\nTrain: {len(train_paths):>6} images  (Au: {sum(1 for l in train_labels if l==0)}, Tp: {sum(1 for l in train_labels if l==1)})')
print(f'Val:   {len(val_paths):>6} images  (Au: {sum(1 for l in val_labels if l==0)}, Tp: {sum(1 for l in val_labels if l==1)})')
print(f'Test:  {len(test_paths):>6} images  (Au: {sum(1 for l in test_labels if l==0)}, Tp: {sum(1 for l in test_labels if l==1)})')
print(f'\nTest loaders will be created per-condition in the evaluation loop.')
print(f'Workers: {NUM_WORKERS}')


# ============================================================
# Cell 10 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 3.3 ELA Comparison Across Recompression Levels
    # ============================================================
    
    # Show how the SAME image looks under different recompression + ELA
    conditions_viz = [None] + QUALITY_FACTORS  # [None, 95, 90, 80, 70]
    condition_labels = ['Original'] + [f'Q={q}' for q in QUALITY_FACTORS]
    
    # Pick 2 tampered and 1 authentic image from test set
    tp_test_idx = [i for i, l in enumerate(test_labels) if l == 1][:2]
    au_test_idx = [i for i, l in enumerate(test_labels) if l == 0][:1]
    viz_indices = tp_test_idx + au_test_idx
    
    fig, axes = plt.subplots(len(viz_indices), len(conditions_viz) + 1, 
    figsize=(4 * (len(conditions_viz) + 1), 4 * len(viz_indices)))
    if len(viz_indices) == 1:
        axes = axes[np.newaxis, :]
    
    for row, idx in enumerate(viz_indices):
        path = test_paths[idx]
        label = test_labels[idx]
        
        # Column 0: Original RGB
        try:
            orig = Image.open(path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            axes[row, 0].imshow(np.array(orig))
        except Exception:
            axes[row, 0].text(0.5, 0.5, 'Load failed', ha='center', va='center')
        axes[row, 0].set_title(f'RGB ({"Tp" if label==1 else "Au"})', fontsize=10)
        axes[row, 0].axis('off')
        
        # Columns 1-5: ELA under each condition
        for col, (qf, qlabel) in enumerate(zip(conditions_viz, condition_labels)):
            if qf is None:
                ela = compute_ela_image(path, quality=ELA_QUALITY)
            else:
                ela = compute_ela_with_recompression(path, qf, ELA_QUALITY)
            ela = ela.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            axes[row, col + 1].imshow(np.array(ela))
            axes[row, col + 1].set_title(f'ELA: {qlabel}', fontsize=10)
            axes[row, col + 1].axis('off')
    
    plt.suptitle(f'{VERSION} --- ELA Under Different Recompression Levels', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
    print('Note: Stronger recompression (lower Q) changes the ELA pattern, potentially confusing the model.')
    
except Exception as _viz_err:
    print(f'[Cell 10] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 12
# ============================================================
# ============================================================
# 4.1 Load Pre-trained P.3 Model
# ============================================================

# Build architecture (no pretrained weights -- will load from checkpoint)
model = smp.Unet(
encoder_name=ENCODER,
encoder_weights=None,
in_channels=IN_CHANNELS,
classes=NUM_CLASSES,
activation=None
)

# Try to load P.3 checkpoint
checkpoint_found = False
search_dirs = ['/kaggle/input', '.', '/content/drive/MyDrive']

for ckpt_dir in search_dirs:
    if not os.path.isdir(ckpt_dir):
        continue
    for fname in os.listdir(ckpt_dir):
        if 'vR.P.3' in fname and fname.endswith('.pth'):
            ckpt_path = os.path.join(ckpt_dir, fname)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            elif 'best_model_state' in ckpt and ckpt['best_model_state'] is not None:
                model.load_state_dict(ckpt['best_model_state'])
            else:
                model.load_state_dict(ckpt)
            checkpoint_found = True
            print(f'Loaded checkpoint: {ckpt_path}')
            if 'best_epoch' in ckpt:
                print(f'  Best epoch: {ckpt["best_epoch"]}')
            if 'best_val_loss' in ckpt:
                print(f'  Best val loss: {ckpt["best_val_loss"]:.4f}')
            break
    if checkpoint_found:
        break

# Also search subdirectories of /kaggle/input
if not checkpoint_found:
    for ckpt_dir in search_dirs:
        if not os.path.isdir(ckpt_dir):
            continue
        for root, dirs, files in os.walk(ckpt_dir):
            for fname in files:
                if 'vR.P.3' in fname and fname.endswith('.pth'):
                    ckpt_path = os.path.join(root, fname)
                    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                    if 'model_state_dict' in ckpt:
                        model.load_state_dict(ckpt['model_state_dict'])
                    elif 'best_model_state' in ckpt and ckpt['best_model_state'] is not None:
                        model.load_state_dict(ckpt['best_model_state'])
                    else:
                        model.load_state_dict(ckpt)
                    checkpoint_found = True
                    print(f'Loaded checkpoint: {ckpt_path}')
                    break
            if checkpoint_found:
                break
        if checkpoint_found:
            break

if not checkpoint_found:
    print('WARNING: No P.3 checkpoint found!')
    print('  Falling back to ImageNet pretrained weights + frozen body + BN unfrozen.')
    print('  Results will NOT represent P.3 performance.')
    # Fallback: load ImageNet weights as P.3 would have started
    model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
    activation=None
    )
    # Apply P.3 freeze strategy
    for param in model.encoder.parameters():
        param.requires_grad = False
    for module in model.encoder.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            for param in module.parameters():
                param.requires_grad = True
            module.track_running_stats = True

model = model.to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f'\nModel: UNet + {ENCODER}')
print(f'Total parameters: {total_params:,}')
print(f'Mode: eval (no training)')
print(f'Checkpoint: {"P.3" if checkpoint_found else "FALLBACK (ImageNet)"}')


# ============================================================
# Cell 14
# ============================================================
# Training components not needed --- evaluation only
print('Training SKIPPED --- using P.3 pre-trained weights')
print(f'Model is in eval mode: {not model.training}')


# ============================================================
# Cell 15
# ============================================================
# ============================================================
# 5.3 Robustness Evaluation Loop
# ============================================================

conditions = [None] + QUALITY_FACTORS  # [None, 95, 90, 80, 70]
robustness_results = {}

for qf in conditions:
    label = 'Original' if qf is None else f'Q={qf}'
    print(f'\n{"="*60}')
    print(f'  Evaluating: {label}')
    print(f'{"="*60}')

    test_ds = CASIARobustnessDataset(
    test_paths, test_labels, ELA_MEAN, ELA_STD,
    mask_size=IMAGE_SIZE, ela_quality=ELA_QUALITY,
    recompress_quality=qf
    )
    test_loader_qf = DataLoader(test_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=True)

    # Run inference
    model.eval()
    all_preds, all_masks, all_labels = [], [], []
    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader_qf, desc=label, leave=False):
            images = images.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(images)
            probs = torch.sigmoid(logits.float())
            all_preds.append(probs.cpu().numpy())
            all_masks.append(masks.numpy())
            all_labels.extend(labels.numpy())

    preds = np.concatenate(all_preds)
    masks_arr = np.concatenate(all_masks)
    labels_arr = np.array(all_labels)

    # Pixel metrics
    binary = (preds > 0.5).astype(np.float32)
    pred_flat = binary.flatten()
    mask_flat = masks_arr.flatten()
    eps = 1e-7
    tp = (pred_flat * mask_flat).sum()
    fp = (pred_flat * (1 - mask_flat)).sum()
    fn = ((1 - pred_flat) * mask_flat).sum()
    pixel_f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    pixel_iou = tp / (tp + fp + fn + eps)
    pixel_prec = tp / (tp + fp + eps)
    pixel_rec = tp / (tp + fn + eps)

    # Pixel AUC
    prob_flat = preds.flatten()
    n_pixels = len(prob_flat)
    if n_pixels > 5_000_000:
        sample_idx = np.random.choice(n_pixels, 5_000_000, replace=False)
        pixel_auc = roc_auc_score(mask_flat[sample_idx], prob_flat[sample_idx])
    else:
        pixel_auc = roc_auc_score(mask_flat, prob_flat) if mask_flat.sum() > 0 and (1-mask_flat).sum() > 0 else 0.0

    # Image-level classification
    img_preds = []
    img_scores = []
    for i in range(len(preds)):
        pm = preds[i, 0]
        bm = (pm > 0.5).astype(np.float32)
        img_preds.append(1 if bm.sum() >= 100 else 0)
        img_scores.append(pm.max())
    img_preds = np.array(img_preds)
    img_scores = np.array(img_scores)
    img_acc = accuracy_score(labels_arr, img_preds)
    img_f1 = f1_score(labels_arr, img_preds, average='macro')
    img_auc = roc_auc_score(labels_arr, img_scores) if len(np.unique(labels_arr)) > 1 else 0.0

    robustness_results[label] = {
    'pixel_f1': pixel_f1, 'pixel_iou': pixel_iou,
    'pixel_prec': pixel_prec, 'pixel_rec': pixel_rec,
    'pixel_auc': pixel_auc,
    'img_acc': img_acc, 'img_f1': img_f1, 'img_auc': img_auc,
    'preds': preds, 'masks': masks_arr, 'labels': labels_arr,
    'img_preds': img_preds, 'img_scores': img_scores,
    }

    print(f'  Pixel F1:  {pixel_f1:.4f}')
    print(f'  Pixel IoU: {pixel_iou:.4f}')
    print(f'  Pixel AUC: {pixel_auc:.4f}')
    print(f'  Img Acc:   {img_acc:.4f} ({img_acc*100:.2f}%)')
    print(f'  Img F1:    {img_f1:.4f}')

# Summary table
print(f'\n{"="*60}')
print(f'  ROBUSTNESS SUMMARY')
print(f'{"="*60}')
print(f'  {"Condition":<12} {"Pixel F1":>10} {"Pixel IoU":>10} {"Img Acc":>10} {"Delta F1":>10}')
baseline_f1 = robustness_results['Original']['pixel_f1']
for cond, res in robustness_results.items():
    delta = res['pixel_f1'] - baseline_f1
    print(f'  {cond:<12} {res["pixel_f1"]:>10.4f} {res["pixel_iou"]:>10.4f} {res["img_acc"]:>10.4f} {delta:>+10.4f}')


# ============================================================
# Cell 17
# ============================================================
# ============================================================
# 6.1 Pixel-Level Metrics --- Detailed Breakdown Per Condition
# ============================================================

print(f'{"="*80}')
print(f'  PIXEL-LEVEL METRICS (Test Set) --- Per Condition')
print(f'{"="*80}')
print(f'  {"Condition":<12} {"Precision":>10} {"Recall":>10} {"F1 (Dice)":>10} {"IoU":>10} {"AUC":>10}')
print(f'  {"-"*62}')

for cond, res in robustness_results.items():
    print(f'  {cond:<12} {res["pixel_prec"]:>10.4f} {res["pixel_rec"]:>10.4f} '
    f'{res["pixel_f1"]:>10.4f} {res["pixel_iou"]:>10.4f} {res["pixel_auc"]:>10.4f}')

print(f'{"="*80}')

# Degradation analysis
baseline_f1 = robustness_results['Original']['pixel_f1']
baseline_iou = robustness_results['Original']['pixel_iou']
print(f'\nDegradation from Original:')
for cond, res in robustness_results.items():
    if cond == 'Original':
        continue
    df1 = res['pixel_f1'] - baseline_f1
    diou = res['pixel_iou'] - baseline_iou
    print(f'  {cond}: F1 {df1:+.4f} ({df1/baseline_f1*100:+.1f}%), IoU {diou:+.4f} ({diou/baseline_iou*100:+.1f}%)')

if USE_WANDB:
    wandb.log({'pixel_f1': pixel_f1, 'pixel_iou': pixel_iou,
    'pixel_precision': pixel_precision, 'pixel_recall': pixel_recall,
    'pixel_auc': pixel_auc})


# ============================================================
# Cell 18
# ============================================================
# ============================================================
# 6.2 Image-Level Classification --- Per Condition
# ============================================================

print(f'{"="*80}')
print(f'  IMAGE-LEVEL CLASSIFICATION (Test Set) --- Per Condition')
print(f'{"="*80}')
print(f'  {"Condition":<12} {"Accuracy":>10} {"Macro F1":>10} {"ROC-AUC":>10}')
print(f'  {"-"*42}')

for cond, res in robustness_results.items():
    print(f'  {cond:<12} {res["img_acc"]:>10.4f} {res["img_f1"]:>10.4f} {res["img_auc"]:>10.4f}')

print(f'{"="*80}')

# Per-condition classification reports
for cond, res in robustness_results.items():
    print(f'\n--- {cond} ---')
    print(classification_report(
    res['labels'], res['img_preds'],
    target_names=['Authentic', 'Tampered']
    ))

if USE_WANDB:
    wandb.log({'image_accuracy': cls_accuracy, 'image_macro_f1': cls_macro_f1,
    'image_roc_auc': cls_auc})


# ============================================================
# Cell 19 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 6.3 Confusion Matrices (One Per Condition)
    # ============================================================
    
    n_cond = len(robustness_results)
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 4))
    
    for ax_idx, (cond, res) in enumerate(robustness_results.items()):
        cm = confusion_matrix(res['labels'], res['img_preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Au', 'Tp'],
        yticklabels=['Au', 'Tp'], ax=axes[ax_idx])
        axes[ax_idx].set_xlabel('Predicted')
        axes[ax_idx].set_ylabel('Actual')
        axes[ax_idx].set_title(f'{cond}\nAcc={res["img_acc"]:.2%}', fontsize=10)
    
    plt.suptitle(f'{VERSION} --- Confusion Matrices Across Conditions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print confusion details
    for cond, res in robustness_results.items():
        cm = confusion_matrix(res['labels'], res['img_preds'])
        tn, fp_cm, fn_cm, tp_cm = cm.ravel()
        print(f'{cond}: TN={tn}, FP={fp_cm}, FN={fn_cm}, TP={tp_cm} | '
        f'FPR={fp_cm/(tn+fp_cm)*100:.1f}%, FNR={fn_cm/(fn_cm+tp_cm)*100:.1f}%')
    
except Exception as _viz_err:
    print(f'[Cell 19] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 20 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 6.4 Degradation Curves: Metrics vs Compression Quality
    # ============================================================
    
    conditions_list = list(robustness_results.keys())
    f1_values = [robustness_results[c]['pixel_f1'] for c in conditions_list]
    iou_values = [robustness_results[c]['pixel_iou'] for c in conditions_list]
    acc_values = [robustness_results[c]['img_acc'] for c in conditions_list]
    auc_values = [robustness_results[c]['pixel_auc'] for c in conditions_list]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = range(len(conditions_list))
    
    # Pixel F1
    axes[0, 0].plot(x, f1_values, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(conditions_list, rotation=15)
    axes[0, 0].set_ylabel('Pixel F1')
    axes[0, 0].set_title('Pixel F1 vs Compression')
    axes[0, 0].grid(True, alpha=0.3)
    for i, v in enumerate(f1_values):
        axes[0, 0].annotate(f'{v:.4f}', (i, v), textcoords='offset points',
        xytext=(0, 10), ha='center', fontsize=9)
    
    # Pixel IoU
    axes[0, 1].plot(x, iou_values, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(conditions_list, rotation=15)
    axes[0, 1].set_ylabel('Pixel IoU')
    axes[0, 1].set_title('Pixel IoU vs Compression')
    axes[0, 1].grid(True, alpha=0.3)
    for i, v in enumerate(iou_values):
        axes[0, 1].annotate(f'{v:.4f}', (i, v), textcoords='offset points',
        xytext=(0, 10), ha='center', fontsize=9)
    
    # Image Accuracy
    axes[1, 0].plot(x, acc_values, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(conditions_list, rotation=15)
    axes[1, 0].set_ylabel('Image Accuracy')
    axes[1, 0].set_title('Image Accuracy vs Compression')
    axes[1, 0].grid(True, alpha=0.3)
    for i, v in enumerate(acc_values):
        axes[1, 0].annotate(f'{v:.2%}', (i, v), textcoords='offset points',
        xytext=(0, 10), ha='center', fontsize=9)
    
    # Pixel AUC
    axes[1, 1].plot(x, auc_values, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(conditions_list, rotation=15)
    axes[1, 1].set_ylabel('Pixel AUC')
    axes[1, 1].set_title('Pixel AUC vs Compression')
    axes[1, 1].grid(True, alpha=0.3)
    for i, v in enumerate(auc_values):
        axes[1, 1].annotate(f'{v:.4f}', (i, v), textcoords='offset points',
        xytext=(0, 10), ha='center', fontsize=9)
    
    plt.suptitle(f'{VERSION} --- Performance Degradation Under Compression', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Relative degradation
    print('Relative degradation from Original baseline:')
    base_f1 = robustness_results['Original']['pixel_f1']
    base_acc = robustness_results['Original']['img_acc']
    for c in conditions_list[1:]:
        r = robustness_results[c]
        print(f'  {c}: F1 {(r["pixel_f1"]-base_f1)/base_f1*100:+.2f}%, Acc {(r["img_acc"]-base_acc)/base_acc*100:+.2f}%')
    
except Exception as _viz_err:
    print(f'[Cell 20] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 22 (visualization — wrapped for headless)
# ============================================================
try:
    # ============================================================
    # 7.1 Prediction Comparison Across Conditions
    # ============================================================
    
    # Pick 14 tampered and 6 authentic from test set (20 total)
    tampered_indices = [i for i, l in enumerate(test_labels) if l == 1]
    authentic_indices = [i for i, l in enumerate(test_labels) if l == 0]
    sample_indices = tampered_indices[:14] + authentic_indices[:6]
    
    conditions_list_all = list(robustness_results.keys())
    n_cond = len(conditions_list_all)
    n_imgs = len(sample_indices)
    
    fig, axes = plt.subplots(n_imgs, n_cond + 2, figsize=(4 * (n_cond + 2), 4 * n_imgs))
    if n_imgs == 1:
        axes = axes[np.newaxis, :]
    
    for row, idx in enumerate(sample_indices):
        path = test_paths[idx]
        label_true = test_labels[idx]
    
        # Col 0: Original RGB
        try:
            orig = Image.open(path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            axes[row, 0].imshow(np.array(orig))
        except Exception:
            axes[row, 0].text(0.5, 0.5, 'Load failed', ha='center', va='center')
        axes[row, 0].set_title(f'RGB ({"Tp" if label_true==1 else "Au"})', fontsize=9)
        axes[row, 0].axis('off')
    
        # Col 1: GT Mask
        gt_mask = get_gt_mask(path, IMAGE_SIZE)
        axes[row, 1].imshow(gt_mask, cmap='hot', vmin=0, vmax=1)
        axes[row, 1].set_title(f'GT Mask', fontsize=9)
        axes[row, 1].axis('off')
    
        # Cols 2+: Predicted mask under each condition
        for col, cond in enumerate(conditions_list_all):
            pred_map = robustness_results[cond]['preds'][idx, 0]
            pred_binary = (pred_map > 0.5).astype(np.float32)
            axes[row, col + 2].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
            axes[row, col + 2].set_title(f'{cond}', fontsize=9)
            axes[row, col + 2].axis('off')
    
    plt.suptitle(f'{VERSION} --- Predictions: RGB | GT | Original | Q95 | Q90 | Q80 | Q70', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    
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
    # 7.2 Per-Image Metric Distribution Across Conditions
    # ============================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for cond, res in robustness_results.items():
        per_img_f1 = []
        per_img_iou = []
        for i in range(len(res['preds'])):
            pred = (res['preds'][i].flatten() > 0.5).astype(np.float32)
            mask = res['masks'][i].flatten()
            tp_i = (pred * mask).sum()
            fp_i = (pred * (1 - mask)).sum()
            fn_i = ((1 - pred) * mask).sum()
            f1_i = (2 * tp_i) / (2 * tp_i + fp_i + fn_i + 1e-7)
            iou_i = tp_i / (tp_i + fp_i + fn_i + 1e-7)
            per_img_f1.append(f1_i)
            per_img_iou.append(iou_i)
    
        # Only tampered images (authentic always have F1~0)
        tampered_f1 = [per_img_f1[i] for i in range(len(res['labels'])) if res['labels'][i] == 1]
        tampered_iou = [per_img_iou[i] for i in range(len(res['labels'])) if res['labels'][i] == 1]
    
        axes[0].hist(tampered_f1, bins=30, alpha=0.4, label=f'{cond} (mean={np.mean(tampered_f1):.3f})')
        axes[1].hist(tampered_iou, bins=30, alpha=0.4, label=f'{cond} (mean={np.mean(tampered_iou):.3f})')
    
    axes[0].set_xlabel('Per-Image Pixel F1')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Tampered Image F1 Distribution by Condition')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Per-Image Pixel IoU')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Tampered Image IoU Distribution by Condition')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{VERSION} --- Per-Image Metric Distributions (Tampered Only)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
except Exception as _viz_err:
    print(f'[Cell 23] Visualization skipped: {_viz_err}')

# ============================================================
# Cell 25
# ============================================================
# ============================================================
# 8.1 Results Summary --- Robustness Table
# ============================================================

print(f'{"="*100}')
print(f'  ROBUSTNESS RESULTS SUMMARY --- {VERSION}')
print(f'{"="*100}')
print()

# Main results table
print(f'{"Condition":<12} {"Pixel F1":>10} {"Pixel IoU":>10} {"Pixel Prec":>12} {"Pixel Rec":>11} '
f'{"Pixel AUC":>10} {"Img Acc":>10} {"Img F1":>9} {"Delta F1":>10}')
print('-' * 100)

baseline_f1 = robustness_results['Original']['pixel_f1']
for cond, res in robustness_results.items():
    delta = res['pixel_f1'] - baseline_f1
    print(f'{cond:<12} {res["pixel_f1"]:>10.4f} {res["pixel_iou"]:>10.4f} {res["pixel_prec"]:>12.4f} '
    f'{res["pixel_rec"]:>11.4f} {res["pixel_auc"]:>10.4f} {res["img_acc"]:>10.4f} '
    f'{res["img_f1"]:>9.4f} {delta:>+10.4f}')

print(f'\n{"="*100}')

# Cross-track comparison
print(f'\nCross-Track Comparison (Original condition):')
orig = robustness_results['Original']
print(f'{"Version":<10} {"Track":<12} {"Encoder":<12} {"Input":<14} '
f'{"Test Acc":<10} {"Pixel-F1":<10} {"IoU":<8}')
print('-' * 78)
print(f'{"vR.P.3":<10} {"Pretrained":<12} {"ResNet-34":<12} {"ELA Q90 384sq":<14} '
f'{"86.79%":<10} {"0.6920":<10} {"0.5291":<8}')
print(f'{"vR.P.10":<10} {"Pretrained":<12} {"ResNet-34":<12} {"ELA Q90 384sq":<14} '
f'{"87.32%":<10} {"0.7277":<10} {"0.5719":<8}')
print(f'{VERSION:<10} {"Pretrained":<12} {"ResNet-34":<12} {"ELA Q90 384sq":<14} '
f'{orig["img_acc"]*100:.2f}{"%-":<7} {orig["pixel_f1"]:<10.4f} {orig["pixel_iou"]:<8.4f}')
print(f'{"="*100}')


# ============================================================
# Cell 27
# ============================================================
# ============================================================
# 10. Save Robustness Results
# ============================================================

# Save results dict (no model checkpoint --- model is from P.3)
results_dict = {
'version': VERSION,
'parent': 'vR.P.3',
'experiment': 'Compression Robustness Testing',
'conditions': {
k: {m: v for m, v in res.items()
if m not in ('preds', 'masks', 'labels', 'img_preds', 'img_scores')}
for k, res in robustness_results.items()
},
'config': {
'version': VERSION,
'encoder': ENCODER,
'image_size': IMAGE_SIZE,
'batch_size': BATCH_SIZE,
'ela_quality': ELA_QUALITY,
'quality_factors': QUALITY_FACTORS,
'seed': SEED,
'checkpoint': 'vR.P.3',
'training': False,
}
}

results_filename = f'{VERSION}_robustness_results.pth'
torch.save(results_dict, results_filename)

print(f'Results saved: {results_filename}')
print(f'File size: {os.path.getsize(results_filename) / 1e6:.1f} MB')
print(f'\n{VERSION} --- Compression Robustness Testing complete.')
print(f'No model was trained. Results are measurements of P.3 under recompression.')

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

