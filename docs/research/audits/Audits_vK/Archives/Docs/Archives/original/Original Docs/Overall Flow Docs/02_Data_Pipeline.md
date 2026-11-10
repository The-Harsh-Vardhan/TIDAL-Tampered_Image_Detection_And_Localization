# 2. Data Pipeline — Implementation Guide

## 2.1 Assignment Requirement

> *"You are responsible for all dataset cleaning, preprocessing, and ensuring mask alignment. Properly split your data into train, validation, and test sets."*

This is a critical evaluation area. The evaluators want to see that you can build a **production-quality data pipeline** — not just load images into a model.

---

## 2.2 Pipeline Architecture

```
Raw Data (Kaggle Download)
    │
    ▼
┌───────────────────────────┐
│ Step 1: Discovery & Pairing│  Scan directories, match images → masks by naming convention
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ Step 2: Validation & Clean │  Check dimensions, exclude 17 misaligned, binarize masks
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ Step 3: Stratified Split   │  85% train / 7.5% val / 7.5% test (stratified by class)
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ Step 4: Dataset Class      │  PyTorch Dataset with on-the-fly loading + augmentation
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ Step 5: DataLoader         │  Batching, shuffling, pin_memory, num_workers
└───────────────────────────┘
```

---

## 2.3 Step 1: Discovery & Pairing

### Image-Mask Pairing Logic

```python
import os
import glob
from pathlib import Path

def discover_dataset(tp_dir, gt_dir, au_dir):
    """
    Discover and pair tampered images with their ground truth masks.
    Also collect authentic images (mask = all zeros, generated dynamically).
    """
    pairs = []
    
    # --- Tampered images ---
    for img_path in sorted(glob.glob(os.path.join(tp_dir, '*'))):
        stem = Path(img_path).stem
        # Mask naming: image stem + '_gt.png'
        mask_path = os.path.join(gt_dir, stem + '_gt.png')
        
        if os.path.exists(mask_path):
            # Determine forgery type from filename
            forgery_type = 'splicing' if '_D_' in stem else 'copy-move'
            pairs.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'is_tampered': True,
                'forgery_type': forgery_type
            })
    
    # --- Authentic images (no mask file; mask is all zeros) ---
    for img_path in sorted(glob.glob(os.path.join(au_dir, '*'))):
        pairs.append({
            'image_path': img_path,
            'mask_path': None,  # Will generate zeros dynamically
            'is_tampered': False,
            'forgery_type': 'authentic'
        })
    
    return pairs
```

### Expected Output
```
Total pairs discovered: ~4,975
  Authentic:  ~1,701
  Tampered:   ~3,274
    Splicing:   ~2,500
    Copy-Move:  ~757
  Unpaired (missing mask): 0  (verify this!)
```

---

## 2.4 Step 2: Validation & Cleaning

### Dimension Validation

```python
from PIL import Image

def validate_pairs(pairs):
    """
    Check image-mask dimension alignment. 
    Return clean pairs with misaligned ones removed.
    """
    clean_pairs = []
    misaligned = []
    
    for pair in pairs:
        if pair['mask_path'] is None:
            # Authentic image — no mask to check
            clean_pairs.append(pair)
            continue
        
        img = Image.open(pair['image_path'])
        mask = Image.open(pair['mask_path'])
        
        if img.size == mask.size:
            clean_pairs.append(pair)
        else:
            misaligned.append({
                'image': pair['image_path'],
                'img_size': img.size,
                'mask_size': mask.size
            })
    
    print(f"Clean pairs: {len(clean_pairs)}")
    print(f"Misaligned (excluded): {len(misaligned)}")
    
    # Log misaligned for documentation
    for m in misaligned:
        print(f"  {os.path.basename(m['image'])}: "
              f"img={m['img_size']}, mask={m['mask_size']}")
    
    return clean_pairs, misaligned
```

**Expected**: 17 misaligned pairs excluded, ~4,958 clean pairs remaining.

### Mask Binarization (Applied During Loading)
```python
import numpy as np

def load_mask(mask_path, image_size):
    """
    Load and binarize a ground truth mask.
    If mask_path is None (authentic image), return all-zeros.
    """
    if mask_path is None:
        # Authentic image: entire mask is 0 (no tampering)
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    mask = np.array(Image.open(mask_path).convert('L'))
    mask = (mask > 128).astype(np.float32)  # Binarize: 0.0 or 1.0
    return mask
```

---

## 2.5 Step 3: Stratified Split

### Requirements
- 85% training / 7.5% validation / 7.5% test
- Stratified by: `is_tampered` (authentic vs. tampered)
- Within tampered: maintain splicing-to-copy-move ratio across splits
- Fixed `random_state=42` for reproducibility

### Implementation

```python
from sklearn.model_selection import train_test_split

def split_dataset(pairs, seed=42):
    """
    Stratified split maintaining class ratios.
    """
    # Create stratification labels
    labels = [p['forgery_type'] for p in pairs]  # 'authentic', 'splicing', 'copy-move'
    
    # First split: 85% train, 15% temp
    train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
        pairs, labels,
        test_size=0.15,
        random_state=seed,
        stratify=labels
    )
    
    # Second split: 50/50 of temp → 7.5% val, 7.5% test
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels
    )
    
    return train_pairs, val_pairs, test_pairs
```

### Expected Split Sizes
| Split | Total | Authentic | Splicing | Copy-Move |
|-------|-------|-----------|----------|-----------|
| Train (85%) | ~4,214 | ~1,446 | ~2,125 | ~643 |
| Val (7.5%) | ~372 | ~128 | ~187 | ~57 |
| Test (7.5%) | ~372 | ~128 | ~187 | ~57 |

### Print Distribution (Include in Notebook)
```python
def print_split_stats(name, pairs):
    total = len(pairs)
    auth = sum(1 for p in pairs if not p['is_tampered'])
    spl = sum(1 for p in pairs if p['forgery_type'] == 'splicing')
    cm = sum(1 for p in pairs if p['forgery_type'] == 'copy-move')
    print(f"{name:>8}: Total={total}, Au={auth}, Splicing={spl}, Copy-Move={cm}")

print_split_stats("Train", train_pairs)
print_split_stats("Val", val_pairs)
print_split_stats("Test", test_pairs)
```

---

## 2.6 Step 4: PyTorch Dataset Class

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TamperingDataset(Dataset):
    def __init__(self, pairs, transform=None, image_size=512):
        """
        Args:
            pairs: List of dicts with 'image_path', 'mask_path', 'is_tampered', 'forgery_type'
            transform: albumentations Compose object
            image_size: Target size for resizing
        """
        self.pairs = pairs
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load image
        image = np.array(Image.open(pair['image_path']).convert('RGB'))
        
        # Load mask (zeros for authentic images)
        if pair['mask_path'] is not None:
            mask = np.array(Image.open(pair['mask_path']).convert('L'))
            mask = (mask > 128).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply augmentations (synchronized for image + mask)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']       # Tensor: (3, H, W)
            mask = transformed['mask']         # Tensor: (H, W)
        
        # Ensure mask has channel dimension: (1, H, W)
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Image-level label
        label = torch.tensor(1.0 if pair['is_tampered'] else 0.0)
        
        return image, mask, label
```

### Key Design Decisions
1. **On-the-fly loading**: Images are loaded per-batch, not pre-loaded into memory. CASIA v2.0 is ~2.6 GB — too large to fit in Colab RAM alongside the model.
2. **Synchronized transforms**: `albumentations` applies identical spatial transforms to both image and mask (via the `image=..., mask=...` API).
3. **Dynamic mask generation**: Authentic images get all-zero masks at load time — no need to pre-create mask files.
4. **Channel dimension**: Mask is returned as `(1, H, W)` to match the model's single-channel output.

---

## 2.7 Step 5: DataLoader Configuration

```python
from torch.utils.data import DataLoader

def create_loaders(train_pairs, val_pairs, test_pairs, 
                   train_transform, val_transform, batch_size=4):
    
    train_dataset = TamperingDataset(train_pairs, transform=train_transform)
    val_dataset = TamperingDataset(val_pairs, transform=val_transform)
    test_dataset = TamperingDataset(test_pairs, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Batch=1 for clean per-image evaluation
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### DataLoader Parameter Justification

| Parameter | Value | Why |
|-----------|-------|-----|
| `batch_size=4` | 4 | With 512×512 input + 6-channel model, 4 fits in T4 16GB VRAM |
| `shuffle=True` (train) | True | Prevents the model from memorizing data order |
| `num_workers=2` | 2 | Colab has 2 CPUs; more workers can cause hangs |
| `pin_memory=True` | True | Pre-allocates tensors in page-locked memory for faster CPU→GPU transfer |
| `drop_last=True` (train) | True | Prevents BatchNorm instability from batch-size-1 final batches |
| `persistent_workers=True` | True | Keeps worker processes alive between epochs (saves init overhead) |
| `batch_size=1` (test) | 1 | Allows per-image metric computation without padding/resizing artifacts |

---

## 2.8 Data Integrity Sanity Checks

Include these in the notebook after creating the DataLoader (visual verification):

```python
# Sanity check: visualize 4 random training samples
import matplotlib.pyplot as plt

batch = next(iter(train_loader))
images, masks, labels = batch

fig, axes = plt.subplots(4, 2, figsize=(10, 20))
for i in range(4):
    # Denormalize image for display
    img = images[i].permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image (label={labels[i].item():.0f})")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(masks[i, 0].numpy(), cmap='gray')
    axes[i, 1].set_title("Ground Truth Mask")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
```

### What to Verify Visually
1. Images are properly loaded (no corruption, correct orientation)
2. Masks align with tampered regions (white pixels overlay the actual manipulation)
3. Authentic images have all-black masks
4. Augmentations look reasonable (not too aggressive)
5. Image normalization is correct (colors should look natural after denormalization)

---

## 2.9 Pipeline Summary

| Stage | Input | Output | Key Operation |
|-------|-------|--------|---------------|
| Discovery | Raw directories | List of pairs (image_path, mask_path, class) | Filename pattern matching |
| Validation | List of pairs | Cleaned pairs (17 removed) | Dimension check |
| Splitting | Cleaned pairs | Train/Val/Test lists | Stratified random split |
| Dataset | Pairs + transforms | `(image_tensor, mask_tensor, label)` tuples | On-the-fly load + augment |
| DataLoader | Dataset objects | Batched tensors on GPU | Shuffling, batching, pinning |
