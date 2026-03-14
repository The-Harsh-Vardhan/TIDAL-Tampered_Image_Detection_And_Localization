# 02 — Dataset and Preprocessing

## Purpose

This document specifies the dataset selection, structure, known issues, and preprocessing steps required before training.

## Dataset: CASIA v2.0

| Property | Value |
|---|---|
| Source | Kaggle |
| Download method | Kaggle API (`kaggle datasets download`) |
| Total usable images | ~4,958 (after excluding misaligned pairs) |
| Authentic images (Au) | ~1,701 |
| Tampered images (Tp) | ~3,257 |
| — Splicing (Tp_D) | ~2,500 |
| — Copy-move (Tp_S) | ~757 |
| Disk size | ~2.6 GB |

## Directory Structure

```
CASIA2/
├── Au/          # Authentic images (no masks)
├── Tp/          # Tampered images
└── Gt/          # Ground truth masks
```

## File Naming Convention

| Type | Image filename | Mask filename |
|---|---|---|
| Tampered | `Tp_D_NRN_nat00025_11037.jpg` | `Tp_D_NRN_nat00025_11037_gt.png` |
| Authentic | `Au_ani_00001.jpg` | No mask (generate all-zero mask) |

Mask pairing rule: mask filename = image stem + `_gt.png`.

Forgery type detection:
- `'_D_'` in filename → splicing
- `'_S_'` in filename → copy-move

## Known Issues and Cleaning

### Issue 1: Resolution Misalignment (17 pairs)

17 tampered images have height/width swapped between the image and its mask (e.g., image is 384×256 but mask is 256×384).

**Resolution:** Exclude these 17 pairs. They represent ~0.5% of the dataset and excluding them preserves mask integrity.

**Validation check:**
```python
from PIL import Image

def is_aligned(image_path, mask_path):
    img = Image.open(image_path)
    msk = Image.open(mask_path)
    return img.size == msk.size
```

### Issue 2: Non-Binary Masks

Ground truth masks contain intermediate grayscale values (gradients near edges) instead of strict 0/255.

**Resolution:** Apply a fixed threshold during preprocessing:
```python
mask = (mask > 128).astype(np.uint8)
```

This produces a clean binary mask (0 = authentic, 1 = tampered).

### Issue 3: Authentic Images Have No Masks

Authentic images do not have corresponding mask files.

**Resolution:** Generate all-zero masks at load time:
```python
if mask_path is None:
    mask = np.zeros((height, width), dtype=np.uint8)
```

## Data Split

| Split | Ratio | Approximate count |
|---|---|---|
| Training | 85% | ~4,214 |
| Validation | 7.5% | ~372 |
| Test | 7.5% | ~372 |

**Stratification:** Maintain the ratio of authentic, splicing, and copy-move images across all splits. Use `sklearn.model_selection.train_test_split` with `stratify` and `random_state=42`.

## Preprocessing Steps

1. **Load image** — Read RGB image using PIL or OpenCV.
2. **Load mask** — Read grayscale mask; generate all-zero mask for authentic images.
3. **Validate alignment** — Confirm image and mask have matching dimensions.
4. **Binarize mask** — Apply threshold: `mask = (mask > 128).astype(np.uint8)`.
5. **Resize** — Resize both image and mask to 512×512. Use bilinear interpolation for images and nearest-neighbor for masks.
6. **Normalize image** — Apply ImageNet normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
7. **Convert to tensor** — Image shape: (3, 512, 512). Mask shape: (1, 512, 512).

Mask interpolation must use **nearest neighbor only** to preserve binary values.

## Dataset Discovery Function

```python
import os
from pathlib import Path

def discover_pairs(dataset_root):
    """
    Discover image-mask pairs and authentic images.
    Returns list of dicts with keys: image_path, mask_path, label, forgery_type.
    """
    tp_dir = os.path.join(dataset_root, 'Tp')
    gt_dir = os.path.join(dataset_root, 'Gt')
    au_dir = os.path.join(dataset_root, 'Au')
    
    pairs = []
    
    # Tampered images with masks
    for img_name in os.listdir(tp_dir):
        img_path = os.path.join(tp_dir, img_name)
        stem = Path(img_name).stem
        mask_name = stem + '_gt.png'
        mask_path = os.path.join(gt_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
        
        forgery_type = 'splicing' if '_D_' in stem else 'copy-move'
        
        pairs.append({
            'image_path': img_path,
            'mask_path': mask_path,
            'label': 1.0,
            'forgery_type': forgery_type,
        })
    
    # Authentic images (no masks)
    for img_name in os.listdir(au_dir):
        img_path = os.path.join(au_dir, img_name)
        pairs.append({
            'image_path': img_path,
            'mask_path': None,
            'label': 0.0,
            'forgery_type': 'authentic',
        })
    
    return pairs
```

## Related Documents

- [03_Data_Pipeline.md](03_Data_Pipeline.md) — Dataset class and data loading
- [04_Model_Architecture.md](04_Model_Architecture.md) — Model that consumes this data
