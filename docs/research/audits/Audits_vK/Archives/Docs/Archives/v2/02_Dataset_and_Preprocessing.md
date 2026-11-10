# 02 — Dataset and Preprocessing

## Purpose

Specify the dataset selection, structure, known issues, and preprocessing steps required before training.

## Dataset: CASIA v2.0

| Property | Value |
|---|---|
| Source | Kaggle |
| Download method | Kaggle API (`kaggle datasets download`) |
| Content | Authentic images, tampered images, ground truth masks |
| Disk size | Approximately 2.6 GB |

The exact image counts should be determined dynamically during the dataset discovery step, not hardcoded.

## Directory Structure

```
CASIA2/
+-- Au/          # Authentic images (no masks)
+-- Tp/          # Tampered images
+-- Gt/          # Ground truth masks
```

## File Naming Convention

| Type | Image filename example | Mask filename example |
|---|---|---|
| Tampered | `Tp_D_NRN_nat00025_11037.jpg` | `Tp_D_NRN_nat00025_11037_gt.png` |
| Authentic | `Au_ani_00001.jpg` | No mask (generate all-zero mask) |

Mask pairing rule: mask filename = image stem + `_gt.png`.

Forgery type detection from filename:
- `'_D_'` in filename -> splicing
- `'_S_'` in filename -> copy-move

## Dataset Cleaning

### Alignment Validation

Some tampered images have mismatched dimensions with their masks (e.g., height/width swapped). The notebook must validate every pair dynamically and log excluded pairs.

```python
from PIL import Image

def validate_pair(image_path, mask_path):
    img = Image.open(image_path)
    msk = Image.open(mask_path)
    return img.size == msk.size
```

Do not hardcode the count of misaligned pairs. Instead, run the validation and report the count found.

### Mask Binarization

Ground truth masks contain intermediate grayscale values. Apply a fixed threshold:

```python
mask = (mask > 128).astype(np.uint8)
```

This produces a clean binary mask: 0 = authentic pixel, 1 = tampered pixel.

### Authentic Images

Authentic images have no corresponding mask files. Generate all-zero masks at load time:

```python
if mask_path is None:
    mask = np.zeros((height, width), dtype=np.uint8)
```

### Corrupted Files

Check that images and masks load successfully. Log and skip any files that fail to open:

```python
img = cv2.imread(image_path)
if img is None:
    # Log and skip this pair
    continue
```

## Data Split

| Split | Ratio |
|---|---|
| Training | 85% |
| Validation | 7.5% |
| Test | 7.5% |

**Stratification:** Maintain the ratio of authentic, splicing, and copy-move images across all splits. Use `sklearn.model_selection.train_test_split` with `stratify` and `random_state=42`.

**Two-step split procedure:**
1. Split full dataset into train (85%) and temp (15%) stratified by forgery type.
2. Split temp into validation (50% of temp) and test (50% of temp) stratified by forgery type.

**Split integrity notes:**

CASIA v2.0 does not publish source-image groupings, so true group-aware splitting is not feasible without manual annotation. This is a known limitation. As a pragmatic measure:
- Use stratified splitting to ensure class balance.
- Persist the split manifest (list of filenames per split) for reproducibility.
- Acknowledge in the notebook that related images may exist across splits.

## Dataset Discovery Function

```python
import os
from pathlib import Path

def discover_pairs(dataset_root):
    """
    Discover image-mask pairs and authentic images.
    Returns list of dicts and counts of excluded pairs.
    """
    tp_dir = os.path.join(dataset_root, 'Tp')
    gt_dir = os.path.join(dataset_root, 'Gt')
    au_dir = os.path.join(dataset_root, 'Au')

    pairs = []
    excluded = []

    # Tampered images with masks
    for img_name in sorted(os.listdir(tp_dir)):
        img_path = os.path.join(tp_dir, img_name)
        stem = Path(img_name).stem
        mask_name = stem + '_gt.png'
        mask_path = os.path.join(gt_dir, mask_name)

        if not os.path.exists(mask_path):
            excluded.append((img_name, 'mask_not_found'))
            continue

        if not validate_pair(img_path, mask_path):
            excluded.append((img_name, 'dimension_mismatch'))
            continue

        forgery_type = 'splicing' if '_D_' in stem else 'copy-move'
        pairs.append({
            'image_path': img_path,
            'mask_path': mask_path,
            'label': 1.0,
            'forgery_type': forgery_type,
        })

    # Authentic images (no masks)
    for img_name in sorted(os.listdir(au_dir)):
        img_path = os.path.join(au_dir, img_name)
        pairs.append({
            'image_path': img_path,
            'mask_path': None,
            'label': 0.0,
            'forgery_type': 'authentic',
        })

    print(f"Valid pairs: {len(pairs)}")
    print(f"Excluded: {len(excluded)}")
    for name, reason in excluded:
        print(f"  {name}: {reason}")

    return pairs
```

## Related Documents

- [03_Data_Pipeline.md](03_Data_Pipeline.md) — Dataset class and data loading
- [04_Model_Architecture.md](04_Model_Architecture.md) — Model that consumes this data
