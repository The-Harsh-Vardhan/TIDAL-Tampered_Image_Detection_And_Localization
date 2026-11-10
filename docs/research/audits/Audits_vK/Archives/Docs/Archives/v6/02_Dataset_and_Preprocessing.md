# Dataset and Preprocessing

---

## Dataset Selection

### Preferred Dataset

| Property | Value |
|---|---|
| Dataset | CASIA Splicing Detection + Localization |
| Source | Kaggle: `sagnikkayalcse52/casia-spicing-detection-localization` |
| URL | https://www.kaggle.com/datasets/sagnikkayalcse52/casia-spicing-detection-localization/data |
| On-disk structure | `New folder/IMAGE/Au/`, `New folder/IMAGE/Tp/`, `New folder/MASK/Au/`, `New folder/MASK/Tp/` |
| Supported forgery types | Splicing (`_D_` in filename), Copy-move (`_S_` in filename) |

### Why This Dataset Was Selected

- It provides direct image-mask correspondence suitable for segmentation training.
- It avoids the manual ground-truth alignment burden of raw CASIA v2.0 variants.
- It includes both authentic and tampered images with localization masks.
- It is pre-mounted on Kaggle — no download step needed.

### Dataset Limitations

1. CASIA does not expose source-image grouping metadata, so related images may still leak across splits.
2. Some masks may contain noisy boundaries inherited from CASIA-derived annotations.
3. The dataset covers classical splicing and copy-move only — GAN edits and deepfakes are not represented.
4. JPEG compression artifacts are part of the data distribution and may both help and hurt forensic localization.
5. The dataset is small by modern standards, increasing overfitting risk.

---

## Dataset Discovery

The notebook discovers the dataset root dynamically using a **case-insensitive** directory walk, accommodating the actual Kaggle layout where directories may be uppercase (`IMAGE/`, `MASK/`) and nested inside intermediate folders (`New folder/`).

```python
KAGGLE_INPUT = '/kaggle/input'

DATASET_ROOT = None
for root, dirs, files in os.walk(KAGGLE_INPUT):
    dirs_lower = [d.lower() for d in dirs]
    if 'image' in dirs_lower and 'mask' in dirs_lower:
        DATASET_ROOT = root
        IMAGE_DIR = dirs[dirs_lower.index('image')]
        MASK_DIR = dirs[dirs_lower.index('mask')]
        break
```

This handles:
- `Image/` or `IMAGE/` or `image/` (any case)
- `Mask/` or `MASK/` or `mask/` (any case)
- Any nesting depth (e.g., `dataset-name/New folder/IMAGE/`)

---

## Dynamic Pair Discovery

The discovery stage:

1. Validates tampered image readability (`is_valid_image()` using `PIL.Image.verify()`).
2. Resolves mask path — exact filename match, with fallback to extension variants (`.png`, `.jpg`, `.bmp`, `.tif`).
3. Validates image-mask dimension agreement.
4. Classifies forgery type: `_D_` → splicing, `_S_` → copy-move, otherwise `unknown` (with warning).
5. Validates authentic image readability.
6. Returns both `pairs` and `excluded` lists with reason strings.

```python
def is_valid_image(filepath):
    '''Check if an image file can be opened and decoded.'''
    try:
        img = Image.open(filepath)
        img.verify()
        return True
    except Exception:
        return False
```

Excluded reasons include:
- `corrupt_image` — image file cannot be opened or decoded
- `corrupt_file` — authentic image cannot be opened
- `mask_not_found` — no matching mask file found
- `dim_mismatch: img=(...) mask=(...)` — spatial dimensions disagree

---

## Mask Binarization

Ground-truth masks are binarized at threshold `> 0`:

```python
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask > 0).astype(np.uint8)
```

This captures **all annotated pixels** including low-intensity annotations that would be lost with a `> 128` threshold. This is applied consistently in both `TamperingDataset` and `ResizeDegradationDataset`.

Authentic images receive all-zero masks:

```python
mask = np.zeros((height, width), dtype=np.uint8)
```

---

## Data Split

| Split | Ratio | Purpose |
|---|---|---|
| Train | 70% | Model fitting |
| Validation | 15% | Threshold selection, checkpoint selection, early stopping |
| Test | 15% | Final reporting only |

```python
train_pairs, temp_pairs = train_test_split(
    pairs,
    test_size=1.0 - CONFIG['train_ratio'],  # 0.30
    random_state=SEED,
    stratify=forgery_labels,
)

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=0.5,
    random_state=SEED,
    stratify=temp_labels,
)
```

Stratification key: `forgery_type` (authentic / splicing / copy-move).

### Data Leakage Verification

After splitting, the notebook verifies zero file overlap:

```python
train_files = set(p['image_path'] for p in train_pairs)
val_files = set(p['image_path'] for p in val_pairs)
test_files = set(p['image_path'] for p in test_pairs)

assert len(train_files & val_files) == 0, 'Train-val leakage detected!'
assert len(train_files & test_files) == 0, 'Train-test leakage detected!'
assert len(val_files & test_files) == 0, 'Val-test leakage detected!'
```

---

## Split Manifest

The split is persisted to `split_manifest.json` for reproducibility:

```python
manifest = {
    'seed': SEED,
    'total_pairs': len(pairs),
    'train_count': len(train_pairs),
    'val_count': len(val_pairs),
    'test_count': len(test_pairs),
    'train_ratio': CONFIG['train_ratio'],
}
```

Saved to: `/kaggle/working/results/split_manifest.json`

---

## Preprocessing Pipeline

1. Load RGB image with `cv2.imread` + `cv2.cvtColor(BGR2RGB)`.
2. Load grayscale mask with `cv2.imread(..., cv2.IMREAD_GRAYSCALE)`.
3. Binarize mask at `> 0`.
4. Generate zero mask for authentic images.
5. Apply albumentations transforms (resize 384×384, augmentations).
6. Normalize with ImageNet statistics: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`.
7. Convert to tensors with `ToTensorV2()`.

---

## Leakage Note

CASIA does not provide source-image grouping metadata. The set-intersection leakage check verifies that no file appears in multiple splits, but content-level leakage (e.g., different crops of the same source image) cannot be detected. Generalization claims should remain conservative.
