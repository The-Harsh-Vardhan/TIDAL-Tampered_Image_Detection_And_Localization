# Dataset and Preprocessing

---

## Dataset Selection

### Preferred Dataset

| Property | Value |
|---|---|
| Dataset | CASIA Splicing Detection + Localization |
| Source | Kaggle: `sagnikkayalcse52/casia-spicing-detection-localization` |
| URL | https://www.kaggle.com/datasets/sagnikkayalcse52/casia-spicing-detection-localization/data |
| Expected structure | `Image/Au/`, `Image/Tp/`, `Mask/Au/`, `Mask/Tp/` |
| Supported forgery types | Splicing, copy-move |

### Why This Dataset Was Selected

- It provides direct image-mask correspondence suitable for segmentation training.
- It avoids the manual ground-truth alignment burden of raw CASIA v2.0 variants.
- It includes both authentic and tampered images with localization masks.
- It is the best fit for a single-notebook Colab workflow.

### Dataset Limitations

1. CASIA does not expose source-image grouping metadata, so related images may still leak across splits.
2. Some masks may contain noisy boundaries inherited from CASIA-derived annotations.
3. The dataset covers classical splicing and copy-move only.
4. JPEG compression artifacts are part of the data distribution and may both help and hurt forensic localization.

---

## Dataset Download

The notebook downloads the dataset into a **slug-specific cache directory** instead of scanning all of `/content`:

```python
DATASET_SLUG = 'sagnikkayalcse52/casia-spicing-detection-localization'
DATASET_CACHE_DIR = os.path.join('/content', DATASET_SLUG.split('/')[-1])
```

This avoids silently binding to a stale or unrelated dataset tree from another session.

---

## Dynamic Discovery

The discovery stage must:

1. Validate tampered image readability.
2. Validate mask readability.
3. Validate image-mask dimension agreement.
4. Exclude `unknown_forgery_type` samples rather than stratifying them.
5. Return both `pairs` and `excluded`.

Reference implementation:

```python
def validate_readable_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        if cv2.imread(image_path) is None:
            return False, 'corrupt_image'
    except Exception:
        return False, 'corrupt_image'
    return True, ''


def validate_readable_mask(mask_path):
    try:
        with Image.open(mask_path) as mask:
            mask.verify()
        if cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) is None:
            return False, 'corrupt_mask'
    except Exception:
        return False, 'corrupt_mask'
    return True, ''


def validate_dimensions(image_path, mask_path):
    with Image.open(image_path) as img, Image.open(mask_path) as mask:
        if img.size != mask.size:
            return False, f'dim_mismatch: img={img.size} mask={mask.size}'
    return True, ''


def discover_pairs(dataset_root):
    pairs = []
    excluded = []
    # Discover tampered samples
    # -> validate image readability
    # -> resolve mask path
    # -> validate mask readability
    # -> validate dimensions
    # -> assign forgery_type or exclude unknowns
    # Discover authentic samples
    # -> validate image readability
    return pairs, excluded
```

Excluded reasons may include:
- `corrupt_image`
- `corrupt_mask`
- `mask_not_found`
- `dim_mismatch: ...`
- `unknown_forgery_type`

---

## Mask Binarization

Ground-truth masks are binarized with a fixed threshold:

```python
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask > 128).astype(np.uint8)
```

Authentic images receive all-zero masks:

```python
mask = np.zeros((height, width), dtype=np.uint8)
```

---

## Data Split

| Split | Ratio | Purpose |
|---|---|---|
| Train | 85% | Model fitting |
| Validation | 7.5% | Threshold selection, checkpoint selection, early stopping |
| Test | 7.5% | Final reporting only |

The first split stage is driven directly by `CONFIG['train_ratio']`.

```python
train_pairs, temp_pairs = train_test_split(
    pairs,
    test_size=1.0 - CONFIG['train_ratio'],
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

Stratification key:
- `authentic`
- `splicing`
- `copy-move`

---

## Split Manifest Reuse

The split manifest is now part of the reproducibility contract:

```python
manifest = {
    'version': 2,
    'dataset_slug': DATASET_SLUG,
    'seed': SEED,
    'train_ratio': CONFIG['train_ratio'],
    'train': [relative_image_path(p) for p in train_pairs],
    'val': [relative_image_path(p) for p in val_pairs],
    'test': [relative_image_path(p) for p in test_pairs],
}
```

Behavior:
- The notebook saves split entries as **relative paths** under `DATASET_ROOT`.
- On reruns, the notebook reloads `split_manifest.json` when it is compatible with the currently discovered dataset.
- If the manifest is incompatible, the notebook recreates the split and writes a normalized manifest.

This makes the manifest the source of truth for repeatable experiments instead of a passive artifact.

---

## Preprocessing Pipeline

1. Load RGB image with `cv2.imread` + `cv2.cvtColor`.
2. Load grayscale mask with `cv2.imread(..., cv2.IMREAD_GRAYSCALE)`.
3. Binarize mask at `> 128`.
4. Generate zero mask for authentic images.
5. Apply albumentations transforms.
6. Normalize with ImageNet statistics.
7. Convert to tensors with `ToTensorV2()`.

---

## Leakage Note

CASIA still lacks source-image grouping metadata, so a true group-aware split is not available. Manifest reuse improves reproducibility, but it does **not** eliminate source-image leakage risk.
