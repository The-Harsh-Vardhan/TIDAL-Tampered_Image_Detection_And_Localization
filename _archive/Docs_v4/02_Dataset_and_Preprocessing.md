# Dataset and Preprocessing

---

## Dataset Selection

| Property | Value |
|---|---|
| Dataset | CASIA v2.0 |
| Source | Kaggle: `sophatvathana/casia-dataset` |
| Approximate size | ~2.6 GB |
| Structure | `Tp/` (tampered), `Au/` (authentic), `Gt/` (ground-truth masks) |
| Forgery types | Splicing (`_D_` in filename), Copy-move (`_S_` in filename) |

---

## Dynamic Discovery

The discovery function must:

1. Scan `Tp/` for tampered images; match each to a mask in `Gt/` by stem + `_gt.png`.
2. Validate that image and mask have identical spatial dimensions.
3. Classify forgery type from filename; log a warning for unrecognized patterns instead of defaulting silently.
4. Scan `Au/` for authentic images; verify each loads without error.
5. **Return** `(pairs, excluded)` — the excluded list is a data structure, not just printed output.

```python
def discover_pairs(dataset_root):
    """
    Returns:
        pairs: list of dicts with keys image_path, mask_path, label, forgery_type
        excluded: list of (filename, reason) tuples
    """
    tp_dir = os.path.join(dataset_root, 'Tp')
    gt_dir = os.path.join(dataset_root, 'Gt')
    au_dir = os.path.join(dataset_root, 'Au')

    pairs = []
    excluded = []

    for img_name in sorted(os.listdir(tp_dir)):
        stem = Path(img_name).stem
        mask_path = os.path.join(gt_dir, stem + '_gt.png')

        if not os.path.exists(mask_path):
            excluded.append((img_name, 'mask_not_found'))
            continue

        # Forgery type guard
        if '_D_' in stem:
            forgery_type = 'splicing'
        elif '_S_' in stem:
            forgery_type = 'copy-move'
        else:
            forgery_type = 'unknown'
            warnings.warn(f"Unrecognized forgery pattern: {img_name}")

        pairs.append({
            'image_path': os.path.join(tp_dir, img_name),
            'mask_path': mask_path,
            'label': 1.0,
            'forgery_type': forgery_type,
        })

    for img_name in sorted(os.listdir(au_dir)):
        img_path = os.path.join(au_dir, img_name)
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            excluded.append((img_name, 'corrupt_file'))
            continue

        pairs.append({
            'image_path': img_path,
            'mask_path': None,
            'label': 0.0,
            'forgery_type': 'authentic',
        })

    return pairs, excluded  # Both returned, not printed
```

---

## Mask Binarization

Ground-truth masks are grayscale PNGs with variable intensity. Binarize with a fixed threshold:

```python
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask > 128).astype(np.uint8)  # Binary: 0 or 1
```

Authentic images get all-zero masks:

```python
mask = np.zeros((height, width), dtype=np.uint8)
```

---

## Data Split

| Split | Ratio | Purpose |
|---|---|---|
| Train | 85% | Model training |
| Validation | 7.5% | Threshold selection, early stopping |
| Test | 7.5% | Final reporting only |

**Procedure:**

```python
# Step 1: train (85%) vs temp (15%)
train_pairs, temp_pairs = train_test_split(
    pairs, test_size=0.15, random_state=42, stratify=forgery_labels
)
# Step 2: temp → val (50%) + test (50%)
val_pairs, test_pairs = train_test_split(
    temp_pairs, test_size=0.5, random_state=42, stratify=temp_labels
)
```

**Stratification key:** `forgery_type` (authentic / splicing / copy-move).

**Manifest persistence:**

```python
manifest = {
    'seed': 42,
    'total_pairs': len(pairs),
    'excluded_count': len(excluded),
    'train': [p['image_path'] for p in train_pairs],
    'val': [p['image_path'] for p in val_pairs],
    'test': [p['image_path'] for p in test_pairs],
}
with open(os.path.join(checkpoint_dir, 'split_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
```

---

## Leakage Note

CASIA v2.0 does not provide source-image groupings. Some tampered images may share the same source image as other tampered or authentic samples. A group-aware split is not possible without external annotation. This is a **known limitation**, mitigated by:

- Stratified splitting to avoid class imbalance across splits.
- Manifest persistence for reproducibility and future audit.

This limitation means generalization claims should be made cautiously.

---

## Validation Summary

Before training, the notebook prints:

- Total valid pairs and per-type counts.
- Number of excluded pairs with reasons (mask not found, dimension mismatch, corrupt file, unknown pattern).
- Per-split class distribution to confirm stratification.
