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
    pairs = []
    excluded = []

    for img_name in sorted(os.listdir(tp_dir)):
        # ... pair matching logic ...
        # Forgery type guard
        if '_D_' in stem:
            forgery_type = 'splicing'
        elif '_S_' in stem:
            forgery_type = 'copy-move'
        else:
            forgery_type = 'unknown'
            warnings.warn(f"Unrecognized forgery pattern: {img_name}")

    # ... authentic scanning ...
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
    'train': [p['image_path'] for p in train_pairs],
    'val': [p['image_path'] for p in val_pairs],
    'test': [p['image_path'] for p in test_pairs],
}
with open(os.path.join(checkpoint_dir, 'split_manifest.json'), 'w') as f:
    json.dump(manifest, f)
```

---

## Leakage Note

CASIA v2.0 does not provide source-image groupings. Some tampered images may share the same source image as other tampered or authentic samples. A group-aware split is not possible without external annotation. This is a **known limitation**, mitigated by:

- Stratified splitting to avoid class imbalance across splits.
- Manifest persistence for reproducibility and future audit.

---

## Validation Summary

Before training, the notebook prints:

- Total valid pairs and per-type counts.
- Number of excluded pairs with reasons (mask not found, dimension mismatch, corrupt file, unknown pattern).
- Per-split class distribution to confirm stratification.
