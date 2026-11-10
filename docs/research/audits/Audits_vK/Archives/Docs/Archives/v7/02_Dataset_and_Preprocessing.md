# Dataset and Preprocessing

---

## Dataset Selection

**CASIA v2.0 — Splicing Detection + Localization**
Kaggle slug: `sagnikkayalcse52/casia-spicing-detection-localization`

**Why CASIA?** It is the most widely used benchmark for image tampering detection in the research literature (cited across all Tier A papers). It provides pixel-level ground-truth masks for tampered regions, which is essential for segmentation-based localization. The assignment explicitly targets this dataset.

**What it contains:**
- Tampered images (prefix `Tp_`) with corresponding binary masks indicating tampered pixels
- Authentic images (prefix `Au_`) with no mask (all-zero ground truth)
- Forgery types: splicing (`_S_`), copy-move (`_C_`), and removal

**Known dataset limitations:**
- **Size:** ~1000 tampered images + authentic images. Small by modern standards — limits generalization claims.
- **Source-image leakage:** Some tampered images share source content with other images in the dataset. The stratified split does not fully prevent semantic overlap. This is a known property of CASIA v2, not a pipeline error.
- **Annotation quality:** Some ground-truth masks have imprecise boundaries or faint regions. Mask binarization (> 0) compensates by treating any non-zero pixel as foreground.
- **Classical tampering only:** No GAN-generated or deepfake content. Results do not extend to AI-generated image manipulation.

---

## Dataset Discovery

The notebook discovers the dataset root dynamically under `/kaggle/input/` (Kaggle) or the Drive download path (Colab) using case-insensitive directory matching.

### Directory Walk

```python
# Case-insensitive search for IMAGE/ and MASK/ directories
for root, dirs, files in os.walk(dataset_root):
    for d in dirs:
        if d.upper() == 'IMAGE':
            image_dir = os.path.join(root, d)
        if d.upper() == 'MASK':
            mask_dir = os.path.join(root, d)
```

This accommodates variations in directory naming (`IMAGE/` vs `Image/`) and nesting (`New folder/`).

### Pair Discovery with Validation

For each tampered image (`Tp_*`), the pipeline:

1. Attempts to find the corresponding mask file (trying `.png`, `.jpg`, `.bmp` extensions)
2. Validates image readability with `cv2.imread()`
3. Validates mask readability
4. Checks dimension agreement between image and mask (height and width must match)
5. Excludes images with unknown forgery type

```python
def is_valid_image(path):
    """Check if an image file can be read by OpenCV."""
    img = cv2.imread(path)
    return img is not None
```

**Dimension check:**
```python
img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if img.shape[:2] != mask.shape[:2]:
    # Skip this pair — dimension mismatch
    mismatched.append(img_path)
    continue
```

Images that fail any check are logged and excluded from the dataset. The notebook prints:
- Total tampered pairs found
- Images excluded (unreadable, dimension mismatch, unknown type)
- Authentic images found

---

## Mask Binarization

Ground-truth masks are binarized using threshold `> 0`:

```python
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask > 0).astype(np.float32)
```

**Why > 0 instead of > 128?** CASIA masks have varying intensity levels. Some masks use faint pixel values (< 128) to indicate tampered regions. A threshold of > 128 would miss these regions entirely. Using > 0 is more lenient and captures all annotated tampering, at the cost of potentially including annotation noise at mask boundaries.

---

## Data Split

**Strategy:** Stratified 70/15/15 split on the binary label (authentic vs. tampered).

```python
CONFIG['train_ratio'] = 0.70
# Remaining 30% split equally → 15% validation, 15% test
```

Stratification ensures the authentic/tampered ratio is preserved across all three splits.

### Split Manifest

The split is persisted to `split_manifest.json`:

```json
{
    "train": ["path/to/img1.jpg", ...],
    "val": ["path/to/img2.jpg", ...],
    "test": ["path/to/img3.jpg", ...],
    "train_labels": [0, 1, 1, ...],
    "val_labels": [0, 1, ...],
    "test_labels": [0, 1, ...]
}
```

On subsequent runs, the notebook reloads `split_manifest.json` when compatible, ensuring reproducibility without re-splitting.

### Data Leakage Verification

```python
train_set = set(train_paths)
val_set = set(val_paths)
test_set = set(test_paths)

assert len(train_set & val_set) == 0, "Train-val overlap detected!"
assert len(train_set & test_set) == 0, "Train-test overlap detected!"
assert len(val_set & test_set) == 0, "Val-test overlap detected!"
```

These assertions run before training starts. Any overlap triggers an immediate failure.

---

## Authentic Image Handling

Authentic images (`Au_*`) have no mask file. They are assigned an all-zero mask (no tampered pixels) and label `0`. In the `TamperingDataset` class:

```python
if self.labels[idx] == 0:  # Authentic
    mask = np.zeros((h, w), dtype=np.float32)
```

This ensures authentic images are included in training (the model must learn to predict all-zeros for clean images) and evaluation (true-negative handling).

---

## Interview: "How do you handle corrupted images?"

The discovery pipeline skips images that fail `cv2.imread()`. This is a corruption guard — if an image file exists but cannot be decoded (truncated download, corrupted bytes), it is logged and excluded rather than causing a crash inside `__getitem__` during training. The same guard applies to mask files.

## Interview: "Why stratified splitting?"

Without stratification, random splitting could give one split disproportionately many authentic images and another disproportionately many tampered images. This would make validation and test metrics unreliable. Stratification preserves the class ratio in every split, ensuring fair evaluation.
