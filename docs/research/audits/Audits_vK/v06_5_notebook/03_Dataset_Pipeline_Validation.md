# Audit 6.5 — Part 3: Dataset Pipeline Validation

## Dataset Overview

| Property | Value |
|---|---|
| Dataset source | CASIA Splicing Detection + Localization (Kaggle) |
| Total files discovered | 12,614 |
| Tampered images | 5,123 (40.6%) |
| Authentic images | 7,491 (59.4%) |
| Excluded/skipped | 0 |
| Forgery types | Splicing: 1,828 / Copy-move: 3,295 |

### Split
| Set | Count | Authentic | Copy-move | Splicing |
|---|---|---|---|---|
| Train | 8,829 | 5,243 | 2,306 | 1,280 |
| Val | 1,892 | 1,124 | 494 | 274 |
| Test | 1,893 | 1,124 | 495 | 274 |

---

## Pipeline Audit

### 1. Image-Mask Pairing ✅

The `discover_pairs()` function implements robust pairing:

- Searches for masks by matching filenames (exact match first, then stem + multiple extensions)
- Validates dimensions match between image and mask (`validate_dimensions()`)
- Validates that images can be opened and decoded (`is_valid_image()`)
- Excludes pairs with missing masks, corrupt images, or dimension mismatches
- **Result: 0 exclusions**, meaning all 5,123 tampered images have valid corresponding masks

**Assessment:** Pairing logic is thorough. No evidence of misalignment.

### 2. Mask Binarization ⚠️ Minor Concern

```python
mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
mask = (mask > 0).astype(np.uint8)
```

The mask is binarized with a threshold of `> 0`, meaning any non-zero pixel is treated as tampered. This is the standard approach for binary segmentation masks.

**Potential issue:** If CASIA masks contain anti-aliased edges (gradual 0→255 transitions), the `> 0` threshold will include edge pixels that are partially tampered. This could make the mask slightly larger than the actual tampered region. However, for training purposes this is generally acceptable and standard practice.

### 3. Resize Handling ⚠️ Key Concern

```python
A.Resize(CONFIG['image_size'], CONFIG['image_size'])  # 384×384
```

Both images and masks are resized to 384×384 using Albumentations' default interpolation:
- Images: `cv2.INTER_LINEAR` (bilinear)
- Masks: `cv2.INTER_NEAREST` (nearest neighbor, via Albumentations mask pipeline)

**Assessment:** Albumentations correctly handles mask interpolation with nearest-neighbor, preventing interpolation artifacts in binary masks. This is correct.

**Concern:** The CASIA dataset has varying image sizes (the sample shows 256×384 images). Resizing non-square images to 384×384 introduces aspect ratio distortion. This is a standard tradeoff in deep learning (preserving aspect ratio adds complexity), but it means the model sees distorted images. This could affect small, thin tampered regions more than large ones.

### 4. Augmentation Pipeline

**Train transforms:**
```
Resize(384,384) → HorizontalFlip(0.5) → VerticalFlip(0.5) → RandomRotate90(0.5) → Normalize → ToTensor
```

**Val/Test transforms:**
```
Resize(384,384) → Normalize → ToTensor
```

**Assessment:**
- ✅ Normalization uses ImageNet statistics (correct for ResNet34 pretrained encoder)
- ✅ Augmentations are applied identically to images and masks (Albumentations handles this)
- ⚠️ Augmentation is **weak** — only geometric transforms, no photometric augmentation (brightness, contrast, saturation, hue), no elastic deformation, no cutout/mixup
- ⚠️ No augmentation addressing compression artifacts (which CASIA images contain)

### 5. Data Split Quality ✅

- **Stratified split** by forgery type — correct
- **70/15/15 ratio** — standard
- **Leakage verification** — explicit assertion confirms no path overlap between train/val/test
- **Manifest saved** — full split recorded in JSON for reproducibility

**Assessment:** Split methodology is rigorous.

### 6. Authentic Image Handling ✅

For authentic images, a zero mask is generated:
```python
mask = np.zeros((h, w), dtype=np.uint8)
```

This is correct — authentic images have no tampered regions.

### 7. DataLoader Configuration ✅

```
batch_size=4, num_workers=2, pin_memory=True, persistent_workers=True
drop_last=True (train), drop_last=False (val/test)
worker_init_fn with seeded generator
```

**Assessment:** Configuration is appropriate for Kaggle T4 environment. `persistent_workers` avoids worker restart overhead. Seeded workers ensure reproducibility.

---

## Impact Analysis

| Issue | Severity | Impact on Metrics |
|---|---|---|
| Aspect ratio distortion | Low | Minor — affects thin regions |
| Weak augmentation | **Medium** | **Major cause of overfitting** |
| Mask binarization threshold | Low | Negligible |
| No photometric augmentation | Medium | Model may overfit to color distributions |

---

## Verdict

The dataset pipeline is **correctly implemented** with good engineering practices (dimension validation, leakage checks, manifest saving). The main weakness is the **limited augmentation strategy**, which is a significant contributor to the overfitting observed in training. The pipeline does not introduce any bugs that would invalidate the metrics.
