# 02 — Data Pipeline Audit

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Summary

The v9 data pipeline is substantially better designed than v8's. It is also completely unvalidated. Better-designed unvalidated pipelines blow up in interesting ways. This section documents what the code says it does and where it will break.

---

## 1. Dataset Discovery

### v8 approach
Walks `/kaggle/input` looking for `IMAGE/` and `MASK/` directories. Validates dimension matching. Infers forgery type from filename tokens (`_D_` → splicing, `_S_` → copy-move). Records missing masks and corrupt files explicitly. Saves a split manifest JSON.

### v9 approach
Much more sophisticated: recursive search across multiple candidate roots (Drive, local content, data dir override), metadata.csv–driven pair loading with fallback to filename inference, more robust normalisation logic, and a label inference pipeline that handles authentic/tampered via filename hints.

### Problems with v9 discovery

**Problem 1 — Multiple root search can silently use the wrong dataset.**  
The v9 code searches `data_roots = [data_dir, drive_root, local_data_root, /content]` in order and uses the first root where `metadata.csv` is found. If a stale metadata.csv exists from a previous experiment in `/content/data/artifacts/`, the pipeline will quietly load the wrong data without any warning. There is no cross-validation that the found metadata matches the current dataset.

**Problem 2 — Authentic image masks are zero-filled on-the-fly.**  
v9 creates zero masks for authentic images at load time. v8 did the same. This is fine. The danger is that the zero mask is the same for every authentic sample, which means if augmentation corrupts the mask-image synchronisation, the failure is invisible — a zero mask always looks "correct" for authentic images.

**Problem 3 — No mask binarization audit.**  
`mask = (mask > 0).astype("float32")` works for clean binary PNGs. CASIA masks are nominally binary but some contain anti-aliased soft edges or palette-encoded values. Neither v8 nor v9 logs the distribution of unique pixel values in a sample of masks before binarization. This should be a mandatory sanity step.

---

## 2. pHash Near-Duplicate Grouping

### What v9 claims to do
Hash every image with `imagehash.phash`, compute Hamming distances within LSH bands, union-find grouping for images within distance threshold 4, split per group to prevent leakage.

### What v9 actually does — correctly

Comparing v9's implementation to the flawed design sketch in the earlier Docs9 planning documents, the actual notebook code is significantly better:

```python
def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
```

This is a real Hamming distance computation. Earlier planning documents showed exact hash string matching. The notebook improved on the plan, which is good.

### What v9 does wrong

**Problem 1 — pHash is computed on every image on every run.**  
12,614 images × PIL open + phash = significant wall-clock time on Colab. There is no caching of computed hashes. Every time the notebook is rerun from cell 1, it recomputes 12,614 hashes. On a cold Colab T4 with drive-mounted data, this is easily 10-20 minutes of setup before training begins.

**Problem 2 — Tuple ordering in candidate_pairs is not guaranteed for exact reproducibility.**  
`candidate_pairs = set()` with `tuple(sorted(...))` produces canonical pairs, but the iteration order of a set is not deterministic across Python versions. Minor issue, but for a "reproducibility" project this matters.

**Problem 3 — Group splits can produce highly imbalanced authentic/tampered ratios.**  
If multiple near-duplicate groups fall disproportionately in one forgery type and the stratification key only uses `label + forgery_type`, the authentic/tampered ratio in each split may drift significantly. The code uses `safe_group_train_test` which falls back to unstratified splitting "if counts.min() < 2". No diagnostic output confirms this fallback did or did not trigger.

---

## 3. Stratified Split

### v8 approach
Simple 70/15/15 by forgery type label. Path-overlap leakage check at the end. Works for simple cases.

### v9 approach
pHash-group-aware stratified split. Substantially more rigorous at preventing near-duplicate leakage.

### Critical gap in v9
There is no leakage verification after splitting. v8 explicitly ran:

```python
assert len(train_paths & val_paths) == 0, 'LEAKAGE: train-val overlap!'
```

v9 trusts the group-assignment logic completely and never cross-checks that no image path appears in multiple splits. Given the complexity of the group assignment code, the absence of this assertion is negligent.

---

## 4. ELA Computation

```python
def compute_ela(image_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    _, encoded = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, decoded)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    return ela_gray
```

### Problems with this implementation

**Problem 1 — ELA is computed on BGR images without pre-normalisation.**  
The input `image_bgr` to `compute_ela` arrives after `cv2.imread`, which is fine. But then the ELA computation occurs before any resizing. If the image is 3000×4000 and gets ELA computed at full resolution before the `A.Resize(384, 384)` in the augment pipeline, the ELA map captures artifact patterns at full resolution that will then be downsampled, mixing artifact signals across pixels. The ELA should be computed on the already-resized image.

**Problem 2 — ELA normalisation is inconsistent with RGB normalisation.**  
RGB channels are normalised with ImageNet mean/std:
```python
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```
The ELA channel is normalised by:
```python
ela_tensor = torch.from_numpy(ela_map.astype(np.float32) / 255.0).unsqueeze(0)
```
Simple `/255` division is not the same normalisation regime. The ELA channel will have a systematically different scale and distribution than the RGB channels when they enter the encoder's first convolution. This is not automatically wrong but it is not documented, not tested, and the pretrained ResNet34 first-layer weights were optimised for a range much narrower than most ELA maps will produce.

**Problem 3 — Augmentation transforms do not apply consistently to ELA.**  
The augmentation pipeline uses Albumentations transforms on `image` and `mask`. ELA is concatenated after augmentation. This means geometric augmentations (horizontal flip, random rotate) are applied to RGB, then ELA is computed from the un-augmented image before flipping, and then concatenated. The ELA channel will not be spatially aligned with the augmented RGB. This is a silent alignment bug that will corrupt the forensic signal on every augmented training sample.

---

## 5. Overall Data Pipeline Rating

| Component | v8 | v9 | Notes |
|-----------|----|----|-------|
| Pair discovery | ✅ Executed | ⚠️ Code only | |
| Mask validation | ✅ Executed | ⚠️ Code only | |
| pHash leakage guard | ❌ Absent | ✅ Present (code) | Good addition, not validated |
| Stratified split | ✅ Simple, verified | ✅ Better, not verified | Missing post-split assertion |
| ELA computation | N/A | ⚠️ Buggy | Alignment and normalisation issues |
| Leakage assertion | ✅ Explicit check | ❌ Missing | Regression |

**Bottom line:** The v9 pipeline has a better conceptual design than v8. It also has at least three concrete bugs in the ELA pipeline (resolution order, normalisation mismatch, augmentation misalignment) and one important omission (no post-split leakage assertion). None of these can be confirmed or denied because the notebook has never been run.
