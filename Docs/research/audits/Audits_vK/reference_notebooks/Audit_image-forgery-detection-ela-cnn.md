# Audit: Image Forgery Detection Using ELA and CNN (+ Variants)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**Files:**
- `image-forgery-detection-using-ela-and-cnn.ipynb` (13.3 MB) — Original, 7 epochs
- `image-forgery-detection-using-ela-and-cnn (1).ipynb` (12.5 MB) — Same model, 24 epochs
- `image-forgery-detection.ipynb` (12.8 MB) — Same model, deprecation fixes
- `image-forgery-detection-using-ela-and-cnn.xpynb` (0 bytes) — Empty file

---

## Notebook Overview

Three variants of the same ELA + CNN image forgery classification notebook. All use the same 2-conv CNN architecture (29.5M params) with ELA preprocessing on CASIA 2.0. The notebooks differ in hyperparameters (epochs, batch size) and minor code fixes.

| Variant | Epochs | Batch Size | Phase 2 Val Acc | Overall Test Acc |
|---|---|---|---|---|
| Original (13.3MB) | 7+7 | 15 | 92.44% | 92.48% |
| Variant (1) (12.5MB) | 24+24 | 32 | 93.64% | 94.68% |
| Variant (2) (12.8MB) | 24+24 | 10 | 93.64% | 93.89% |
| .xpynb (0 bytes) | N/A | N/A | N/A | Empty file |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CASIA 2.0 |
| Authentic subsample | 2,100 (from 7,354 total) |
| Tampered | 2,064 |
| Total used | 4,164 |
| Split | 80/20 → 3,331 train / 833 val |
| Image Size | 128×128 |
| Preprocessing | ELA at JPEG quality=91, resized, /255.0 |

**Data subsampling:** Only 2,100 of 7,354 authentic images are used to roughly balance classes. This discards 70% of authentic data.

---

## Model Architecture Review

All three variants use the identical architecture from `build_model()`:

| Layer | Output Shape | Params |
|---|---|---|
| Conv2D(32, 5×5, valid, relu) | (124, 124, 32) | 2,432 |
| Conv2D(32, 5×5, valid, relu) | (120, 120, 32) | 25,632 |
| MaxPool2D(2×2) | (60, 60, 32) | 0 |
| Dropout(0.25) | (60, 60, 32) | 0 |
| Flatten | 115,200 | 0 |
| Dense(256, relu) | 256 | **29,491,456** |
| Dropout(0.5) | 256 | 0 |
| Dense(2, softmax) | 2 | 514 |
| **Total** | | **29,520,034** |

This is the same architecture as `notebook-casia-2.ipynb`. The massive Flatten→Dense layer dominates parameter count.

---

## Training Pipeline Review

All variants use a two-phase training scheme:

### Phase 1: fit_generator (augmented)
- `ImageDataGenerator` with featurewise centering, featurewise std normalization, rotation_range=10
- Optimizer: Nadam
- Loss: categorical_crossentropy
- **Val accuracy stuck at 48.86% in ALL variants** (the augmented validation generator breaks evaluation)

### Phase 2: model.fit (non-augmented)
- Recompiled with Adam + binary_crossentropy
- EarlyStopping: `monitor='val_acc'` (deprecated, never triggers)

### The 48.86% Validation Bug

In Phase 1, validation accuracy is exactly 48.86% for every epoch across all three variants. This is because `ImageDataGenerator` with `featurewise_center=True` and `featurewise_std_normalization=True` computes statistics from the training data, but the validation generator uses different (wrong) statistics. In variant (2), partial improvement appears at epoch 22 (52.70%), suggesting the smaller batch_size=10 slightly mitigates the issue.

---

## Evaluation Metrics Review

### Per-class manual evaluation on full CASIA 2.0

| Metric | Original | Variant (1) | Variant (2) |
|---|---|---|---|
| Fake detection | 98.69% (2037/2064) | 99.37% (2051/2064) | 99.18% (2047/2064) |
| Real detection | 90.74% (6673/7354) | 93.36% (6866/7354) | 92.41% (6796/7354) |
| **Overall accuracy** | **92.48%** | **94.68%** | **93.89%** |

**CASIA1 cross-dataset test (variant 1):** Real=100%, Fake=100% (only a handful of samples tested).

---

## Visualization Assessment

- ELA visualization examples (showing original vs ELA image)
- Wavelet denoising exploration (BayesShrink, VisuShrink) — not used in training
- Training accuracy/loss curves
- No confusion matrix, no per-class precision/recall plots

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| ELA Implementation | **Good** | Quality=91 with PIL resave + OpenCV variant (cv2 with SCALE=15) |
| Model Design | **Poor** | 29.5M params from Flatten→Dense; same issue as notebook-casia-2 |
| Phase 1 Validation | **Broken** | Val stuck at 48.86% due to augmentation statistics mismatch |
| EarlyStopping | **Broken** | Monitors deprecated `val_acc`; never triggers |
| Variant Management | **Poor** | Three near-identical copies with no version control |
| Dead Code | **Misleading** | Config class references Xception + 224×224, never used |
| File Handling | **Poor** | Temp file `temp_file_name.jpg` not cleaned up |
| Variable Naming | **Bug** | `filename` vs `file_name` collision in evaluation loop |

---

## Strengths

1. **92-95% classification accuracy** — solid results for the ELA+CNN approach
2. **ELA quality=91** — good parameter choice for forensic signal extraction
3. **Two ELA implementations** — PIL-based (`convert_to_ela_image`) and OpenCV-based (`compute_ela_cv`) for comparison
4. **Wavelet denoising exploration** — shows awareness of alternative preprocessing techniques
5. **CASIA1 cross-dataset evaluation** — tests generalization beyond the training dataset

---

## Weaknesses

1. **Validation broken in Phase 1** — 48.86% accuracy persists across all epochs in all variants
2. **29.5M parameters** — absurdly large for a 4,164-sample binary classification
3. **Three nearly identical copies** with no clear documentation of what changed between them
4. **Classification only** — no pixel-level localization
5. **Data subsampling** — discards 70% of authentic images
6. **Temp file I/O** — writes JPEG to disk for every ELA computation
7. **Variable name collision** — `filename` and `file_name` used interchangeably in evaluation, causing only .jpg files to be processed (misses .png)

---

## Critical Issues

1. **Phase 1 validation at constant 48.86%.** The `featurewise_center` and `featurewise_std_normalization` parameters cause the validation generator to use training-set statistics. When `shuffle=False` on the validation generator, the per-batch statistics are computed from validation data, creating a mismatch. `fit_generator` evaluates on these incorrectly normalized batches, producing a garbage 48.86%.

2. **Variable name collision in evaluation.** The code uses `filename` for the loop variable but checks `or filename.endswith('png')` (correct variable) vs `file_name` (different variable from outer scope). In practice, this means `.png` files may be skipped during manual evaluation, inflating precision/recall for `.jpg`-only predictions.

3. **Empty .xpynb file (0 bytes).** The file `image-forgery-detection-using-ela-and-cnn.xpynb` exists but contains no data. It appears to be a corrupted save or accidental creation.

---

## Suggested Improvements

1. Fix the validation generator by using `validation_data=(X_val, Y_val)` directly instead of a generator
2. Replace Flatten→Dense with GlobalAveragePooling2d (reduces 29.5M → ~8K params)
3. Use all authentic images (don't subsample to 2,100)
4. Add proper augmentation in a way that doesn't break validation statistics
5. Fix the `val_acc` → `val_accuracy` EarlyStopping monitor
6. Delete the .xpynb empty file and deduplicate the three variants
7. Fix the filename/file_name variable collision

---

## Roast Section

When you look at these three notebooks side by side, the question isn't "which one is best?" — it's "why do all three exist?" They share the same architecture, the same dataset, the same 29.5M-parameter Dense layer, and the same validation bug where Phase 1 accuracy is permanently frozen at 48.86%. The only differences are epoch count (7 vs 24) and batch size (10 vs 15 vs 32). This is hyperparameter search by file duplication — the slowest, least traceable method possible.

The 48.86% validation bug is a clinic in how data generators can silently destroy your evaluation. `featurewise_center=True` normalizes each batch based on statistics fitted to the training set, but when the validation generator processes data independently, the normalization doesn't match. The result: every epoch reports exactly 48.86% validation accuracy, and nobody notices because the training accuracy looks fine. It's like grading exams in a foreign language — the students might be doing great, but you can't tell because you're reading it wrong.

The model itself is the familiar 29.5M-parameter elephant: two conv layers, one pooling operation, and a Dense layer larger than ResNet-34. 99.9% of the parameters live in a single matrix multiplication that maps 115,200 flattened features to 256 hidden units. This model doesn't learn features — it memorizes pixel patterns with brute force.

The Config class promising Xception at 224×224 is still hanging around at the top of all three notebooks, completely unused, like a menu item that was never delivered. Someone started with ambitious architectural plans and settled for a 2-layer CNN at 128×128.

**Bottom line:** 92-95% accuracy is the best result among the external reference notebooks, but it's achieved through memorization (29.5M params on 4K images) with broken validation. The same result could be achieved with 100× fewer parameters and proper evaluation.
