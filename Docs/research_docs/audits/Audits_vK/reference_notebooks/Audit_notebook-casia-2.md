# Audit: Notebook CASIA-2

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `notebook-casia-2.ipynb` (4.9 MB)

---

## Notebook Overview

A Keras-based ELA + CNN notebook for binary image forgery classification on CASIA 2.0. Uses a simple 2-conv CNN with ELA preprocessing at 128×128 resolution. Achieves 90.5% test accuracy but shows clear overfitting and has several notable engineering issues.

| Attribute | Value |
|---|---|
| Cell Count | 64 (many empty cells at end) |
| Model | Sequential CNN (2 Conv + 1 Dense) |
| Parameters | **29,520,034** (all trainable) |
| Dataset | CASIA 2.0 (12,477 images) |
| Task | Binary classification (authentic vs tampered) |
| Preprocessing | ELA at JPEG quality=91 |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CASIA 2.0 |
| Authentic images | 7,354 |
| Tampered images | 5,123 |
| Total | 12,477 |
| Train | 8,983 (90% × 80%) |
| Validation | 2,246 (90% × 20%) |
| Test | 1,248 (10%) |
| Image Size | 128×128 |
| ELA Quality | 91 |

**Split is 90/10 train-test, then 80/20 train-val from the 90% portion.** This gives a very small test set (1,248) compared to modern standards (15% = ~1,872).

---

## Model Architecture Review

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

**The Flatten → Dense connection has 29.5M parameters.** This single layer accounts for 99.9% of all model parameters. The model is massively overparameterized — a 115,200 × 256 weight matrix for a dataset of 8,983 training images.

---

## Training Pipeline Review

The model is trained **twice** in sequence (without weight reinitialization):

### Phase 1: fit_generator + Nadam
| Component | Value |
|---|---|
| Optimizer | Nadam |
| Loss | categorical_crossentropy |
| Augmentation | ImageDataGenerator (featurewise_center, featurewise_std_norm, rotation=10°) |
| Epochs | 25 |
| Batch Size | 32 |

### Phase 2: model.fit + Adam (continues from Phase 1 weights)
| Component | Value |
|---|---|
| Optimizer | Adam (lr=1e-4, decay=1e-4/25) |
| Loss | binary_crossentropy |
| EarlyStopping | monitor=`val_acc` (deprecated), patience=2 |
| Epochs | 25 |
| Batch Size | 32 |

**Critical:** The model is compiled and trained twice with different losses (`categorical_crossentropy` → `binary_crossentropy`) without reinitializing weights. Phase 2 continues from Phase 1's learned weights, essentially creating an informal two-phase training schedule.

---

## Evaluation Metrics Review

### Phase 2 Training Progression

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.3857 | 82.56% | 0.3315 | 88.16% |
| 8 | 0.1842 | 92.07% | 0.2202 | **90.96%** |
| 11 | 0.1595 | 92.96% | 0.2148 | 90.96% |
| 25 | 0.1015 | 94.71% | 0.3078 | 90.47% |

### Test Results

| Metric | Value |
|---|---|
| Test Accuracy | **90.54%** |
| Weighted F1 | 0.91 |
| Tampered Precision | 0.84 |
| Tampered Recall | 0.92 |
| Authentic Precision | 0.95 |
| Authentic Recall | 0.89 |

---

## Visualization Assessment

The notebook includes:
- Training accuracy/loss curves
- Classification reports (sklearn)
- No confusion matrix visualization (only text)
- No ELA visualization examples
- Naive localization attempt: thresholding ELA difference at pixel value > 50 (not learned)

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| ELA Implementation | **Good** | Quality=91, proper JPEG re-save and diff |
| Model Design | **Poor** | 29.5M params from single Flatten→Dense layer |
| Training Setup | **Broken** | Two-phase training without reinitialization |
| Early Stopping | **Broken** | Monitors `val_acc` (deprecated, should be `val_accuracy`) |
| Overfitting Control | **Poor** | Train loss=0.10 vs val loss=0.31 at epoch 25 |
| Dead Code | **Misleading** | Config class references Xception + 224×224, never used |
| Documentation | **Fair** | Some markdown cells, but key design decisions unexplained |

---

## Strengths

1. **Good ELA implementation** — JPEG quality=91 is a reasonable choice for forensic signal extraction
2. **90.5% test accuracy** — best among the external Kaggle reference notebooks
3. **Two-phase training** — while unintentional, the Nadam→Adam switch with learning rate decay is similar to a warmup strategy
4. **Classification report** — properly uses sklearn for per-class metrics

---

## Weaknesses

1. **29.5M parameters for binary classification** — the Flatten→Dense layer is absurdly large
2. **Clear overfitting** — train loss drops to 0.10 while val loss rises from 0.21 to 0.31
3. **No localization** — only classification; localization attempt is a naive pixel threshold
4. **128×128 resolution** — too small for detecting subtle tampering artifacts
5. **EarlyStopping monitors deprecated `val_acc`** — callback likely never triggers in modern TF/Keras
6. **Two-phase training** — switching loss functions without reinitialization is methodologically unsound
7. **Dead Config class** — references Xception and 224×224 that are never used

---

## Critical Issues

1. **Flatten→Dense bottleneck (29.5M params).** With only 2 conv layers and 1 pooling layer, the spatial dimensions are barely reduced (60×60×32 = 115,200), creating an enormous Dense layer. Adding more pooling layers or using GlobalAveragePooling would reduce this to ~32 parameters × 256 = 8,192 — a 3,600× reduction.

2. **Overfitting trajectory.** Validation loss increases from epoch 8 onward while training loss continues decreasing. The model memorizes the training set. The EarlyStopping callback should have stopped training at epoch 8, but it monitors the deprecated `val_acc` metric.

3. **Loss function switch without reinitialization.** Phase 1 trains with categorical_crossentropy (Nadam), Phase 2 recompiles with binary_crossentropy (Adam). Since the weights are not reset, Phase 2 starts from Phase 1's converged state. This is not standard practice and makes it unclear which loss function actually drove the learning.

4. **Naive localization.** The `find_manipulated_region` function just thresholds ELA pixel differences at 50 — this is not learned, not validated, and produces binary masks with no evaluation against ground truth.

---

## Suggested Improvements

1. Replace Flatten→Dense with GlobalAveragePooling2d to reduce parameters 3,600×
2. Use a deeper encoder (ResNet, EfficientNet) with pretrained weights
3. Add proper localization via segmentation (U-Net decoder)
4. Fix EarlyStopping to monitor `val_accuracy` (or `val_loss`)
5. Add proper augmentation (flips, color jitter, JPEG compression)
6. Increase resolution to 256×256 or higher
7. Remove dead Config class and Xception references
8. Add more pooling layers before the Dense layer

---

## Roast Section

This notebook achieves 90% accuracy through the time-honored tradition of brute force — 29.5 million parameters thrown at a binary classification problem. The Flatten→Dense layer alone contains 29.5M weights, which is more parameters than ResNet-34 (21.8M) or EfficientNet-B0 (5.3M). The model compensates for having only 2 convolutional layers and 1 pooling operation by memorizing the training set pixel-by-pixel.

The overfitting curve tells the whole story: train loss cruises down to 0.10 while validation loss climbs from 0.21 to 0.31. The EarlyStopping callback was supposed to prevent this, but it monitors `val_acc` — a metric name that Keras deprecated years ago. So the callback sits there, watching for a metric that never arrives, while the model overfits with impunity.

The two-phase training is the accidental hero. Someone trained the model with Nadam + categorical_crossentropy for 25 epochs, then recompiled with Adam + binary_crossentropy for another 25 — without realizing that recompiling doesn't reset the weights. The result is an informal transfer learning scheme: Phase 1 provides warm initialization, Phase 2 fine-tunes with a smaller LR and different loss. This accident probably contributes 5-10% of the model's accuracy.

The Config class at the top promises Xception at 224×224 but delivers a 2-conv CNN at 128×128. It's the code equivalent of a restaurant menu with lobster that actually serves instant noodles.

The "localization" is a single function that thresholds ELA pixel differences at 50. It's never evaluated, never compared to ground truth, and would fail immediately on any image where the tampered region has similar compression characteristics to the rest of the image. This is localization by wishful thinking.

**Bottom line:** 90% accuracy is respectable for a simple ELA+CNN approach, but the model achieves it by memorizing rather than generalizing. Replace the 29.5M Dense layer with GlobalAveragePooling, add proper augmentation and early stopping, and the same accuracy could be achieved with 100K parameters.
