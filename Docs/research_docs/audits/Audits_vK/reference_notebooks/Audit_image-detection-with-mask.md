# Audit: Image Detection With Mask

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `image-detection-with-mask (1).ipynb` (8.5 MB)

---

## Notebook Overview

A PyTorch dual-head notebook that performs both **classification and segmentation** for image tampering detection on the CASIA Splicing dataset. Uses a custom `UNetWithClassifier` (same architecture as the vK.10.x series) with two training runs. Classification reaches 89.75% accuracy, but the segmentation head barely learns (Dice stuck at ~0.57).

| Attribute | Value |
|---|---|
| Cell Count | ~60+ cells |
| Model | `UNetWithClassifier` (PyTorch) |
| Parameters | ~31.6M (same as vK.10.x) |
| Dataset | CASIA Splicing Detection + Localization |
| Task | Dual: binary classification + pixel-level segmentation |
| Image Size | 256×256 |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CASIA Splicing Detection + Localization |
| Total | 8,829 + 1,892 + 1,893 |
| Train | 8,829 |
| Validation | 1,892 |
| Test | 1,893 |
| Augmentation | HFlip, VFlip, RandomBrightnessContrast, Normalize |
| Library | Albumentations + ToTensorV2 |

Uses the same CASIA dataset and splits as the vK.10.x series. Contains Arabic comments in code cells.

---

## Model Architecture Review

```
UNetWithClassifier (custom PyTorch nn.Module)

Encoder:
  inc:   DoubleConv(3, 64)     — [Conv2d(3×3)+BN+ReLU] ×2
  down1: Down(64, 128)         — MaxPool2d(2) + DoubleConv
  down2: Down(128, 256)
  down3: Down(256, 512)
  down4: Down(512, 1024)

Decoder (skip connections via ConvTranspose2d):
  up1: Up(1024, 512)           — ConvTranspose2d + concat + DoubleConv
  up2: Up(512, 256)
  up3: Up(256, 128)
  up4: Up(128, 64)

Segmentation Head:
  outc: Conv2d(64, 1, kernel_size=1)

Classification Head (from bottleneck):
  AdaptiveAvgPool2d(1) → Linear(1024, 512) → ReLU → Dropout(0.5) → Linear(512, 2)
```

This is the **same architecture** as the vK.10.3b series. No pretrained encoder — all 31.6M parameters trained from scratch.

---

## Training Pipeline Review

### Run 1 (Simple Setup)

| Component | Configuration |
|---|---|
| Optimizer | Adam (lr=1e-4) |
| Loss | 1.0×CrossEntropyLoss(class_weights) + 1.0×BCEWithLogitsLoss |
| Scheduler | ReduceLROnPlateau (mode=max, patience=3, factor=0.5) |
| Epochs | 30 |
| Batch Size | 8 |

**Results:** Val Acc=71.88%, Val Dice=0.5949. Segmentation head essentially not learning.

### Run 2 (Improved Loss)

| Component | Configuration |
|---|---|
| Loss | FocalLoss(class_weights) + 0.5×BCE + 0.5×DiceLoss |
| Scheduler | CosineAnnealingLR (T_max=10) |
| Epochs | 50 |

**Results:** Val Acc=89.22%, Test Acc=89.75%, Test Dice=0.5673. Classification improved significantly but segmentation still stuck.

---

## Evaluation Metrics Review

### Run 1

| Metric | Value |
|---|---|
| Best Val Accuracy | 71.88% |
| Val Dice | 0.5949 |
| Test Accuracy | 71.88% |
| Test Dice | 0.5949 |

### Run 2

| Metric | Value |
|---|---|
| Best Val Accuracy | **89.22%** (epoch 47) |
| Val Dice | 0.5628 |
| Test Accuracy | **89.75%** |
| Test Dice | **0.5673** |

**Dice of ~0.57 indicates the segmentation head is barely learning.** For reference, a model predicting all-zeros on a dataset with ~40% tampered images would get Dice ≈ 0.57 (because authentic images have zero masks, contributing perfect scores). The segmentation head may be predicting near-zero for all pixels.

---

## Visualization Assessment

Limited information from cell outputs. No detailed visualization assessment possible from the available data.

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Architecture | **Fair** | Same custom UNet as vK.10.x, from scratch |
| Dataset Loading | **Good** | Albumentations pipeline, proper transforms |
| Training: Run 1 | **Poor** | BCE only for segmentation, no dice loss |
| Training: Run 2 | **Good** | FocalLoss + Dice combination |
| Code Quality | **Fair** | Contains Arabic comments (bilingual) |
| Data Leak | **Critical** | TEST_CSV reads from val CSV file |

---

## Strengths

1. **Dual-head architecture** — performs both classification and localization, same approach as the vK.10.x series
2. **Two training runs** — demonstrates iterative improvement (Run 1 → Run 2)
3. **FocalLoss in Run 2** — addresses class imbalance properly
4. **Albumentations pipeline** — modern augmentation library with proper transforms
5. **89.75% classification accuracy** — competitive result for image-level detection

---

## Weaknesses

1. **Segmentation Dice stuck at ~0.57** — the localization head is essentially not learning
2. **No pretrained encoder** — 31.6M parameters from scratch
3. **No ELA input** — only RGB channels
4. **Data leak in test evaluation** — `TEST_CSV` points to `val_metadata.csv`
5. **No threshold optimization** — uses default threshold=0.5
6. **Bilingual code** — Arabic comments reduce accessibility for non-Arabic readers

---

## Critical Issues

1. **Data leak: TEST_CSV = val_metadata.csv.** The code at line ~490 sets `TEST_CSV` to the validation CSV file path, meaning the "test" evaluation is actually re-evaluating on the validation set. The reported test metrics (89.75% accuracy) are therefore **leaking validation data into the test evaluation**.

2. **Segmentation head not learning.** Dice ≈ 0.57 across both runs suggests the segmentation head is predicting near-zero masks. The combined loss (classification + segmentation) is dominated by the classification head, which drives accuracy up while the segmentation gradient signal is too weak. Without `pos_weight` on the BCE loss, the overwhelming majority of zero-mask pixels from authentic images suppresses the tampered-region gradients.

3. **No pretrained encoder.** Training a 31.6M parameter U-Net from scratch on ~8,800 images is insufficient for learning meaningful low-level features. The vK.11.x series demonstrated that switching to a pretrained ResNet34 encoder doubles segmentation performance.

---

## Suggested Improvements

1. Fix `TEST_CSV` to point to the actual test CSV file
2. Use pretrained encoder (ResNet34 via SMP) — this alone would likely double segmentation Dice
3. Add ELA as a 4th input channel for forensic signal
4. Add `pos_weight` to BCE loss for segmentation imbalance
5. Add threshold optimization on validation set
6. Add edge loss (Sobel) to improve boundary precision
7. Add gradient accumulation for larger effective batch size
8. Translate Arabic comments to English for broader accessibility

---

## Roast Section

This notebook is the "before" picture in the vK.10.x evolution story. Same UNetWithClassifier architecture, same CASIA dataset, same 31.6M-parameter from-scratch training, and the same result: classification works (89.75%) but segmentation doesn't (Dice=0.57). The model learns to say "this image is tampered" without learning to point at where.

The most critical bug is the quietest: `TEST_CSV` points to `val_metadata.csv`. The proudly reported 89.75% test accuracy is actually validation accuracy wearing a test-accuracy name tag. The real test performance is unknown because it was never evaluated. This is the data science equivalent of grading your own homework and submitting the results.

The Dice score of 0.57 deserves special attention. On a dataset where ~40% of images are authentic (all-zero masks), a model that predicts all-zero masks would achieve roughly Dice=0.58 on the mixed set (because (2×0)/(0+0) is defined as 1.0 for empty masks in most implementations). The segmentation head is likely producing near-zero outputs for everything — and the Dice "score" is almost entirely driven by the trivial true-negative contribution of authentic images.

Run 2's improvement from 72% → 90% classification accuracy through FocalLoss + CosineAnnealing is genuinely good engineering — but it only fixes the classification head. The segmentation head gets DiceLoss added in Run 2 and still doesn't improve (0.5949 → 0.5628, actually worse). The fundamental problem — 31.6M parameters from scratch — is not addressable by loss function changes.

**Bottom line:** Good experimental iteration (Run 1 → Run 2) on the classification task, but segmentation requires pretrained features, pos_weight, and/or ELA input to break above trivial performance. Fix the test CSV leak before reporting any results.
