# Technical Audit: vK.7.1 (Run 01)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `vk-7-1-tampered-image-detection-and-localization-run-01.ipynb` (~11.7MB)

---

## 1. Architecture

| Attribute | Value |
|---|---|
| Model | `UNetWithClassifier` — same custom U-Net |
| Encoder | 4-stage: `3→64→128→256→512→1024` |
| Decoder | Transpose convolution with skip connections |
| Classifier Head | `AdaptiveAvgPool2d(1)→Linear(1024,512)→ReLU→Dropout(0.5)→Linear(512,2)` |
| Parameters | ~15.7M (all trainable, **no pretrained weights**) |
| Input | 3-channel RGB, 256×256 |

**Same architecture as all previous runs — no change.**

---

## 2. Training Pipeline

| Parameter | Value |
|---|---|
| Optimizer | `Adam(lr=1e-4)` |
| Scheduler | `CosineAnnealingLR(T_max=10)` |
| Cls Loss | `FocalLoss(alpha=class_weights, gamma=2.0)` |
| Seg Loss | `0.5×BCE + 0.5×Dice` |
| Loss Weights | α=1.5, β=1.0 |
| Batch Size | 8 |
| Epochs | 50 |
| AMP | No |
| Gradient Clipping | Yes (max_norm=5.0) |
| Early Stopping | No |
| DataParallel | No |

**This notebook retains the dual-block structure from vK.1–vK.3:**
- **Run 1 (Block 1):** Buggy data split — trains on 1,893 test samples for 30 epochs
- **Run 2 (Block 2):** Correct data split — trains on 8,829 samples for 50 epochs

---

## 3. Data Pipeline

| Attribute | Value |
|---|---|
| Dataset | CASIA-2 Upgraded |
| Split | 70/15/15 stratified |
| Block 1 Data Leak | **Still present** (trains on test_metadata.csv) |
| Augmentations | HFlip, BrightnessContrast, GaussNoise, JpegCompression, ShiftScaleRotate |

---

## 4. Evaluation — Actual Numbers

### Run 2 (Block 2 — correct split):

**Training Progress:**

| Epoch | Val Acc | Val Dice | Val IoU | Val F1 |
|---|---|---|---|---|
| 1 | 0.5201 | 0.5949 | 0.5949 | 0.5949 |
| 10 | 0.6195 | 0.5881 | 0.5881 | 0.5881 |
| 22 | 0.8256 | 0.5379 | 0.5294 | 0.5379 |
| 36 | 0.8906 | 0.5631 | 0.5471 | 0.5631 |
| 50 | 0.8943 | 0.5846 | 0.5603 | 0.5846 |

**Final Test Results:**

| Metric | Value |
|---|---|
| Test Accuracy | **0.8986** |
| Test Dice (all) | **0.5761** |
| Test IoU (all) | **0.5526** |
| Test F1 (all) | **0.5761** |

### Key Observations:
- Dice starts at 0.5949 (all-zero mask plateau), drops as model starts predicting non-zero masks, then slowly recovers — classic pattern of metric inflation masking early all-zero prediction
- **No tampered-only metrics reported** — regression from v6.5's honest reporting
- Classification accuracy matches vK.3-run: 89.86%
- Metrics are virtually identical to vK.3-run, suggesting the same model behavior

### Evaluation Features:

| Feature | Present? |
|---|---|
| Tampered-only metrics | **No** — regression from v6.5 |
| Threshold optimization | No |
| Confusion matrix | No |
| ROC/PR curves | No |
| Forgery-type breakdown | No |
| Robustness testing | No |
| Grad-CAM | No |

---

## 5. Strengths

1. Added W&B integration for experiment tracking
2. Added 4-panel visualization grid (Original/GT/Pred/Overlay)
3. 89.9% classification accuracy — consistent with vK.3
4. Complete 50-epoch training with no errors

---

## 6. Weaknesses

1. **Block 1 data leakage bug STILL present** — persists from vK.1
2. **No tampered-only metrics** — actually regressed from v6.5
3. **No pretrained encoder** — same training-from-scratch limitation
4. **All-sample Dice inflation** fully present, no warning or annotation
5. No AMP, no DataParallel, no early stopping
6. Checkpoint based on accuracy, not segmentation quality

---

## 7. Roast

vK.7.1 is a time capsule. It's essentially vK.3 with slightly different markdown, still faithfully reproducing both the Block 1 data leakage bug and the Dice inflation formula. The team had already discovered tampered-only metrics in v6.5 and then... forgot? Regressed? The 11.7MB file size suggests this was an actual execution with full output cells, and those outputs show the same old story: 89.9% accuracy, 0.58 inflated Dice, zero evidence that the segmentation head learned anything meaningful. The W&B logs captured the training curves perfectly — a flatline of mediocrity that nobody seems to have looked at too closely.
