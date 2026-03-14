# Technical Audit: vK.1 / vK.2 / vK.3 / vK.3-run-01

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**Scope:** Early-generation notebooks — documentation evolution with identical training pipeline

---

## 1. Structural Overview

All four notebooks share a **dual-block structure** — two complete, independent training pipelines executed sequentially:

- **Block 1 ("Source-Preserved Prior Experiment"):** Simpler training with a **known-buggy data split**, basic losses, 30 epochs.
- **Block 2 ("Effective Submission Run"):** Corrected data split, FocalLoss + BCE+Dice, 50 epochs, W&B integration.

The code in Block 1 is identical across all four notebooks. Block 2 is nearly identical across vK.1–vK.3 except for comments and documentation. vK.3-run-01 is the actual Kaggle execution of vK.3.

---

## 2. Architecture

| Attribute | Value |
|---|---|
| Model | `UNetWithClassifier` — custom U-Net with classification head |
| Encoder | 4-stage: `3→64→128→256→512→1024` (bottleneck) |
| Decoder | Transpose convolution upsampling with skip connections |
| Classifier Head | `AdaptiveAvgPool2d(1)→Linear(1024,512)→ReLU→Dropout(0.5)→Linear(512,2)` |
| Parameters | ~15.7M (all trainable, **no pretrained weights**) |
| Input | 3-channel RGB, 256×256 |

**Critique:**
- Vanilla U-Net from scratch with no pretrained encoder on a 12K-image dataset — severely data-inefficient
- No connection between classification and segmentation heads (no multi-task gating or attention)
- No ELA, frequency, or noise-based input channels

---

## 3. Training Pipeline

### Block 1

| Parameter | Value |
|---|---|
| Optimizer | `Adam(lr=1e-4)` |
| Scheduler | `ReduceLROnPlateau(mode="max", factor=0.5, patience=3)` |
| Cls Loss | `CrossEntropyLoss(weight=class_weights)` |
| Seg Loss | `BCEWithLogitsLoss` |
| Loss Weights | α=1.0, β=1.0 |
| Epochs | 30, Batch Size 8, No AMP, No gradient clipping |

### Block 2

| Parameter | Value |
|---|---|
| Optimizer | `Adam(lr=1e-4)` |
| Scheduler | `CosineAnnealingLR(T_max=10)` |
| Cls Loss | `FocalLoss(alpha=class_weights, gamma=2.0)` |
| Seg Loss | `0.5×BCE + 0.5×Dice` |
| Loss Weights | α=1.5, β=1.0 |
| Epochs | 50, Batch Size 8, Gradient clipping (max_norm=5.0) |

---

## 4. Data Pipeline

| Attribute | Value |
|---|---|
| Dataset | CASIA-2 Upgraded (12,614 images: 7,491 Au + 5,123 Tp) |
| Split | 70/15/15 stratified (`random_state=42`) |
| Train/Val/Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256×256 |
| Augmentation (Block 2) | HFlip, BrightnessContrast, GaussNoise, JpegCompression, ShiftScaleRotate |

### CRITICAL BUG — Data Leakage in Block 1

```python
TRAIN_CSV = "/kaggle/working/test_metadata.csv"    # TRAINS ON TEST SET!
TEST_CSV  = "/kaggle/working/val_metadata.csv"      # TESTS ON VAL SET!
```

Block 1 trains on **1,893 test samples** and evaluates on validation. Block 2 corrects this.

---

## 5. Evaluation — Actual Numbers (vK.3-run-01)

### Block 1 (buggy split, 30 epochs on 1,893 samples):

| Metric | Value |
|---|---|
| Best Val Acc | 0.7077 |
| Test Acc | 0.7077 (actually evaluated on val set) |
| Test Dice | 0.5949 (**inflated**) |

### Block 2 (correct split, 50 epochs on 8,829 samples):

| Metric | Value |
|---|---|
| Test Acc | **0.8986** |
| Test Dice | **0.5761** (inflated) |
| Test IoU | **0.5526** (inflated) |
| Test F1 | **0.5761** (inflated) |

### CRITICAL BUG — Metric Inflation

The `dice_coef` function computes over ALL samples including authentic. For authentic images (all-zero GT masks), if the model predicts all-zero: `Dice = (0+ε)/(0+ε) = 1.0`. With ~59.4% authentic samples, a model predicting all-zero everywhere gets Dice ≈ 0.594.

**The 0.5949 Dice plateau in Block 1 IS the all-zero-mask degenerate solution.** The real tampered-only Dice is likely 0.15–0.25.

---

## 6. What Changed Between Versions

| Aspect | vK.1 | vK.2 | vK.3 | vK.3-run |
|---|---|---|---|---|
| Code Logic | Baseline | +IoU/F1/W&B reporting | = vK.2 | = vK.3 |
| Documentation | None | Markdown cells added | English docstrings | = vK.3 |
| Block 1 Data Leak | Present | Present | Present | Executed |
| Dice Inflation | Present | Present | Present | Visible in outputs |
| Seeding | None | None | None | None |

**Verdict:** vK.1→vK.3 is purely a documentation maturity progression. Zero algorithmic or architectural improvements. All bugs persist.

---

## 7. Engineering Quality

| Criterion | Rating | Notes |
|---|---|---|
| Reproducibility | **FAIL** | No seeding for PyTorch/NumPy/random |
| Config Management | **POOR** | Scattered constants, no unified CONFIG dict |
| Checkpoint Strategy | **FLAWED** | Saves on best val accuracy, NOT best segmentation |
| Code Modularity | **POOR** | Flat notebook, massive Block 1/Block 2 duplication |
| AMP | **ABSENT** | No mixed-precision training |
| Early Stopping | **ABSENT** | Runs all epochs regardless of convergence |
| DataParallel | **ABSENT** | Single GPU only |

---

## 8. Bugs & Issues

| # | Severity | Issue |
|---|---|---|
| 1 | **CRITICAL** | Block 1 data leakage — trains on test set |
| 2 | **CRITICAL** | Dice/IoU/F1 inflated by authentic samples (0/0 = 1.0) |
| 3 | **HIGH** | No seeding — results not reproducible |
| 4 | **HIGH** | Checkpoint based on accuracy, not segmentation quality |
| 5 | **MEDIUM** | No early stopping — potential overtraining |
| 6 | **MEDIUM** | Deprecated JpegCompression API |
| 7 | **LOW** | No AMP — slower than necessary |
| 8 | **LOW** | 60% code duplication between blocks |

---

## 9. Roast

The vK.1–vK.3 series is the **documentation equivalent of putting lipstick on a pig.** Three version increments that change exactly zero lines of training code. The Block 1 data leakage shipped across all versions — nobody noticed the notebook was literally training on the test set. The Dice metric was designed to deceive: a model predicting blank masks scores 0.59 and nobody questioned it. The 89.9% classification accuracy is real but irrelevant to the assignment — the assignment asks for *localization*, and the segmentation head learned nothing meaningful. No seeding, no AMP, no early stopping, no pretrained encoder. This is an engineering prototype with three coats of documentation paint.
