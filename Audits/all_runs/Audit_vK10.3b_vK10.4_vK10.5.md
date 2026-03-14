# Technical Audit: vK.10.3b / vK.10.4 / vK.10.5

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**Files:**
- `vk-10-3b-tampered-image-detection-and-localization-run-01.ipynb` (~137KB)
- `vk-10-4-tampered-image-detection-and-localization-run-01.ipynb` (~410KB)
- `vk-10-5-tampered-image-detection-and-localization.ipynb` (~415KB)

---

## 1. Architecture (Identical Across All Three)

| Attribute | Value |
|---|---|
| Model | `UNetWithClassifier` — custom U-Net with classifier head |
| Encoder | `DoubleConv(3,64)→Down(64,128)→Down(128,256)→Down(256,512)→Down(512,1024)` |
| Decoder | `Up(1024,512)→Up(512,256)→Up(256,128)→Up(128,64)→OutConv(64,1)` |
| Classifier | `AdaptiveAvgPool2d(1)→Linear(1024, 2)` |
| Parameters | **31,563,459** (all trainable, **no pretrained weights**) |
| Input | 3-channel RGB, 256×256 |

**Note:** Parameter count increased from ~15.7M (vK.1–vK.7.5) to **31.6M** — the classifier head was simplified (removed 512-dim hidden layer, dropout) but the encoder/decoder appears unchanged. The exact increase source needs verification.

---

## 2. Training Pipeline

| Parameter | vK.10.3b | vK.10.4 | vK.10.5 |
|---|---|---|---|
| Optimizer | Adam(lr=1e-4, wd=1e-5) | Same | Same |
| Scheduler | CosineAnnealingLR | Same | Same |
| Cls Loss | BCE (α=1.5) | Same | Same |
| Seg Loss | BCE + Dice (β=1.0) | Same | Same |
| Batch Size | 32 (auto-scaled) | 32 (auto-scaled) | 32 (auto-scaled) |
| Max Epochs | 50 | 50 | 50 |
| AMP | **Yes** | **Yes** | **Yes** |
| Gradient Clipping | Yes | Yes | Yes |
| Early Stopping | **Yes** (patience=10, on tampered Dice) | **Yes** | **Yes** |
| **DataParallel** | **No** | **No** | **Yes** |

### Key Engineering Improvements Over vK.1–vK.7.5:

1. **AMP (Automatic Mixed Precision)** — first notebooks to enable this
2. **Early stopping on tampered Dice** — correct stopping criterion
3. **VRAM-based batch auto-scaling** — adapts to GPU memory
4. **Centralized CONFIG dictionary** — all hyperparameters in one place
5. **Full reproducibility seeding** — Python, NumPy, PyTorch, cuDNN
6. **Three-file checkpoint system** — best, last, resume
7. **DataParallel (vK.10.5 only)** — uses both Kaggle T4 GPUs
8. **`get_base_model()` (vK.10.5 only)** — properly unwraps DataParallel for state_dict

---

## 3. Data Pipeline

| Attribute | Value |
|---|---|
| Dataset | CASIA-2 Upgraded |
| Split | 70/15/15 stratified |
| Train/Val/Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256×256 |
| Augmentation | HFlip, BrightnessContrast, GaussNoise, ImageCompression, ShiftScaleRotate |
| Metadata Caching | Yes (CSV-based) |
| Worker Seeding | Yes |
| **Block 1 Data Leak** | **REMOVED** — single-block structure |

**Major improvement:** The vK.10.x series finally removed the dual-block structure and the Block 1 data leakage bug.

---

## 4. Evaluation — Actual Numbers

### vK.10.3b

| Metric | Value |
|---|---|
| Accuracy | 0.5061 |
| Dice (all) | 0.5781 |
| **Dice (tampered)** | **0.0004** |
| **IoU (tampered)** | **0.0002** |
| **F1 (tampered)** | **0.0004** |
| ROC-AUC | 0.6069 |
| Epochs completed | ~10 (early stopped) |
| Best epoch | Unknown |

### vK.10.4

| Metric | Value |
|---|---|
| Accuracy | 0.4675 |
| Dice (all) | 0.5938 |
| **Dice (tampered)** | **0.0000** |
| **IoU (tampered)** | **0.0000** |
| **F1 (tampered)** | **0.0000** |
| ROC-AUC | 0.6534 |
| Epochs completed | 10 (early stopped) |
| Best epoch | 1 (Dice = 0.0000) |

### vK.10.5

| Metric | Value |
|---|---|
| Accuracy | 0.4791 |
| Dice (all) | 0.5724 |
| **Dice (tampered)** | **0.0006** |
| **IoU (tampered)** | **0.0003** |
| **F1 (tampered)** | **0.0006** |
| ROC-AUC | 0.6201 |
| Epochs completed | ~10 (early stopped) |
| Best epoch | 1 (Dice = 0.0021) |

### All Three Are Total Segmentation Failures

Tampered-only Dice is 0.0000–0.0006 across all three runs. The model predicts near-zero masks for everything. Classification accuracy is ~47–50% — **worse than random chance** for binary classification.

---

## 5. Epoch-by-Epoch Evidence (vK.10.5)

| Epoch | Train Dice(tam) | Val Dice(tam) | Val AUC |
|---|---|---|---|
| 1 | 0.0115 | **0.0021** | 0.6215 |
| 2 | 0.0003 | 0.0000 | 0.6571 |
| 3 | 0.0006 | 0.0000 | 0.6710 |
| 4 | 0.0016 | 0.0000 | 0.6635 |
| 5–10 | 0.0000–0.0006 | 0.0000 | 0.66–0.72 |

Best model saved at epoch 1 (the only epoch with non-zero val Dice). Training never recovers. Early stopping correctly terminates after patience=10.

---

## 6. Root Cause Analysis — Why Did vK.10.x Collapse?

The vK.10.x series has **worse segmentation than vK.3** (which achieved 0.58 inflated Dice, implying ~0.15–0.25 real tampered Dice). Root causes:

1. **31.6M parameters trained from scratch on 8,829 images** — a data-to-parameter ratio of 0.00028. This is catastrophically unfavorable. The model cannot converge without pretrained features.

2. **Batch size 32 (auto-scaled) with LR=1e-4** — the larger batch compared to vK.3's batch=8 means each gradient update averages over 4× more samples, effectively reducing the learning rate. Without LR scaling, convergence is slower.

3. **Early stopping at patience=10 with tampered Dice starting near 0** — the model gets 10 epochs to show improvement from a 0.0021 baseline. When it doesn't (because the model needs more epochs with a scratch encoder), training terminates. In contrast, vK.3 ran all 50 epochs without early stopping.

4. **Combined loss gradient conflict** — the segmentation and classification losses may fight each other in early training, preventing either head from converging.

---

## 7. What Changed Between Versions

| Change | vK.10.3b | vK.10.4 | vK.10.5 |
|---|---|---|---|
| Data Visualization section | No | **Added** | Present |
| DataParallel | No | No | **Yes** |
| `get_base_model()` | No | No | **Yes** |
| Multi-GPU batch scaling | Single GPU | Single GPU | Total VRAM |
| Architecture | Same | Same | Same |
| Hyperparameters | Same | Same | Same |

The progression is purely engineering improvements — DataParallel, data visualization, multi-GPU support. **None of these address the fundamental model learning failure.**

---

## 8. Engineering Quality

| Criterion | Rating | Notes |
|---|---|---|
| CONFIG Management | **EXCELLENT** | Centralized dict with all hyperparameters |
| Reproducibility | **EXCELLENT** | Full seeding for all RNGs + cuDNN |
| Checkpoint System | **GOOD** | Three-file system: best/last/resume |
| AMP | **GOOD** | Properly implemented with GradScaler |
| Early Stopping | **GOOD** | Correct metric (tampered Dice), but patience may be too low |
| DataParallel (vK.10.5) | **GOOD** | Properly implemented with `get_base_model()` |
| Batch Auto-Scaling | **GOOD** | VRAM-aware dynamic batch sizing |
| Data Leakage | **FIXED** | Block 1 bug finally eliminated |

### Missing Evaluation Features

| Feature | Present? |
|---|---|
| Threshold optimization | No |
| Confusion matrix | No |
| ROC/PR curves | No |
| Forgery-type breakdown | No |
| Mask-size stratification | No |
| Robustness testing | No |
| Grad-CAM | No |
| Shortcut learning checks | No |

---

## 9. Roast

The vK.10.x series is the triumph of engineering over science. Someone spent three iterations building a beautiful CONFIG system, implementing AMP, adding DataParallel with a proper unwrapper, creating a three-file checkpoint system with resume capability, seeding every random number generator in existence — and then pointed this pristine engineering at a **31.6M parameter model trained from scratch on 8,800 images** and watched it produce tampered Dice = 0.0000.

The early stopping works perfectly — it correctly identifies that the model is learning nothing and mercifully terminates training. The checkpoint system perfectly saves the best model — which is the epoch 1 model with Dice = 0.0021. The VRAM-based batch scaling elegantly auto-tunes the batch size — to 32, which further hampers convergence.

This is the engineering equivalent of building a Formula 1 pit crew for a car with no engine. Every system works as designed. The overall system produces zero useful output. **A pretrained ResNet34 encoder would have fixed everything in one line:** `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet')`.
