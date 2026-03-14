# Technical Audit: vK.10.6 (Run 01)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `vk-10-6-tampered-image-detection-and-localization-run-01.ipynb` (~2.6MB, 85K lines)
**Platform:** Kaggle, 2× Tesla T4 GPUs, ~4h 18m runtime

---

## 1. Architecture

| Attribute | Value |
|---|---|
| Model | `UNetWithClassifier` — custom U-Net (same as vK.10.5) |
| Encoder | `DoubleConv(3,64)→Down(64,128)→Down(128,256)→Down(256,512)→Down(512,1024)` |
| Decoder | `Up(1024,512)→Up(512,256)→Up(256,128)→Up(128,64)→OutConv(64,1)` |
| Classifier | `AdaptiveAvgPool2d(1)→Linear(1024,512)→ReLU→Dropout(0.5)→Linear(512,2)` |
| Parameters | **31,563,459** (all trainable, **no pretrained weights**) |
| Input | 3-channel RGB, 256×256 |

**No architectural changes from vK.10.5.** Same custom U-Net trained from scratch. vK.10.6 is pure evaluation additions.

---

## 2. Training Pipeline

| Parameter | vK.10.5 | vK.10.6 | Change |
|---|---|---|---|
| Optimizer | Adam(lr=1e-4, wd=1e-4) | Same | — |
| Scheduler | CosineAnnealingLR(T_max=50) | Same | — |
| Cls Loss | FocalLoss(γ=2, weights=[0.84,1.23]) | Same | — |
| Seg Loss | 0.5×BCE + 0.5×Dice | Same | — |
| Combined Loss | 1.5×cls + 1.0×seg | Same | — |
| Batch Size | 32 (auto-scaled) | 32 (auto-scaled) | — |
| **Max Epochs** | **50** | **100** | **2× increase** |
| **Early Stop Patience** | **10** | **30** | **3× increase** |
| AMP | Yes | Yes | — |
| Gradient Clipping | max_norm=5.0 | Same | — |
| DataParallel | Yes (2 GPUs) | Yes (2 GPUs) | — |

### Key Training Change: 100 Epochs with Patience 30

vK.10.6 doubled the epoch budget (50→100) and tripled early stopping patience (10→30). This is significant because vK.10.5 hit patience at ~epoch 10 before the model had a chance to learn. vK.10.6 ran all 100 epochs — early stopping never triggered because the model kept improving through epoch 99.

### CosineAnnealingLR Double-Cycle Artifact

With `T_max=50` over 100 epochs, the scheduler completes one full cosine cycle at epoch 50 (LR drops to near-zero), then the LR rises back up for epochs 51–100. This is an unconventional double-cycle schedule that was likely unintentional — but it actually helped, as the second LR peak coincided with the best epoch (99).

---

## 3. Data Pipeline

| Attribute | Value |
|---|---|
| Dataset | CASIA 2.0 Upgraded (`harshv777/casia2-0-upgraded-dataset`) |
| Split | 70/15/15 stratified by label |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | 256×256 |
| Data Leakage Check | **Explicit path overlap assertion — PASSED** (NEW in vK.10.6) |

### Augmentations

| Transform | Details |
|---|---|
| Resize | 256×256 |
| HorizontalFlip | p=0.5 |
| RandomBrightnessContrast | p=0.3 |
| GaussNoise | p=0.25 |
| ImageCompression | quality 50–90, p=0.25 |
| Affine | translate=2%, scale=0.9–1.1, rotate=±10°, p=0.5 |
| Normalize | ImageNet stats |

Same augmentation pipeline as vK.10.5.

---

## 4. Evaluation — Exact Numbers

### Test Results at Default Threshold (0.50)

| Metric | Value |
|---|---|
| Test Accuracy | **0.8357** |
| Image-Level AUC-ROC | **0.9057** |
| Dice (all) | 0.4853 |
| IoU (all) | 0.4638 |
| **Dice (tampered-only)** | **0.1946** |
| **IoU (tampered-only)** | **0.1418** |
| **F1 (tampered-only)** | **0.1946** |

### After Threshold Optimization (optimal = 0.15)

| Metric | Default (0.50) | Optimal (0.15) | Delta |
|---|---|---|---|
| Tampered Dice/F1 | 0.1946 | **0.2213** | +0.0267 |
| Tampered IoU | 0.1418 | **0.1554** | +0.0136 |

### Pixel-Level AUC

| Metric | Value |
|---|---|
| Pixel-Level AUC-ROC (tampered only) | **0.7083** |
| Image-Level AUC-ROC (all) | **0.9057** |

### Per-Forgery-Type Breakdown (threshold=0.15)

| Type | Count | F1 |
|---|---|---|
| Copy-move | 260 | **0.4110** |
| Splicing | 509 | **0.1244** |

**Interesting reversal from v6.5/v8:** vK.10.6 detects copy-move (F1=0.41) much better than splicing (F1=0.12), the opposite of v6.5 (splicing=0.59, copy-move=0.31). This likely reflects different feature learning between the custom from-scratch encoder and the pretrained ResNet34.

### Mask-Size Stratification (threshold=0.15)

| Bucket | Count | F1 |
|---|---|---|
| Tiny (<2%) | 289 | 0.0972 |
| Small (2–5%) | 190 | 0.2115 |
| Medium (5–15%) | 171 | 0.3403 |
| Large (>15%) | 119 | 0.3675 |

Strong size-performance correlation. Tiny forgeries (38% of tampered images) are nearly undetectable.

### Shortcut Learning Checks

| Test | F1 | Delta | Verdict |
|---|---|---|---|
| Baseline | 0.2213 | — | — |
| Mask randomization | 0.0658 | -0.1555 | **PASS** |
| Boundary erosion (1px) | 0.2221 | +0.0007 | **PASS** |

Model uses image content, not shortcuts. Predictions are not boundary-dependent.

### Robustness Results (threshold=0.15)

| Condition | F1 | Delta from Clean |
|---|---|---|
| Clean | 0.2213 | — |
| JPEG QF=70 | 0.2037 | -0.0176 |
| JPEG QF=50 | 0.1682 | -0.0531 |
| Noise σ=10 | 0.1986 | -0.0227 |
| Noise σ=25 | 0.1728 | -0.0485 |
| Blur k=3 | 0.2074 | -0.0139 |
| **Blur k=5** | **0.0748** | **-0.1465** |
| Resize 0.75× | 0.2132 | -0.0081 |
| Resize 0.50× | 0.2062 | -0.0151 |

Blur k=5 is catastrophic (-66% of clean F1). JPEG and noise cause moderate drops. Resize is well-tolerated.

### Training Progression (Selected Epochs)

| Epoch | Train Loss | Val Acc | Val AUC | Val Dice(tam) |
|---|---|---|---|---|
| 1 | 0.9352 | 0.4059 | 0.6619 | 0.0000 |
| 10 | 0.8194 | 0.4995 | 0.6722 | 0.0000 |
| 20 | 0.8047 | 0.5835 | 0.7629 | 0.0321 |
| 30 | 0.7866 | 0.7278 | 0.8044 | 0.1103 |
| 50 | 0.7601 | 0.7209 | 0.8300 | 0.1405 |
| 97 | 0.7598 | 0.7997 | 0.8853 | 0.1682 |
| **99** | — | **0.8277** | **0.8981** | **0.1811** |
| 100 | 0.7024 | 0.8224 | 0.8971 | 0.1726 |

**Model was still improving at epoch 99.** Tampered Dice was 0.0000 for the first ~12 epochs, then slowly climbed. The model had NOT converged — more epochs would likely improve further. This vindicates the patience=30 and max_epochs=100 changes.

---

## 5. Evaluation Features

| Feature | vK.10.5 | vK.10.6 | Notes |
|---|---|---|---|
| Tampered-only metrics | Yes | Yes | |
| Data leakage verification | No | **Yes** | Path overlap assertion, PASSED |
| Threshold optimization | No | **Yes** | Sweep 0.05–0.80, optimal=0.15 |
| Pixel-level AUC-ROC | No | **Yes** | 0.7083 (tampered only) |
| Confusion matrix | No | **Yes** | Seaborn heatmap |
| ROC curve | No | **Yes** | With AUC annotation |
| PR curve | No | **Yes** | With AP annotation |
| Forgery-type breakdown | No | **Yes** | Splicing vs copy-move |
| Mask-size stratification | No | **Yes** | 4 buckets: tiny/small/medium/large |
| Shortcut learning checks | No | **Yes** | Mask randomization + boundary erosion |
| Failure case analysis | No | **Yes** | 10 worst predictions with metadata |
| Grad-CAM | No | **Yes** | Encoder bottleneck (down4.conv.block) |
| Robustness testing | No | **Yes** | 8 degradation conditions |
| W&B integration | Yes | Yes | Online mode |

**vK.10.6 adds 12 new evaluation features** — making it the most comprehensively evaluated notebook in the entire vK.x series and matching v6.5/v8's evaluation depth.

---

## 6. Engineering Quality

| Criterion | Rating | Notes |
|---|---|---|
| CONFIG System | **Excellent** | Centralized dict, all hyperparameters |
| Reproducibility | **Excellent** | Full seeding, cuDNN deterministic, worker seeds |
| Checkpoint System | **Excellent** | Three-file (best/last/periodic) with auto-resume |
| DataParallel | **Good** | 2 GPUs with `get_base_model()` unwrapper |
| AMP | **Good** | autocast + GradScaler |
| Batch Auto-Scaling | **Good** | VRAM-based dynamic sizing |
| W&B Tracking | **Good** | Online mode with offline fallback |
| Evaluation Suite | **Excellent** | 12+ analysis tools |
| Documentation | **Good** | Table of contents, section numbering, docstrings |

---

## 7. Comparison vs vK.10.5 and Earlier Runs

| Metric | vK.10.5 | vK.10.6 | v6.5 (best) |
|---|---|---|---|
| Architecture | Custom (scratch) | Custom (scratch) | **SMP ResNet34 (pretrained)** |
| Parameters | 31.6M | 31.6M | 24.4M |
| Epochs Run | ~10 (ES) | **100** | 25 (ES) |
| Test Accuracy | 0.4791 | **0.8357** | 0.8246 |
| Image AUC | 0.6201 | **0.9057** | 0.8703 |
| Tam F1 | 0.0006 | **0.2213** | **0.4101** |
| Tam IoU | 0.0003 | **0.1554** | **0.3563** |
| Pixel AUC | — | **0.7083** | — |
| Eval Features | 0 | **12** | 10 |

**vK.10.6 vs vK.10.5:** Massive improvement. Accuracy 0.48→0.84, AUC 0.62→0.91, Tam-F1 0.0006→0.22 (370× improvement). The 100-epoch training with patience=30 finally let the from-scratch model learn. But still well below v6.5's 0.41 (pretrained).

**vK.10.6 vs v6.5:** Classification is now **better** (Acc: 0.84 vs 0.82, AUC: 0.91 vs 0.87) thanks to the dedicated classifier head. But segmentation is still **~50% worse** (Tam-F1: 0.22 vs 0.41) because training from scratch can't match pretrained features.

---

## 8. Bugs and Red Flags

### Critical

1. **No pretrained encoder** — 31.6M parameters trained from scratch on 8,829 images. This is the root cause of the segmentation gap vs v6.5. With pretrained ResNet34, this notebook's comprehensive evaluation suite would produce meaningful results.

2. **CONFIG documentation mismatch** — The markdown summary table says `max_epochs=50`, `patience=10`, `scheduler_T_max=10`, but actual CONFIG has `max_epochs=100`, `patience=30`, `scheduler_T_max=50`. The documentation was not updated when training parameters were changed.

3. **VRAM threshold table mismatch** — Markdown says `>=15GB: batch=32` but code uses `>=28GB: 32, >=20GB: 24, >=15GB: 16`.

4. **CosineAnnealingLR double-cycle** — `T_max=50` over 100 epochs creates an unintentional double cosine cycle. The LR drops to near-zero at epoch 50, then rises back to 1e-4 by epoch 100. This accidentally helped (best epoch was 99, right at the second LR peak) but was likely unintentional and should be documented.

5. **Model had not converged** — Still improving at epoch 99. Tampered Dice was on an upward trajectory. More epochs or multiple cosine restart cycles (CosineAnnealingWarmRestarts) could push performance further.

### Moderate

6. **Splicing F1=0.1244 is very poor** — The model detects copy-move (F1=0.41) but fails on splicing (F1=0.12), opposite to v6.5. Without pretrained features, the model can detect geometry-based copy-move artifacts but not the subtler splicing boundaries.

7. **Blur k=5 robustness collapse** — F1 drops 66% under moderate blur, suggesting the model relies on high-frequency edge features that blur destroys.

8. **stderr suppression** — `os.dup2(devnull, 2)` silently hides all C-level stderr warnings from CUDA/PyTorch.

9. **Section numbering inconsistency** — Mixed numbering (sections 2.1, 2.2 coexist with 4.4, 4.5 from an older scheme).

---

## 9. Roast

vK.10.6 is the "what if we just let it cook longer?" experiment — and it actually worked, sort of. Taking vK.10.5's infrastructure (which produced Tam-F1=0.0006 after 10 epochs of despair) and simply running it for 100 epochs with patience=30 produced Tam-F1=0.22 — a 370× improvement from literally the same architecture. The early stopping in vK.10.5 was correct in its diagnosis (the model isn't learning) but premature in its conclusion (it just needed more time).

The accidental double cosine schedule is chef's kiss — someone set `T_max=50` for a 100-epoch run, creating a learning rate that tanks to zero at epoch 50 and then climbs back up. The model's best epoch (99) sits right at the second LR peak. This is the training equivalent of accidentally leaving the oven on overnight and finding perfectly slow-cooked brisket in the morning.

Classification is now genuinely good: AUC=0.91 beats v6.5's 0.87. The dedicated classifier head (which v6.5 lacks) pays off for image-level detection. But segmentation at F1=0.22 is still half of v6.5's 0.41 — proving that even 100 epochs of training from scratch can't match what a pretrained ImageNet encoder provides in 25 epochs.

The evaluation suite is the real star. 12 new analysis features make this the most thoroughly evaluated run in the vK.x series — and the results are actually interesting. The copy-move > splicing reversal (opposite of v6.5) reveals that the from-scratch encoder learns different features than ResNet34. The mask-size stratification shows tiny forgeries are hopeless (F1=0.10). The robustness results reveal a fatal weakness to blur.

**Bottom line:** vK.10.6 proves three things: (1) the from-scratch model CAN learn if given enough time, (2) it still can't match pretrained, and (3) the comprehensive evaluation suite built for this notebook is exactly what the project needed from the start. Now combine this evaluation suite with v6.5's pretrained encoder and you have a real submission.
