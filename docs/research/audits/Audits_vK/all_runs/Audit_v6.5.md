# Technical Audit: v6.5 (Run 01)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `v6-5-tampered-image-detection-localization run-01.ipynb` (~494KB)
**Platform:** Kaggle, 2× Tesla T4 GPUs

---

## 1. Architecture

| Attribute | Value |
|---|---|
| Model | `smp.Unet` (Segmentation Models PyTorch) |
| Encoder | **ResNet34, pretrained on ImageNet** |
| Encoder Channels | [64, 64, 128, 256, 512] (standard ResNet34) |
| Decoder | SMP default U-Net decoder with 5 skip connections |
| Classifier Head | **None** — image-level detection derived from `probs.view(-1).max()` |
| Parameters | **24,436,369** (all trainable, encoder initialized from ImageNet) |
| Input | 3-channel RGB, **384×384** |
| Output | 1-channel logits, 384×384 (sigmoid applied externally) |

### Architecture Notes

- v6.5 represents a **complete architecture departure** from the vK.x series. While vK.1–vK.3 and vK.7.x use a custom `UNetWithClassifier` trained from scratch at 256×256, v6.5 uses the SMP library with a pretrained encoder.
- No dedicated classification head exists. Image-level detection is performed by checking if the maximum predicted pixel probability exceeds the threshold — a crude but functional approach that avoids adding a separate classification loss.
- All 24.4M parameters are trainable (encoder not frozen), which is standard for fine-tuning with differential learning rates.

---

## 2. Training Pipeline

| Parameter | Value |
|---|---|
| Optimizer | **AdamW** with differential LR |
| Encoder LR | 1e-4 |
| Decoder + Seg Head LR | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | **None** |
| Loss | `BCEDiceLoss` = `BCEWithLogitsLoss` + `(1 - Dice)` |
| Dice Computation | **Batch-level** (sum across entire batch, not per-sample) |
| Batch Size | 4 (physical) × 4 (accumulation) = **16 effective** |
| Max Epochs | 50 |
| Epochs Run | **25** (early stopping triggered) |
| Best Epoch | **15** (val F1 = 0.7289) |
| AMP | **Yes** (`autocast('cuda')` + `GradScaler`) |
| Gradient Clipping | **Yes** (`max_norm=1.0`) |
| Early Stopping | **Yes** (patience=10 on val Pixel-F1) |
| DataParallel | **Yes** (2 GPUs) |

### Loss Function Detail

```
L = BCEWithLogitsLoss(logits, targets) + (1 - Dice(sigmoid(logits), targets))
Dice = (2 × intersection + 1.0) / (sum_pred + sum_gt + 1.0)
```

The Dice component is computed on the full batch as a single sum (not per-sample averaged), which biases the loss toward images with larger tampered regions and underweights tiny forgeries.

---

## 3. Data Pipeline

| Attribute | Value |
|---|---|
| Dataset | CASIA Splicing Detection + Localization (Kaggle) |
| Total Valid Pairs | 12,614 |
| Authentic / Tampered | 7,491 (59.4%) / 5,123 (40.6%) |
| Copy-move / Splicing | 3,295 (26.1%) / 1,828 (14.5%) |
| Split | 70/15/15, stratified by forgery_type |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | **384×384** (resized from native ~256×384) |
| Data Leakage Check | **Explicit path-level overlap assertions — PASSED** |
| Split Manifest | Saved to JSON for reproducibility |

### Augmentations

| Stage | Transforms |
|---|---|
| Train | Resize(384) → HFlip(0.5) → VFlip(0.5) → RandomRotate90(0.5) → Normalize(ImageNet) → ToTensorV2 |
| Val/Test | Resize(384) → Normalize(ImageNet) → ToTensorV2 |

**Augmentation weakness:** Geometric transforms only — no color jitter, no noise injection, no JPEG compression, no elastic deformation. This directly limits robustness performance.

### Mask Handling

- Authentic images: zero mask (`np.zeros`)
- Tampered masks: binarized at threshold 0 (`(mask > 0).astype(uint8)`)

---

## 4. Evaluation — Exact Numbers

### Test Set Results (optimal threshold = 0.1327)

**Image-Level:**

| Metric | Value |
|---|---|
| Image Accuracy | **0.8246** |
| Image AUC-ROC | **0.8703** |

**Pixel-Level (all 1,893 images):**

| Metric | Value |
|---|---|
| Mixed Pixel-F1 | **0.7208 ± 0.4158** |
| Mixed Pixel-IoU | **0.6989 ± 0.4194** |
| Precision | **0.7455** |
| Recall | **0.7634** |

**Pixel-Level (769 tampered images only):**

| Metric | Value |
|---|---|
| **Tampered Pixel-F1** | **0.4101 ± 0.4148** |
| **Tampered Pixel-IoU** | **0.3563 ± 0.3798** |

**Per-Forgery-Type Breakdown:**

| Type | Count | F1 (mean ± std) |
|---|---|---|
| Splicing | 274 | **0.5901 ± 0.3850** |
| Copy-move | 495 | **0.3105 ± 0.3968** |

### Threshold Optimization

- 50-point sweep on validation set
- Optimal threshold: **0.1327** (far below default 0.5)
- Val F1 at default 0.5: 0.7289; at optimal 0.1327: **0.7344**

### Robustness Results (threshold = 0.1327)

| Condition | Mixed F1 (mean ± std) | Delta from Clean |
|---|---|---|
| clean | 0.7208 ± 0.4158 | — |
| jpeg_qf70 | 0.5912 ± 0.4913 | -0.1296 |
| jpeg_qf50 | **0.5938 ± 0.4911** | -0.1269 |
| gaussian_noise_light | **0.5938 ± 0.4911** | -0.1270 |
| gaussian_noise_heavy | **0.5938 ± 0.4911** | -0.1270 |
| gaussian_blur | 0.5881 ± 0.4717 | -0.1326 |
| resize_0.75x | 0.6631 ± 0.4461 | -0.0576 |
| resize_0.5x | 0.6134 ± 0.4672 | -0.1073 |

### Training Progression

| Epoch | Train Loss | Val Loss | Val F1 | Val IoU |
|---|---|---|---|---|
| 1 | 1.0902 | 1.0520 | 0.3871 | 0.3413 |
| 5 | 0.8430 | 0.8963 | 0.6108 | 0.5700 |
| 10 | 0.6813 | 0.8124 | 0.7037 | 0.6782 |
| **15** | **0.6338** | **0.8107** | **0.7289** | **0.7088** |
| 20 | 0.5477 | 0.9997 | 0.7109 | 0.6888 |
| 25 | 0.5075 | 1.1975 | 0.6671 | 0.6378 |

**Overfitting:** Clear after epoch 15. Train loss drops 0.63→0.51 while val loss spikes 0.81→1.20. The train/val gap doubles, confirming significant overfitting. Early stopping correctly terminated training.

---

## 5. Evaluation Features

| Feature | Present? | Notes |
|---|---|---|
| Tampered-only metrics | **Yes** | First run with honest per-class metrics |
| Threshold optimization | **Yes** | 50-point val sweep, optimal=0.1327 |
| Forgery-type breakdown | **Yes** | Splicing vs copy-move F1 |
| Robustness testing | **Yes** | 8 degradation conditions |
| Grad-CAM | **Yes** | Hook-based on encoder.layer4, saved as PNG |
| Failure case analysis | **Yes** | Top-10 worst predictions with metadata |
| Diagnostic overlays | **Yes** | TP(green)/FP(red)/FN(blue) color coding |
| Data leakage verification | **Yes** | Path-level overlap assertions |
| W&B integration | **Yes** | Full: per-epoch metrics, plots, robustness, artifact |
| Split manifest | **Yes** | Saved to JSON |
| Confusion matrix | No | |
| PR curves (plotted) | No | AUC-ROC computed numerically but not plotted |
| Mask-size stratification | **Partial** | Only in failure analysis context |
| Shortcut detection | No | |

---

## 6. Engineering Quality

| Criterion | Rating | Notes |
|---|---|---|
| CONFIG System | **Excellent** | Central dict with all hyperparams + feature flags |
| Device Abstraction | **Good** | `setup_device()` handles GPU detection |
| Reproducibility | **Good** | SEED=42, worker seeding, cuDNN deterministic |
| Checkpoint System | **Good** | best_model.pt + last_checkpoint.pt with DataParallel prefix handling |
| Code Organization | **Good** | 56 cells, modular functions (`train_one_epoch`, `validate_model`) |
| Artifact Inventory | **Good** | Final cell verifies all expected outputs exist |
| Gradient Accumulation | **Good** | Handles partial final batch correctly |

---

## 7. Strengths

1. **First and only project run to use a pretrained encoder** (alongside v8) — immediately produced the best segmentation results in the entire project
2. **Most comprehensive evaluation suite** of any run: threshold optimization, forgery-type breakdown, Grad-CAM, robustness testing, failure analysis, diagnostic overlays
3. **Best tampered-only F1 across all runs** at 0.4101 — weak by published benchmarks but 680× better than vK.10.5
4. **Differential learning rates** (encoder: 1e-4, decoder: 1e-3) — correct transfer learning practice
5. **Data leakage verification** — explicit overlap assertions, unlike the vK.x series
6. **Professional engineering** — CONFIG dict, AMP, gradient accumulation, seeding, W&B, checkpoints

---

## 8. Weaknesses and Red Flags

### Critical

1. **No LR scheduler** — neither ReduceLROnPlateau nor CosineAnnealing. The oscillating val loss after epoch 15 is a classic symptom. This is the most significant training pipeline gap.

2. **Suspicious robustness results** — `jpeg_qf50`, `gaussian_noise_light`, and `gaussian_noise_heavy` all produce **identical F1 = 0.5938** down to 4 decimal places. Three very different degradations producing the same F1 is statistically near-impossible. This strongly suggests a bug in the robustness evaluation pipeline.

3. **Copy-move F1 = 0.3105 is near-failure** — The model detects splicing (F1=0.59) much better than copy-move (F1=0.31). Since copy-move is 64% of tampered images (495 of 769), this drags down overall tampered F1 significantly.

4. **Optimal threshold = 0.1327** — Far below the expected 0.3–0.5 range, indicating the model's probability calibration is poor. It produces very low confidence predictions, requiring an extreme threshold to activate.

### Moderate

5. **Batch-level Dice computation** biases loss toward large tampered regions and underweights tiny forgeries — directly contributing to the poor performance on small masks.

6. **Weak augmentation pipeline** — Geometric transforms only (flip, rotate90). No color jitter, noise injection, or JPEG compression augmentation. The 13% robustness drop under JPEG compression confirms this gap.

7. **Mixed-set metric inflation** — F1=0.7208 (mixed) vs 0.4101 (tampered-only). The 1,124 authentic images get perfect F1=1.0 by predicting all-zeros, inflating the headline number. The real segmentation performance is the tampered-only 0.4101.

8. **No dedicated classification head** — Image-level accuracy (0.8246) relies on max pixel probability, which is suboptimal compared to a proper classifier head with pooled features.

9. **All encoder parameters trainable from epoch 1** — No frozen warmup phase. With only 8,829 training images, freezing the encoder for 3–5 epochs could reduce overfitting.

10. **High variance across samples** — All F1 metrics have std > 0.38, meaning per-image performance is extremely inconsistent.

---

## 9. Roast

v6.5 is the best run in this entire project, and it's still mediocre. That's not a contradiction — it's a damning summary of the project trajectory. Someone finally made the right architectural decision (pretrained ResNet34), built a comprehensive evaluation suite (threshold sweep, Grad-CAM, robustness testing, forgery breakdown), implemented proper engineering (AMP, gradient accumulation, DataParallel, early stopping), and the result was... tampered F1 = 0.41. Published CASIA-2 benchmarks with similar architectures hit 0.65+.

The irony is that v6.5 did almost everything right and still underperformed — because the things it got wrong are the training subtleties: no LR scheduler (so the model overshoots and oscillates after epoch 15), no training-time noise/JPEG augmentation (so the 13% robustness drop was predictable), batch-level Dice (so tiny forgeries get ignored), and no encoder warmup (so early gradient updates may corrupt pretrained features).

The robustness evaluation has a bug — three different degradations producing identical F1=0.5938 means either the perturbations aren't being applied or the model collapses to a fixed prediction pattern under any noise. Either way, the robustness numbers are unreliable.

Copy-move detection at F1=0.31 is the biggest practical failure: 64% of tampered images are copy-move, and the model basically can't find them. The threshold landing at 0.1327 tells you the model is chronically under-confident — it knows something is there but can't commit to a strong prediction.

**Bottom line:** v6.5 proved that a pretrained encoder is the right approach. It just needs better training (scheduler, augmentations, per-sample Dice) and the copy-move problem solved. Everything the vK.10.x series should have built on — and instead abandoned.
