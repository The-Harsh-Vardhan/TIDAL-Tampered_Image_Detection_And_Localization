# Technical Audit: v8 (Run 01)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `v8-tampered-image-detection-localization-run-01.ipynb` (~554KB)
**Platform:** Kaggle, 2× Tesla T4 GPUs, ~61 minutes total runtime

---

## 1. Architecture

| Attribute | Value |
|---|---|
| Model | `smp.Unet` (same as v6.5) |
| Encoder | **ResNet34, pretrained on ImageNet** |
| Parameters | **24,436,369** (same as v6.5) |
| Input | 3-channel RGB, **384×384** |
| Output | 1-channel logits, 384×384 |
| Classifier Head | None (same as v6.5 — image-level from max pixel probability) |
| DataParallel | Yes (2 GPUs) |

**No architectural changes from v6.5.** Same SMP U-Net, same encoder, same parameter count.

---

## 2. Training Pipeline

| Parameter | v6.5 | v8 | Change |
|---|---|---|---|
| Optimizer | AdamW (enc:1e-4, dec:1e-3) | Same | — |
| Scheduler | **None** | **ReduceLROnPlateau**(patience=3, factor=0.5, min_lr=1e-6) | **Added** |
| BCE pos_weight | None | **30.01** | **Added** |
| Dice Computation | Batch-level | **Per-sample** | **Fixed** |
| Batch Size | 4 (eff. 16) | **64** (eff. **256**) | **16× increase** |
| Accumulation Steps | 4 | 4 | — |
| Max Epochs | 50 | 50 | — |
| Epochs Run | 25 (ES) | **27** (ES) | — |
| Best Epoch | 15 | **17** | — |
| AMP | Yes | Yes | — |
| Gradient Clipping | Yes (1.0) | Yes (1.0) | — |
| Early Stopping | Yes (patience=10) | Yes (patience=10) | — |
| Encoder Warmup | Not available | **Configurable** (set to 0 — disabled) | **Infrastructure added** |

### Loss Function

```
L = BCEWithLogitsLoss(logits, targets, pos_weight=30.01) + mean(per_sample_dice_loss_i)
per_sample_dice_loss_i = 1 - (2×intersection_i + 1.0) / (sum_pred_i + sum_gt_i + 1.0)
```

**Key change:** pos_weight=30.01 was computed from raw pixel ratios including authentic images: `bg_pixels(1.33B) / fg_pixels(44.3M) = 30.01`. This massively inflates the weight because authentic images (59.4%) contribute only background pixels. Computing on tampered images only would give a much lower ratio.

### Augmentations (expanded from v6.5)

| v6.5 | v8 (additions in bold) |
|---|---|
| Resize(384) | Resize(384) |
| HFlip(0.5) | HFlip(0.5) |
| VFlip(0.5) | VFlip(0.5) |
| RandomRotate90(0.5) | RandomRotate90(0.5) |
| — | **ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)** |
| — | **ImageCompression(quality_lower=50, quality_upper=95, p=0.3)** |
| — | **GaussNoise(var_limit=(10,50), p=0.3)** |
| — | **GaussianBlur(blur_limit=(3,5), p=0.2)** |
| Normalize + ToTensorV2 | Normalize + ToTensorV2 |

---

## 3. Data Pipeline

| Attribute | Value |
|---|---|
| Dataset | CASIA Splicing Detection + Localization (same as v6.5) |
| Split | 70/15/15 stratified (identical counts to v6.5) |
| Train / Val / Test | 8,829 / 1,892 / 1,893 |
| Image Size | 384×384 |
| Data Leakage Check | **Explicit path-level verification — PASSED** |
| Split Manifest | Saved to JSON |

Identical data pipeline to v6.5 except for the expanded augmentation suite.

---

## 4. Evaluation — Exact Numbers

### Comparison vs v6.5

| Metric | v6.5 | v8 | Delta | % Change |
|---|---|---|---|---|
| Image Accuracy | 0.8246 | **0.7190** | -0.1056 | -12.8% |
| Image AUC-ROC | 0.8703 | **0.8170** | -0.0533 | -6.1% |
| Mixed Pixel-F1 | 0.7208 | **0.5181** | -0.2027 | -28.1% |
| **Tampered Pixel-F1** | **0.4101** | **0.2949** | **-0.1152** | **-28.1%** |
| Tampered Pixel-IoU | 0.3563 | **0.2321** | -0.1242 | -34.9% |
| Splicing F1 | 0.5901 | **0.5758** | -0.0143 | -2.4% |
| **Copy-move F1** | **0.3105** | **0.1394** | **-0.1711** | **-55.1%** |
| Best Val F1 | 0.7289 | **0.3585** | -0.3704 | -50.8% |
| Optimal Threshold | 0.1327 | **0.7500** | +0.6173 | — |

**v8 is a clear regression from v6.5 across every single metric.**

### Mask-Size Stratification (NEW in v8)

| Bucket | Count | F1 (mean ± std) |
|---|---|---|
| Tiny (<2%) | 295 | 0.1432 ± 0.2590 |
| Small (2–5%) | 180 | 0.2429 ± 0.3139 |
| Medium (5–15%) | 152 | 0.4057 ± 0.3463 |
| Large (>15%) | 142 | 0.5573 ± 0.3446 |

Performance scales linearly with mask size. Tiny forgeries (38% of tampered images) are nearly undetectable at F1=0.14.

### Shortcut Learning Checks (NEW in v8)

| Test | Result | Verdict |
|---|---|---|
| Mask randomization | F1 = 0.0772 (expected ~0.0–0.1) | **PASS** |
| Boundary erosion Δ | -0.0040 | **PASS** |
| Boundary dilation Δ | -0.0169 | **PASS** |

The model uses image content for decisions, not spurious mask correlations. Boundary sensitivity is minimal — predictions are not just boundary artifacts.

### Robustness Results (threshold = 0.7500)

| Condition | Mixed F1 | Delta from Clean |
|---|---|---|
| clean | 0.5181 | — |
| jpeg_qf70 | 0.5338 | **+0.0157** |
| jpeg_qf50 | 0.5092 | -0.0090 |
| gaussian_noise_light | 0.3878 | -0.1303 |
| gaussian_noise_heavy | 0.3894 | -0.1287 |
| gaussian_blur | 0.4755 | -0.0426 |
| resize_0.75x | 0.4650 | -0.0531 |
| resize_0.5x | 0.4731 | -0.0450 |

**JPEG robustness dramatically improved** (gap: 0.009 vs v6.5's 0.13) — the ImageCompression training augmentation worked. However, **Gaussian noise robustness worsened** (gap: 0.13 despite GaussNoise augmentation).

### Training Progression

| Epoch | Train Loss | Val Loss | Val F1 |
|---|---|---|---|
| 1 | 2.2359 | 2.1599 | 0.0826 |
| 5 | 1.9101 | 2.0898 | 0.1361 |
| 10 | 1.7657 | 2.0678 | 0.3335 |
| **17** | **1.5790** | **2.1219** | **0.3585** |
| 20 | 1.5163 | 2.2036 | 0.2830 |
| 27 | 1.4455 | 2.2630 | 0.3198 |

**Severe overfitting:** Train loss monotonically decreases while val loss increases from epoch ~10 onward. Val F1 is highly volatile (0.08→0.33→0.13→0.36 swings between epochs), indicating unstable convergence.

**LR reductions occurred at:** Epoch 14 (enc: 1e-4→5e-5), epoch 21 (→2.5e-5), epoch 25 (→1.25e-5).

---

## 5. Evaluation Features

| Feature | v6.5 | v8 | Notes |
|---|---|---|---|
| Tampered-only metrics | Yes | Yes | |
| Threshold optimization | Yes (50-pt) | Yes (15-pt, 0.05–0.80) | |
| Forgery-type breakdown | Yes | Yes | |
| Mask-size stratification | Partial | **Yes** | Tiny/small/medium/large buckets |
| Shortcut detection | No | **Yes** | Mask randomization + boundary sensitivity |
| Robustness testing | Yes | Yes | Same 8 conditions |
| Grad-CAM | Yes | Yes | encoder.layer4 |
| Failure case analysis | Yes | Yes | Top-10 worst with metadata |
| Diagnostic overlays | Yes | Yes | TP/FP/FN color coding |
| W&B integration | Yes | Yes | |
| Confusion matrix | No | No | |
| PR curves (plotted) | No | No | |

---

## 6. Engineering Quality

| Criterion | Rating | Notes |
|---|---|---|
| CONFIG System | **Excellent** | Central dict with all hyperparams + feature flags |
| Reproducibility | **Excellent** | Full seeding + cuDNN deterministic (v6.5's `benchmark` bug fixed) |
| Checkpoint System | **Good** | best + last + periodic (every 10 epochs) |
| LR Scheduler | **Good** | ReduceLROnPlateau with monitoring |
| Gradient Norm Logging | **Good** | New — logs pre-clip gradient norms per accumulation step |
| LR Tracking | **Good** | New — logs encoder/decoder LR per epoch |
| Code Organization | **Good** | 14 numbered sections with markdown headers |
| Alternative Losses | Present | FocalDiceLoss, TverskyDiceLoss defined but unused (dead code) |

---

## 7. Root Cause Analysis — Why v8 Regressed

### Primary Cause: pos_weight=30.01

The BCE pos_weight was computed from **all** training pixels including authentic images:

```
bg_pixels = 1,328,847,646  (includes ALL authentic pixels)
fg_pixels = 44,277,628     (only tampered regions)
pos_weight = 1,328,847,646 / 44,277,628 = 30.01
```

This is methodologically wrong. Since 59.4% of images are authentic (all-background), the ratio is grossly inflated. Computing on tampered images only would give a much lower positive weight (~3–5×). The 30× weighting forces the model to massively over-predict "tampered" pixels, which:
- Distorts the probability distribution (optimal threshold jumps from 0.13 to 0.75)
- Increases false positives on authentic regions
- Destroys the delicate balance the Dice loss tries to maintain

The notebook itself flags this issue: *"pos_weight may be too aggressive. Consider reducing."*

### Secondary Cause: Effective Batch Size 256 Without LR Rescaling

The effective batch increased 16× (16→256) while the base learning rates stayed the same (enc:1e-4, dec:1e-3). While AdamW is more robust to batch size than SGD, a 16× increase still matters. The ReduceLROnPlateau scheduler can only reduce LR (not increase it), so it cannot compensate for an initial LR that is already too low for the batch size.

### Interaction Effect

The pos_weight pushes the model to over-predict → the high threshold (0.75) then aggressively clips predictions → net result is worse segmentation. The model is torn between the BCE telling it "predict more tampered!" and the Dice loss telling it "match the actual mask boundaries." This conflict, combined with the high effective batch averaging out gradients, prevents stable convergence — explaining the violent val F1 oscillations (0.08→0.33→0.13→0.36).

---

## 8. What v8 Got Right (Despite the Regression)

1. **ReduceLROnPlateau** — Fills v6.5's biggest gap (no scheduler). The right idea, just overwhelmed by other issues.
2. **Per-sample Dice** — Fixes v6.5's batch-level Dice bias. Each image contributes equally regardless of mask size.
3. **Expanded augmentations** — ColorJitter, ImageCompression, GaussNoise, GaussianBlur. The JPEG robustness gap improved from 13% to <1%.
4. **Mask-size stratified evaluation** — Reveals the strong size-performance correlation (tiny F1=0.14, large F1=0.56).
5. **Shortcut learning validation** — Both tests pass, confirming model legitimacy.
6. **Encoder warmup infrastructure** — Ready to use even though disabled in this run.
7. **Gradient norm logging** — New diagnostic for monitoring training stability.
8. **cuDNN deterministic mode fixed** — v6.5 had `cudnn.benchmark = True` contradicting deterministic mode; v8 fixes this.

---

## 9. Roast

v8 is a textbook case of fixing the right problems with the wrong magnitudes. Someone looked at v6.5 and correctly identified its gaps: no scheduler (added ReduceLROnPlateau), weak augmentations (added 4 new transforms), batch-level Dice bias (switched to per-sample), no shortcut validation (added two tests). Every single one of these changes is directionally correct. Then they computed pos_weight from the wrong pixel population and got 30.01 instead of ~4.0, 16×'d the effective batch without rescaling LR, and watched the model's tampered F1 crater from 0.41 to 0.29.

The copy-move collapse tells the whole story: F1 dropped from 0.31 to 0.14 — a 55% regression on the majority forgery type. The model learned to scream "tampered!" at everything (thanks to pos_weight=30) and then rely on a 0.75 threshold to filter out the noise. For splicing (which has stronger artifacts), this still kind of works (F1=0.58, barely down from 0.59). For copy-move (whose artifacts are subtler), the over-prediction is catastrophic.

The mask-size stratification results are the silver lining: they clearly show that tiny forgeries (<2% area, F1=0.14) are the frontier problem. The shortcut detection passing is genuine validation. The JPEG robustness gap going from 13% to <1% proves the augmentations work. The infrastructure improvements (scheduler, per-sample Dice, gradient logging, encoder warmup option) are all worth carrying forward.

**The fix is simple:** recompute pos_weight on tampered images only (~4.0), reduce effective batch to 32–64, and v8's codebase would likely beat v6.5. All the right ingredients are here — just assembled with the wrong proportions.
