# Audit: v8 Tampered Image Detection and Localization (Variants)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**Files:**
- `v8-tampered-image-detection-localization.ipynb` (95 KB) — Base template, not executed
- `v8-tampered-image-detection-localization-kaggle.ipynb` (108 KB) — Kaggle template, not executed
- `v8-tampered-image-detection-localization-colab.ipynb` (110 KB) — Colab template, not executed
- `v8-tampered-image-detection-localization-run-01.ipynb` (555 KB) — **Executed Kaggle run**

---

## Notebook Overview

The v8 family extends v6.5 with pos_weight, per-sample Dice loss, ReduceLROnPlateau, and photometric augmentation. Three platform-specific templates (base, Kaggle, Colab) and one executed run. **The result is a regression:** Tampered-F1 dropped from 0.4101 (v6.5) to 0.2949 (v8), making this the cautionary tale of the project.

| Variant | Cells | Executed? | Key Difference |
|---|---|---|---|
| Base template | ~56 | No | Master/base version |
| Kaggle template | 56 (41c/15m) | No | Kaggle paths + secrets |
| Colab template | 58 (43c/15m) | No | Drive mount + Kaggle API download |
| **Run-01** | **56 (41c/15m)** | **Yes (27 epochs)** | **batch=64, eff=256, full results** |

---

## Dataset Pipeline Review

Same dataset as v6.5:

| Property | Value |
|---|---|
| Dataset | CASIA Splicing Detection + Localization |
| Total | 12,614 (5,123 tampered + 7,491 authentic) |
| Split | 70/15/15 (stratified) |
| Image Size | 384×384 |
| Input Channels | 3 (RGB — no ELA) |

**Augmentation (expanded from v6.5):**

| Transform | v6.5 | v8 |
|---|---|---|
| HorizontalFlip | Yes | Yes |
| VerticalFlip | Yes | Yes |
| RandomRotate90 | Yes | Yes |
| **ColorJitter** | No | **Yes** |
| **ImageCompression** | No | **Yes** |
| **GaussNoise(10-50)** | No | **Yes (p=0.3)** |
| **GaussianBlur(3-5)** | No | **Yes (p=0.2)** |
| Normalize | ImageNet | ImageNet |

---

## Model Architecture Review

Same SMP architecture as v6.5:

| Attribute | Value |
|---|---|
| Decoder | `smp.Unet` |
| Encoder | `resnet34` (ImageNet pretrained) |
| Input Channels | 3 (RGB) |
| Output Classes | 1 (binary mask) |
| DataParallel | Yes (multi-GPU) |

---

## Training Pipeline Review

### Changes from v6.5

| Component | v6.5 | v8 | Impact |
|---|---|---|---|
| Batch Size | 4 | **64** | 16× increase |
| Effective Batch | 16 | **256** | 16× increase |
| pos_weight | No | **Yes** | Mask imbalance correction |
| Dice Loss | Batch-level | **Per-sample** | Better gradient signal |
| LR Scheduler | None | **ReduceLROnPlateau** (p=3, f=0.5) | Adaptive LR |
| Augmentation | Spatial only | **Spatial + photometric** | More robust |

### Run-01 Training Progression

| Epoch | Val F1 | Event |
|---|---|---|
| 1 | 0.0826 | Very low start (pos_weight distortion) |
| 7 | 0.2225 | Slow improvement |
| 10 | 0.3335 | — |
| **17** | **0.3585** | **Best model** |
| 27 | — | Early stopping (patience=10) |

**LR Decay:** Encoder LR decayed from 1e-4 to 1.25e-5, decoder from 1e-3 to 1.25e-4.

---

## Evaluation Metrics Review

### Threshold Optimization

| Parameter | v6.5 | v8 |
|---|---|---|
| Optimal Threshold | 0.1327 | **0.7500** |
| Warning | — | "Threshold 0.7500 is above 0.55. pos_weight may be too aggressive" |

The threshold **swung from 0.13 (v6.5) to 0.75 (v8)** — a massive overcorrection. v6.5 under-predicted (threshold too low), v8 over-predicts (threshold too high).

### Test Set Results (threshold=0.75)

| Metric | v6.5 (thr=0.13) | **v8 (thr=0.75)** | Delta |
|---|---|---|---|
| Tampered-only F1 | **0.4101** | 0.2949 | **-0.1152** |
| Tampered-only IoU | **0.3563** | 0.2321 | -0.1242 |
| Image Accuracy | **0.8246** | 0.7190 | -0.1056 |
| Image AUC-ROC | **0.8703** | 0.8170 | -0.0533 |
| Mixed-set F1 | **0.7208** | 0.5181 | -0.2027 |

**Every single metric regressed from v6.5.** This is a comprehensive performance degradation.

### Forgery-Type Breakdown

| Type | v6.5 F1 | v8 F1 | Delta |
|---|---|---|---|
| Splicing | **0.5901** | 0.5758 | -0.0143 |
| Copy-move | **0.3105** | 0.1394 | **-0.1711** |

Splicing held roughly steady, but copy-move detection **collapsed** from 0.31 to 0.14.

### Mask-Size Stratification (v8 only)

| Bucket | Count | F1 |
|---|---|---|
| Tiny (<2%) | 295 | 0.1432 |
| Small (2-5%) | 180 | 0.2429 |
| Medium (5-15%) | 152 | 0.4057 |
| Large (>15%) | 142 | 0.5573 |

Clear size-performance correlation. Tiny forgeries (38% of test tampered images) are nearly undetectable.

### Robustness Results

| Condition | v6.5 Mixed-F1 | v8 Mixed-F1 | v6.5 Delta | v8 Delta |
|---|---|---|---|---|
| Clean | 0.7208 | 0.5181 | — | — |
| JPEG QF=70 | 0.5912 | 0.5338 | -0.1296 | **+0.0157** |
| JPEG QF=50 | 0.5938 | 0.5092 | -0.1269 | -0.0090 |
| Noise (light) | 0.5938 | 0.3878 | -0.1270 | **-0.1303** |
| Noise (heavy) | 0.5938 | 0.3894 | -0.1270 | -0.1287 |
| Blur | 0.5881 | 0.4755 | -0.1326 | -0.0426 |

**JPEG robustness improved** (from -13% to -1%) thanks to ImageCompression augmentation. But **noise robustness unchanged** despite GaussNoise augmentation, and **clean performance dropped 20%**.

### Failure Analysis

- Worst 10: Mean F1=0.0000, Mean GT mask area=0.0214
- 9/10 are copy-move, 1 splicing — same pattern as v6.5

---

## Visualization Assessment

Templates have no visualizations (not executed). Run-01 includes:
- Training loss/F1 curves
- Threshold optimization curve (with warning flag)
- Per-forgery-type breakdown table
- Mask-size stratification analysis
- Robustness comparison table
- Failure case analysis (10 worst predictions)

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Architecture | **Excellent** | Same proven SMP UNet + ResNet34 |
| pos_weight | **Overcorrected** | Threshold pushed to 0.75 — too aggressive |
| LR Scheduler | **Good** | ReduceLROnPlateau added (missing in v6.5) |
| Augmentation | **Good** | Photometric transforms added |
| Effective Batch | **Excessive** | 256 may be too large for 8,830 samples |
| Template Management | **Good** | Clean platform separation (base/Kaggle/Colab) |
| Evaluation | **Excellent** | Forgery breakdown, mask-size stratification, robustness |

---

## Strengths

1. **JPEG robustness improved** — ImageCompression augmentation reduced JPEG sensitivity from -13% to -1%
2. **ReduceLROnPlateau added** — adaptive LR scheduling (missing in v6.5)
3. **Per-sample Dice loss** — better gradient signal than batch-level Dice
4. **Platform templates** — clean separation of base/Kaggle/Colab environments
5. **Mask-size stratification** — new evaluation feature showing size-dependent performance
6. **pos_weight concept** — correct identification of the class imbalance problem

---

## Weaknesses

1. **Comprehensive regression** — every metric worse than v6.5
2. **pos_weight too aggressive** — threshold swung from 0.13 to 0.75
3. **Effective batch size 256** — too large for ~8,830 training samples (27 batches/epoch)
4. **Copy-move F1 collapsed to 0.14** — worse than v6.5's already poor 0.31
5. **No ELA input** — same limitation as v6.5
6. **No classification head** — still derives image-level detection from mask thresholding

---

## Critical Issues

1. **pos_weight overcorrection.** The pos_weight is computed from training mask pixel ratios (foreground/background). Since tampered regions are typically 5-15% of image area, the pos_weight is very large (10-20×), causing the model to over-predict tampered regions. This pushes the optimal threshold to 0.75 — meaning 75% of the model's "tampered" predictions are actually threshold noise. The fix: cap pos_weight at a reasonable maximum (e.g., 5.0) or use Focal Loss instead.

2. **Effective batch size 256 is excessive.** With only ~8,830 training samples and batch=256, there are only 34 gradient updates per epoch. This drastically reduces the number of optimization steps, slowing convergence. v6.5's effective batch of 16 gave ~552 updates/epoch — 16× more frequent weight updates.

3. **Multiple changes at once.** v8 changed pos_weight, batch size, Dice mode, scheduler, and augmentation simultaneously. When the result regressed, it's impossible to isolate which change caused the degradation. This violates the principle of changing one variable at a time.

---

## Suggested Improvements

1. Cap pos_weight at 5.0 or use Focal Loss for segmentation
2. Reduce effective batch size to 16-32 (same as v6.5)
3. Ablate changes individually: test pos_weight alone, batch size alone, etc.
4. Add ELA as 4th input channel
5. Add dedicated classification head
6. Reduce accumulation_steps from 4 to 1 with batch=64 (effective=64 instead of 256)

---

## Roast Section

v8 is the over-engineering cautionary tale. Take a model that works (v6.5, Tam-F1=0.41), add every technique from the textbook — pos_weight, per-sample Dice, LR scheduling, photometric augmentation, larger batch — and watch it get worse at everything. Tam-F1 dropped to 0.29, accuracy dropped to 0.72, AUC dropped to 0.82. The notebook even prints its own warning: "Threshold 0.7500 is above 0.55. pos_weight may be too aggressive." It diagnosed the problem and did nothing about it.

The pos_weight overcorrection tells the whole story. v6.5 had a threshold of 0.13 — too low, meaning the model under-predicted. The correct response was to gently nudge the predictions upward. Instead, v8 applied a pos_weight of 10-20×, launching the predictions into the stratosphere and requiring a threshold of 0.75 to rein them back in. It's like fixing a leaky faucet with a fire hose.

The effective batch size of 256 is another silent killer. With 8,830 training samples and batch=256, the model sees its entire training set in 34 steps. Each epoch is essentially "glance at the data 34 times and update weights." v6.5 did 552 updates per epoch — 16× more granular optimization. The larger batch size was presumably intended for training stability, but it traded convergence speed for nothing.

The three platform templates (base, Kaggle, Colab) are the best-engineered part of this project. Clean separation, platform-specific auth/paths, shared architecture. It's a shame the shared architecture performs worse than its predecessor.

Copy-move F1 of 0.14 deserves a moment of silence. v6.5 already struggled with copy-move (0.31), but v8 made it dramatically worse. The photometric augmentation (color jitter, JPEG compression, noise, blur) may actually hurt copy-move detection by destroying the subtle pixel-level patterns that differentiate copied regions from originals.

**Bottom line:** v8 demonstrates that more techniques ≠ better results. The JPEG robustness improvement is genuine (augmentation works), but the pos_weight overcorrection and oversized batch destroyed everything else. The lesson: change one thing at a time, validate each change, and don't ship a model that warns you about its own threshold.
