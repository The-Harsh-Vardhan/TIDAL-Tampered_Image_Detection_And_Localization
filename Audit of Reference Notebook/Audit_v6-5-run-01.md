# Audit: v6.5 Tampered Image Detection and Localization (Run 01)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `v6-5-tampered-image-detection-localization run-01.ipynb` (494 KB)

---

## Notebook Overview

The fully executed first run of v6.5 — the **best-performing notebook in the entire pre-vK.11.x lineage**. Uses SMP UNet with pretrained ResNet34 encoder, achieving Tampered-F1=0.4101 and Image-level AUC=0.8703. This notebook established the performance ceiling that the vK.10.x from-scratch series struggled to match.

| Attribute | Value |
|---|---|
| Cell Count | 56 (42 code, 14 markdown) |
| Model | SMP UNet + ResNet34 (ImageNet pretrained) |
| Dataset | CASIA Splicing Detection + Localization (12,614 images) |
| Task | Pixel-level segmentation + image-level detection |
| Image Size | 384×384 |
| Training | 25 epochs (early stopped at epoch 25, best @ epoch 15) |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CASIA Splicing Detection + Localization |
| Tampered | 5,123 |
| Authentic | 7,491 |
| Total | 12,614 |
| Split | 70/15/15 (stratified) |
| Image Size | 384×384 |
| Input Channels | 3 (RGB — no ELA) |

**Augmentation (spatial only):**

| Transform | Parameters |
|---|---|
| Resize | 384×384 |
| HorizontalFlip | p=0.5 |
| VerticalFlip | p=0.5 |
| RandomRotate90 | p=0.5 |
| Normalize | ImageNet mean/std |

No photometric augmentation (brightness, contrast, noise, JPEG compression). This limits robustness to real-world degradations.

---

## Model Architecture Review

| Attribute | Value |
|---|---|
| Decoder | `smp.Unet` |
| Encoder | `resnet34` (ImageNet pretrained) |
| Input Channels | 3 (RGB) |
| Output Classes | 1 (binary mask) |
| DataParallel | Optional (multi-GPU) |

Segmentation-only architecture — no dedicated classification head. Image-level detection is derived from thresholding the predicted mask (if any pixel exceeds threshold, the image is classified as tampered).

---

## Training Pipeline Review

| Component | Configuration |
|---|---|
| Optimizer | AdamW (encoder_lr=1e-4, decoder_lr=1e-3, wd=1e-4) |
| Loss | BCEDiceLoss (batch-level Dice, no pos_weight) |
| AMP | Enabled |
| Max Epochs | 50 |
| Patience | 10 |
| Effective Batch Size | 16 (batch=4 × accumulation=4) |
| Max Grad Norm | 1.0 |
| LR Scheduler | **None** |
| W&B | Online (run ID: 466u2z1a) |

### Training Progression

| Epoch | Val F1 | Event |
|---|---|---|
| 1 | 0.4663 | Strong start (pretrained features) |
| 9 | 0.6911 | Rapid improvement |
| 11 | 0.7140 | — |
| **15** | **0.7289** | **Best model** |
| 25 | — | Early stopping (patience=10) |

**Note:** Val F1 here is **mixed-set** (includes authentic), not tampered-only. The high F1 is inflated by true-negative contribution from authentic images.

---

## Evaluation Metrics Review

### Threshold Optimization

| Parameter | Value |
|---|---|
| Sweep range | 50 thresholds, 0.1 to 0.9 |
| **Optimal threshold** | **0.1327** |
| Best val F1 at threshold | 0.7344 |

The extremely low threshold (0.13) reveals that the model's sigmoid outputs are skewed toward zero — even tampered pixels have low probabilities. This is the symptom of missing `pos_weight` in the BCE loss.

### Test Set Results (threshold=0.1327)

| Metric | Mixed-Set (1,893) | Tampered-Only (769) |
|---|---|---|
| Pixel-F1 | 0.7208 ± 0.42 | **0.4101 ± 0.41** |
| Pixel-IoU | 0.6989 ± 0.42 | **0.3563 ± 0.38** |
| Precision | 0.7455 | — |
| Recall | 0.7634 | — |

| Metric | Value |
|---|---|
| Image-Level Accuracy | **0.8246** |
| Image-Level AUC-ROC | **0.8703** |

### Forgery-Type Breakdown

| Type | Count | F1 |
|---|---|---|
| Splicing | 274 | **0.5901** ± 0.39 |
| Copy-move | 495 | **0.3105** ± 0.40 |

Splicing is detected much better than copy-move — pretrained features excel at detecting semantic inconsistencies (splicing) but struggle with geometric duplication (copy-move).

### Robustness Results

| Condition | Mixed-F1 | Delta |
|---|---|---|
| Clean | 0.7208 | — |
| JPEG QF=70 | 0.5912 | **-0.1296** |
| JPEG QF=50 | 0.5938 | -0.1269 |
| Gaussian noise (light) | 0.5938 | -0.1270 |
| Gaussian noise (heavy) | 0.5938 | -0.1270 |
| Gaussian blur | 0.5881 | **-0.1326** |
| Resize 0.75× | 0.6631 | -0.0576 |
| Resize 0.5× | 0.6134 | -0.1073 |

~13% F1 drop across JPEG, noise, and blur degradations — moderate fragility.

### Failure Analysis

- Bottom 10: Mean F1=0.0000, Mean GT mask area=0.0961
- 6/10 failures on tiny masks (<2% area)
- 8/10 failures on copy-move forgeries

---

## Visualization Assessment

The notebook includes:
- Training loss/F1 curves
- Threshold optimization curve
- ROC curve with AUC annotation
- Prediction panels (original, GT mask, predicted mask, overlay)
- Failure case analysis (10 worst predictions)
- Robustness comparison table

**Comprehensive visualization suite** — well-documented qualitative results.

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Architecture | **Excellent** | SMP UNet + pretrained ResNet34 |
| Training | **Good** | Differential LR, AMP, gradient accumulation |
| Evaluation | **Excellent** | Threshold optimization, forgery breakdown, robustness |
| LR Scheduler | **Missing** | Fixed LR, no schedule |
| pos_weight | **Missing** | Causes low optimal threshold (0.13) |
| Classification Head | **Missing** | No dedicated cls head; derives from mask threshold |
| Augmentation | **Limited** | Spatial only, no photometric |

---

## Strengths

1. **Tam-F1=0.4101** — best pixel-level localization in the pre-vK.11.x lineage
2. **AUC=0.8703** — strong image-level detection
3. **Comprehensive evaluation** — threshold optimization, forgery-type breakdown, robustness testing, failure analysis
4. **Pretrained features deliver** — strong performance from epoch 1 (val_F1=0.47)
5. **Splicing F1=0.59** — good localization of splicing forgeries
6. **Clean notebook structure** — 14 markdown sections with clear documentation

---

## Weaknesses

1. **Missing pos_weight** — sigmoid outputs skewed to zero, optimal threshold is 0.13 (should be ~0.5)
2. **No LR scheduler** — training stalled after epoch 15
3. **No classification head** — image-level detection derived from mask thresholding (fragile)
4. **Copy-move F1=0.31** — poor on geometric forgeries
5. **~13% robustness drop** — vulnerable to JPEG compression, noise, and blur
6. **No ELA input** — only RGB, missing forensic signal
7. **Spatial-only augmentation** — no brightness/contrast/noise augmentation

---

## Critical Issues

1. **Missing pos_weight in BCE loss.** The segmentation maps (~95%+ background pixels) cause BCE to heavily penalize false positives while being lenient on false negatives. The model learns to predict near-zero for everything, leading to optimal threshold=0.13. Adding `pos_weight` would shift the sigmoid outputs toward 0.5, improving threshold interpretability.

2. **No LR scheduler.** The model's best epoch (15) was followed by 10 epochs of stagnation before early stopping. ReduceLROnPlateau could have potentially escaped this plateau and pushed F1 higher.

3. **Copy-move detection weakness (F1=0.31).** Pretrained ResNet34 features are good at detecting semantic anomalies (splicing) but poor at detecting geometric redundancy (copy-move). This is a fundamental limitation of the encoder choice — copy-move detection benefits from self-correlation features not present in ImageNet representations.

---

## Suggested Improvements

1. Add `pos_weight` to BCEDiceLoss (tampered pixel ratio / background pixel ratio)
2. Add ReduceLROnPlateau scheduler (patience=3, factor=0.5)
3. Add photometric augmentation (brightness, contrast, JPEG compression, Gaussian noise)
4. Add ELA as 4th input channel
5. Add dedicated classification head (FC layers on bottleneck features)
6. Add per-sample Dice loss (instead of batch-level)
7. Increase max epochs to 100 with patience=15

---

## Roast Section

v6.5 is the notebook that proved pretrained features are the answer. While the vK.10.x series was training 31.6M parameters from scratch and getting Tam-Dice=0.0004, v6.5 loaded ResNet34's ImageNet weights and hit Tam-F1=0.41 in 15 epochs of fine-tuning. The pretrained encoder provides such a strong initialization that the model achieves meaningful segmentation from epoch 1 (val_F1=0.47) — before the from-scratch UNet even produces non-zero predictions.

But v6.5 has its own demons. The optimal threshold of 0.1327 is a confession: the model is so reluctant to predict "tampered" that you have to lower the bar to 13% confidence before it starts flagging anything. This is the direct consequence of training without pos_weight on masks that are 95%+ background. The BCE loss effectively tells the model: "predicting zero is almost always right" — and the model obliges.

The splicing-vs-copy-move disparity (0.59 vs 0.31) is architecturally revealing. ResNet34 excels at detecting "this part of the image looks semantically different" (splicing) but fails at "this part of the image looks identical to another part" (copy-move). Copy-move detection fundamentally requires self-correlation computation — matching feature patches within the same image — which is not what an ImageNet encoder was designed for.

The evaluation suite is the most thorough in the reference collection: threshold optimization, forgery-type analysis, robustness testing under 8 degradation conditions, and failure case autopsy. This evaluation framework directly influenced the comprehensive suites in vK.10.6 and vK.11.x.

**Bottom line:** v6.5 established the performance ceiling (Tam-F1=0.41) and evaluation standard for the project. The vK.11.x series inherits its architecture (SMP UNet + ResNet34) and adds what v6.5 lacked: ELA input, pos_weight, LR scheduling, a dedicated classification head, and encoder freeze warmup.
