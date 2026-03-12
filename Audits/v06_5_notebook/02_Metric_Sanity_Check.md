# Audit 6.5 — Part 2: Metric Sanity Check

## Reported Metrics

### Validation (best epoch 15)
| Metric | Value |
|---|---|
| Pixel-F1 | 0.7289 |
| Pixel-IoU | 0.7088 |

### Test Set (threshold=0.1327)
| Metric | Mixed-set (1893) | Tampered-only (769) |
|---|---|---|
| Pixel-F1 | 0.7208 ± 0.4158 | 0.4101 ± 0.4148 |
| Pixel-IoU | 0.6989 ± 0.4194 | 0.3563 ± 0.3798 |
| Precision | 0.7455 | — |
| Recall | 0.7634 | — |

### Image-level
| Metric | Value |
|---|---|
| Accuracy | 0.8246 |
| AUC-ROC | 0.8703 |

### Forgery Type Breakdown
| Type | Count | F1 |
|---|---|---|
| Splicing | 274 | 0.5901 ± 0.3850 |
| Copy-move | 495 | 0.3105 ± 0.3968 |

---

## Sanity Check: Are These Metrics Realistic?

### Mixed-set Pixel-F1 = 0.7208

**Appears inflated.** Here's why:

The test set contains 1124 authentic images + 769 tampered images. Authentic images have all-zero ground truth masks, and when the model predicts mostly zeros for them, the `compute_pixel_f1` function returns `1.0` (both pred and GT are empty → defined as perfect).

This means ~59% of the test set automatically scores F1=1.0 regardless of model quality. The mixed-set F1 is:

```
Mixed F1 ≈ (1124 × 1.0 + 769 × 0.41) / 1893 ≈ 0.76
```

The actual reported value (0.7208) is close to this estimate, confirming the inflation mechanism. **The tampered-only F1 (0.4101) is the true measure of localization quality.**

### Tampered-only Pixel-F1 = 0.4101

**Realistic and concerning.** This is below what well-tuned segmentation models achieve on CASIA (literature reports 0.45–0.65 for basic U-Net architectures), but not unreasonably so for a first training run without LR scheduling, strong augmentation, or FPN/attention mechanisms.

### Copy-move F1 = 0.3105

**Expected to be low, but this is very low.** Copy-move forgeries involve duplicating regions within the same image, so the textures match perfectly. This is fundamentally harder than splicing (where source/target textures differ). However, F1=0.31 suggests near-random performance on this subcategory.

### Threshold = 0.1327

**This is a red flag indicator — but not a bug.**

A threshold of 0.13 means the model's probability outputs are concentrated in the low range. For most well-calibrated segmentation models, the optimal threshold is near 0.5. A threshold this low indicates:

1. The sigmoid outputs are poorly calibrated — the model rarely predicts probabilities above 0.5 for tampered regions
2. The model is "uncertain" about its tampered predictions even when correct
3. This could be caused by class imbalance (see Part 4) — authentic pixels vastly outnumber tampered pixels within each image

This is not a bug in the evaluation, but it is a **model quality concern**.

### Image-level AUC-ROC = 0.8703

**Realistic and reasonable.** The image-level classification uses max probability as the score, achieving 87% AUC. This suggests the model correctly identifies tampering presence even when localization is imprecise.

### Standard Deviation Analysis

All metrics show very high standard deviation (±0.38 to ±0.42), which means:
- Some images are predicted very well (F1 near 1.0)
- Others are complete failures (F1 = 0.0)
- The model performance is highly inconsistent across samples

---

## Comparison to Suspicious Metric Patterns

| Red Flag | Present? | Details |
|---|---|---|
| IoU > 0.95 early | ❌ | IoU starts at 0.44, peaks at 0.71 |
| F1 = 1.0 | ❌ | Peak F1 = 0.73 on val |
| Loss → 0 quickly | ❌ | Train loss at best epoch = 0.63, not near zero |
| Val metrics >> train metrics | ❌ | Val loss is consistently higher |
| Identical metrics across epochs | ❌ | Metrics vary naturally |
| Zero variance in metrics | ❌ | High variance (±0.41) |

**No metric bugs or evaluation errors detected.**

---

## Metric Reliability Verdict

| Metric | Trustworthy? | Notes |
|---|---|---|
| Mixed-set Pixel-F1 | ⚠️ Misleading | Inflated by authentic images (F1=1.0 by definition) |
| Tampered-only F1 | ✅ Trustworthy | True localization quality measure |
| Splicing F1 | ✅ Trustworthy | Reasonable for the architecture |
| Copy-move F1 | ✅ Trustworthy | Genuinely poor performance |
| Image-level AUC | ✅ Trustworthy | Standard evaluation |
| Threshold | ✅ Valid | Low but not a bug |

**The metrics are computed correctly.** The main concern is presentation: mixed-set F1 should not be reported as the primary metric because it obscures the actual localization performance. Tampered-only F1 (0.41) is the honest metric.
