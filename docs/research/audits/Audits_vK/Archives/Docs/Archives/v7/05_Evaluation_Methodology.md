# Evaluation Methodology

---

## Metrics

### Pixel-Level Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **Pixel-F1** (primary) | 2·TP / (2·TP + FP + FN) | Harmonic mean of precision and recall; primary ranking metric |
| **Pixel-IoU** | TP / (TP + FP + FN) | Intersection over union; complementary overlap metric |
| **Pixel-Precision** | TP / (TP + FP) | Fraction of predicted-tampered pixels that are actually tampered |
| **Pixel-Recall** | TP / (TP + FN) | Fraction of actually-tampered pixels that are correctly detected |

All pixel metrics are computed per-image, then averaged across the evaluation split.

### Image-Level Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **Image Accuracy** | Correct / Total | Binary classification: is the image tampered? |
| **Image AUC-ROC** | Area under ROC curve | Threshold-independent discrimination quality |

Image-level detection uses the **top-k mean** tamper score (top 1% of pixel probabilities). An image is classified as tampered if this score exceeds a decision threshold.

---

## Threshold Selection Protocol

The binarization threshold that converts probability maps to binary masks is **not fixed at 0.5**. It is selected via a sweep on the validation set:

```
Sweep range:  0.10 to 0.90
Step size:    0.02
Total steps:  ~40 thresholds
Metric:       Mean Pixel-F1 across validation set
Selection:    Threshold with highest mean Pixel-F1
```

**Why sweep instead of fixed 0.5?** The optimal threshold depends on the model's calibration and the class imbalance in the data. A model trained with BCE+Dice on CASIA may produce probability maps where the optimal F1 occurs at threshold 0.35 or 0.65, not 0.5. The sweep finds this empirically.

**Threshold usage:**
1. Selected on the validation set after training completes
2. Applied to the test set for final evaluation (no per-test tuning)
3. Reused for robustness testing (no per-degradation tuning)

This ensures the threshold is never optimized on the test data.

---

## True-Negative Handling

For authentic images (all-zero ground truth), the model should predict an all-zero mask. When both prediction and ground truth are all-zero:

- **Pixel-F1** = 1.0 (no FP, no FN — perfect prediction)
- **Pixel-Precision** = 1.0 (no false positives)
- **Pixel-Recall** = 1.0 (no false negatives)
- **Pixel-IoU** = 1.0

**Why 1.0 instead of 0.0?** Mathematically, 0/0 is undefined. But semantically, predicting "no tampering" when there is no tampering is a correct prediction. Reporting 0.0 would penalize the model for being correct. The convention of returning 1.0 for true negatives is standard in segmentation evaluation.

---

## Reporting Views

Metrics are reported from three perspectives:

### 1. Mixed Set (All Images)

Includes both authentic and tampered images. This reflects the model's overall performance in a realistic scenario where both clean and tampered images are present.

### 2. Tampered-Only Set

Includes only tampered images. This isolates the model's localization quality — how well it segments tampered regions when we know an image is tampered.

### 3. Forgery-Type Breakdown

Metrics split by forgery type (splicing `_S_`, copy-move `_C_`). This identifies whether the model is systematically weaker on one type. Research suggests splicing and copy-move leave different forensic traces (compression artifacts vs. repeated patterns).

---

## Evaluation Pipeline

After training completes:

1. **Reload best checkpoint** (`best_model.pt`)
2. **Threshold sweep** on validation set → select best threshold
3. **Test evaluation** using selected threshold:
   - Pixel-F1, Pixel-IoU, Precision, Recall (mixed + tampered-only)
   - Image-level accuracy and AUC-ROC
   - Forgery-type breakdown
4. **Save results** to `results_summary.json`

```json
{
    "config": { ... },
    "threshold": 0.42,
    "test_mixed": {
        "pixel_f1": 0.xxx,
        "pixel_iou": 0.xxx,
        "pixel_precision": 0.xxx,
        "pixel_recall": 0.xxx
    },
    "test_tampered_only": {
        "pixel_f1": 0.xxx,
        ...
    },
    "image_accuracy": 0.xxx,
    "image_auc_roc": 0.xxx,
    "forgery_breakdown": {
        "splicing": { ... },
        "copy_move": { ... }
    }
}
```

---

## Global vs. Per-Image Metrics

**Per-image averaging** is used: metrics are computed for each image individually, then averaged. This gives equal weight to every image regardless of its mask size.

Alternative: **global pixel accumulation** sums TP/FP/FN across all images before computing F1. This would weight larger masks more heavily. Per-image averaging is preferred because it treats each tampered image as equally important.

---

## Interview: "Why Pixel-F1 as the primary metric instead of IoU?"

Both F1 and IoU measure spatial overlap. They rank models identically (F1 is a monotonic transformation of IoU: F1 = 2·IoU / (1+IoU)). F1 is chosen as primary because:
1. It is more commonly reported in the tamper detection literature (Tier A papers)
2. Its values are more interpretable (0.7 F1 means precision and recall are balanced around 0.7)
3. The assignment evaluation criteria reference F1-based metrics

IoU is reported as a secondary metric for completeness.

## Interview: "How do you prevent data leakage in evaluation?"

Three mechanisms:
1. **Split isolation:** Train/val/test splits are created once and persisted to `split_manifest.json`. Assertions verify zero path overlap.
2. **Threshold isolation:** The threshold is selected on the validation set, never the test set.
3. **No per-degradation tuning:** Robustness tests reuse the validation-selected threshold — no threshold adaptation per degradation condition.
