# Audit 6.5 — Part 4: Class Imbalance Analysis

## The Problem

In tampered image detection, there are **two levels of class imbalance**:

1. **Image-level:** 59.4% authentic vs 40.6% tampered — relatively balanced
2. **Pixel-level (within tampered images):** Tampered regions typically cover 1–20% of the image. The vast majority of pixels are background (non-tampered).

The pixel-level imbalance is the critical one for segmentation.

---

## Pixel-Level Imbalance in This Dataset

From the failure analysis output:
- Worst 10 predictions have mean GT mask area of **0.0961** (9.6% of image)
- 6 of 10 worst cases have mask area < 2%
- This confirms that tampered regions are **sparse** — most pixels in each image are non-tampered

For a 384×384 image (147,456 pixels), a 2% mask covers only ~2,949 pixels. A model that predicts "all background" would achieve:
- Pixel accuracy: **98%**
- Pixel-F1 for tampered class: **0.0**

---

## How the Notebook Handles Imbalance

### 1. BCE + Dice Loss ✅ Partially Addresses

```python
class BCEDiceLoss(nn.Module):
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)  # Standard BCE
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return bce_loss + dice_loss
```

**BCE component:** Treats every pixel equally. With severe class imbalance, the gradient is dominated by background pixels (98% of total), pushing the model toward predicting low probabilities everywhere. **No pos_weight correction is applied.**

**Dice component:** Operates on the overlap between prediction and ground truth, making it naturally robust to class imbalance. The Dice loss gives equal importance to precision and recall of the foreground class, regardless of how many background pixels exist.

**Combined:** The BCE pulls the model toward conservative (low-probability) predictions while Dice pulls toward accurate foreground segmentation. This tension may explain the very low optimal threshold (0.1327) — the BCE component suppresses probability outputs.

### 2. Threshold Tuning ✅ Correctly Implemented

The notebook performs a 50-point threshold sweep on the validation set:
```
Best threshold: 0.1327
Best val F1 at threshold: 0.7344
```

This is critical for class-imbalanced segmentation. The default 0.5 threshold would be inappropriate here. The sweep correctly optimizes for Pixel-F1, which is the right target metric.

**However:** A threshold of 0.13 is unusually low even for imbalanced segmentation. This suggests the sigmoid outputs are concentrated near zero — the model is having difficulty producing confident positive predictions.

### 3. No BCE Weighting ❌ Missing

The notebook uses `nn.BCEWithLogitsLoss()` without `pos_weight`. For this dataset, applying `pos_weight` proportional to the background-to-foreground ratio would significantly help:

```python
# Example: if tampered pixels are ~5% of total
pos_weight = torch.tensor([19.0])  # (1 - 0.05) / 0.05
nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This is a **significant omission** and likely a major contributor to the low threshold and weak tampered-only F1.

### 4. No Focal Loss ⚠️ Optional but Relevant

Focal Loss is specifically designed for extreme class imbalance (from RetinaNet). It down-weights easy examples (background pixels correctly classified with high confidence) and focuses training on hard examples (tampered region boundaries). This is highly relevant here.

---

## Are Metrics Inflated Due to Background Dominance?

### Mixed-set F1 = 0.7208 — **Yes, inflated**

The `compute_pixel_f1` function defines:
```python
if gt.sum() == 0 and pred.sum() == 0:
    return 1.0  # True negative = perfect score
```

This means every authentic image where the model predicts nothing (or nearly nothing) scores F1=1.0. With 1,124 authentic images in the test set (59.4%), this automatically inflates the mixed-set average.

**Estimated inflation:**
- Authentic images contribute ~1,124 × 1.0 = 1,124 to the F1 sum
- Tampered images contribute ~769 × 0.41 = 315.3
- Mixed average: (1124 + 315.3) / 1893 = **0.760** (close to reported 0.721)

The gap between estimated (0.76) and reported (0.72) suggests some authentic images receive false positive predictions, reducing their F1 below 1.0.

### Tampered-only F1 = 0.4101 — **Not inflated, trustworthy**

This metric only includes images with actual tampered regions, so background dominance doesn't inflate the average (though it still affects per-image scores via the intersection calculation).

### IoU — Same inflation pattern

All-zero predictions on authentic images score IoU=1.0, inflating the mixed-set IoU just like F1.

---

## Impact Summary

| Factor | Effect on Metrics | Severity |
|---|---|---|
| No BCE pos_weight | Model outputs low probabilities | **High** |
| Authentic F1=1.0 by definition | Mixed-set metrics inflated | **High** (presentation issue) |
| Dice loss presence | Partially compensates for imbalance | Positive |
| Threshold tuning done | Recovers some performance | Positive |
| No Focal loss | Misses hard-example mining | Medium |

---

## Recommendation

The class imbalance handling is **partially addressed** (Dice loss + threshold tuning) but **incomplete** (no BCE weighting, no Focal loss). The most impactful fix would be adding `pos_weight` to the BCE component, which requires computing the class ratios across the training set. This single change could substantially improve the tampered-only F1 and shift the optimal threshold closer to 0.5.
