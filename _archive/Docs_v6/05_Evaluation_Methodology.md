# Evaluation Methodology

---

## Metrics

### Pixel-Level (Localization)

| Metric | Role |
|---|---|
| Pixel-F1 | Primary localization metric |
| Pixel-IoU | Secondary localization metric |
| Pixel Precision | False-positive sensitivity |
| Pixel Recall | False-negative sensitivity |

### Image-Level (Detection)

| Metric | Role | Input |
|---|---|---|
| Image Accuracy | Binary detection correctness | `topk_mean(prob_map) >= threshold` |
| Image AUC-ROC | Threshold-independent ranking | `topk_mean(prob_map)` |

The image-level score is the **mean of the top-k pixel probabilities**, not `max(prob_map)`.

---

## Threshold Protocol

1. Sweep 50 thresholds from `0.1` to `0.9` on the **validation set only**.
2. Select the threshold that maximizes mean validation Pixel-F1.
3. Use that threshold for:
   - pixel-level binarization
   - image-level detection
   - robustness evaluation
4. Freeze the threshold before any test-set reporting.

---

## Authentic Image Handling (True-Negative Consistency)

For all metrics, when both prediction and ground truth are empty (true negative):

| Metric | True-Negative Value | Rationale |
|---|---|---|
| Pixel-F1 | 1.0 | Correct: nothing to detect, nothing detected |
| Pixel-IoU | 1.0 | Correct: empty intersection over empty union |
| Precision | 1.0 | Correct: no false positives |
| Recall | 1.0 | Correct: no false negatives |

This eliminates the inconsistency where F1/IoU returned 1.0 but precision/recall returned 0.0 for correct authentic predictions.

---

## Reporting Views

### 1. Mixed-Set

Report across all test images (authentic + tampered):
- Pixel-F1 (mean ± std)
- Pixel-IoU (mean ± std)
- Pixel Precision (mean)
- Pixel Recall (mean)
- Image Accuracy
- Image AUC-ROC

### 2. Tampered-Only

Report on tampered images only:
- Pixel-F1 (mean ± std)
- Pixel-IoU (mean ± std)
- Pixel Precision (mean)
- Pixel Recall (mean)

### 3. Forgery-Type Breakdown

Report Pixel-F1 separately for:
- Splicing
- Copy-move

---

## Results Reporting Format

```text
TEST SET RESULTS (threshold=X.XXXX)

Mixed-set (N images):
  Pixel-F1:    X.XXXX ± X.XXXX
  Pixel-IoU:   X.XXXX ± X.XXXX
  Precision:   X.XXXX
  Recall:      X.XXXX

Tampered-only (M images):
  Pixel-F1:    X.XXXX ± X.XXXX
  Pixel-IoU:   X.XXXX ± X.XXXX
  Precision:   X.XXXX
  Recall:      X.XXXX

Image-level:
  Accuracy:    X.XXXX
  AUC-ROC:     X.XXXX
```

---

## Explainability in Evaluation

Evaluation is complemented by:
- Prediction-mask overlays (TP/FP/FN color-coded)
- Grad-CAM heatmaps
- Failure-case analysis

These are diagnostic tools, not replacement metrics. They help validate whether the model focuses on plausible tampered regions.
