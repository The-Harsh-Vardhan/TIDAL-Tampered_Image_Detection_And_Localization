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

The notebook is now **threshold-aware during training** as well: checkpoint selection and early stopping use the best validation Pixel-F1 from the sweep rather than a fixed `0.5` threshold.

---

## Authentic Image Handling

For Pixel-F1 / IoU:

| Ground Truth | Prediction | Result |
|---|---|---|
| Empty | Empty | perfect (`1.0`) |
| Empty | Non-empty | failure (`0.0`) |

For Precision / Recall:
- per-image Precision/Recall on empty-empty authentic samples are undefined
- mixed-set Precision/Recall are therefore reported as **global pixel metrics**
- tampered-only Precision/Recall are reported separately

This avoids misleading zeros from correct all-zero authentic predictions.

---

## Reporting Views

### 1. Mixed-Set

Report across all test images:
- Pixel-F1
- Pixel-IoU
- global pixel Precision
- global pixel Recall
- Image Accuracy
- Image AUC-ROC

### 2. Tampered-Only

Report on tampered images only:
- Pixel-F1
- Pixel-IoU
- Pixel Precision
- Pixel Recall

### 3. Forgery-Type Breakdown

Report Pixel-F1 separately for:
- splicing
- copy-move

---

## Evaluation Function Interface

```python
def evaluate(model, test_loader, test_pairs, device, threshold):
    """
    Returns keys including:
        pixel_f1_mean, pixel_f1_std,
        pixel_iou_mean, pixel_iou_std,
        precision_mean, recall_mean,
        tampered_f1_mean, tampered_f1_std,
        tampered_iou_mean, tampered_iou_std,
        tampered_precision_mean, tampered_recall_mean,
        image_accuracy, image_auc_roc,
        threshold_used, num_test_images, num_tampered_images
    """
```

---

## Metric Implementation Notes

- Pixel-F1 and IoU are computed per image with explicit empty-mask handling.
- Mixed-set Precision/Recall are computed from accumulated TP/FP/FN counts over the full split.
- Tampered-only Precision/Recall are computed from accumulated TP/FP/FN counts over the tampered subset.
- Image-level scores use the top-k mean of the probability map.

Reference sketch:

```python
tp, fp, fn = compute_confusion_counts(pred_mask, gt_mask)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

tamper_score = compute_image_tamper_score(prob_map)
is_tampered = tamper_score >= threshold
```

---

## Results Reporting Format

```text
TEST SET RESULTS (threshold=X.XXXX)

Mixed-set (N images):
  Pixel-F1:  X.XXXX ± X.XXXX
  Pixel-IoU: X.XXXX ± X.XXXX
  Precision: X.XXXX   # global pixel precision
  Recall:    X.XXXX   # global pixel recall

Tampered-only (M images):
  Pixel-F1:  X.XXXX ± X.XXXX
  Pixel-IoU: X.XXXX ± X.XXXX
  Precision: X.XXXX
  Recall:    X.XXXX

Image-level:
  Accuracy: X.XXXX
  AUC-ROC:  X.XXXX
```

---

## Explainability in Evaluation

Evaluation is complemented by:
- prediction-mask overlays
- Grad-CAM heatmaps
- failure-case analysis

These are not replacement metrics, but they help validate whether the model is focusing on plausible tampered regions.
