# Evaluation Methodology

---

## Metrics

### Pixel-Level (Localization)

| Metric | Role | Formula |
|---|---|---|
| **Pixel-F1** | Primary, checkpoint selection | 2·TP / (2·TP + FP + FN) |
| **Pixel-IoU** | Localization quality | TP / (TP + FP + FN) |
| **Pixel Precision** | False-positive rate | TP / (TP + FP) |
| **Pixel Recall** | False-negative rate | TP / (TP + FN) |

### Image-Level (Detection)

| Metric | Role | Input |
|---|---|---|
| **Image Accuracy** | Binary classification correctness | `max(prob_map) >= threshold` |
| **Image AUC-ROC** | Threshold-independent ranking | `max(prob_map)` as continuous score |

---

## Threshold Protocol

1. **Selection:** Sweep 50 thresholds from 0.1 to 0.9 on the **validation set**. Pick the threshold that maximizes mean Pixel-F1.
2. **Application:** The selected `pixel_threshold` is used for:
   - Pixel-level mask binarization: `pred_mask = (prob_map > threshold).float()`
   - Image-level detection: `is_tampered = max(prob_map) >= threshold`
3. **Frozen for test:** The threshold is fixed before any test-set evaluation. No per-degradation or per-split tuning.

**Design decision:** Using a single threshold for both pixel and image-level decisions keeps the system simple. The image-level detection inherits the localization operating point, which is appropriate for this assignment scope. If future work shows that a separate image-level threshold improves detection accuracy, it can be added as a second sweep.

---

## Authentic Image Handling

When the ground-truth mask is all-zero (authentic image):

| Prediction | Pixel-F1 |
|---|---|
| All-zero (correct) | 1.0 |
| Non-zero (false alarm) | 0.0 |

This convention means authentic images can inflate mixed-set metrics. Both reporting views are required.

---

## Reporting Views

### 1. Mixed-Set (all test images)

Reports metrics across authentic + tampered images. Includes Image Accuracy and AUC-ROC.

### 2. Tampered-Only

Reports pixel-level metrics (F1, IoU, Precision, Recall) on tampered images only. This is the honest localization measure.

### 3. Forgery-Type Breakdown

Report Pixel-F1 separately for splicing and copy-move subsets. This reveals whether the model handles both manipulation types.

---

## Evaluation Function Interface

```python
def evaluate(model, test_loader, device, threshold):
    """
    Args:
        model: trained smp.Unet
        test_loader: DataLoader for test split
        device: torch device
        threshold: float — validation-selected threshold for both pixel and image decisions

    Returns:
        dict with keys:
            pixel_f1_mean, pixel_f1_std,
            pixel_iou_mean, pixel_iou_std,
            precision_mean, recall_mean,
            tampered_f1_mean, tampered_f1_std,
            tampered_iou_mean, tampered_iou_std,
            image_accuracy, image_auc_roc,
            threshold_used, num_test_images, num_tampered_images
    """
```

**Note:** The function takes a single `threshold` argument. Per the threshold protocol above, this value is used for both pixel binarization and image-level classification.

---

## Metric Implementations

```python
def compute_pixel_f1(pred, gt, eps=1e-8):
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return (2 * precision * recall / (precision + recall + eps)).item()


def compute_iou(pred, gt, eps=1e-8):
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / (union + eps)).item()
```

---

## Results Reporting Format

```
TEST SET RESULTS (threshold=X.XXX)

Mixed-set (N images):
  Pixel-F1:  X.XXXX ± X.XXXX
  Pixel-IoU: X.XXXX ± X.XXXX
  Precision: X.XXXX
  Recall:    X.XXXX

Tampered-only (M images):
  Pixel-F1:  X.XXXX ± X.XXXX
  Pixel-IoU: X.XXXX ± X.XXXX

Image-level:
  Accuracy:  X.XXXX
  AUC-ROC:   X.XXXX

Forgery-type breakdown:
  Splicing (K images):  F1=X.XXXX ± X.XXXX
  Copy-move (J images): F1=X.XXXX ± X.XXXX
```
