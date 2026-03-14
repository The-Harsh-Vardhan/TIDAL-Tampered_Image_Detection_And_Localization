# 06 — Evaluation Methodology

## Purpose

Define evaluation metrics, their implementations, and the evaluation protocol.

## Metric Summary

| Priority | Metric | Level | Used for |
|---|---|---|---|
| Primary | Pixel-F1 | Pixel | Model selection (best checkpoint) |
| Primary | Pixel-IoU | Pixel | Localization quality |
| Secondary | Pixel Precision | Pixel | False positive analysis |
| Secondary | Pixel Recall | Pixel | False negative analysis |
| Secondary | Image Accuracy | Image | Classification performance |
| Secondary | Image AUC-ROC | Image | Threshold-independent classification |
| Supplementary | Oracle-F1 | Pixel | Analysis only — never for model selection |

## Pixel-Level Metrics

### Pixel-F1

```python
def compute_pixel_f1(pred, gt, eps=1e-8):
    pred = pred.flatten()
    gt = gt.flatten()

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Authentic images: both empty = correct -> 1.0; pred nonempty = wrong -> 0.0
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0

    return f1.item()
```

### Pixel-IoU

```python
def compute_iou(pred, gt, eps=1e-8):
    pred = pred.flatten()
    gt = gt.flatten()

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection

    if union == 0:
        return 1.0

    return (intersection / (union + eps)).item()
```

### Precision and Recall

```python
def compute_precision_recall(pred, gt, eps=1e-8):
    pred = pred.flatten()
    gt = gt.flatten()

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    precision = (tp / (tp + fp + eps)).item()
    recall = (tp / (tp + fn + eps)).item()
    return precision, recall
```

## Image-Level Metrics

Image-level detection is derived from the predicted probability map. No separate classification head is used.

```python
tamper_score = prob_map.max().item()  # Or top-k mean probability
is_tampered = tamper_score >= threshold
```

The image-level threshold may differ from the pixel binarization threshold. Both should be selected on the validation set.

### Image Accuracy

```python
def compute_image_accuracy(scores, labels, threshold=0.5):
    preds = (np.array(scores) >= threshold).astype(int)
    return (preds == np.array(labels)).mean()
```

### Image AUC-ROC

```python
from sklearn.metrics import roc_auc_score

def compute_image_auc(scores, labels):
    return roc_auc_score(labels, scores)
```

## Reporting: Two Views

Report localization metrics in two views to avoid authentic-image inflation:

| View | Description | Why |
|---|---|---|
| **Mixed-set** | All test images (authentic + tampered) | Reflects end-to-end behavior |
| **Tampered-only** | Only tampered test images | True localization quality without inflated scores from easy authentic images |

Authentic images that predict all-zero correctly return F1=1.0 and IoU=1.0, which inflates mixed-set averages. The tampered-only view gives the honest localization assessment.

## Threshold Selection

**Rule:** The operating threshold is selected on the validation set. The test set is used for reporting only.

Validation threshold search:

```python
def find_best_threshold(model, val_loader, device, n_thresholds=50):
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    # ... sweep and find threshold maximizing mean Pixel-F1 on validation set
    return best_threshold, best_f1
```

Oracle-F1 on the test set may be reported as supplementary analysis but must not be used for model selection or threshold tuning.

## Full Evaluation Pipeline

```python
def evaluate(model, test_loader, device, threshold=0.5):
    model.eval()
    all_f1, all_iou = [], []
    tampered_f1, tampered_iou = [], []
    precisions, recalls = [], []
    image_scores, image_labels = [], []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > threshold).float()

            for pred, gt, prob, label in zip(preds, masks, probs, labels):
                f1 = compute_pixel_f1(pred, gt)
                iou = compute_iou(pred, gt)
                p, r = compute_precision_recall(pred, gt)

                all_f1.append(f1)
                all_iou.append(iou)
                precisions.append(p)
                recalls.append(r)

                if label.item() == 1.0:
                    tampered_f1.append(f1)
                    tampered_iou.append(iou)

                image_scores.append(prob.max().item())
                image_labels.append(int(label.item()))

    return {
        'pixel_f1_mean': np.mean(all_f1),
        'pixel_f1_std': np.std(all_f1),
        'pixel_iou_mean': np.mean(all_iou),
        'pixel_iou_std': np.std(all_iou),
        'tampered_f1_mean': np.mean(tampered_f1),
        'tampered_iou_mean': np.mean(tampered_iou),
        'precision_mean': np.mean(precisions),
        'recall_mean': np.mean(recalls),
        'image_accuracy': compute_image_accuracy(image_scores, image_labels, threshold),
        'image_auc_roc': compute_image_auc(image_scores, image_labels),
        'threshold_used': threshold,
        'num_test_images': len(all_f1),
        'num_tampered_images': len(tampered_f1),
    }
```

## Related Documents

- [05_Training_Strategy.md](05_Training_Strategy.md) — Validation during training
- [07_Visualization_and_Results.md](07_Visualization_and_Results.md) — Visualization of results
