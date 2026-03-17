# 06 — Evaluation Methodology

## Purpose

This document defines all evaluation metrics, their implementations, and the evaluation protocol for the project.

## Metric Hierarchy

| Priority | Metric | Level | Used for |
|---|---|---|---|
| Primary | Pixel-F1 | Pixel | Model selection (best checkpoint) |
| Primary | Pixel-IoU (Jaccard) | Pixel | Localization quality |
| Secondary | Pixel Precision | Pixel | False positive analysis |
| Secondary | Pixel Recall | Pixel | False negative analysis |
| Secondary | Image-level Accuracy | Image | Classification performance |
| Secondary | Image-level AUC-ROC | Image | Threshold-independent classification |
| Optional | Oracle-F1 | Pixel | Best achievable F1 across thresholds |

## Pixel-Level Metrics

### Pixel-F1 Score

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

```python
def compute_pixel_f1(pred, gt, eps=1e-8):
    """
    Compute pixel-level F1 between binary prediction and ground truth.
    Both inputs: tensors with values in {0, 1}, shape (1, H, W) or (H, W).
    """
    pred = pred.flatten()
    gt = gt.flatten()
    
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    # Handle case where both pred and gt are all-zero (authentic image)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0
    
    return f1.item()
```

### Pixel-IoU (Jaccard Index)

$$\text{IoU} = \frac{TP}{TP + FP + FN}$$

```python
def compute_iou(pred, gt, eps=1e-8):
    """
    Compute Intersection over Union for binary masks.
    """
    pred = pred.flatten()
    gt = gt.flatten()
    
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    
    if union == 0:
        return 1.0  # Both masks are empty
    
    return (intersection / (union + eps)).item()
```

### Pixel Precision and Recall

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

Image-level detection is derived from the predicted mask. No separate classification head is used.

### Score Derivation

```python
tamper_score = predicted_probability_map.max()  # Per-image score
is_tampered = tamper_score >= threshold
```

### Image-Level Accuracy

```python
def compute_image_accuracy(scores, labels, threshold=0.5):
    preds = (np.array(scores) >= threshold).astype(int)
    return (preds == np.array(labels)).mean()
```

### Image-Level AUC-ROC

```python
from sklearn.metrics import roc_auc_score

def compute_image_auc(scores, labels):
    return roc_auc_score(labels, scores)
```

## Oracle-F1 (Threshold Search)

Oracle-F1 finds the threshold that maximizes mean Pixel-F1 across all test images. This is reported for analysis only — the final model uses a threshold selected on the **validation set**.

```python
def find_oracle_threshold(model, dataloader, device, n_thresholds=50):
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    all_probs = []
    all_gts = []
    
    model.eval()
    with torch.no_grad():
        for images, masks, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_gts.append(masks)
    
    all_probs = torch.cat(all_probs)
    all_gts = torch.cat(all_gts)
    
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (all_probs > t).float()
        f1s = [compute_pixel_f1(p, g) for p, g in zip(preds, all_gts)]
        mean_f1 = np.mean(f1s)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_t = t
    
    return best_t, best_f1
```

**Important:** Run `find_oracle_threshold` on the **validation set** to select the operating threshold. Then report Oracle-F1 on the test set only as supplementary information. Do not use test-set Oracle-F1 to select the threshold.

## Full Evaluation Pipeline

```python
def evaluate(model, test_loader, device, threshold=0.5):
    model.eval()
    pixel_f1s, pixel_ious = [], []
    precisions, recalls = [], []
    image_scores, image_labels = [], []
    
    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > threshold).float()
            
            for pred, gt, prob, label in zip(preds, masks, probs, labels):
                pixel_f1s.append(compute_pixel_f1(pred, gt))
                pixel_ious.append(compute_iou(pred, gt))
                p, r = compute_precision_recall(pred, gt)
                precisions.append(p)
                recalls.append(r)
                image_scores.append(prob.max().item())
                image_labels.append(int(label.item()))
    
    results = {
        'pixel_f1_mean': np.mean(pixel_f1s),
        'pixel_f1_std': np.std(pixel_f1s),
        'pixel_iou_mean': np.mean(pixel_ious),
        'pixel_iou_std': np.std(pixel_ious),
        'precision_mean': np.mean(precisions),
        'recall_mean': np.mean(recalls),
        'image_accuracy': compute_image_accuracy(image_scores, image_labels, threshold),
        'image_auc_roc': compute_image_auc(image_scores, image_labels),
        'threshold_used': threshold,
        'num_test_images': len(pixel_f1s),
    }
    
    return results
```

## Reporting Format

Present results as a summary table in the notebook:

| Metric | Value |
|---|---|
| Pixel-F1 (mean ± std) | e.g., 0.62 ± 0.18 |
| Pixel-IoU (mean ± std) | e.g., 0.50 ± 0.16 |
| Pixel Precision | e.g., 0.68 |
| Pixel Recall | e.g., 0.57 |
| Image Accuracy | e.g., 0.83 |
| Image AUC-ROC | e.g., 0.87 |
| Threshold used | e.g., 0.45 |
| Test set size | e.g., 372 |

## Related Documents

- [05_Training_Pipeline.md](05_Training_Pipeline.md) — Validation during training
- [07_Visual_Results.md](07_Visual_Results.md) — Visualization of predictions
- [08_Robustness_Testing.md](08_Robustness_Testing.md) — Evaluation under degradations
