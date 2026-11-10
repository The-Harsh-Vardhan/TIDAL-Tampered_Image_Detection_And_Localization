# 6. Performance Metrics — Evaluation Implementation Guide

## 6.1 Assignment Requirement

> *"Evaluate on appropriate metrics (F1, IoU, and any other relevant metric you see fit to evaluate on)."*

Metrics must cover both pixel-level localization quality and image-level detection accuracy.

---

## 6.2 Metric Overview

| Metric | Level | What It Measures | Why We Need It |
|--------|-------|------------------|----------------|
| **Pixel-F1** | Pixel | Harmonic mean of precision & recall on tampered pixels | Primary quality metric — directly aligns with assignment requirements |
| **Pixel-IoU** | Pixel | Intersection over Union of predicted vs. ground truth mask | Measures spatial overlap; standard in segmentation literature |
| **Image-Level AUC-ROC** | Image | Classification ability: tampered vs. authentic | Shows the model can detect WHETHER an image is tampered (even if localization isn't perfect) |
| **Image-Level Accuracy** | Image | Binary classification accuracy | Simple intuitive metric for evaluators |
| **Oracle-F1** | Pixel | Best F1 over all thresholds (upper-bound performance) | Decouples model quality from threshold selection |

---

## 6.3 Metric 1: Pixel-Level F1 Score

### Definition
$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Where:
- Precision = TP / (TP + FP) — "Of pixels predicted as tampered, how many actually are?"
- Recall = TP / (TP + FN) — "Of actually tampered pixels, how many did we find?"

### Implementation

```python
def compute_pixel_f1(pred_mask, gt_mask, threshold=0.5):
    """
    Compute pixel-level F1 score.
    
    Args:
        pred_mask: (H, W) probability map in [0, 1]
        gt_mask: (H, W) binary ground truth {0, 1}
        threshold: float, binarization threshold
    
    Returns:
        f1, precision, recall: float values
    """
    pred_binary = (pred_mask >= threshold).float()
    
    tp = (pred_binary * gt_mask).sum()
    fp = (pred_binary * (1 - gt_mask)).sum()
    fn = ((1 - pred_binary) * gt_mask).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1.item(), precision.item(), recall.item()
```

### Handling Authentic Images (Zero Mask)
For authentic images (fully zero ground truth), there are no tampered pixels:
- If the model correctly predicts all zeros → **F1 = 1.0** (perfect)
- If the model predicts any positive → **F1 = 0.0** (false alarm)

```python
def compute_pixel_f1_safe(pred_mask, gt_mask, threshold=0.5):
    """F1 with proper handling of authentic images."""
    pred_binary = (pred_mask >= threshold).float()
    
    # Authentic image: no tampered pixels in ground truth
    if gt_mask.sum() == 0:
        # Perfect if model also predicts no tampering
        return (1.0, 1.0, 1.0) if pred_binary.sum() == 0 else (0.0, 0.0, 1.0)
    
    tp = (pred_binary * gt_mask).sum()
    fp = (pred_binary * (1 - gt_mask)).sum()
    fn = ((1 - pred_binary) * gt_mask).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1.item(), precision.item(), recall.item()
```

---

## 6.4 Metric 2: Pixel-Level IoU (Jaccard Index)

### Definition
$$IoU = \frac{|Pred \cap GT|}{|Pred \cup GT|} = \frac{TP}{TP + FP + FN}$$

### Implementation

```python
def compute_pixel_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute pixel-level IoU for tampered class.
    
    Args:
        pred_mask: (H, W) probability map in [0, 1]
        gt_mask: (H, W) binary ground truth {0, 1}
        threshold: float
    
    Returns:
        iou: float
    """
    pred_binary = (pred_mask >= threshold).float()
    
    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection
    
    if union == 0:
        return 1.0  # Both empty → perfect agreement
    
    return (intersection / (union + 1e-8)).item()
```

---

## 6.5 Metric 3: Image-Level AUC-ROC

### Concept
Collapse each predicted mask into a single score representing "how tampered is this image?" and compute AUC-ROC against the image-level binary label (tampered vs. authentic).

### Score Aggregation Strategy
Use `max(predicted_mask)` as the image-level tampering score:
- Authentic images → model outputs near-zero everywhere → low max
- Tampered images → model outputs high values in tampered region → high max

```python
from sklearn.metrics import roc_auc_score, roc_curve

def compute_image_level_auc(pred_masks, gt_masks):
    """
    Compute image-level AUC-ROC.
    
    Args:
        pred_masks: list of (H, W) probability maps
        gt_masks: list of (H, W) binary ground truths
    
    Returns:
        auc: float, image_scores: list, image_labels: list
    """
    image_scores = []
    image_labels = []
    
    for pred, gt in zip(pred_masks, gt_masks):
        # Image-level score: maximum prediction value
        score = pred.max().item()
        image_scores.append(score)
        
        # Image-level label: 1 if any pixel is tampered
        label = 1 if gt.sum() > 0 else 0
        image_labels.append(label)
    
    auc = roc_auc_score(image_labels, image_scores)
    return auc, image_scores, image_labels
```

---

## 6.6 Metric 4: Image-Level Accuracy

```python
def compute_image_accuracy(pred_masks, gt_masks, threshold=0.5, pixel_threshold=0.5):
    """
    Binary classification accuracy: is this image tampered or authentic?
    
    Args:
        pred_masks: list of (H, W) probability maps
        gt_masks: list of (H, W) binary ground truths
        threshold: image-level score threshold for "tampered" decision
        pixel_threshold: threshold for binarizing pixel predictions
    
    Returns:
        accuracy: float, correct: int, total: int
    """
    correct = 0
    total = len(pred_masks)
    
    for pred, gt in zip(pred_masks, gt_masks):
        # Image-level prediction
        score = pred.max().item()
        pred_label = 1 if score >= threshold else 0
        
        # Image-level ground truth
        gt_label = 1 if gt.sum() > 0 else 0
        
        correct += int(pred_label == gt_label)
    
    return correct / total, correct, total
```

---

## 6.7 Metric 5: Oracle-F1 (Best Threshold)

### Why
The default threshold of 0.5 is arbitrary. Oracle-F1 finds the threshold that maximizes F1, showing the model's upper-bound performance independent of threshold selection.

```python
import numpy as np

def compute_oracle_f1(pred_masks, gt_masks, num_thresholds=50):
    """
    Find the threshold that maximizes average F1 across all images.
    
    Args:
        pred_masks: list of (H, W) probability maps
        gt_masks: list of (H, W) binary ground truths
        num_thresholds: number of thresholds to evaluate
    
    Returns:
        best_f1: float, best_threshold: float, all_f1s: list
    """
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    mean_f1s = []
    
    for t in thresholds:
        f1s = []
        for pred, gt in zip(pred_masks, gt_masks):
            f1, _, _ = compute_pixel_f1_safe(pred, gt, threshold=t)
            f1s.append(f1)
        mean_f1s.append(np.mean(f1s))
    
    best_idx = np.argmax(mean_f1s)
    return mean_f1s[best_idx], thresholds[best_idx], list(zip(thresholds, mean_f1s))
```

---

## 6.8 Complete Evaluation Pipeline

```python
def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Full evaluation pipeline computing all metrics.
    
    Returns a dict with all metric values.
    """
    model.eval()
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu()  # (B, H, W)
            all_preds.extend([p for p in probs])
            all_gts.extend([m.squeeze(0) for m in masks])
    
    # === Pixel-Level Metrics ===
    pixel_f1s, pixel_ious = [], []
    precisions, recalls = [], []
    
    for pred, gt in zip(all_preds, all_gts):
        f1, prec, rec = compute_pixel_f1_safe(pred, gt, threshold)
        pixel_f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        pixel_ious.append(compute_pixel_iou(pred, gt, threshold))
    
    # === Image-Level Metrics ===
    auc, _, _ = compute_image_level_auc(all_preds, all_gts)
    img_acc, _, _ = compute_image_accuracy(all_preds, all_gts, threshold)
    
    # === Oracle F1 ===
    oracle_f1, oracle_thresh, _ = compute_oracle_f1(all_preds, all_gts)
    
    results = {
        'pixel_f1_mean': np.mean(pixel_f1s),
        'pixel_f1_std': np.std(pixel_f1s),
        'pixel_iou_mean': np.mean(pixel_ious),
        'pixel_iou_std': np.std(pixel_ious),
        'precision_mean': np.mean(precisions),
        'recall_mean': np.mean(recalls),
        'image_auc_roc': auc,
        'image_accuracy': img_acc,
        'oracle_f1': oracle_f1,
        'oracle_threshold': oracle_thresh,
        'threshold_used': threshold,
        'num_test_images': len(all_preds),
    }
    
    return results, all_preds, all_gts
```

---

## 6.9 Results Display

```python
def print_results(results):
    """Pretty-print evaluation results."""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Pixel-Level Metrics':=^60}")
    print(f"  F1 Score:     {results['pixel_f1_mean']:.4f} ± {results['pixel_f1_std']:.4f}")
    print(f"  IoU:          {results['pixel_iou_mean']:.4f} ± {results['pixel_iou_std']:.4f}")
    print(f"  Precision:    {results['precision_mean']:.4f}")
    print(f"  Recall:       {results['recall_mean']:.4f}")
    print(f"\n{'Image-Level Metrics':=^60}")
    print(f"  AUC-ROC:      {results['image_auc_roc']:.4f}")
    print(f"  Accuracy:     {results['image_accuracy']:.4f}")
    print(f"\n{'Threshold Analysis':=^60}")
    print(f"  Used:         {results['threshold_used']:.2f}")
    print(f"  Oracle F1:    {results['oracle_f1']:.4f} (at t={results['oracle_threshold']:.3f})")
    print(f"\n  Test Images:  {results['num_test_images']}")
    print("=" * 60)
```

---

## 6.10 Per-Category Breakdown

Evaluate separately on splicing vs. copy-move:

```python
def evaluate_by_category(model, test_dataset, device, threshold=0.5):
    """
    Separate evaluation for splicing and copy-move images.
    """
    categories = {'splicing': [], 'copy_move': []}
    
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, mask = test_dataset[idx]
            filename = test_dataset.filenames[idx]  # Assumes stored in dataset
            
            image = image.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                logits = model(image)
            pred = torch.sigmoid(logits).squeeze().cpu()
            mask = mask.squeeze()
            
            # CASIA naming: Tp_D_*_S_* = splicing, Tp_D_*_C_* = copy-move
            if '_S_' in filename:
                categories['splicing'].append((pred, mask))
            elif '_C_' in filename:
                categories['copy_move'].append((pred, mask))
    
    for cat_name, pairs in categories.items():
        if not pairs:
            continue
        preds, gts = zip(*pairs)
        f1s = [compute_pixel_f1_safe(p, g, threshold)[0] for p, g in pairs]
        print(f"\n{cat_name.upper()} ({len(pairs)} images):")
        print(f"  Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
```

---

## 6.11 Performance Targets

Based on CASIA v2.0 benchmarks from literature:

| Metric | Conservative (Passing) | Good | Excellent |
|--------|----------------------|------|-----------|
| **Pixel-F1** | 0.55–0.60 | 0.60–0.70 | 0.70+ |
| **Pixel-IoU** | 0.40–0.48 | 0.48–0.55 | 0.55+ |
| **Image AUC** | 0.80–0.85 | 0.85–0.92 | 0.92+ |
| **Image Accuracy** | 0.75–0.80 | 0.80–0.88 | 0.88+ |
| **Oracle-F1** | 0.60–0.65 | 0.65–0.75 | 0.75+ |

> **Note**: These targets are for a well-trained model. If your metrics are significantly below "Conservative," check data pipeline issues before tuning hyperparameters.

---

## 6.12 Threshold Selection Strategy

For the final reported results, use the **Oracle threshold found on the validation set** and apply it to the test set:

```python
# Step 1: Find oracle threshold on validation set
_, oracle_thresh_val, _ = compute_oracle_f1(val_preds, val_gts)

# Step 2: Apply that threshold to test set
test_results, _, _ = evaluate_model(model, test_loader, device, threshold=oracle_thresh_val)
```

This is valid because the threshold is tuned on validation data (not test data), avoiding information leakage.
