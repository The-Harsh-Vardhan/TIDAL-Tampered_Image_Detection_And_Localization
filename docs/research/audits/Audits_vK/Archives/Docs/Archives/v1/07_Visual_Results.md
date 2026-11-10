# 07 — Visual Results

## Purpose

This document specifies the required visualizations for the notebook, their layout, and the sample selection strategy.

## Required Visualizations

### 1. Prediction Comparison Grid (Critical)

The primary visual output. Shows model predictions alongside ground truth for qualitative assessment.

**Layout:** 4 columns per row.

| Column 1 | Column 2 | Column 3 | Column 4 |
|---|---|---|---|
| Original Image | Ground Truth Mask | Predicted Heatmap | Binary Prediction Overlay |

**Specifications:**
- Show 6–8 rows (diverse samples).
- Predicted heatmap: raw sigmoid output displayed as a colormap (e.g., `plt.cm.hot`).
- Binary overlay: red highlight (40% opacity) on tampered regions over the original image.
- Include the per-image Pixel-F1 score as a label on each row.

**Sample selection strategy:**
- 2 rows: best predictions (highest F1)
- 2 rows: median predictions (middle F1)
- 2 rows: worst predictions (lowest F1)
- 2 rows: correctly classified authentic images (all-zero prediction)

This gives honest coverage of model behavior — not just cherry-picked successes.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_grid(images, gt_masks, pred_probs, threshold=0.5, n_samples=8):
    """
    Plot a grid of predictions vs ground truth.
    
    Args:
        images: (N, 3, H, W) tensor, ImageNet-normalized
        gt_masks: (N, 1, H, W) tensor
        pred_probs: (N, 1, H, W) tensor, sigmoid probabilities
        threshold: float
        n_samples: int
    """
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    # ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(n_samples):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * std + mean).clip(0, 1)
        gt = gt_masks[i, 0].numpy()
        prob = pred_probs[i, 0].numpy()
        pred = (prob > threshold).astype(np.float32)
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted heatmap
        axes[i, 2].imshow(prob, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Predicted Heatmap')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = img.copy()
        overlay[pred > 0] = overlay[pred > 0] * 0.6 + np.array([1, 0, 0]) * 0.4
        f1 = compute_pixel_f1(
            torch.tensor(pred).unsqueeze(0),
            torch.tensor(gt).unsqueeze(0),
        )
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Overlay (F1={f1:.3f})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 2. Training Curves (Critical)

Track training progress and detect overfitting.

```python
def plot_training_curves(train_losses, val_losses, val_f1s, val_ious, best_epoch):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0].plot(epochs, train_losses, label='Train Loss')
    axes[0].plot(epochs, val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation F1
    axes[1].plot(epochs, val_f1s, color='green')
    axes[1].axvline(x=best_epoch + 1, color='red', linestyle='--', label=f'Best (epoch {best_epoch + 1})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pixel-F1')
    axes[1].set_title('Validation Pixel-F1')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Validation IoU
    axes[2].plot(epochs, val_ious, color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Pixel-IoU')
    axes[2].set_title('Validation Pixel-IoU')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 3. ROC Curve (Medium Priority)

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(image_scores, image_labels):
    fpr, tpr, _ = roc_curve(image_labels, image_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Image-Level ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 4. F1 vs. Threshold (Medium Priority)

```python
def plot_f1_vs_threshold(thresholds, mean_f1s, oracle_threshold):
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, mean_f1s)
    plt.axvline(x=oracle_threshold, color='red', linestyle='--', label=f'Oracle ({oracle_threshold:.2f})')
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Default (0.50)')
    plt.xlabel('Threshold')
    plt.ylabel('Mean Pixel-F1')
    plt.title('F1 Score vs. Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_vs_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()
```

## Display Order in Notebook

1. Training curves (loss, F1, IoU)
2. Metrics summary table
3. F1 vs. threshold + ROC curve
4. Prediction grid: best cases
5. Prediction grid: average cases
6. Prediction grid: failure cases
7. Prediction grid: authentic image verification
8. Robustness results (if applicable — see [08_Robustness_Testing.md](08_Robustness_Testing.md))

## Saving Figures

All figures are saved as PNG files with `dpi=150` for clear rendering. This allows figures to be included in reports or shared outside the notebook.

## Related Documents

- [06_Evaluation_Methodology.md](06_Evaluation_Methodology.md) — Metrics used in visualizations
- [08_Robustness_Testing.md](08_Robustness_Testing.md) — Robustness evaluation plots
