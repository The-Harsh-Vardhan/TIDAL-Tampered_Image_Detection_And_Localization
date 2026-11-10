# 7. Visual Results — Visualization Implementation Guide

## 7.1 Assignment Requirement

> *"Include a few visual results showcasing the model's output on test images."*
> *"Show predicted masks alongside the ground truth."*

Good visual results are the most impactful part of the notebook. Evaluators will look at these before reading any metric.

---

## 7.2 Required Visualizations

| Visualization | Priority | Purpose |
|--------------|----------|---------|
| **4-Column Comparison Grid** | **Critical** | Original / Ground Truth / Predicted Heatmap / Binary Mask overlay |
| **Training Curves** | **Critical** | Loss + Metrics vs. epoch (proves training converged) |
| **Confusion Examples** | High | Best predictions, worst predictions, near-threshold cases |
| **Authentic Image Check** | High | Show model correctly outputs blank mask for authentic images |
| **ROC Curve** | Medium | Visual proof of image-level detection quality |
| **F1 vs Threshold** | Medium | Justifies threshold selection |

---

## 7.3 Visualization 1: 4-Column Comparison Grid

This is the **most important visualization**. Include 6–8 rows showing diverse examples.

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def visualize_predictions(images, gt_masks, pred_masks, 
                          num_samples=6, threshold=0.5, figsize=(20, None)):
    """
    Create a 4-column visualization grid:
    Col 1: Original image
    Col 2: Ground truth mask
    Col 3: Predicted probability heatmap
    Col 4: Binary prediction overlaid on image
    
    Args:
        images: list of (3, H, W) tensors (normalized)
        gt_masks: list of (H, W) tensors (binary)
        pred_masks: list of (H, W) tensors (probability in [0,1])
        num_samples: number of rows
        threshold: binarization threshold for column 4
    """
    num_samples = min(num_samples, len(images))
    fig_height = num_samples * 3.5
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, fig_height))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    column_titles = ['Original Image', 'Ground Truth', 
                     'Predicted Heatmap', 'Prediction Overlay']
    
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold')
    
    for i in range(num_samples):
        # Denormalize image for display
        img = images[i].permute(1, 2, 0).numpy()  # (H, W, 3)
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + 
                       np.array([0.485, 0.456, 0.406]), 0, 1)
        
        gt = gt_masks[i].numpy()
        pred = pred_masks[i].numpy()
        pred_binary = (pred >= threshold).astype(np.float32)
        
        # Col 1: Original image
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        
        # Col 2: Ground truth mask (green = tampered)
        axes[i, 1].imshow(gt, cmap='Greens', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        
        # Col 3: Predicted heatmap (continuous probability)
        im = axes[i, 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        
        # Col 4: Binary prediction overlaid on original
        axes[i, 3].imshow(img)
        # Red overlay where predicted tampered
        overlay = np.zeros((*pred_binary.shape, 4))
        overlay[pred_binary > 0] = [1, 0, 0, 0.4]  # Red with 40% opacity
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis('off')
        
        # Compute per-image F1 and show on the left
        f1, _, _ = compute_pixel_f1_safe(
            torch.tensor(pred), torch.tensor(gt), threshold)
        axes[i, 0].set_ylabel(f'F1={f1:.3f}', fontsize=12, rotation=0,
                               labelpad=50, va='center')
    
    plt.tight_layout()
    plt.savefig('prediction_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### Sample Selection Strategy

Show a **curated mix** of results to be honest and informative:

```python
def select_showcase_samples(pred_masks, gt_masks, threshold=0.5, n_per_category=2):
    """
    Select samples that tell the complete story:
    - 2 best predictions (highest F1)
    - 2 average predictions (median F1)
    - 2 worst predictions (lowest F1)
    - 2 authentic images (correctly classified)
    """
    f1_scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        f1, _, _ = compute_pixel_f1_safe(pred, gt, threshold)
        f1_scores.append(f1)
    
    # Sort by F1
    sorted_indices = np.argsort(f1_scores)
    
    # Separate tampered and authentic
    tampered_indices = [i for i in sorted_indices if gt_masks[i].sum() > 0]
    authentic_indices = [i for i in sorted_indices if gt_masks[i].sum() == 0]
    
    selected = []
    # Best tampered predictions
    selected.extend(tampered_indices[-n_per_category:])
    # Median tampered predictions
    mid = len(tampered_indices) // 2
    selected.extend(tampered_indices[mid:mid+n_per_category])
    # Worst tampered predictions
    selected.extend(tampered_indices[:n_per_category])
    # Authentic images (correctly predicted as clean)
    correct_auth = [i for i in authentic_indices 
                    if pred_masks[i].max() < threshold]
    selected.extend(correct_auth[:n_per_category])
    
    return selected
```

---

## 7.4 Visualization 2: Training Curves

```python
def plot_training_curves(train_losses, val_losses, val_f1s, val_ious):
    """
    Plot training and validation curves side by side.
    
    Args:
        train_losses: list of float (per-epoch)
        val_losses: list of float (per-epoch)
        val_f1s: list of float (per-epoch)
        val_ious: list of float (per-epoch)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 curve
    axes[1].plot(epochs, val_f1s, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pixel-F1')
    axes[1].set_title('Validation Pixel-F1')
    axes[1].grid(True, alpha=0.3)
    best_epoch = np.argmax(val_f1s) + 1
    best_f1 = max(val_f1s)
    axes[1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    axes[1].annotate(f'Best: {best_f1:.4f}\n(Epoch {best_epoch})',
                     xy=(best_epoch, best_f1),
                     xytext=(best_epoch + 3, best_f1 - 0.05),
                     arrowprops=dict(arrowstyle='->', color='gray'),
                     fontsize=10)
    
    # IoU curve
    axes[2].plot(epochs, val_ious, 'm-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Pixel-IoU')
    axes[2].set_title('Validation Pixel-IoU')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### What the Curves Should Look Like (Quality Check)

| Pattern | Meaning | Action |
|---------|---------|--------|
| Train ↓, Val ↓, both converge | **Healthy training** | Continue or stop if plateau |
| Train ↓, Val ↑ (divergence) | **Overfitting** | Add dropout, augmentation, weight decay, or stop early |
| Both flat from start | **Underfitting** | Increase learning rate, check data pipeline, unfreeeze more layers |
| Train ↓ then spikes | **Learning rate too high** | Reduce LR or use warmup |
| Val F1 has jagged oscillations | **Batch size too small** | Increase effective batch via gradient accumulation |

---

## 7.5 Visualization 3: ROC Curve

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(image_scores, image_labels):
    """Plot ROC curve for image-level tampered/authentic classification."""
    fpr, tpr, thresholds = roc_curve(image_labels, image_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Image-Level Tampering Detection ROC', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 7.6 Visualization 4: F1 vs. Threshold

```python
def plot_f1_vs_threshold(thresholds, f1_scores, oracle_thresh):
    """Plot F1 as a function of binarization threshold."""
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
    plt.axvline(x=oracle_thresh, color='r', linestyle='--', linewidth=1.5,
                label=f'Oracle Threshold = {oracle_thresh:.3f}')
    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5,
                label='Default (0.5)')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Mean Pixel-F1', fontsize=12)
    plt.title('Pixel-F1 vs. Binarization Threshold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_vs_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 7.7 Visualization 5: Loss Component Breakdown

Track individual loss components during training:

```python
def plot_loss_components(epoch_losses_bce, epoch_losses_dice, epoch_losses_edge):
    """Show how each component of the hybrid loss behaves during training."""
    epochs = range(1, len(epoch_losses_bce) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_losses_bce, label='BCE Loss', linewidth=2)
    plt.plot(epochs, epoch_losses_dice, label='Dice Loss', linewidth=2)
    plt.plot(epochs, epoch_losses_edge, label='Edge Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Loss Component Breakdown', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_components.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 7.8 Visualization Display Order in Notebook

In the final notebook, present visualizations in this order:

1. **Training curves** — "The model trained successfully"
2. **Loss component breakdown** — "Each loss component contributed meaningfully"
3. **Metrics summary table** — "Here are the numbers"
4. **F1 vs. threshold + ROC curve** — "Threshold analysis and classification performance"
5. **4-column grid: Best predictions** — "What the model does well"
6. **4-column grid: Average predictions** — "Typical performance"
7. **4-column grid: Failure cases** — "Where the model struggles" (with analysis)
8. **Authentic image check** — "The model doesn't hallucinate tampering"

### Why This Order
Evaluators scan notebooks top-to-bottom. Leading with training convergence proof builds confidence. Then quantitative results set expectations. Then visuals confirm the numbers. Ending with failure analysis shows maturity and honesty.

---

## 7.9 Figure Formatting Standards

```python
# Apply consistent style across all plots
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})
```

All figures should have:
- White background (not Colab's default gray)
- Legible axis labels and title
- Grid lines at 30% opacity
- Saved at 150 DPI (readable but not excessive file size)
- `bbox_inches='tight'` to avoid clipping labels
