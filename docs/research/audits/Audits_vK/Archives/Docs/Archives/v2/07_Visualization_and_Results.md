# 07 — Visualization and Results

## Purpose

Specify the required visualizations and their layout. The assignment explicitly requires: Original Image, Ground Truth, Predicted output, and Overlay Visualization.

## Required Visualizations

### 1. Prediction Comparison Grid (Required)

The primary visual deliverable. Four columns per row:

| Column 1 | Column 2 | Column 3 | Column 4 |
|---|---|---|---|
| Original Image | Ground Truth Mask | **Binary Predicted Mask** | Overlay |

**The third column must be the binary predicted mask**, not a probability heatmap. The assignment asks for "Predicted output" which is the final mask. A heatmap may be shown separately as supplementary analysis.

**Overlay:** Red highlight (40% opacity) on regions predicted as tampered, drawn on the original image.

**Sample selection:** Show 6-8 rows with honest coverage:
- 2 rows: best predictions (highest F1)
- 2 rows: median predictions
- 2 rows: worst predictions (failure cases)
- 2 rows: correctly classified authentic images

Include per-image Pixel-F1 as a label on each row.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_grid(images, gt_masks, pred_masks, pred_probs=None,
                         threshold=0.5, n_samples=8):
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(n_samples):
        # Denormalize image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * std + mean).clip(0, 1)
        gt = gt_masks[i, 0].cpu().numpy()
        pred = pred_masks[i, 0].cpu().numpy()

        # Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Ground truth
        axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Binary predicted mask
        axes[i, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Predicted Mask')
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

### 2. Training Curves (Required)

```python
def plot_training_curves(train_losses, val_losses, val_f1s, val_ious, best_epoch):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label='Train Loss')
    axes[0].plot(epochs, val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_f1s, color='green')
    axes[1].axvline(x=best_epoch + 1, color='red', linestyle='--',
                    label=f'Best epoch {best_epoch + 1}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pixel-F1')
    axes[1].set_title('Validation Pixel-F1')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_ious, color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Pixel-IoU')
    axes[2].set_title('Validation Pixel-IoU')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 3. Optional Supplementary Plots

These are useful but not required by the assignment:

- **Probability heatmap grid** — show raw sigmoid output per image as a supplementary view.
- **ROC curve** — image-level true positive rate vs. false positive rate with AUC.
- **F1 vs. threshold** — sweep thresholds and plot mean Pixel-F1.

## Display Order in Notebook

1. Training curves (loss, F1, IoU)
2. Metrics summary table (mixed-set and tampered-only)
3. Prediction grid: best, median, failure, authentic samples
4. Optional: probability heatmap grid
5. Optional: ROC curve, F1-vs-threshold plot
6. Optional: robustness results (see 08_Robustness_Testing.md)

## Related Documents

- [06_Evaluation_Methodology.md](06_Evaluation_Methodology.md) — Metrics definitions
- [08_Robustness_Testing.md](08_Robustness_Testing.md) — Robustness plots
