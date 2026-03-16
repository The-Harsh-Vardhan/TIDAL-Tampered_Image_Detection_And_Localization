# Visualization and Explainability

---

## Required Visualizations

### 1. Prediction Grid (4-column layout)

| Column 1 | Column 2 | Column 3 | Column 4 |
|---|---|---|---|
| Original Image | Ground Truth Mask | **Binary Predicted Mask** | Overlay |

**Sample selection (6–8 rows):**
- 2 rows: highest Pixel-F1 (best predictions on tampered images)
- 2 rows: median Pixel-F1 (typical performance)
- 2 rows: lowest Pixel-F1 (failure cases)
- 2 rows: authentic images (verify no false alarms)

Each row labeled with its Pixel-F1 score and forgery type.

**Column 3 is the binary predicted mask** — not a probability heatmap. This is the required deliverable per the assignment ("Predicted output").

**Column 4 overlay:**
```python
overlay = original.copy()
overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.6 + np.array([1, 0, 0]) * 0.4
```

### 2. Training Curves

Three subplots in a single row:

1. **Loss** (train + val on same axes)
2. **Validation Pixel-F1** with vertical line at best epoch
3. **Validation Pixel-IoU**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
```

Save as `training_curves.png`.

### 3. F1-vs-Threshold Sweep

Plot mean Pixel-F1 (validation set) vs. threshold, with:
- Best threshold marked with vertical dashed line
- Clean axis labels and grid

Save as `f1_vs_threshold.png`.

---

## Optional Visualizations

These are produced if time allows or for Phase 2:

### 4. Probability Heatmap Grid

Shows raw `sigmoid(logits)` as a heatmap alongside the binary mask — useful for understanding model confidence.

### 5. Image-Level ROC Curve

Plot ROC curve using `max(prob_map)` as the scoring function and true image labels:

```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(image_labels, image_scores)
roc_auc = auc(fpr, tpr)
```

### 6. Robustness Bar Chart

Bar chart comparing Pixel-F1 under each degradation condition. Clean baseline highlighted.

### 7. ELA Visualization (Phase 2)

Side-by-side: Original | ELA map | GT mask | Prediction — shows whether ELA helps identify tampered regions.

---

## Display Order in Notebook

1. Training curves
2. F1-vs-threshold
3. Metrics summary (text table)
4. Prediction grid
5. (Optional) ROC curve
6. (Optional) Robustness results
7. (Optional) ELA visualization
