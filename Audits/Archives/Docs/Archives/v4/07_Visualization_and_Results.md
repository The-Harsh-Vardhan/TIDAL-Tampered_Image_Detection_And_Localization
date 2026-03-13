# Visualization and Results

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

**What it shows:** Whether the model's binary segmentation matches the ground-truth tampering region. Best/worst rows highlight what the model gets right and where it fails, which helps diagnose systematic error patterns (e.g., missing small spliced regions, over-segmenting copy-move boundaries).

### 2. Training Curves

Three subplots in a single row:

1. **Loss** (train + val on same axes) — Shows convergence and overfitting behavior.
2. **Validation Pixel-F1** with vertical line at best epoch — Shows when peak performance was reached.
3. **Validation Pixel-IoU** — Provides a complementary quality check.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
```

Save as `training_curves.png`.

**What it shows:** Whether training converged, whether early stopping triggered at the right time, and whether the models learning trajectory is healthy (smooth loss decrease without divergence).

### 3. F1-vs-Threshold Sweep

Plot mean Pixel-F1 (validation set) vs. threshold, with:
- Best threshold marked with vertical dashed line
- Clean axis labels and grid

Save as `f1_vs_threshold.png`.

**What it shows:** How sensitive pixel-level performance is to the binarization threshold. A flat curve around the optimum suggests stability; a sharp peak suggests fragility.

---

## Optional Visualizations

These are produced if time allows or for Phase 2. Each has a specific diagnostic purpose.

### 4. Probability Heatmap Grid

Shows raw `sigmoid(logits)` as a heatmap alongside the binary mask.

**What it shows:** Model confidence across the image. High-confidence false positives and low-confidence true positives reveal failure modes that the binary mask alone cannot expose.

### 5. Image-Level ROC Curve

Plot ROC curve using `max(prob_map)` as the scoring function and true image labels:

```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(image_labels, image_scores)
roc_auc = auc(fpr, tpr)
```

**What it shows:** Whether the image-level scoring function (`max(prob_map)`) separates authentic from tampered images at various thresholds.

### 6. Robustness Bar Chart

Bar chart comparing Pixel-F1 under each degradation condition. Clean baseline highlighted.

**What it shows:** Which degradation types cause the largest performance drop, indicating which forensic cues the model relies on.

### 7. ELA Visualization (Phase 2)

Side-by-side: Original | ELA map | GT mask | Prediction — shows whether ELA helps identify tampered regions.

**What it shows:** How ELA maps visually relate to tampered regions. Useful for qualitative assessment of whether adding ELA as a 4th channel provides additional forensic signal.

### 8. Feature Map Inspection (Phase 2)

Display intermediate encoder feature maps (e.g., after the first conv block) for a tampered and an authentic image side-by-side.

```python
# Hook to capture intermediate activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.encoder.layer1.register_forward_hook(get_activation('layer1'))
```

**What it shows:** Whether early encoder layers respond differently to tampered vs. authentic regions. This provides a basic model-interpretability signal without requiring gradient-based methods. It is not a formal explainability analysis but offers more insight than output-only visualization.

> **Scope note:** This project does not implement formal explainability methods (e.g., Grad-CAM, SHAP, integrated gradients). Feature-map inspection is the closest interpretability tool offered. The document title reflects this — "Visualization and Results" rather than "Explainability."

---

## Display Order in Notebook

1. Training curves
2. F1-vs-threshold
3. Metrics summary (text table)
4. Prediction grid
5. (Optional) Probability heatmap
6. (Optional) ROC curve
7. (Optional) Robustness results
8. (Optional) Feature map inspection / ELA visualization
