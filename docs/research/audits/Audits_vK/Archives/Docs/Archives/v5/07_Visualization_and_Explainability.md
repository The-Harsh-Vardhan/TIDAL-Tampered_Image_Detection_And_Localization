# Visualization and Explainability

This document covers both required visualizations and Explainable AI (XAI) components. Explainability methods are integrated into the visualization and evaluation stages to verify that the model focuses on actual tampered regions.

---

## Explainable AI — Design Rationale

For a tamper detection system, it is critical to verify that the model's predictions are grounded in actual forensic evidence (e.g., boundary artifacts, compression inconsistencies) rather than spurious correlations (e.g., image content, color distribution).

Three lightweight explainability methods are used:

| Method | Stage | Purpose |
|---|---|---|
| **Grad-CAM heatmaps** | Visualization / Evaluation | Show which spatial regions drive the encoder's activations |
| **Prediction mask overlays** | Visualization | Visually assess whether predicted masks align with ground-truth tampered regions |
| **Failure case analysis** | Evaluation | Identify systematic error patterns in worst predictions |

**Design choice:** These methods were selected because they are lightweight, require no external frameworks, and provide actionable insight for a segmentation-based pipeline. Heavy frameworks like SHAP or LIME are excluded — they are designed for classification tasks and add significant computational overhead without clear benefit for dense pixel-level prediction.

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

**What it shows:** Whether the model's binary segmentation matches the ground-truth tampering region. Best/worst rows highlight what the model gets right and where it fails.

**Reference:** The `image-detection-with-mask.ipynb` notebook uses a similar overlay approach: `overlay[mask_pred==1] = [1, 0, 0]` with alpha blending at `0.6 * img + 0.4 * overlay`.

### 2. Training Curves

Three subplots in a single row:

1. **Loss** (train + val on same axes) — Shows convergence and overfitting behavior.
2. **Validation Pixel-F1** with vertical line at best epoch — Shows when peak performance was reached.
3. **Validation Pixel-IoU** — Provides a complementary quality check.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
```

Save as `training_curves.png`.

### 3. F1-vs-Threshold Sweep

Plot mean Pixel-F1 (validation set) vs. threshold, with:
- Best threshold marked with vertical dashed line
- Clean axis labels and grid

Save as `f1_vs_threshold.png`.

**What it shows:** How sensitive pixel-level performance is to the binarization threshold. A flat curve around the optimum suggests stability; a sharp peak suggests fragility.

---

## Explainable AI Visualizations

### 4. Grad-CAM Heatmaps

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps showing which spatial regions of the encoder's feature maps contribute most to the model's output. For segmentation, we adapt Grad-CAM to the encoder's final block.

```python
class GradCAM:
    """Grad-CAM for segmentation encoder features."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Generate Grad-CAM heatmap for the segmentation output."""
        self.model.eval()
        output = self.model(input_tensor)
        # Use mean of segmentation logits as scalar target
        target = output.mean()
        self.model.zero_grad()
        target.backward()

        # Compute weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        # Resize to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()
```

**Usage in notebook:**
```python
# Apply to encoder's final layer
grad_cam = GradCAM(model, model.encoder.layer4)

# Generate heatmap for a test image
cam = grad_cam.generate(image_tensor.unsqueeze(0).to(device))

# Visualize: Original | Grad-CAM | GT Mask | Predicted Mask
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original")
axes[1].imshow(cam, cmap='jet', alpha=0.7)
axes[1].set_title("Grad-CAM")
axes[2].imshow(gt_mask, cmap='gray')
axes[2].set_title("Ground Truth")
axes[3].imshow(pred_mask, cmap='gray')
axes[3].set_title("Prediction")
```

**What it shows:** Whether the encoder focuses on tampered regions (boundary artifacts, compression anomalies) versus irrelevant image content (objects, textures). A well-performing model should show high Grad-CAM activation in regions that overlap with the ground-truth mask.

**Diagnostic value:**
- **High activation on tampered regions** → Model uses correct forensic cues
- **High activation on non-tampered regions** → Model may rely on spurious correlations
- **Diffuse activation everywhere** → Model lacks spatial discrimination

### 5. Prediction Mask Overlays

For each sample in the prediction grid, overlay the predicted binary mask on the original image with color coding:

```python
def create_diagnostic_overlay(original, pred_mask, gt_mask):
    """Create a color-coded overlay showing TP, FP, FN regions."""
    overlay = original.copy().astype(np.float32) / 255.0
    # True Positives — green
    tp_mask = (pred_mask > 0) & (gt_mask > 0)
    overlay[tp_mask] = overlay[tp_mask] * 0.5 + np.array([0, 1, 0]) * 0.5
    # False Positives — red
    fp_mask = (pred_mask > 0) & (gt_mask == 0)
    overlay[fp_mask] = overlay[fp_mask] * 0.5 + np.array([1, 0, 0]) * 0.5
    # False Negatives — blue
    fn_mask = (pred_mask == 0) & (gt_mask > 0)
    overlay[fn_mask] = overlay[fn_mask] * 0.5 + np.array([0, 0, 1]) * 0.5
    return overlay
```

**Color coding:**
- **Green**: True Positive (correctly detected tampered pixels)
- **Red**: False Positive (incorrectly flagged as tampered)
- **Blue**: False Negative (missed tampered pixels)

**What it shows:** Spatially where the model succeeds and fails, making it easy to spot patterns (e.g., consistently missing small spliced regions, false alarms near edges).

### 6. Failure Case Analysis

Systematically analyze the N worst predictions (lowest Pixel-F1) to identify error patterns:

```python
def analyze_failure_cases(predictions, n_worst=10):
    """Analyze worst predictions to identify systematic error patterns."""
    sorted_preds = sorted(predictions, key=lambda p: p['pixel_f1'])
    worst = sorted_preds[:n_worst]

    analysis = {
        'forgery_types': [p['forgery_type'] for p in worst],
        'mean_mask_area': np.mean([p['gt_mask_area'] for p in worst]),
        'mean_f1': np.mean([p['pixel_f1'] for p in worst]),
        'common_patterns': []
    }

    # Check for small-region failures
    small_mask_count = sum(1 for p in worst if p['gt_mask_area'] < 0.02)
    if small_mask_count > n_worst // 2:
        analysis['common_patterns'].append('Fails on small tampered regions (<2% area)')

    # Check for forgery-type bias
    from collections import Counter
    type_counts = Counter(analysis['forgery_types'])
    dominant_type = type_counts.most_common(1)[0]
    if dominant_type[1] > n_worst * 0.7:
        analysis['common_patterns'].append(f'Disproportionately fails on {dominant_type[0]}')

    return analysis
```

**What it shows:** Whether failures are systematic (e.g., model consistently fails on copy-move) or random. This guides future improvement priorities.

---

## Optional Visualizations

### 7. Probability Heatmap Grid

Shows raw `sigmoid(logits)` as a heatmap alongside the binary mask.

**What it shows:** Model confidence across the image. High-confidence false positives and low-confidence true positives reveal failure modes that the binary mask alone cannot expose.

### 8. Image-Level ROC Curve

Plot ROC curve using the image-level **top-k mean tamper score** and true image labels:

```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(image_labels, image_scores)
roc_auc = auc(fpr, tpr)
```

### 9. Robustness Bar Chart

Bar chart comparing Pixel-F1 under each degradation condition. Clean baseline highlighted.

### 10. Feature Map Inspection (Phase 2)

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

**What it shows:** Whether early encoder layers respond differently to tampered vs. authentic regions. This provides a basic model-interpretability signal complementing the Grad-CAM analysis above.

### 11. ELA Visualization (Phase 2)

Side-by-side: Original | ELA map | GT mask | Prediction — shows whether ELA helps identify tampered regions.

---

## Display Order in Notebook

1. Training curves
2. F1-vs-threshold
3. Metrics summary (text table)
4. Prediction grid (4-column)
5. Grad-CAM heatmaps (selected samples)
6. Diagnostic overlays (TP/FP/FN color-coded)
7. Failure case analysis summary
8. (Optional) Probability heatmap
9. (Optional) ROC curve
10. (Optional) Robustness results
11. (Optional) Feature map inspection / ELA visualization
