# Visualization and Explainability

This document covers both required visualizations and Explainable AI (XAI) components. Explainability methods are used to verify that the model focuses on actual tampered regions rather than spurious correlations.

---

## Explainable AI — Design Rationale

For a tamper detection system, it is critical to verify that predictions are grounded in forensic evidence (e.g., boundary artifacts, compression inconsistencies) rather than irrelevant image content. Three lightweight explainability methods are used:

| Method | Stage | Purpose |
|---|---|---|
| **Grad-CAM heatmaps** | Visualization | Show which spatial regions drive encoder activations |
| **Prediction mask overlays** | Visualization | Assess whether predicted masks align with ground-truth |
| **Failure case analysis** | Evaluation | Identify systematic error patterns in worst predictions |

**Design choice:** These methods are lightweight, require no external frameworks, and provide actionable insight for a segmentation-based pipeline. Heavy frameworks like SHAP or LIME are excluded — they are designed for classification tasks and add significant overhead without clear benefit for dense pixel-level prediction.

**Honest positioning:** Grad-CAM and overlays are diagnostic tools, not rigorous explainability methods. They support model inspection but do not provide formal causal explanations.

---

## Required Visualizations

### 1. Prediction Grid (4-column layout)

| Column 1 | Column 2 | Column 3 | Column 4 |
|---|---|---|---|
| Original Image | Ground Truth Mask | Binary Predicted Mask | Overlay |

**Sample selection (6–8 rows):**
- 2 rows: highest Pixel-F1 (best predictions on tampered images)
- 2 rows: median Pixel-F1 (typical performance)
- 2 rows: lowest Pixel-F1 (failure cases)
- 2 rows: authentic images (verify no false alarms)

Each row labeled with its Pixel-F1 score and forgery type. An `n_rows == 0` guard prevents crash on empty subsets.

**Overlay:** Red-tinted tampered regions blended with original image.

### 2. Training Curves

Three subplots in a single row:

1. **Loss** (train + val) — convergence and overfitting behavior.
2. **Validation Pixel-F1** with vertical line at best epoch.
3. **Validation Pixel-IoU** — complementary quality check.

Saved as `training_curves.png`.

### 3. F1-vs-Threshold Sweep

Plot mean Pixel-F1 (validation set) vs. threshold, with best threshold marked. A flat curve around the optimum suggests stability; a sharp peak suggests fragility.

Saved as `f1_vs_threshold.png`.

---

## Explainable AI Visualizations

### 4. Grad-CAM Heatmaps

Grad-CAM generates heatmaps showing which spatial regions of the encoder's feature maps contribute most to the model's output.

```python
class GradCAM:
    '''Grad-CAM for segmentation encoder features.'''

    def __init__(self, model, target_layer):
        # Register forward hook to capture activations
        # Register backward hook to capture gradients
        pass

    def generate(self, input_tensor):
        # Forward pass → mean of logits as scalar target → backward
        # Safety: verify hooks captured data (warn + return None if not)
        # Compute weights = gradients.mean(dim=(2,3))
        # CAM = ReLU(sum(weights * activations))
        # Normalize to [0,1], bilinear upsample to input resolution
        # Wrapped in try/except for graceful failure
        pass
```

**Target layer:** `model.encoder.layer4` (final encoder block).

**Safety checks (v5.1):**
- `try/except` around `generate()` — prevents notebook crash on hook failures
- None-check on gradients/activations — warns and returns None if hooks failed
- Visualization code handles None-valued Grad-CAM gracefully

**Diagnostic interpretation:**
- High activation on tampered regions → model uses correct forensic cues
- High activation on non-tampered regions → possible spurious correlations
- Diffuse activation → model lacks spatial discrimination

### 5. Diagnostic Overlays (TP/FP/FN)

Color-coded overlay showing prediction quality:

```python
def create_diagnostic_overlay(original, pred_mask, gt_mask):
    '''Colour-coded diagnostic overlay.'''
    # Green: True Positive (correctly detected tampered pixels)
    # Red:   False Positive (incorrectly flagged as tampered)
    # Blue:  False Negative (missed tampered pixels)
    pass
```

**What it shows:** Spatially where the model succeeds and fails, making it easy to spot patterns (e.g., consistently missing small spliced regions, false alarms near edges).

### 6. Failure Case Analysis

Systematically analyze the N worst predictions (lowest Pixel-F1):

```python
def analyze_failure_cases(predictions, n_worst=10):
    '''Analyze worst predictions to identify systematic error patterns.'''
    # Check for small-region failures (< 2% area)
    # Check for forgery-type bias (disproportionate failures on one type)
    # Return analysis dict with patterns
    pass
```

**What it shows:** Whether failures are systematic (e.g., consistently fails on copy-move) or random.

---

## Display Order in Notebook

1. Training curves
2. F1-vs-threshold sweep
3. Prediction grid (4-column)
4. Grad-CAM heatmaps
5. Diagnostic overlays
6. Failure case analysis
7. Robustness bar chart
