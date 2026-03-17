# Validation Experiments

This document describes experiments designed to test whether the model learns genuine forensic features rather than shortcuts or artifacts. These are diagnostic experiments — they validate the model's reasoning, not its raw performance.

---

## 1. Mask Randomization Test

### Purpose

If the model achieves similar performance when ground-truth masks are replaced with random masks during training, it suggests the model is not learning from the mask signal at all — it may be exploiting image-level shortcuts (e.g., JPEG quality differences between authentic and tampered images, filename patterns in metadata, or systematic visual differences unrelated to tampering).

### Protocol

1. **Train a baseline model** with correct ground-truth masks (the normal pipeline).
2. **Train a randomized model** with the same images but randomly shuffled masks — each tampered image gets a random binary mask from a different image.
3. **Compare validation Pixel-F1** between the two models.

### Expected Outcome

| Scenario | Baseline F1 | Random-Mask F1 | Interpretation |
|---|---|---|---|
| Model learns masks | High | Near zero | ✅ Model uses spatial mask information |
| Model ignores masks | High | Similar to baseline | ⚠️ Model exploits shortcuts, not forensic features |
| Both low | Low | Low | Model fails regardless — check data pipeline |

### Implementation Notes

```python
# Shuffle mask assignments for tampered images only
import random

randomized_pairs = list(zip(tampered_images, tampered_masks))
random.shuffle(randomized_pairs)
random_images, random_masks = zip(*randomized_pairs)

# Train with random_masks instead of original masks
# Use identical CONFIG, seed, and training procedure
```

The randomization should be applied **once** before training (not per-epoch) to ensure the randomized model has a consistent (albeit wrong) training signal.

### Why This Matters

CASIA v2 has known biases:
- Tampered images may systematically differ from authentic images in JPEG quality, resolution, or color statistics
- The `Tp_` prefix in filenames is stripped by the pipeline, but other statistical differences remain
- Source-image leakage means some tampered regions share content with training authentic images

If the model exploits these biases instead of learning mask-level forensic features, it would perform well on CASIA but fail on any other dataset. The mask randomization test catches this.

---

## 2. Shortcut Learning Detection

### What Shortcuts Could Exist?

| Potential Shortcut | Signal | Risk Level |
|---|---|---|
| JPEG quality mismatch | Tampered images may have different compression levels | Medium |
| Resolution differences | Tampered and authentic images may have different native resolutions | Low (all resized to 384) |
| Color statistics | Spliced regions may shift overall color distribution | Medium |
| Filename encoding | `Tp_` vs `Au_` prefix — stripped by pipeline but worth verifying | Low |
| Source-image overlap | CASIA reuses source images across tampered variants | High (not mitigable) |

### Detection Approach

1. **Grad-CAM analysis:** If Grad-CAM consistently highlights non-tampered regions (e.g., image borders, uniform backgrounds), the model may rely on global statistics rather than localized forensic evidence.
2. **Authentic-only false positive rate:** If the model produces false positives concentrated in specific spatial locations (e.g., always flags the center region), it has learned a spatial prior rather than content-based detection.
3. **Forgery-type performance gap:** A large gap between splicing F1 and copy-move F1 suggests the model relies on forgery-type-specific artifacts rather than general tampering detection.

### Reporting

The failure case analysis in `07_Visualization_and_Explainability.md` already identifies systematic error patterns. This section extends that analysis by explicitly checking for shortcut indicators:

```python
def check_shortcut_indicators(predictions, labels, forgery_types):
    """Check for signs of shortcut learning."""
    # 1. Compute false positive spatial distribution on authentic images
    # 2. Compare splicing vs copy-move F1 scores
    # 3. Check if high-confidence predictions cluster in specific image regions
    return analysis
```

---

## 3. Boundary Artifact Risk

### The Risk

Splicing creates visible boundary artifacts where the tampered region meets the original image. A model could learn to detect these boundaries without understanding tampering — it would detect any sharp transition, including natural edges.

### Why This Matters for CASIA

CASIA v2 masks are binary annotations of tampered regions. The boundaries in these masks correspond to where the splice was inserted. If the model only detects the boundary artifact (not the tampered content), it would:
- Succeed on splicing (which creates clear boundaries)
- Fail on copy-move (where boundaries may be blended)
- Produce hollow predictions (correct boundary, empty interior)

### Diagnostic Checks

1. **Prediction solidity:** Compare the ratio of filled pixels to boundary pixels in predicted masks. Hollow predictions (high edge-to-interior ratio) suggest boundary dependence.

```python
def compute_prediction_solidity(pred_mask):
    """Measure how solid (filled) the prediction is vs. boundary-only."""
    from scipy import ndimage
    eroded = ndimage.binary_erosion(pred_mask, iterations=2)
    interior_ratio = eroded.sum() / max(pred_mask.sum(), 1)
    return interior_ratio  # Low value → hollow/boundary-only prediction
```

2. **Edge-dilated mask comparison:** Dilate ground-truth mask boundaries by N pixels. If the model's predictions concentrate within the dilated boundary region, it detects edges rather than regions.

3. **Copy-move vs. splicing consistency:** Copy-move boundaries are often better blended. If the model's F1 drops significantly on copy-move, boundary dependence is likely.

### Expected Outcomes

| Indicator | Healthy Model | Boundary-Dependent Model |
|---|---|---|
| Prediction solidity | > 0.5 (mostly filled regions) | < 0.2 (hollow predictions) |
| Copy-move vs. splicing F1 gap | < 0.10 | > 0.25 |
| Predictions within boundary band | < 50% of predicted pixels | > 80% of predicted pixels |

---

## 4. Summary Table

| Experiment | Tests For | Pass Criterion |
|---|---|---|
| Mask randomization | Shortcut learning | Random-mask F1 << baseline F1 |
| Grad-CAM spatial analysis | Spurious region attention | Activation concentrates on tampered regions |
| Authentic false positive pattern | Spatial prior exploitation | No systematic spatial clustering of FPs |
| Prediction solidity | Boundary-only detection | Solidity > 0.5 for majority of predictions |
| Forgery-type gap | Type-specific artifact dependence | Splicing-vs-copy-move F1 gap < 0.10 |

---

## Implementation Status

These validation experiments are **diagnostic tools**, not part of the standard training pipeline. They can be run after the baseline model is trained:

- Mask randomization requires a separate training run (~50% of normal training time with early stopping)
- Shortcut detection uses the existing evaluation predictions
- Boundary artifact analysis uses the existing test predictions and ground-truth masks

All experiments are designed to run in the same notebook environment (Kaggle T4 / Colab T4+) with no additional dependencies.

---

## Interview: "How do you know the model actually learns tampering detection?"

Three lines of evidence:
1. **Mask randomization:** If shuffling masks drops F1 to near zero, the model depends on correct spatial mask supervision — it learned *where* tampering is, not just *whether* an image is tampered.
2. **Grad-CAM:** If heatmaps highlight tampered regions specifically, the model's attention aligns with forensic evidence.
3. **Prediction solidity:** If predictions are filled regions (not just edges), the model detects tampered content, not just boundary artifacts.

No single test is conclusive. Together, they provide converging evidence that the model's performance reflects genuine forensic learning rather than dataset biases.
