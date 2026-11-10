# 05 — Evaluation Methodology Evolution

## Purpose

Document how evaluation strategy evolved from Docs7 design → Audit critique → Run01 reality, and define improvements for v8.

---

## Phase 1: Docs7 Design

### Metric Suite

| Metric | Level | Role |
|---|---|---|
| Pixel-F1 | Pixel | **Primary** — harmonic mean of precision/recall |
| Pixel-IoU | Pixel | Secondary — region overlap |
| Precision | Pixel | Component metric |
| Recall | Pixel | Component metric |
| Image Accuracy | Image | Binary correct/incorrect |
| Image AUC-ROC | Image | Threshold-independent ranking |

### Evaluation Protocol

- **Threshold sweep:** 0.1 to 0.9 (9 candidates), selected by max mean Pixel-F1 on validation
- **Empty-mask handling:** If both prediction and ground truth are empty, return F1=1.0 (authentic correctly classified)
- **Per-image averaging:** Compute F1 per image, then average across test set
- **Reporting:** Mixed-set (all images), tampered-only, per-forgery-type (splicing, copy-move)
- **Image-level detection:** `tamper_score = max(prob_map)`, then binary threshold
- **Robustness evaluation:** Same metrics under JPEG, noise, blur, resize degradation

### Visualization

- 4-panel: Original | Ground Truth | Prediction | Overlay
- Best/Typical/Worst examples selected by Pixel-F1
- Grad-CAM heatmaps

## Phase 2: Audit Critique

### Findings

| Finding | Severity | Source |
|---|---|---|
| Empty-mask F1=1.0 inflates mixed-set metrics | CRITICAL | Audit6 Pro §03 Finding 2 |
| Mixed-set reported as primary metric hides true localization quality | CRITICAL | Audit6 Pro §03 Finding 2 |
| Single threshold for both pixel and image tasks | MEDIUM | Audit6 Pro §03 Finding 3 |
| No boundary metrics (BF1, boundary IoU) | HIGH | Audit6 Pro §03 Finding 1 |
| No cross-dataset evaluation | HIGH | Audit6 Pro §03 Finding 7 |
| No per-mask-size performance stratification | MEDIUM | Audit6 Pro §01 Finding 6 |
| Grad-CAM uses `output.mean()` — not region-specific | LOW | Audit6 Pro §03 Finding 4 |
| No quantitative XAI evaluation | LOW | Audit6 Pro §03 Finding 5 |
| Robustness suite tests nuisance transforms, not forgery generalization | MEDIUM | Audit6 Pro §03 Finding 6 |
| Image-level detection is heuristic (max prob), not learned | HIGH | Audit6 Pro §02 Finding 8 |

### The Metric Inflation Problem

This was the audit's most damaging finding. The evaluation framework was technically correct but practically misleading:

- Mixed-set has 1,124 authentic images (59%) that automatically score F1=1.0
- Only 769 tampered images (41%) contribute actual localization signal
- Reporting mixed F1 = 0.72 as primary metric makes the model look much better than its true tampered-only F1 = 0.41

## Phase 3: Run01 Evidence

### The Inflation Confirmed

| Metric | Mixed-set (1893) | Tampered-only (769) | Δ |
|---|---|---|---|
| Pixel-F1 | 0.7208 | 0.4101 | **−0.3107** |
| Pixel-IoU | 0.6989 | 0.3563 | −0.3426 |

The gap of 0.31 between mixed and tampered-only F1 confirms that ~43% of the apparent score comes from authentic images.

### Threshold Sweep Results

- Best threshold: **0.1327** — far below the normal 0.3–0.5 range
- This means: at the "optimal" threshold, any pixel with >13% predicted probability is classified as tampered
- Implication: The model's probability calibration is broken (outputs are too conservative)

### Per-Forgery Performance

| Type | F1 | Note |
|---|---|---|
| Splicing | 0.5901 | Moderate — model has some capability |
| Copy-move | 0.3105 | Near-failure — systematic weakness |

### Image-Level Detection

| Metric | Value |
|---|---|
| Accuracy | 0.8246 |
| AUC-ROC | 0.8703 |

Image-level detection works better than pixel localization, but relies on a heuristic (max probability) rather than a learned classifier.

### Failure Analysis

- Worst 10 predictions: all have F1=0.0
- 8/10 are copy-move, 6/10 have mask area <2%
- These are complete failures, not partial misalignment

---

## v8 Evaluation Improvements

### P0: Critical Changes

**1. Lead with Tampered-Only Metrics**

```python
# Evaluation reporting order:
print("=" * 60)
print("PRIMARY — Tampered-Only Localization")
print(f"  Pixel-F1:  {tampered_f1:.4f} ± {tampered_f1_std:.4f}")
print(f"  Pixel-IoU: {tampered_iou:.4f} ± {tampered_iou_std:.4f}")
print(f"  N images:  {n_tampered}")
print()
print("SECONDARY — Mixed-Set (includes authentic F1=1.0)")
print(f"  Pixel-F1:  {mixed_f1:.4f} ± {mixed_f1_std:.4f}")
```

**2. Expand Threshold Sweep Range**

```python
# Current: np.arange(0.1, 1.0, 0.1) — only 9 candidates
# v8: finer sweep with lower range
thresholds = np.concatenate([
    np.arange(0.05, 0.30, 0.05),  # catch low-threshold cases
    np.arange(0.30, 0.80, 0.05),  # normal range
])
```

**3. Per-Forgery-Type Reporting as Standard**

Always report splicing and copy-move F1 separately — not as supplementary data.

### P1: Important Changes

**4. Add Mask-Size Stratification**

```python
size_buckets = {
    'tiny (<2%)': lambda r: r < 0.02,
    'small (2-5%)': lambda r: 0.02 <= r < 0.05,
    'medium (5-15%)': lambda r: 0.05 <= r < 0.15,
    'large (>15%)': lambda r: r >= 0.15,
}
# Report F1 per bucket for tampered images
```

Run01 showed 6/10 worst failures have mask area <2%. This stratification quantifies the problem.

**5. Add Boundary F1 (BF1)**

Boundary F1 measures prediction quality at tampered-region edges, where localization precision matters most.

```python
from skimage.segmentation import find_boundaries

def boundary_f1(pred_mask, gt_mask, tolerance=2):
    pred_boundary = find_boundaries(pred_mask)
    gt_boundary = find_boundaries(gt_mask)
    # Dilate boundaries by tolerance pixels
    # Compute precision/recall/F1 on boundary pixels
```

**6. Separate Image-Level Threshold Calibration**

Use a dedicated threshold for image-level detection, optimized for image-level F1 or desired precision/recall tradeoff, rather than reusing the segmentation threshold.

### P2: Moderate Changes

**7. Add Precision-Recall Curves**

Plot PR curves for both pixel-level and image-level tasks. These show the full operating characteristic, not just the single-threshold snapshot.

**8. Add Confidence-Stratified Analysis**

Group predictions by model confidence (mean probability of tampered pixels) and report accuracy within confidence bins. This reveals calibration quality.

---

## Metric Reporting Template for v8

```
=== V8 EVALUATION REPORT ===

PRIMARY: Tampered-Only Localization (N=xxx)
  Pixel-F1:     x.xxxx ± x.xxxx
  Pixel-IoU:    x.xxxx ± x.xxxx
  Boundary-F1:  x.xxxx ± x.xxxx

BY FORGERY TYPE:
  Splicing (N=xxx):   F1 = x.xxxx ± x.xxxx
  Copy-move (N=xxx):  F1 = x.xxxx ± x.xxxx

BY MASK SIZE:
  Tiny (<2%):    F1 = x.xxxx (N=xxx)
  Small (2-5%):  F1 = x.xxxx (N=xxx)
  Medium (5-15%): F1 = x.xxxx (N=xxx)
  Large (>15%):  F1 = x.xxxx (N=xxx)

IMAGE-LEVEL DETECTION:
  Accuracy:   x.xxxx
  AUC-ROC:    x.xxxx
  Threshold:  x.xxxx (separate from pixel threshold)

SECONDARY: Mixed-Set (N=xxx, includes authentic)
  Pixel-F1:  x.xxxx ± x.xxxx

ROBUSTNESS:
  [per degradation condition]

CALIBRATION:
  Optimal pixel threshold: x.xxxx
  Expected range: 0.30-0.50 (if outside, flag)
```
