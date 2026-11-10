# 06 — Visualization Audit

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Summary

Visualisation is explicitly required by the assignment. The assignment text says: "provide clear visual results comparing the Original Image, Ground Truth, Predicted output, and an Overlay Visualization."

v8 run-01 satisfies this requirement. v9 does not, because v9 has not been run.

---

## v8 Visualisation — What Exists

The v8 run-01 notebook contains code and **executed output cells** for:

1. **Training batch sanity check** — 2-row grid showing raw images and their binary masks before augmentation. Confirms that masks are loaded and aligned.

2. **F1 vs threshold curve** — Printed in the notebook showing the relationship between threshold choice and tampered-only F1. Confirms the threshold scan ran.

3. **4-panel prediction grid** — For selected test samples (best, typical, worst per forgery type): Original | GT Mask | Predicted Mask | Overlay. The overlay uses `overlay[pred_bool] = overlay[pred_bool] * 0.6 + np.array([1, 0, 0]) * 0.4` — semi-transparent red highlighting of predicted tampered regions.

All of these are rendered output cells in the notebook file. A reviewer sees them immediately on opening the notebook on GitHub or Colab.

---

## v9 Visualisation — What Exists

The v9 notebook defines:

```python
def make_overlay(rgb, gt_mask, pred_mask):
    # red channel for predictions, green channel for ground truth

def select_visualization_rows(per_image_rows, num_examples):
    # best/typical/worst selection

def visualize_predictions(per_image_rows, split_df, num_examples, output_path):
    # 5-column grid: Original | ELA | GT Mask | Prediction | Overlay
```

This is a better visualisation design than v8: 5 columns instead of 4, includes ELA channel, colour-codes GT (green) vs prediction (red) in overlay.

**The single problem:** `"outputs": []` on every cell. The visualisation functions were never called. No output images exist.

---

## The 5-Column Grid Design

When executed, v9 visualisations will be richer than v8 because:

1. Column 3 (ELA) shows what the forensic channel looks like — this is genuinely useful for understanding why the model predicts what it predicts
2. Overlay uses green for GT and red for predicted, making false positives and false negatives visually distinct
3. `select_visualization_rows` picks worst/typical/best examples by per-image F1 score, ensuring the display is representative and not cherry-picked

### One design problem

The `select_visualization_rows` function tries to find images at the 10th, 50th, and 90th percentile of the per-image F1 distribution. If the test set has very few samples at extremes (all heavily imbalanced toward good predictions, or none converge), the percentile selection may cluster multiple examples at the same accuracy tier, producing redundant visualisation.

More importantly: for authentic images, the "ground truth mask" is all-zero. Showing authentic images in the 5-column grid produces a row where columns 3-5 are all black, which is visually useless. The selection logic should prioritise tampered images for visualisation rows. It is not obvious from the code that this restriction is enforced.

---

## ELA Visualisation

The `cmap="hot"` colormap for ELA maps is a good choice — it makes high ELA regions (likely tampered edges) visually prominent. This is better than a grayscale ELA, which often looks like noise.

No other issues with the ELA visualisation design. The design is sound. The execution doesn't exist.

---

## Robustness Visualisation

v9 includes `robustness_df` visualisation defined in the main experiment runner. Bar charts showing F1 degradation per condition are generated.

v8 had similar bar charts and they were executed. v9's are not.

---

## Visualisation Rating

| Visual Component | v8 run-01 | v9 Colab | Notes |
|----------------|-----------|----------|-------|
| Training batch sanity grid | ✅ Executed | ⚠️ Not present | v9 skipped this |
| Loss curve visualisation | ✅ Executed | ⚠️ Not visible | No training ran |
| Threshold F1 curve | ✅ Executed | ⚠️ Not visible | |
| 4-panel prediction grid | ✅ Executed | ❌ Never produced | Assignment requirement |
| GT/Pred overlay | ✅ Executed | ❌ Never produced | Assignment requirement |
| ELA channel visualisation | N/A | ❌ Never produced | New in v9, not executed |
| Robustness bar charts | ✅ Executed | ❌ Never produced | |
| Design quality | Good (4 col) | Better (5 col) | v9 design is better |
| Assignment compliance | ✅ Met | ❌ Not met | |

---

## Critical Point

The assignment literally requires visuals. Not code that would produce visuals. Not functions that define how visuals would be generated. Actual image output.

v9 flunks the most explicitly named requirement in the assignment specification because nobody ran the notebook.

This is not a subtle gap. It is the most obvious failure in the entire v9 submission.
