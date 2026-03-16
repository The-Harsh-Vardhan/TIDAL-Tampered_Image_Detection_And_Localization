# 05 — Evaluation Methodology Roast

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Summary

The v9 evaluation framework is the most sophisticated component of the entire notebook. It is rigorous in design: tampered-only primary metrics, learned classification head evaluation, Boundary-F1, PR curves, mask-size stratification, per-forgery-type breakdown, mask randomization, and a full robustness suite.

It is also completely hypothetical because the notebook has never been run.

This section evaluates the evaluation design itself and identifies what will and will not work when someone finally executes it.

---

## 1. Segmentation Metrics

### What v9 computes
- IoU (tampered-only and mixed-set)
- Dice / F1 (tampered-only and mixed-set)
- Boundary F1 with 2-pixel dilation tolerance

### Boundary F1 implementation

```python
def boundary_f1_score(pred_mask, gt_mask, tolerance=2):
    pred_boundary = find_boundaries(pred_mask.astype(bool), mode="inner")
    gt_boundary = find_boundaries(gt_mask.astype(bool), mode="inner")
    structure = np.ones((2*tolerance+1, 2*tolerance+1), dtype=bool)
    pred_dilated = binary_dilation(pred_boundary, structure=structure)
    gt_dilated = binary_dilation(gt_boundary, structure=structure)
    ...
```

This is a correct implementation of the soft boundary metric used in instance segmentation literature (similar to the Boundary IoU from Cheng et al., 2021). The 2-pixel tolerance is reasonable for 384×384 images.

**Issue:** Boundary F1 degenerates when the predicted mask is empty (model predicts no tampered pixels). The code handles this with:
```python
if pred_boundary.sum() == 0:
    precision = 1.0 if gt_boundary.sum() == 0 else 0.0
```

Precision = 1.0 when both predicted and ground-truth boundary are empty is correct for authentic images but will silently produce precision=1.0 for tampered images where the model completely failed to predict a mask. This will inflate precision in failure cases. The Boundary-F1 for "model predicts nothing" cases should be 0.0, not 1.0.

---

## 2. Image-Level Detection

### v9 approach
Learned classification head → sigmoid → threshold scanning for best image-level F1.

This is the right approach. The assignment requires image-level detection. A learned head is substantially better than `mask.max()`.

### Problems

**Problem 1 — Threshold is selected on validation, applied to test.**  
This is the correct methodology. However, the threshold grid `0.05 → 0.80` in steps of `0.05` gives 15 candidate thresholds. At batch size 4, the validation set has meaningful statistical noise. There is no cross-validation of the threshold. The selected threshold could be overfitted to the validation set.

**Problem 2 — Heuristic score is still reported alongside learned score.**  
The code computes both `cls_prob` (learned head) and `mask_max_prob` (heuristic). Both are embedded in `per_image_df`. This is useful for comparison, but the evaluation report needs to be explicit about which one is used for the primary detection metric. The v9 notebook is not explicit about this. If the heuristic happens to perform better on the specific val split, will it be used instead? The logic is unclear.

---

## 3. PR Curves

v9 includes precision-recall curve computation using `precision_recall_curve` from sklearn.

**Issue:** PR curves on highly imbalanced datasets where authentic images have all-zero true masks can produce misleading curve shapes. The macro-averaged PR AUC for segmentation can look good even when copy-move performance is terrible, if the splicing half of the test set performs well. v9's code computes PR curves but the interpretation guidance is absent.

---

## 4. Mask Randomization Test

### Design
Shuffle the test set masks, assign them to the wrong images, re-evaluate. If metrics stay high, the model is detecting image-texture patterns rather than the correct tampering regions.

### What this proves — and what it does not

This test correctly identifies the binary question: are predictions coupled to image content?

What it does not prove:
- The model is NOT using dataset-specific artifacts (e.g., CASIA always uses the same JPEG quality factor, so ELA features could correlate with "was saved in a CASIA-standard way" rather than "was tampered with")
- The model generalises to unseen manipulation types
- The model does not exploit systematic mask-image positional biases in CASIA (tampered regions in CASIA are often in predictable positions — central, foreground)

The notebook comments on this correctly in the narrative but the implementation summary does NOT acknowledge the limitations listed above. A reviewer reading just the results table would think the randomization test validates forensic learning. It does not.

---

## 5. Robustness Suite

The suite applies 8 degradation conditions: clean, jpeg_qf70, jpeg_qf50, gaussian_noise_light, gaussian_noise_heavy, gaussian_blur, resize_0.75x, resize_0.5x.

This is the same test suite as v8. Good — it enables comparison.

### Problem: No comparison against v8 numbers

v8 run-01 produced actual robustness numbers. v9 defines the same suite but has no baseline execution. The robustness suite only adds value if the v9 numbers can be compared against v8 to demonstrate that the new training pipeline actually improved robustness. As currently structured, the suite will produce a table of numbers that exist in isolation, with no prior to compare against.

---

## 6. Per-Forgery-Type Evaluation

The notebook tracks metrics separated by `forgery_type` (splicing vs copy-move). This is valuable for honest calibration reporting.

### The only forthright call in v9's documentation

The docs correctly note that copy-move F1 was 0.31 in the v8 run-01, which is genuinely terrible. v9 acknowledges this is likely to remain weak even after the improvements. That honesty is correct.

**But the evaluation only demonstrates this honesty if the notebook is executed.** Without a run, saying "we expect copy-move to remain weak" is neither honest nor dishonest. It is just text.

---

## 7. Evaluation Compared to v8

| Evaluation Component | v8 run-01 | v9 Colab | Status |
|---------------------|-----------|----------|--------|
| Tampered-only primary metrics | ✅ | ✅ (code) | Tie |
| Boundary F1 | ❌ | ✅ (code) | Improvement (code only) |
| PR curves | ❌ | ✅ (code) | Improvement (code only) |
| Learned classification metrics | ❌ | ✅ (code) | Improvement (code only) |
| Mask randomization test | ❌ | ✅ (code) | Improvement (code only) |
| Robustness suite | ✅ Executed | ✅ (code) | v8 has evidence, v9 does not |
| Actual numeric results | ✅ Present | ❌ Absent | **REGRESSION** |
| Accuracy/AUC numbers in notebook | ✅ Present | ❌ Absent | **REGRESSION** |

The evaluation design improved. The evaluation output regressed completely to nothing.
