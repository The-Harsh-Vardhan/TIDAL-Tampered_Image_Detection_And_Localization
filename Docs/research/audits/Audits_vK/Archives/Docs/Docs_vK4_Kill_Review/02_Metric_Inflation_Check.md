# 02 — Metric Inflation Check

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Finding 1: Mixed-Set F1/IoU Inflation — 🔴 CRITICAL

The primary reported metrics (`pixel_f1_mean`, `pixel_iou_mean`) in `evaluate_test()` (Cell 31) are computed over **ALL** test samples, including **authentic images** (label=0).

For authentic images:
- Ground-truth mask = all zeros (no tampering)
- If the model correctly predicts all zeros → `compute_pixel_f1` returns **1.0**

This means every correctly-classified authentic image contributes a **perfect F1 score of 1.0** to the average. If 50% of the test set is authentic and the model gets them all right, the "mixed" F1 is artificially pulled up by ~0.5 even if tampered localization is mediocre.

**The tampered-only F1 (`tampered_f1_mean`) is the only honest localization metric**, but it's reported secondary and the "all" metric appears first in the results printout.

> **Impact:** A mixed-set F1 of 0.75 could mean tampered-only F1 of only 0.50. This is textbook metric inflation.

## Finding 2: Batch-Averaged vs Sample-Averaged Validation Metrics — ⚠️ MEDIUM

The `validate()` function (Cell 24) computes `dice_coef_batch`, `iou_coef_batch`, `f1_coef_batch` per-batch and then averages across batches (`dice_sum / num_batches`). This is **not** the same as per-sample averaging if batch sizes vary (last incomplete batch). With `drop_last=False` on val/test loaders, the last batch is likely smaller, giving it disproportionate weight.

## Finding 3: Threshold Tuned on Validation — ✅ CORRECT

The threshold sweep (Cell 30) is correctly performed on the **validation set**, and the selected threshold is then applied to the **test set**. This is proper methodology. No leakage here.

## Finding 4: `compute_pixel_f1` Edge Case — ⚠️ LOW

When both `gt` and `pred` are all zeros, `compute_pixel_f1` returns 1.0 (line 761). This is a defensible design choice (true negative = perfect), but it inflates the mixed-set metric as described in Finding 1.

## Finding 5: No Confidence Intervals — ⚠️ LOW

All metrics are reported as point estimates. No bootstrap confidence intervals, no std across samples. This makes it impossible to judge statistical significance.

---

## Verdict

**The primary reported "all" metrics are inflated by authentic samples scoring perfect F1/IoU.** The tampered-only F1 is the honest metric but is buried in secondary position.

**Severity: HIGH** — The most prominent metric is misleading.
